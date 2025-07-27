# 2025/7/17
# zhangzhong

# 我需要写一个可以按照一定的比例，混合多个数据集
# 带有shuffle
# 我们先用GPT2自带的tokenizer吧
# 之后再换成我们自己train出来的


from datasets import (
    load_dataset,
    Dataset,
    IterableDataset,
    interleave_datasets,
    load_dataset_builder,
    DatasetBuilder,
)

from enum import StrEnum
import multiprocessing
from torch.utils.data import DataLoader
from transformers.tokenization_utils import PreTrainedTokenizer
from datasets.distributed import split_dataset_by_node
from torchdata.stateful_dataloader import StatefulDataLoader


# tokenizer 应该有外部传入 减少依赖
# 不过，tokenizer的类型是什么呢
class PackingDataset:
    def __init__(
        self,
        datasets: list[Dataset | IterableDataset],
        tokenizer: PreTrainedTokenizer,
        sampling_probabilities: list[float],
        shuffle=True,
    ):
        # self.dataset_configs = dataset_configs
        self.datasets = datasets
        self.sampling_probabilities = sampling_probabilities
        # ratios 可以根据DatasetConfig计算出来
        # if sample_probabilities is None:
        #     self.sample_probabilities = calculate_sampling_probabilities(
        #         dataset_configs=dataset_configs, dataset_infos=dataset_infos
        #     )
        # else:
        #     self.sample_probabilities = sample_probabilities
        if sum(self.sampling_probabilities) != 1.0:
            raise ValueError("Ratios must sum to 1.0")

        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        # self.batch_size = batch_size

        self.eos_token_id = self.tokenizer.encode(tokenizer.eos_token)[0]
        # 检查比例是否正确

        # 咱们就用一半的核心吧
        self.num_procs = multiprocessing.cpu_count() // 2

    # 首先混合所有的数据集，并提出去所有的需要做预训练的文本到text这个字段

    def tokenize(self, examples):
        return self.tokenizer(examples["text"])

    def pack(self, examples, context_length: int):
        # 这里的examples是一个batch
        # 我们需要把每个example的tokens拼接起来
        # 然后切分为1024的长度
        # packed_texts = []
        # for text in examples["input_ids"]:
        #     packed_text = tokenizer.eos_token.join()
        #     packed_texts.append(packed_text)
        # list of int 要怎么做join ？
        packed_input_ids: list[int] = []
        for input_ids in examples["input_ids"]:
            packed_input_ids.extend(input_ids)
            packed_input_ids.append(self.eos_token_id)  # 添加eos token
        # 不对，没必要这么处理，加上了就加上了呗
        # packed_texts.append(examples["input_ids"][-1])  # 最后一个input_ids不需要添加eos token
        print(f"Packed input IDs length: {len(packed_input_ids)}")
        # 然后切分成1024的长度
        # 最后一行，如果不足1024，就不要了。
        cutted_packed_input_ids: list[list[int]] = []
        for i in range(0, len(packed_input_ids), context_length):
            cutted_packed_input_ids.append(packed_input_ids[i : i + context_length])
        # 去掉最后一个
        if len(cutted_packed_input_ids[-1]) < context_length:
            cutted_packed_input_ids.pop(-1)
        # cutted_packed_input_ids.pop(-1)  # 最后一行如果不足1024，就不要了。
        # 然后统计一下cutted_packed_input_ids的token数量
        total_tokens = sum(len(ids) for ids in cutted_packed_input_ids)
        print(f"Total tokens after packing: {total_tokens}")
        print(f"Number of packed sequences: {len(cutted_packed_input_ids)}")
        # 这里返回的就是一个batch

        return {"packed_input_ids": cutted_packed_input_ids}

    def get_interleaved_dataset(self):
        interleaved_dataset = interleave_datasets(
            datasets=self.datasets,  # type: ignore
            probabilities=self.sampling_probabilities,
            # TODO: 这个种子是必须设置的
            # 否则我们checkpoint之后，拿到的数据可能会不一样
            seed=42,  # 设置随机种子以确保可重复性
            stopping_strategy="all_exhausted",  # 当所有数据集都被耗尽时停止
        )
        return interleaved_dataset

    def get_tokenized_dataset(self):
        interleaved_dataset = self.get_interleaved_dataset()
        tokenized_dataset = interleaved_dataset.map(
            self.tokenize, batched=True, remove_columns=["text"]
        )
        return tokenized_dataset

    def get_packed_dataset(self, context_length: int):
        tokenized_dataset = self.get_tokenized_dataset()
        packed_dataset = tokenized_dataset.map(
            self.pack,
            batched=True,
            remove_columns=["input_ids", "attention_mask"],
            # https://huggingface.co/docs/datasets/v4.0.0/en/package_reference/main_classes#datasets.Dataset.map.fn_kwargs
            fn_kwargs={"context_length": context_length},
        )
        packed_dataset = packed_dataset.rename_column("packed_input_ids", "input_ids")
        return packed_dataset

    def get_splited_dataset(self, context_length: int, rank: int, world_size: int):
        packed_dataset = self.get_packed_dataset(context_length=context_length)
        # 这里我们需要把数据集分成world_size份
        # 每一份的rank是rank
        splited_dataset = split_dataset_by_node(
            packed_dataset,
            rank=rank,
            world_size=world_size,
        )
        return splited_dataset

    # len 和 getitem不需要自己写
    # 我们就写一个get_dataset就行了
    # TODO: maybe refactor this function for better testing
    def to_torch_dataloader(self, batch_size: int, context_length: int) -> DataLoader:
        # 创建一个空的Dataset
        # TODO
        # 然后根据load所有的数据集
        # load的过程很慢，可以异步加载？
        # 不过先不急

        # interleave所有的数据集
        # 并且设置采样率
        # 不过这个采样率要怎么设置才正确呢？
        # 采样率弄一个大概就行了
        # 因为数据集的大小是固定的
        # 这个可以作为一个参数和数据集保存起来
        # 比较好的方式是定义一个枚举，把数据集的一些信息保存起来

        # TODO:
        # shuffle是有必要的吗？
        # 先不用了

        interleaved_dataset = interleave_datasets(
            datasets=self.datasets,  # type: ignore
            probabilities=self.sampling_probabilities,
            seed=42,  # 设置随机种子以确保可重复性
            stopping_strategy="all_exhausted",  # 当所有数据集都被耗尽时停止
        )

        tokenized_dataset = interleaved_dataset.map(
            self.tokenize, batched=True, remove_columns=["text"]
        )

        packed_dataset = tokenized_dataset.map(
            self.pack,
            batched=True,
            remove_columns=["input_ids", "attention_mask"],
            # https://huggingface.co/docs/datasets/v4.0.0/en/package_reference/main_classes#datasets.Dataset.map.fn_kwargs
            fn_kwargs={"context_length": context_length},
        )
        packed_dataset = packed_dataset.rename_column("packed_input_ids", "input_ids")

        return DataLoader(
            dataset=packed_dataset.with_format("torch"),  # type: ignore
            batch_size=batch_size,
        )

    def to_torch_dataset(self, context_length: int):
        interleaved_dataset = interleave_datasets(
            datasets=self.datasets,  # type: ignore
            probabilities=self.sampling_probabilities,
            seed=42,  # 设置随机种子以确保可重复性
            stopping_strategy="all_exhausted",  # 当所有数据集都被耗尽时停止
        )

        # batched ！？
        tokenized_dataset = interleaved_dataset.map(
            self.tokenize, batched=True, remove_columns=["text"]
        )

        packed_dataset = tokenized_dataset.map(
            self.pack,
            batched=True,
            remove_columns=["input_ids", "attention_mask"],
            # https://huggingface.co/docs/datasets/v4.0.0/en/package_reference/main_classes#datasets.Dataset.map.fn_kwargs
            fn_kwargs={"context_length": context_length},
        )
        packed_dataset = packed_dataset.rename_column("packed_input_ids", "input_ids")
        # return packed_dataset.with_format("torch")
        return packed_dataset

    def to_distributed_dataset(self, context_length, rank: int, world_size: int):
        dataset = self.to_torch_dataset(context_length)
        dataset = split_dataset_by_node(
            dataset,
            rank=rank,
            world_size=world_size,
        )
        return dataset.with_format("torch")

    def to_stateful_dataloader(
        self, batch_size: int, context_length: int, rank: int, world_size: int
    ) -> StatefulDataLoader:
        dataset = self.get_splited_dataset(
            context_length=context_length, rank=rank, world_size=world_size
        ).with_format("torch")
        # dataset = self.to_distributed_dataset(
        #     context_length=context_length, rank=rank, world_size=world_size
        # )
        return StatefulDataLoader(
            dataset=dataset, batch_size=batch_size, drop_last=True
        )
