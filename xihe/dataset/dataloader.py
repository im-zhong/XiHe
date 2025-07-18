# 2025/7/17
# zhangzhong

# 我需要写一个可以按照一定的比例，混合多个数据集
# 带有shuffle
# 我们先用GPT2自带的tokenizer吧
# 之后再换成我们自己train出来的


from datasets import load_dataset, Dataset, IterableDataset, interleave_datasets

from enum import StrEnum
import multiprocessing
from torch.utils.data import DataLoader


# 一共有三个东西！
# path, name, split
# 这个东西的名字直接定义成dataset的名字吧，更方便一点
class DatasetEnum(StrEnum):
    C4 = "allenai/c4"
    WIKIPEDIA = "wikimedia/wikipedia"
    TINY_CODES = "nampdn-ai/tiny-codes"
    GUTENBERG = "eminorhan/gutenberg_en"
    STACK_EXCHANGE = "donfu/oa-stackexchange"
    ARXIV = "common-pile/arxiv_papers"


# 定义一个类包含dataset的信息
class DatasetInfo:
    def __init__(
        self, name: str | None, size: int, preprocess_fn, features: list[str] = []
    ):
        self.name = name
        self.size = size
        self.features = features
        self.preprocess_fn = preprocess_fn  # 预处理函数


class DatasetConfig:
    def __init__(
        self,
        name: DatasetEnum,
        split: str,
        num_epochs: int,
        # sample_probability: float,
        to_iterable: bool = True,
    ) -> None:
        self.name = name
        self.split = split
        self.num_epochs = num_epochs
        # self.sample_probability: float = sample_probability
        self.to_iterable: bool = to_iterable


# TODO；
# 这个东西在训练下一个模型的时候肯定会变
# 不过具体怎么变我也没有办法预知
# 所以现在就先这么写吧
# TODO: 数据集的大小需要正确的设置一下
dataset_infos: dict[DatasetEnum, DatasetInfo] = {
    # https://huggingface.co/datasets/allenai/c4
    DatasetEnum.C4: DatasetInfo(
        name="en",
        size=1000000,
        preprocess_fn=lambda examples: {
            "text": [text.strip() for text in examples["text"]]
        },
    ),
    # https://huggingface.co/datasets/wikimedia/wikipedia
    DatasetEnum.WIKIPEDIA: DatasetInfo(
        name="20231101.en",
        size=500000,
        preprocess_fn=lambda examples: {
            "text": [text.strip() for text in examples["text"]]
        },
    ),
    # https://huggingface.co/datasets/nampdn-ai/tiny-codes
    DatasetEnum.TINY_CODES: DatasetInfo(
        name=None,
        size=200000,
        preprocess_fn=lambda examples: {
            "text": [
                f"{prompt} {response}".strip()
                for prompt, response in zip(examples["prompt"], examples["response"])
            ]
        },
    ),
    # https://huggingface.co/datasets/eminorhan/gutenberg_en
    DatasetEnum.GUTENBERG: DatasetInfo(
        name="chunk_size_1024",
        size=300000,
        preprocess_fn=lambda examples: {
            "text": [text.strip() for text in examples["text"]]
        },
    ),
    # https://huggingface.co/datasets/donfu/oa-stackexchange
    DatasetEnum.STACK_EXCHANGE: DatasetInfo(
        name=None,
        size=150000,
        preprocess_fn=lambda examples: {
            "text": [
                f"{instruction} {response}".strip()
                for instruction, response in zip(
                    examples["INSTRUCTION"], examples["RESPONSE"]
                )
            ]
        },
    ),
    # https://huggingface.co/datasets/common-pile/arxiv_papers
    DatasetEnum.ARXIV: DatasetInfo(
        name=None,
        size=400000,
        preprocess_fn=lambda examples: {
            "text": [text.strip() for text in examples["text"]]
        },
    ),
}


# 不对哎，这个函数没法考虑split这个参数的影响。。。
# 哎，有点烦，算了，不管了
def calculate_sampling_probabilities(
    dataset_configs: list[DatasetConfig], dataset_infos: dict[DatasetEnum, DatasetInfo]
) -> list[float]:
    # 返回的长度应该和dataset_configs的长度一致
    total_size: int = sum(
        dataset_infos[dataset.name].size for dataset in dataset_configs
    )
    probabilities: list[float] = [
        dataset_infos[dataset.name].size / total_size for dataset in dataset_configs
    ]
    return probabilities


# tokenizer 应该有外部传入 减少依赖
# 不过，tokenizer的类型是什么呢
class PackingDataset:
    def __init__(
        self,
        dataset_configs: list[DatasetConfig],
        tokenizer,
        sample_probabilities: list[float] | None = None,
        shuffle=True,
    ):
        self.dataset_configs = dataset_configs

        # ratios 可以根据DatasetConfig计算出来
        if sample_probabilities is None:
            self.sample_probabilities = calculate_sampling_probabilities(
                dataset_configs=dataset_configs, dataset_infos=dataset_infos
            )
        else:
            self.sample_probabilities = sample_probabilities
        if sum(self.sample_probabilities) != 1.0:
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
        # cutted_packed_input_ids.pop(-1)  # 最后一行如果不足1024，就不要了。
        # 然后统计一下cutted_packed_input_ids的token数量
        total_tokens = sum(len(ids) for ids in cutted_packed_input_ids)
        print(f"Total tokens after packing: {total_tokens}")
        print(f"Number of packed sequences: {len(cutted_packed_input_ids)}")
        # 这里返回的就是一个batch

        return {"packed_input_ids": cutted_packed_input_ids}

    # 现在只需要实现一个函数
    # 根据config中的结果，返回一个数据集就行了
    # 可以先实现一个简单的函数
    # 根据一个DatasetConfig返回一个Dataset
    def get_dataset_from_config(
        self, dataset_config: DatasetConfig
    ) -> Dataset | IterableDataset:
        # 根据DatasetConfig加载数据集
        dataset_info: DatasetInfo = dataset_infos[dataset_config.name]
        dataset = load_dataset(
            path=dataset_config.name,
            name=dataset_info.name,
            split=dataset_config.split,
        )
        assert isinstance(dataset, Dataset), "Loaded dataset is not a Dataset instance"
        # 需不需要转成iterable dataset?
        # 这个应该由一个参数来控制

        # 不同的数据集有不同的预处理方式
        # 应该由一个函数提供

        # 预处理数据集
        # 要不预处理这一步，不变成iterable了吧
        preprocessed_dataset = dataset.map(
            function=dataset_info.preprocess_fn,
            batched=True,
            batch_size=4096,
            # TODO: when debugging, do not use multiprocessing
            # num_proc=self.num_procs,  # 使用多核处理
            remove_columns=dataset.column_names,
        )

        if dataset_config.to_iterable:
            return preprocessed_dataset.to_iterable_dataset()
        return preprocessed_dataset

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

        datasets = [
            self.get_dataset_from_config(dataset_config)
            for dataset_config in self.dataset_configs
        ]

        interleaved_dataset = interleave_datasets(
            datasets=datasets,  # type: ignore
            probabilities=self.sample_probabilities,
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
