# 2025/7/19
# zhangzhong

from xihe.dataset import (
    PackingDataset,
    create_dataset,
    calculate_sampling_probabilities,
)
from typing import Any
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
import os
from datasets import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

# https://github.com/huggingface/datasets/issues/5360
# 还好搜了下，有现成的
from datasets.distributed import split_dataset_by_node
import random


def test_create_dataset() -> None:
    dataset = create_dataset(
        path="wikimedia/wikipedia",
        name="20231101.en",
        split="train[:1024]",
    )
    print(dataset)


# 我还想实现一个功能
# 就是这些dataset builder可以保存在本地
# 就保存在一个cache目录里面就好了
# 这样可以节省下载的时间，节省跑单元测试的时间
def test_calculate_sampling_probabilities() -> None:
    wikipedia: dict[str, Any] = {
        "path": "wikimedia/wikipedia",
        "name": "20231101.en",
        "num_epochs": 2,
    }
    c4: dict[str, Any] = {
        "path": "allenai/c4",
        "name": "en",
        "num_epochs": 1,
    }
    sample_probabilities: list[float] = calculate_sampling_probabilities(
        pathes=[dataset["path"] for dataset in [wikipedia, c4]],
        names=[dataset["name"] for dataset in [wikipedia, c4]],
        num_epochs=[dataset["num_epochs"] for dataset in [wikipedia, c4]],
    )
    print(sample_probabilities)


# 需要测试两个东西，一个是streaming + pytorch.dataset 的结合
# 另外一个就是distributed 问了chatgpt


def test_distributed_sampler() -> None:
    print("Testing DistributedSampler...", flush=True)
    # must create process group to use distributed sampler
    rank = 0
    # must set the MASTER_ADDR and MASTER_PORT environment variables
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # PyTorch’s distributed initialization is a blocking operation:
    # Each process waits until all other processes in the group have joined.
    # does every process need to call the init_process_group?
    # 	1.	Registers the process with the collective communication backend (nccl, gloo, etc.)
    # 	2.	Establishes communication among all processes in the group
    # 	3.	Synchronizes group metadata, like rank, world_size, and connection info

    # If even one process doesn’t call it, all others will hang waiting for it — since the group is incomplete.
    torch.distributed.init_process_group(backend="gloo", rank=rank, world_size=1)
    # torch.cuda.set_device(rank)

    print(f"Initialized process group with rank {rank}", flush=True)

    # Test the DistributedSampler with a PackingDataset
    wikipedia: dict[str, Any] = {
        "path": "wikimedia/wikipedia",
        "name": "20231101.en",
        "split": "train[:1024]",
        # "num_epochs": 2,
    }
    c4: dict[str, Any] = {
        "path": "allenai/c4",
        "name": "en",
        "split": "train[:1024]",
        # "num_epochs": 1,
    }

    packing_dataset = PackingDataset(
        datasets=[
            create_dataset(**wikipedia),
            create_dataset(**c4),
        ],
        tokenizer=AutoTokenizer.from_pretrained("gpt2"),
        sampling_probabilities=[0.5, 0.5],
    )
    dataset = packing_dataset.to_torch_dataset(
        batch_size=8,
        context_length=1024,
    )
    # 为了测试，显式的设置world_size为4
    # 卧槽！牛逼！
    dataset = split_dataset_by_node(
        dataset,
        rank=rank,
        world_size=4,
    )
    dataset = dataset.with_format("torch")
    # Sampler that restricts data loading to a subset of the dataset.
    # https://discuss.pytorch.org/t/distributedsampler/90205
    # indices = indices[self.rank:self.total_size:self.num_replicas]
    # It is especially useful in conjunction with torch.nn.parallel.DistributedDataParallel.
    # and load a subset of the original dataset that is exclusive to it.
    # num_replicas: Number of processes participating in distributed training. By default, world_size is retrieved from the current distributed group.
    # 但是我们这里为了测试，显式的设置为4
    # TypeError: object of type 'IterableDataset' has no len()
    # ！！！果然是这样，所以没法用IterableDataset，
    # 不过，DistSampler也只是让不同的进程去读不同的子集而已
    # 只要我们的iterable dataset支持shard，也可以自己包装成一个类似的东西
    # 就是不用这个dist sampler了
    # sampler = DistributedSampler(dataset, num_replicas=4, rank=rank)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=8,
        # sampler=sampler,
    )
    for batch in dataloader:
        input_ids = batch["input_ids"]
        assert input_ids.shape == (8, 1024)
        break  # Only test one batch for simplicity

    torch.distributed.destroy_process_group()


# test dataloader resume
# https://huggingface.co/docs/datasets/stream#save-a-dataset-checkpoint-and-resume-iteration
def test_iterable_dataset_resume() -> None:
    print("Testing dataloader resume...", flush=True)

    iterable_dataset = Dataset.from_dict({"a": range(6)}).to_iterable_dataset(
        num_shards=3
    )
    for idx, example in enumerate(iterable_dataset):
        print(example)
        if idx == 2:
            state_dict = iterable_dataset.state_dict()
            print("checkpoint")
            break

    iterable_dataset = Dataset.from_dict({"a": range(6)}).to_iterable_dataset(
        num_shards=3
    )
    iterable_dataset.load_state_dict(state_dict)
    print("restart from checkpoint")
    for example in iterable_dataset:
        print(example)
    # This is a placeholder for the actual test implementation
    # You would typically create a DataLoader and simulate a resume scenario
    # For example, by saving the state of the DataLoader and then resuming it
    # after some interruption.
    pass  # Implement your test logic here


# 全面转成iterable的dataset吧
# 这样不需要等待数据集下完，也可以进行训练
#


def test_stateful_dataloader() -> None:
    iterable_dataset = Dataset.from_dict({"a": range(64)}).to_iterable_dataset(
        num_shards=4
    )
    dataloader = StatefulDataLoader(iterable_dataset, batch_size=4, num_workers=4)

    state_dict: dict[str, Any] = {}
    for idx, batch in enumerate(dataloader):
        print(f"Batch {idx}: {batch}")
        if idx == 3:
            state_dict = dataloader.state_dict()
            print("checkpoint")
            break

    dataloader = StatefulDataLoader(iterable_dataset, batch_size=4, num_workers=4)
    dataloader.load_state_dict(state_dict)
    print("restart from checkpoint")
    for idx, batch in enumerate(dataloader):
        print(f"Batch {idx}: {batch}")
        if idx == 3:
            break

    # 分析一下这个输出
    # 感觉这个stateful dataloader是可以完成我们的需求的
    #


# 写一个随机文本生成器作为测试数据集


# 想一下什么样的数据集比较容易测试
# 随机指定一个长度 min max
# 第一个样本就是A0 A0 A0 A0 A0
# 第二个样本就是A1 A1 A1 A1 A1
# 这样吧
def gen_random_text_dataset(
    prefix: str, min_len: int, max_len: int, size: int
) -> list[str]:
    random.seed(42)  # For reproducibility
    dataset = []

    for i in range(size):
        length = random.randint(min_len, max_len)
        text = f"{prefix}" + (f"{i}" * length)
        dataset.append(text.strip())

    return dataset


# ok! 完全弄懂了！
def test_packing_dataset() -> None:
    # 好像只要dataset包含text这一个列就可以作为packding dataset的参数了

    dsa = gen_random_text_dataset("A", 5, 10, 8)
    print(dsa)

    dsA = Dataset.from_dict(
        {"text": gen_random_text_dataset("A", 5, 10, 1000)}
    ).to_iterable_dataset(num_shards=4)
    dsB = Dataset.from_dict(
        {"text": gen_random_text_dataset("B", 5, 10, 1023)}
    ).to_iterable_dataset(num_shards=4)
    dsC = Dataset.from_dict(
        {"text": gen_random_text_dataset("C", 5, 10, 839)}
    ).to_iterable_dataset(num_shards=4)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    packing_dataset = PackingDataset(
        datasets=[dsA, dsB, dsC],
        tokenizer=tokenizer,
        sampling_probabilities=[0.5, 0.3, 0.2],
    )

    # 我们想要测试没有经过tokenizer的数据流
    #
    interleaved_dataset = packing_dataset.get_interleaved_dataset()
    print("interleaved dataset: ", interleaved_dataset)

    for idx, example in enumerate(interleaved_dataset):
        print(f"Example {idx}: {example}")
        if idx == 5:
            break

    tokenized_dataset = packing_dataset.get_tokenized_dataset()
    print("tokenized dataset: ", tokenized_dataset)

    for idx, example in enumerate(tokenized_dataset):
        print(f"Tokenized Example {idx}: {example}")
        if idx == 5:
            break

    # 到了这里 shard仍然有4个
    # 这意味着我们是可以做split by node的
    # 看看效果吧
    context_length = 8
    packed_dataset = packing_dataset.get_packed_dataset(context_length=context_length)
    print("packed dataset: ", packed_dataset)
    for idx, example in enumerate(packed_dataset):
        print(f"Packed Example {idx}: {example}")
        if idx == 5:
            break

    # 我要弄清楚这个管线到底是怎么工作的！
    # 怎么日志还输出了好多次呢？
    # 关键应该在map上！
    # 我们设置了map的batched=True
    # map还有一个默认参数是batch_size
    # 这意味着每次都会读取batch_size个example
    # 然后调用map函数，也就是tokenize和packing
    # 这就解释了为什么会有多次输出

    # 我们把数据集split一下 看看输出的都是什么

    world_size = 4
    splited_ds0 = packing_dataset.get_splited_dataset(
        context_length=context_length, rank=0, world_size=world_size
    )
    # 现在他的shard就只有1了
    print("Splitted Dataset for rank 0: ", splited_ds0)
    for idx, example in enumerate(splited_ds0):
        print(f"Splitted Example {idx}: {example}")
        print(
            f"Decoded Example {idx}: {tokenizer.decode(example['input_ids'], skip_special_tokens=True)}"
        )

    splited_ds1 = packing_dataset.get_splited_dataset(
        context_length=context_length, rank=1, world_size=world_size
    )
    print("Splitted Dataset for rank 1: ", splited_ds1)
    for idx, example in enumerate(splited_ds1):
        print(f"Splitted Example {idx}: {example}")
        # decode ids
        print(
            f"Decoded Example {idx}: {tokenizer.decode(example['input_ids'], skip_special_tokens=True)}"
        )

    # 我如果可以把这些inputids 都变回文字 会好一些
    # 可以看到到底发生了什么

    # 怎么把print重定向到文件里面呢
    # 这些测试的输出太长了 被vscode给truncate了
    # uv run pytest -s -k test_packing_dataset > test_packing_dataset.txt
    # ok!
    # 然后用dataloader来获取batch 就ok了
    # TODO: 把这个封装成get_dataloader
    # 然后把这里的代码改成调用这个函数
    print("Using DataLoader to get batches from rank 2...")
    # splited_ds2 = packing_dataset.get_splited_dataset(
    #     context_length=context_length, rank=2, world_size=world_size
    # ).with_format("torch")
    # dataloader = StatefulDataLoader(
    #     splited_ds2,
    #     batch_size=4,
    #     drop_last=True,  # 如果最后一个batch不满4个就丢弃
    # )
    dataloader = packing_dataset.to_stateful_dataloader(
        batch_size=4,
        context_length=context_length,
        rank=2,
        world_size=world_size,
    )
    for idx, batch in enumerate(dataloader):
        assert batch["input_ids"].shape == (4, context_length)
        print(f"Batch {idx}: {batch}")
        print(
            f"Decoded Batch {idx}: {[tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['input_ids']]}"
        )
