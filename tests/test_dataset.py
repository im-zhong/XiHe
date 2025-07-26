# 2025/7/19
# zhangzhong

from xihe.dataset import (
    PackingDataset,
    DatasetEnum,
    create_dataset,
    calculate_sampling_probabilities,
)
from typing import Any
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, DistributedSampler
import torch
import os
from datasets import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

# https://github.com/huggingface/datasets/issues/5360
# 还好搜了下，有现成的
from datasets.distributed import split_dataset_by_node


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
    print(f"restart from checkpoint")
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
    print(f"restart from checkpoint")
    for idx, batch in enumerate(dataloader):
        print(f"Batch {idx}: {batch}")
        if idx == 3:
            break

    # 分析一下这个输出
    # 感觉这个stateful dataloader是可以完成我们的需求的
    #
