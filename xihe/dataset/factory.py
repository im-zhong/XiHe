# 2025/7/26
# zhangzhong

from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils import PreTrainedTokenizer
from .dataset import create_dataset, calculate_sampling_probabilities
from .dataloader import PackingDataset
from xihe.schemas import DatasetArgs


# 最好把对config的依赖给去掉
def create_dataloader(
    tokenizer: PreTrainedTokenizer,
    rank: int,
    batch_size: int,
    context_length: int,
    world_size: int,
    datasets_args: list[DatasetArgs],
    sampling_probabilities: list[float] | None = None,
) -> StatefulDataLoader:
    datasets = [
        create_dataset(
            path=dataset.path,
            name=dataset.name,
            split=dataset.split,
            streaming=dataset.streaming,
            num_shards=world_size,
        )
        for dataset in datasets_args
    ]

    if sampling_probabilities is None:
        # sampling_probabilities: list[float] = []
        # if config.dataloader.sampling_probabilities:
        #     sampling_probabilities = config.dataloader.sampling_probabilities
        # else:
        # 如果没有提供采样概率，就计算一下
        sampling_probabilities = calculate_sampling_probabilities(
            pathes=[dataset.path for dataset in datasets_args],
            names=[dataset.name for dataset in datasets_args],
            num_epochs=[dataset.num_epochs for dataset in datasets_args],
        )

    # # Initialize dataset and dataloader
    dataset = PackingDataset(
        datasets=datasets,
        tokenizer=tokenizer,
        sampling_probabilities=sampling_probabilities,
    )
    dataloader = dataset.to_stateful_dataloader(
        batch_size=batch_size,
        context_length=context_length,
        rank=rank,
        world_size=world_size,
    )
    return dataloader
