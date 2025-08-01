# 2025/7/19
# zhangzhong

# 我需要写一个可以按照一定的比例，混合多个数据集
# 带有shuffle
# 我们先用GPT2自带的tokenizer吧
# 之后再换成我们自己train出来的

# TODO: 把datasets.ipynb里面的两个函数加到这里面，应该就可以通过测试了


from collections.abc import Callable
from enum import StrEnum
from typing import Any

from datasets import (
    Dataset,
    IterableDataset,
    load_dataset,
    load_dataset_builder,
)
from datasets import (
    DatasetInfo as HDatasetInfo,
)


# 一共有三个东西！
# path, name, split
# 这个东西的名字直接定义成dataset的名字吧，更方便一点
class DatasetEnum(StrEnum):
    C4 = "allenai/c4"  # 这个东西好像不叫name来着，叫path
    WIKIPEDIA = "wikimedia/wikipedia"
    TINY_CODES = "nampdn-ai/tiny-codes"
    GUTENBERG = "eminorhan/gutenberg_en"
    STACK_EXCHANGE = "donfu/oa-stackexchange"
    ARXIV = "common-pile/arxiv_papers"


# TODO: 这里的名字和datasets里面的名字重复了
# 定义一个类包含dataset的信息
class DatasetInfo:
    def __init__(
        self,
        # name: str | None,
        preprocess_fn: Callable[..., dict[str, Any]],
        size: float | None = None,
        # features: list[str] = [],
    ) -> None:
        # self.name = name
        # self.size = size
        # self.features = features
        self.size: float | None = size
        self.preprocess_fn = preprocess_fn  # 预处理函数


# class DatasetConfig:
#     def __init__(
#         self,
#         name: DatasetEnum,
#         split: str,
#         num_epochs: int,
#         # sample_probability: float,
#         to_iterable: bool = True,
#     ) -> None:
#         self.name = name
#         self.split = split
#         self.num_epochs = num_epochs
#         # self.sample_probability: float = sample_probability
#         self.to_iterable: bool = to_iterable


# TODO；
# 这个东西在训练下一个模型的时候肯定会变
# 不过具体怎么变我也没有办法预知
# 所以现在就先这么写吧
# TODO: 数据集的大小需要正确的设置一下

# 采样率还是应该自己计算出来
# 除非在配置中直接提供
# 而预处理函数，应该对于同一个数据集的所有name都是一样的才对
# 所以用dataset path作为key是合适的
dataset_infos: dict[DatasetEnum, DatasetInfo] = {
    # https://huggingface.co/datasets/allenai/c4
    DatasetEnum.C4: DatasetInfo(
        # name="en",
        # size=1000000,
        preprocess_fn=lambda examples: {
            "text": [text.strip() for text in examples["text"]]
        },
    ),
    # https://huggingface.co/datasets/wikimedia/wikipedia
    DatasetEnum.WIKIPEDIA: DatasetInfo(
        # name="20231101.en",
        # size=500000,
        # 在这里提供features吧，可以是None 也可以是一个list[str]
        # 或者就用[] 表示none吧
        # 然后再提供一个num_gbytes: float
        # 这样比较方便计算，不会因为数值太大导致溢出
        # 但是features其实自己手动写并不合适
        # 我们在读取样本的时候，就一定可以获得features
        #
        preprocess_fn=lambda examples: {
            "text": [text.strip() for text in examples["text"]]
        },
    ),
    # https://huggingface.co/datasets/nampdn-ai/tiny-codes
    DatasetEnum.TINY_CODES: DatasetInfo(
        # name=None,
        # size=200000,
        preprocess_fn=lambda examples: {
            "text": [
                f"{prompt} {response}".strip()
                for prompt, response in zip(
                    examples["prompt"], examples["response"], strict=False
                )
            ]
        },
    ),
    # https://huggingface.co/datasets/eminorhan/gutenberg_en
    DatasetEnum.GUTENBERG: DatasetInfo(
        # name="chunk_size_1024",
        # size=300000,
        size=62.0,  # 21 GB
        preprocess_fn=lambda examples: {
            "text": [text.strip() for text in examples["text"]]
        },
    ),
    # https://huggingface.co/datasets/donfu/oa-stackexchange
    DatasetEnum.STACK_EXCHANGE: DatasetInfo(
        # name=None,
        # size=150000,
        preprocess_fn=lambda examples: {
            "text": [
                f"{instruction} {response}".strip()
                for instruction, response in zip(
                    examples["INSTRUCTION"], examples["RESPONSE"], strict=False
                )
            ]
        },
    ),
    # https://huggingface.co/datasets/common-pile/arxiv_papers
    DatasetEnum.ARXIV: DatasetInfo(
        # name=None,
        # size=400000,
        size=21.0,
        preprocess_fn=lambda examples: {
            "text": [text.strip() for text in examples["text"]]
        },
    ),
}


def get_dataset_size_from_hf(
    path: str, name: str | None, split: str = "train"
) -> float | None:
    builder = load_dataset_builder(path=path, name=name)
    info: HDatasetInfo = builder.info
    if info.splits is None:
        return None
    split_info = info.splits.get(split)
    if split_info is None:
        return None
    return split_info.num_bytes / 1024 / 1024 / 1024  # 转换为 GB


# 获得某个数据集的大小
def get_dataset_size(path: str, name: str | None, split: str = "train") -> float:
    size = get_dataset_size_from_hf(path, name, split)
    if size is None and path in dataset_infos:
        # 如果在dataset_infos中找到了，就返回size
        size = dataset_infos[DatasetEnum(path)].size

    if size is None:
        # 如果还是没有找到，就抛出异常
        msg = f"Dataset {path} size is not available."
        raise ValueError(msg)
    return size


def get_dataset_features(
    path: str,
    name: str | None,
    split: str = "train",
    streaming: bool = True,  # noqa: FBT001, FBT002
) -> list[str]:
    # builder = load_dataset_builder(path=path, name=name)
    # info: DatasetInfo = builder.info
    # if info.features is None:
    #     raise ValueError("Features are not available.")
    # return list(info.features.keys())
    # 而且不管有没有用streaming 这个方法都可以正确的获得features
    # 卧槽，还得是我呀，哈哈哈！
    ds = load_dataset(path=path, name=name, split=split, streaming=streaming)
    match ds:
        case Dataset():
            return list(ds.features.keys())
        case IterableDataset():
            example = next(iter(ds))
            if not isinstance(example, dict):
                msg = "Example is not a dictionary."
                raise TypeError(msg)
            return list(example.keys())
        case _:
            msg = f"Unsupported dataset type: {type(ds)}"
            raise TypeError(msg)


# 不对哎，这个函数没法考虑split这个参数的影响。。。
# 哎，有点烦，算了，不管了
def calculate_sampling_probabilities(
    pathes: list[str],
    names: list[str | None],
    num_epochs: list[int],
) -> list[float]:
    # 这个函数需要改成从 dataset_builder 中获取size
    # 然后结合num epochs 来计算出采样概率

    # 查看cache目录中对应的文件是否存在
    # 如果存在，就直接加载进来就行了
    # 如果不存在，我们在从hugging face上下载
    # 否则每次单元测试都花费太多时间了
    # 好像开了offline mode之后，测试会快很多
    # 那就没必要写这个东西了

    # dataset_builders: list[DatasetBuilder] = [
    #     load_dataset_builder(path=path, name=name)
    #     for path, name in zip(pathes, names, strict=False)
    # ]
    # dataset_sizes: list[int] = []
    # for dataset_builder in dataset_builders:
    #     if dataset_builder.info.size_in_bytes is None:
    #         # 如果size_in_bytes为None，说明数据集无法通过builder获取大小
    #         # 直接报异常
    #         msg = f"Dataset {dataset_builder.info.dataset_name} size is not available."
    #         raise ValueError(msg)
    #     dataset_sizes.append(dataset_builder.info.size_in_bytes)

    dataset_sizes: list[float] = [
        get_dataset_size(path, name) for path, name in zip(pathes, names, strict=False)
    ]

    # 如果数据集无法通过builder获取大小
    # 直接报异常就行了

    # 返回的长度应该和dataset_configs的长度一致
    total_size: float = sum(
        size * num_epoch
        for size, num_epoch in zip(dataset_sizes, num_epochs, strict=False)
    )
    sampling_probabilities: list[float] = [
        (size * num_epoch) / total_size
        for size, num_epoch in zip(dataset_sizes, num_epochs, strict=False)
    ]
    return sampling_probabilities


# TODO: 这个函数目前默认加载本地数据集，如果没有就会从huggingface上下载
# 应该增加一个streaming参数
# 现在只需要实现一个函数
# 根据config中的结果，返回一个数据集就行了
# 可以先实现一个简单的函数
# 根据一个DatasetConfig返回一个Dataset
# tips: 当streaming为True是，num_shards参数无效
def create_dataset(  # noqa: PLR0913
    path: str,
    name: str | None,
    split: str,
    map_batch_size: int,
    streaming: bool = True,  # noqa: FBT001, FBT002
    num_shards: int = 1,
) -> IterableDataset:
    # 因为我们需要对数据进行预处理
    # 所以实际上只有我们实验过的数据集才可以使用
    # 也就是必须在我们定义的枚举里面
    assert DatasetEnum(path) in dataset_infos, f"Dataset {path} is not supported."

    # 根据DatasetConfig加载数据集
    # dataset_info: DatasetInfo = dataset_infos[dataset_config.name]
    dataset = load_dataset(
        path=path,
        name=name,
        split=split,
        streaming=streaming,
        # # 这两个应该是可以注释掉的，因为设置了HF_HOME环境变量, 现在默认的cache目录就是这个
        # cache_dir="/data2/huggingface/datasets",
        # # 还有就是设置了这个num_proc也没用，在不使用streaming的时候
        # 在使用streaming模式的时候，使用num_proc会报错，反正生成tokens的速度是够的
        # 还是算了
        # num_proc=8,
    )

    if not streaming:
        assert isinstance(dataset, Dataset), f"Loaded dataset {path} is not an Dataset."
        # TODO: 数据集本身是有一个shards的，转换成iterable dataset之后，会保留这个shard吗？
        # 不过也并不重要吧，应该是可以获取数据集原本的shard的
        dataset = dataset.to_iterable_dataset(num_shards=num_shards)
    assert isinstance(dataset, IterableDataset), (
        f"Loaded dataset {path} is not an IterableDataset."
    )

    # assert isinstance(dataset, Dataset), "Loaded dataset is not a Dataset instance"
    # 需不需要转成iterable dataset?
    # 这个应该由一个参数来控制

    # 不同的数据集有不同的预处理方式
    # 应该由一个函数提供

    # 预处理数据集
    # 要不预处理这一步，不变成iterable了吧
    # 这里应该对于streaming和local的数据的处理方式是不同的
    # TODO：对于一个iterable dataset，可能不能通过dataset.column_names来获取列名
    # 而且通过dataset loader也不一定能够获取
    # 获取的最稳定的，最正确的办法，就是iter一下，拿到example
    # 看看所有的features
    # TODO: create dataset根据streaming模式判断怎么获取features name
    # features = get_dataset_features(path, name, split)
    return dataset.map(
        function=dataset_infos[DatasetEnum(path)].preprocess_fn,
        batched=True,
        batch_size=map_batch_size,
        # TODO: when debugging, do not use multiprocessing
        # num_proc=self.num_procs,  # 使用多核处理
        remove_columns=dataset.column_names,
        # remove_columns=features,
    )
