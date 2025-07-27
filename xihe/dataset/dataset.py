# 2025/7/19
# zhangzhong

# 我需要写一个可以按照一定的比例，混合多个数据集
# 带有shuffle
# 我们先用GPT2自带的tokenizer吧
# 之后再换成我们自己train出来的


from datasets import (
    load_dataset,
    IterableDataset,
    load_dataset_builder,
    DatasetBuilder,
)

from enum import StrEnum


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


# 定义一个类包含dataset的信息
class DatasetInfo:
    def __init__(
        self,
        # name: str | None,
        # size: int,
        preprocess_fn,
        # features: list[str] = [],
    ):
        # self.name = name
        # self.size = size
        # self.features = features
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
                for prompt, response in zip(examples["prompt"], examples["response"])
            ]
        },
    ),
    # https://huggingface.co/datasets/eminorhan/gutenberg_en
    DatasetEnum.GUTENBERG: DatasetInfo(
        # name="chunk_size_1024",
        # size=300000,
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
                    examples["INSTRUCTION"], examples["RESPONSE"]
                )
            ]
        },
    ),
    # https://huggingface.co/datasets/common-pile/arxiv_papers
    DatasetEnum.ARXIV: DatasetInfo(
        # name=None,
        # size=400000,
        preprocess_fn=lambda examples: {
            "text": [text.strip() for text in examples["text"]]
        },
    ),
}


# 不对哎，这个函数没法考虑split这个参数的影响。。。
# 哎，有点烦，算了，不管了
def calculate_sampling_probabilities(
    pathes: list[str],
    names: list[str],
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

    dataset_builders: list[DatasetBuilder] = [
        load_dataset_builder(path=path, name=name) for path, name in zip(pathes, names)
    ]
    dataset_sizes: list[int | None] = [
        dataset_builder.info.size_in_bytes for dataset_builder in dataset_builders
    ]
    if None in dataset_sizes:
        raise ValueError(
            "Some dataset sizes are None. Please check the dataset paths and names."
        )

    # 返回的长度应该和dataset_configs的长度一致
    total_size: int = sum(
        size * num_epoch for size, num_epoch in zip(dataset_sizes, num_epochs)
    )
    sampling_probabilities: list[float] = [
        (size * num_epoch) / total_size
        for size, num_epoch in zip(dataset_sizes, num_epochs)
    ]
    return sampling_probabilities


# TODO: 这个函数目前默认加载本地数据集，如果没有就会从huggingface上下载
# 应该增加一个streaming参数
# 现在只需要实现一个函数
# 根据config中的结果，返回一个数据集就行了
# 可以先实现一个简单的函数
# 根据一个DatasetConfig返回一个Dataset
def create_dataset(
    path: str, name: str, split: str, streaming: bool
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
    )
    # assert isinstance(dataset, Dataset), "Loaded dataset is not a Dataset instance"
    # 需不需要转成iterable dataset?
    # 这个应该由一个参数来控制

    # 不同的数据集有不同的预处理方式
    # 应该由一个函数提供

    # 预处理数据集
    # 要不预处理这一步，不变成iterable了吧
    preprocessed_dataset = dataset.map(
        function=dataset_infos[DatasetEnum(path)].preprocess_fn,
        batched=True,
        batch_size=4096,
        # TODO: when debugging, do not use multiprocessing
        # num_proc=self.num_procs,  # 使用多核处理
        remove_columns=dataset.column_names,
    )

    if not streaming:
        return preprocessed_dataset.to_iterable_dataset()
    return preprocessed_dataset
