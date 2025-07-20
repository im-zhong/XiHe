# 2025/7/19
# zhangzhong

# 先想一下大概有哪些需要配置的参数
# 以及我们要如何配置这些个参数？
# 不同的模型训练，应该是需要不同的参数的
# 所以参数配置最好用一个配置文件来做
# 尽可能少的依赖复杂的库
# 我们就只使用了datasets 和 pytorch
# 尽量手写算法，这样可以更好的学习底层的原理

# 配置文件用什么呢 toml yaml ?
# 我还是倾向于用toml的

# 把大模型相关的参数都写到里面

# 我们需要保存tokenizer文件 需要保存checkpoint
# 保存在哪里呢
# 定义一个pedantic model


import tomllib

from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any


# 这个不能叫做ModelConfig
# 我们还有TrainerConfig
# EvalConfig 等等的东西
# 还有就是这个东西
from pydantic import BaseModel, Field
from typing import Literal


class PathConfig(BaseModel):
    cache_dir: Path = Field(
        default=Path("./cache"),
        description="Directory to store cached datasets and models.",
    )
    checkpoint_dir: Path = Field(
        default=Path("./checkpoints"),
        description="Directory to save model checkpoints.",
    )


# 现在看起来也没有必要定义那个枚举了
class DatasetConfig(BaseModel):
    name: str = Field(..., description="Name of the dataset.")
    path: str = Field(..., description="Path to the dataset.")
    split: str = Field(..., description="Dataset split =")
    num_epochs: int = Field(..., description="Number of epochs for training.")
    to_iterable: bool = Field(
        True,
        description="Whether to convert the dataset to an iterable format.",
    )


class DataLoaderConfig(BaseModel):
    batch_size: int = Field(..., description="Batch size for the DataLoader.")
    sampling_probabilities: list[float] | None = Field(
        None,
        description="Sampling probabilities for datasets. If None, will be calculated based on dataset sizes.",
    )
    datasets: list[DatasetConfig] = Field(
        ...,
        description="List of datasets to be used in the DataLoader.",
    )


class TokenizerConfig(BaseModel):
    tokenizer_name: str = Field(..., description="The name or path of the tokenizer.")
    vocab_size: int = Field(..., description="Size of the tokenizer vocabulary.")


class ModelConfig(BaseModel):
    model_name: str = Field(..., description="Name of the model.")
    context_length: int = Field(
        ..., description="Maximum context length for the model."
    )
    num_layers: int = Field(..., description="Number of transformer layers.")
    hidden_size: int = Field(..., description="Hidden size of transformer layers.")
    num_heads: int = Field(..., description="Number of attention heads.")
    intermediate_size: int = Field(
        ..., description="Size of the intermediate (feed-forward) layers."
    )
    dtype: str = Field(
        "float32",
        description="Data type for model parameters. Default is 'float32'.",
    )
    mixed_precision: bool = Field(
        False,
        description="Whether to use mixed precision training. Default is False.",
    )
    low_precision_dtype: str = Field(
        "bfloat16",
        description="Data type for low precision training. Default is 'bfloat16'.",
    )


class TrainerConfig(BaseModel):
    checkpoint_dir: str = Field(..., description="Directory to save checkpoints.")
    batch_size: int = Field(..., description="Training batch size.")
    warmup_steps: int = Field(
        ..., description="Number of warmup steps for LR scheduling."
    )
    total_steps: int = Field(..., description="Total number of training steps.")
    device: str = Field(
        "cuda",
        description="Device to run the training on. Default is 'cuda'.",
    )


class OptimizerConfig(BaseModel):
    optimizer_name: Literal["AdamW"] = Field(
        ..., description="Name of the optimizer. Currently only 'AdamW' is supported."
    )
    initial_lr: float = Field(
        ..., description="Initial learning rate at training start."
    )
    max_lr: float = Field(..., description="Peak learning rate during cosine schedule.")
    final_lr: float = Field(..., description="Final learning rate after cosine decay.")
    weight_decay: float = Field(..., description="Weight decay coefficient.")
    max_grad_norm: float = Field(..., description="Maximum norm for gradient clipping.")


class Config(BaseModel):
    model: ModelConfig = Field(..., description="Model configuration.")
    tokenizer: TokenizerConfig = Field(..., description="Tokenizer configuration.")
    trainer: TrainerConfig = Field(..., description="Trainer configuration.")
    optimizer: OptimizerConfig = Field(..., description="Optimizer configuration.")
    dataloader: DataLoaderConfig = Field(
        ...,
        description="DataLoader configuration, including datasets and sampling probabilities.",
    )
    path: PathConfig = Field(
        default=PathConfig(),
        description="Paths for cache, checkpoints, and other resources.",
    )


def load_config(conf_file: Path) -> Config:
    """
    Load the model configuration from a TOML file.

    Args:
        file_path (str): Path to the TOML configuration file.

    Returns:
        ModelConfig: An instance of ModelConfig with the loaded parameters.
    """
    with open(file=conf_file, mode="rb") as f:
        config_data: dict[str, Any] = tomllib.load(f)
    return Config(**config_data)
