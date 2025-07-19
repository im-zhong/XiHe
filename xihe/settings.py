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


class ModelConfig(BaseModel):
    model_name: str = Field(..., description="The name of the model to be used.")
    tokenizer_path: str = Field(..., description="Path to the tokenizer files.")
    checkpoint_path: str = Field(..., description="Path to save the model checkpoints.")
    max_length: int = Field(512, description="Maximum length of input sequences.")
    batch_size: int = Field(32, description="Batch size for training.")
    learning_rate: float = Field(5e-5, description="Learning rate for the optimizer.")
    num_epochs: int = Field(3, description="Number of epochs for training.")
    warmup_steps: int = Field(
        2000, description="Number of warmup steps for learning rate scheduling."
    )
    weight_decay: float = Field(0.01, description="Weight decay for the optimizer.")
    logging_steps: int = Field(100, description="Number of steps between logging.")
    save_steps: int = Field(
        500, description="Number of steps between saving checkpoints."
    )
    max_steps: int = Field(200000, description="Maximum number of training steps.")

    num_layers: int = Field(12, description="Number of layers in the model.")
    hidden_size: int = Field(768, description="Size of the hidden layers.")
    num_heads: int = Field(12, description="Number of attention heads in the model.")
    intermediate_size: int = Field(3072, description="Size of the intermediate layers.")
    vocab_size: int = Field(32000, description="Size of the vocabulary.")
    grad_clip: float = Field(
        1.0,
        description="Maximum norm for gradient clipping to prevent exploding gradients.",
    )


def load_config(file_path: str) -> ModelConfig:
    """
    Load the model configuration from a TOML file.

    Args:
        file_path (str): Path to the TOML configuration file.

    Returns:
        ModelConfig: An instance of ModelConfig with the loaded parameters.
    """
    with open(file_path, "rb") as f:
        config_data = tomllib.load(f)
    return ModelConfig(**config_data)
