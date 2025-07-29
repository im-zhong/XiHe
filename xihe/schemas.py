# 2025/7/26
# zhangzhong

from pydantic import BaseModel, Field


# 所有的模块都可以依赖models，这个是ok的
# 包括config模块
# 这个要改, 咱们要支持streaming模式，而且默认最终一定会用iterable dataset
# 现在看起来也没有必要定义那个枚举了
class DatasetArgs(BaseModel):
    path: str = Field(..., description="Path to the dataset.")
    name: str | None = Field(default=None, description="Name of the dataset.")
    split: str = Field(..., description="Dataset split =")
    num_epochs: int = Field(..., description="Number of epochs for training.")
    streaming: bool = Field(
        default=False,
        description="Whether to use streaming mode for the dataset.",
    )
    # 所以这个to_iterable参数是多余的
    # to_iterable: bool = Field(
    #     True,
    #     description="Whether to convert the dataset to an iterable format.",
    # )
