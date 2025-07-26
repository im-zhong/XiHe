# 2025/7/26
# zhangzhong

from xihe.dataset import create_dataloader
from xihe.tokenizer import create_tokenizer
from xihe.schemas import DatasetArgs


def test_create_dataloader() -> None:
    tokenizer = create_tokenizer("gpt2")
    datasets_args = [
        DatasetArgs(
            path="allenai/c4",
            name="en",
            split="train[:1024]",
            num_epochs=1,
            streaming=False,
        ),
        DatasetArgs(
            path="wikimedia/wikipedia",
            name="20231101.en",
            split="train[:1024]",
            num_epochs=2,
            streaming=False,
        ),
    ]

    dataloader = create_dataloader(
        tokenizer=tokenizer,
        rank=0,
        batch_size=4,
        context_length=512,
        world_size=4,
        datasets_args=datasets_args,
    )

    assert dataloader is not None
    print("Dataloader created successfully.")
