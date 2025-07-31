# 2025/7/26
# zhangzhong

from xihe.dataset import create_dataloader
from xihe.schemas import DatasetArgs
from xihe.tokenizer import create_tokenizer


def test_create_dataloader() -> None:
    tokenizer = create_tokenizer("gpt2")
    datasets_args = [
        DatasetArgs(
            path="allenai/c4",
            name="en",
            # streaming=True, 下，不支持slice
            # split="train[:1024]",
            split="train",
            num_epochs=1,
            streaming=True,
        ),
        DatasetArgs(
            path="wikimedia/wikipedia",
            name="20231101.en",
            # split="train[:1024]",
            split="train",
            num_epochs=2,
            streaming=True,
        ),
    ]

    dataloader = create_dataloader(
        tokenizer=tokenizer,
        rank=0,
        batch_size=4,
        map_batch_size=1024,
        context_length=512,
        world_size=4,
        seed=42,
        datasets_args=datasets_args,
        sampling_probabilities=[0.5, 0.5],  # Example sampling probabilities
    )

    assert dataloader is not None
    print("Dataloader created successfully.")
