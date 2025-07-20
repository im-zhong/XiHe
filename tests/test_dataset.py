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


# def test_mixed_dataset():
#     # Test the MixedDataset class
#     # datasets = [
#     #     {"name": "dataset1", "size": 1000, "num_epochs": 1},
#     #     {"name": "dataset2", "size": 2000, "num_epochs": 2},
#     # ]

#     # 我们得先加载dataset才行

#     packing_dataset = PackingDataset(
#         dataset_configs=[
#             DatasetConfig(name=DatasetEnum.C4, split="train[:1024]", num_epochs=1),
#             DatasetConfig(
#                 name=DatasetEnum.WIKIPEDIA, split="train[:1024]", num_epochs=2
#             ),
#         ],
#         tokenizer=AutoTokenizer.from_pretrained("gpt2"),
#         sample_probabilities=[0.5, 0.5],
#     )

#     batch_size = 8
#     context_length = 1024
#     dataloader = packing_dataset.to_torch_dataloader(
#         batch_size=batch_size, context_length=context_length
#     )
#     for batch in dataloader:
#         # assert "input_ids" in batch
#         # assert "attention_mask" in batch
#         # assert len(batch["input_ids"]) == 8  # Check batch size
#         input_ids = batch["input_ids"]
#         assert input_ids.shape == (batch_size, context_length)
#         break  # Only test one batch for simplicity
