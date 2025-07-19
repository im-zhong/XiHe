# 2025/7/19
# zhangzhong

from xihe.dataset import PackingDataset, DatasetEnum, DatasetConfig
from transformers import AutoTokenizer


def test_mixed_dataset():
    # Test the MixedDataset class
    # datasets = [
    #     {"name": "dataset1", "size": 1000, "num_epochs": 1},
    #     {"name": "dataset2", "size": 2000, "num_epochs": 2},
    # ]

    packing_dataset = PackingDataset(
        dataset_configs=[
            DatasetConfig(name=DatasetEnum.C4, split="train[:1024]", num_epochs=1),
            DatasetConfig(
                name=DatasetEnum.WIKIPEDIA, split="train[:1024]", num_epochs=2
            ),
        ],
        tokenizer=AutoTokenizer.from_pretrained("gpt2"),
        sample_probabilities=[0.5, 0.5],
    )

    batch_size = 8
    context_length = 1024
    dataloader = packing_dataset.to_torch_dataloader(
        batch_size=batch_size, context_length=context_length
    )
    for batch in dataloader:
        # assert "input_ids" in batch
        # assert "attention_mask" in batch
        # assert len(batch["input_ids"]) == 8  # Check batch size
        input_ids = batch["input_ids"]
        assert input_ids.shape == (batch_size, context_length)
        break  # Only test one batch for simplicity
