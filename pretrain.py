# 2025/7/20
# zhangzhong


# 我们希望怎么使用这个脚本呢？
# pretrain.py --config config.yaml
# 剩下的所有配置都写在配置文件里面就行了呗
# 必要的时候，可以提供一些额外的参数，用来覆盖config里面的配置足够了


from xihe.model import Transformer
from xihe.settings import ModelConfig, load_config
from xihe.trainer import TransformerTrainer
from xihe.dataset import (
    PackingDataset,
    create_dataset,
    calculate_sampling_probabilities,
)
from pathlib import Path


import argparse
import torch
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer


def create_tokenizer(tokenizer_name: str) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(tokenizer_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain a Transformer model")
    parser.add_argument(
        "--conf", "-f", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(Path(args.conf))

    # # Initialize model
    model = Transformer(
        vocab_size=config.tokenizer.vocab_size,
        max_seq_len=config.model.context_length,
        num_layers=config.model.num_layers,
        hidden_size=config.model.hidden_size,
        num_heads=config.model.num_heads,
        intermediate_size=config.model.intermediate_size,
    )

    # get tokenizer from tokenizer configs
    # 但是我们不能直接依赖TokenizerConfig这个类
    # 要写一个函数来根据配置获取tokenizer
    # tokenizer = get_tokenizer(config.tokenizer)
    tokenizer = create_tokenizer(config.tokenizer.tokenizer_name)

    # 还需要设置dataset
    # 我们使用哪些dataset，和使用多少，在datasetconfig里面写上就行了
    datasets = [
        create_dataset(
            path=dataset.path,
            name=dataset.name,
            split=dataset.split,
        )
        for dataset in config.dataloader.datasets
    ]

    sampling_probabilities: list[float] = []
    if config.dataloader.sampling_probabilities:
        sampling_probabilities = config.dataloader.sampling_probabilities
    else:
        # 如果没有提供采样概率，就计算一下
        sampling_probabilities = calculate_sampling_probabilities(
            pathes=[dataset.path for dataset in config.dataloader.datasets],
            names=[dataset.name for dataset in config.dataloader.datasets],
            num_epochs=[dataset.num_epochs for dataset in config.dataloader.datasets],
        )

    # # Initialize dataset and dataloader
    dataset = PackingDataset(
        datasets=datasets,
        tokenizer=tokenizer,
        sampling_probabilities=sampling_probabilities,
    )

    # # Initialize the trainer
    # trainer = TransformerTrainer(
    #     model=model,
    #     settings=config,
    #     scheduler=lr_scheduler,
    #     dataloader=dataloader,
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    # )

    # Start training
    # trainer.train()
