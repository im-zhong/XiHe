d# 2025/7/20
# zhangzhong


# 我们希望怎么使用这个脚本呢？
# pretrain.py --config config.yaml
# 剩下的所有配置都写在配置文件里面就行了呗
# 必要的时候，可以提供一些额外的参数，用来覆盖config里面的配置足够了


from torch.optim.lr_scheduler import LambdaLR
from xihe.model import Transformer
from xihe.settings import load_config
from xihe.trainer import (
    TransformerTrainer,
    create_optimizer,
    create_cosine_lr_scheduler,
)
from xihe.dataset import (
    PackingDataset,
    create_dataset,
    calculate_sampling_probabilities,
)
from pathlib import Path
import wandb
from wandb.sdk.wandb_run import Run

import argparse
import torch
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from torch.optim import Optimizer, Adam, AdamW


def create_tokenizer(tokenizer_name: str) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(tokenizer_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain a Transformer model")
    parser.add_argument(
        "--conf", "-f", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()
    # TODO: tmp set this code, if we impl ddp, should delete this
    # ? 这设置的不是torch 创建tensor的默认设备？
    # 这个并不是用来设置默认设备的，pytorch没有提供这个功能
    # 这个是设置默认的cuda设备的，也就是当你指定 cuda 但是没有指定 cuda:id 的时候的默认id
    torch.cuda.set_device(0)  # Set the default CUDA device to 0

    # Load configuration
    config = load_config(Path(args.conf))
    # 在这里创建wandb更好
    wandb.login()
    run: Run = wandb.init(
        project="My LLM",
        config=config.model_dump(),  # Track hyperparameters and metadata
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
    dataloader = dataset.to_torch_dataloader(
        batch_size=config.dataloader.batch_size,
        context_length=config.model.context_length,
    )

    # # Initialize model
    # TODO: vocab size应该是不用设置的
    # 应该可以通过tokenizer对象来获取才对
    model = Transformer(
        vocab_size=config.tokenizer.vocab_size,
        max_seq_len=config.model.context_length,
        num_layers=config.model.num_layers,
        hidden_size=config.model.hidden_size,
        num_heads=config.model.num_heads,
        intermediate_size=config.model.intermediate_size,
        device=config.trainer.device,  # Pass the device from config
    )
    model = model.to(config.trainer.device)

    # 这里需要创建optimizer和scheduler
    optimizer: Optimizer = create_optimizer(
        name=config.optimizer.optimizer_name,
        learning_rate=config.optimizer.initial_lr,
        weight_decay=config.optimizer.weight_decay,
        parameters=model.parameters(),
    )
    lr_scheduler: LambdaLR = create_cosine_lr_scheduler(
        optimizer=optimizer,
        warmup_steps=config.trainer.warmup_steps,
        total_steps=config.trainer.total_steps,
        initial_lr=config.optimizer.initial_lr,
        max_lr=config.optimizer.max_lr,
        final_lr=config.optimizer.final_lr,
    )

    # # Initialize the trainer
    trainer = TransformerTrainer(
        vocab_size=config.tokenizer.vocab_size,
        model=model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        dataloader=dataloader,
        device=config.trainer.device,
        # dtype=config.model.dtype,
        run=run,
    )

    # Start training
    trainer.train()
