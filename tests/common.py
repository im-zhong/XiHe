# 2025/7/26
# zhangzhong
# 这里面放一些单元测试可以复用的代码吧

from xihe.settings import (
    CheckpointConfig,
    Config,
    DataLoaderConfig,
    DatasetArgs,
    ModelConfig,
    OptimizerConfig,
    TokenizerConfig,
    TrainerConfig,
    WandbConfig,
)

# [dataloader]
# sampling_probabilities = [0.5, 0.5]
# # 先不管这些设置了，等基础的代码可以跑起来在说吧
# # streaming = false
# batch_size = 8

# [[dataloader.datasets]]
# path = "allenai/c4"
# name = "en"
# split = "train[:1024]"
# streaming = false
# num_epochs = 1

# [[dataloader.datasets]]
# path = "wikimedia/wikipedia"
# name = "20231101.en"
# split = "train[:1024]"
# streaming = false
# num_epochs = 2


# [tokenizer]
# # tokenizer_file = "gpt2"
# tokenizer_name = "gpt2"
# vocab_size = 50257


# [model]
# model_name = "XiHe"
# context_length = 1024
# num_layers = 2
# hidden_size = 768
# num_heads = 12
# intermediate_size = 3072
# dtype = "float32"
# mixed_precision = false
# low_precision_dtype = "bfloat16"


# [trainer]
# checkpoint_dir = "checkpoints"
# batch_size = 8
# warmup_steps = 2000
# total_steps = 200000
# device = "cuda"


# # we only use cosine scheduler
# [optimizer]
# optimizer_name = "AdamW"
# initial_lr = 5e-5
# max_lr = 1e-4
# final_lr = 1e-5
# weight_decay = 0.01
# max_grad_norm = 1.0

# [wandb]
# entity = "im-zhong-org"
# project = "myllm-pretrain-test"
# id = "a2ab6ca6-6964-42bc-8710-271c8d48f502"


def generate_testing_config() -> Config:
    return Config(
        model=ModelConfig(
            model_name="XiHe",
            context_length=1024,
            num_layers=2,
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,
            # dtype="float32",
            mixed_precision=False,
            # low_precision_dtype="bfloat16",
        ),
        tokenizer=TokenizerConfig(
            tokenizer_name="gpt2",
            # vocab_size=50257,
        ),
        trainer=TrainerConfig(
            checkpoint_dir="checkpoints",
            batch_size=8,
            warmup_steps=2000,
            total_steps=200000,
            device="cuda",
        ),
        optimizer=OptimizerConfig(
            optimizer_name="AdamW",
            initial_lr=5e-5,
            max_lr=1e-4,
            final_lr=1e-5,
            weight_decay=0.01,
            max_grad_norm=1.0,
        ),
        wandb=WandbConfig(
            entity="im-zhong-org",
            project="myllm-pretrain-test",
            id="a2ab6ca6-6964-42bc-8710-271c8d48f502",
        ),
        dataloader=DataLoaderConfig(
            batch_size=8,
            sampling_probabilities=[0.5, 0.5],
            datasets=[
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
            ],
        ),
        checkpoint=CheckpointConfig(
            keep_num=5,
            save_steps=1000,
        ),
    )
