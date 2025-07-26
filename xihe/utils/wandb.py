# 2025/7/26
# zhangzhong

import wandb
from wandb.sdk.wandb_run import Run
from xihe.settings import Config, WandbConfig


def init_wandb_run(config: Config) -> Run:
    # Load configuration
    # config = load_config(Path(args.conf))
    # 在这里创建wandb更好
    # wandb.login()
    run: Run = wandb.init(
        entity=config.wandb.entity,
        project=config.wandb.project,
        id=config.wandb.id,
        config=config.model_dump(),  # Track hyperparameters and metadata
        # resume="never",  # "must" to resume an existing run, "never" to start a new one
    )
    return run


def load_wnadb_run(wandb_config: WandbConfig) -> Run:
    # wandb.login()
    run: Run = wandb.init(
        entity=wandb_config.entity,
        project=wandb_config.project,
        id=wandb_config.id,
        resume="must",
    )
    return run
