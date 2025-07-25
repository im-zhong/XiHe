# 2025/7/25
# zhangzhong

import wandb


def test_wandb() -> None:
    run = wandb.init(project="test_project")
    print(run.entity)
