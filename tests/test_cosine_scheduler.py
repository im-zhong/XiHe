# 2025/7/20
# zhangzhong
# TODO
# 用ipynb不方便测试，就直接生成图片就行了
# 生成到cache里面就行，不用上传


import matplotlib.pyplot as plt
import torch
from torch.optim import Adam

from xihe.defs import defs
from xihe.optimizer.optimizer import create_cosine_lr_scheduler


def test_cosine_scheduler() -> None:
    # Parameters for the cosine scheduler
    total_steps = 100
    warmup_steps = 10
    initial_lr = 0.01
    max_lr = 0.1
    final_lr = 0.0

    # Create the cosine scheduler
    scheduler = create_cosine_lr_scheduler(
        optimizer=Adam(params=[torch.tensor([])], lr=initial_lr),  # Dummy optimizer
        initial_lr=initial_lr,
        final_lr=final_lr,
        max_lr=max_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    # Generate learning rates for each epoch
    lrs = []
    for _ in range(total_steps):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    # Plot the learning rate schedule
    plt.figure(figsize=(10, 5))
    plt.plot(range(total_steps), lrs, label="Cosine Scheduler with Warmup")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Cosine Learning Rate Scheduler with Warmup")
    plt.legend()
    plt.grid()

    # Save the plot to a file
    defs.cache_dir.mkdir(exist_ok=True)
    plt.savefig(".cache/cosine_scheduler_plot.png")
