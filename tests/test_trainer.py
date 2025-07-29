# 2025/7/26
# zhangzhong

from pathlib import Path

import torch
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from tests.common import generate_testing_config
from xihe.ckpt import ckpt_defs, load_ckpt_from_path
from xihe.model import Transformer
from xihe.optimizer import create_cosine_lr_scheduler, create_optimizer
from xihe.trainer import BasicGPTTrainer, DistributedGPTTrainer


def test_basic_gpt_trainer() -> None:
    # tokenizer = create_tokenizer("gpt2")
    # print(f"Tokenizer: {tokenizer}")

    torch.cuda.set_device(0)

    # rank = 0
    # world_size = 4
    # batch_size = 4
    # context_length = 1024
    # dataloader = create_dataloader(
    #     tokenizer=tokenizer,
    #     rank=rank,
    #     batch_size=batch_size,
    #     context_length=context_length,
    #     world_size=world_size,  # Assuming single GPU for this test
    #     datasets_args=[
    #         DatasetArgs(
    #             path="allenai/c4",
    #             name="en",
    #             split="train[:1024]",
    #             num_epochs=1,
    #             streaming=False,
    #         ),
    #         DatasetArgs(
    #             path="wikimedia/wikipedia",
    #             name="20231101.en",
    #             split="train[:1024]",
    #             num_epochs=2,
    #             streaming=False,
    #         ),
    #     ],
    # )
    # print(f"Dataloader: {dataloader}")

    # gpt3 small
    # vocab_size = 50257  # GPT-3 uses a vocabulary size of 50257
    # hidden_size = 768
    # num_layers = 12
    # num_heads = 12
    # intermediate_size = hidden_size * 4
    # context_length = 1024

    # gpt3 medium: 454166528, 454M
    vocab_size = 32000  # GPT-3 uses a vocabulary size of 50257
    hidden_size = 1024
    num_layers = 24
    num_heads = 16
    intermediate_size = hidden_size * 4
    context_length = 1024
    device = "cuda:0"
    # 最大的batchsize就是8
    # 而且context length只能是1024
    # 显存根本不够啊
    batch_size = 32
    accumulation_gradient_steps = 32 // 4

    model = Transformer(
        vocab_size=vocab_size,
        max_seq_len=context_length,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        device=device,  # Pass the device from config
    )
    model = model.to(device)
    # 可以compile哎, 这个是什么功能？
    # ！不仅更快了，显存占用也更低了！这个东西好啊！！！
    # 大概是3.66it/s
    # 显存只有12G了
    # model = torch.compile(model)
    # 不做compile
    # 2.8it/s
    # 显存占用是15G
    print(f"Model: {model}")

    optimizer_name = "adamw"
    initial_lr = 3.0e-4
    weight_decay = 0.01
    optimizer: Optimizer = create_optimizer(
        name=optimizer_name,
        learning_rate=initial_lr,
        weight_decay=weight_decay,
        parameters=model.parameters(),
    )
    print(f"Optimizer: {optimizer}")

    warmup_steps = 10
    total_steps = 20
    max_lr = 1e-3
    final_lr = 1e-5

    lr_scheduler: LambdaLR = create_cosine_lr_scheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        initial_lr=initial_lr,
        max_lr=max_lr,
        final_lr=final_lr,
    )
    print(f"LR Scheduler: {lr_scheduler}")

    # 不用这个显存占用更大，跑的更慢！
    # 同样的测试条件
    # enable grad scaler: 显存占用17G, 53s
    # disable grad scaler: 显存占用20G, 1m46s
    grad_scaler = GradScaler(device=device)  # Enable mixed precision training

    trainer = BasicGPTTrainer(
        vocab_size=vocab_size,
        # dataloader=dataloader,
        model=model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        grad_scaler=grad_scaler,
        # rank=rank,
        # world_size=world_size,
        device=device,
        accumulation_gradient_steps=accumulation_gradient_steps,
    )

    for _ in tqdm(range(total_steps), desc="Training Steps", total=total_steps):
        batch = {
            "input_ids": torch.randint(0, vocab_size, (batch_size, context_length))
        }
        # batch["input_ids"] = batch["input_ids"].to(device)
        trainer.train_step(batch=batch)

    # state_dict = trainer.get_state_dict(step=0)
    # print(f"Trainer state dict: {state_dict}")


def test_distributed_trainer_from_scratch() -> None:
    # 那么这里就要先构造一个config对象
    # 咱们最好不要从example_conf来读
    # 就在这里构造一个config对象就行了 减少对外部的依赖
    config = generate_testing_config()
    config.trainer.total_steps = 10  # Set total steps for testing
    config.checkpoint.save_steps = 5

    print(f"Config: {config}")
    rank = 0
    world_size = 1
    trainer = DistributedGPTTrainer(
        rank=rank,
        world_size=world_size,  # For testing, we can use a single process
        run=None,  # No wandb run for this test
        config=config,  # Pass the config directly
    )

    trainer.train(config=config)


# TODO: 怎么model的state dict出错了呢？
def test_distributed_trainer_from_ckpt() -> None:
    start_step = 5
    project = "myllm-pretrain-test"

    ckpt_path: Path = ckpt_defs.get_step_ckpt_path(
        project_dir=ckpt_defs.get_ckpts_dir(project=project),
        step=start_step,
    )
    print(f"Checkpoint path: {ckpt_path}")
    checkpoint = load_ckpt_from_path(ckpt_path)
    assert checkpoint is not None, "Checkpoint should not be None"
    # get xxx state
    # 我感觉还是给封装一下比较好吧
    # 做一个Checkpoint
    config = checkpoint.get_config()
    config.trainer.total_steps = start_step + 5  # Set total steps for testing

    rank = 0
    world_size = 1
    trainer = DistributedGPTTrainer(
        rank=rank,
        world_size=world_size,  # For testing, we can use a single process
        run=None,  # No wandb run for this test
        config=config,  # Pass the config directly
    )

    # 需要一个方法让训练停止
    # 我们可以设置一个max step就行了呀
    # TODO: just finish this function, we are almost done!
    trainer.train(
        config=config,  # Pass the config directly
        ckpt=checkpoint,  # Load from the checkpoint
    )
