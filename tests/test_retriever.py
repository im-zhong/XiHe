from dataclasses import dataclass

import torch
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from xihe.model import Transformer
from xihe.optimizer import create_cosine_lr_scheduler, create_optimizer
from xihe.trainer import BasicGPTTrainer


@dataclass
class RetrieverConfig_medium:
    gpu_num = 3
    batch_size = 16
    gradient_accumulation_steps = 8
    sequence_length = 1024
    learning_rate = 6e-4
    min_lr = 6e-5
    vocab_size = 32000
    num_layers = 12
    hidden_size = 768
    num_heads = 12
    beta1 = 0.9
    beta2 = 0.95
    weight_decay = 1e-1
    warmup_iters = 2000
    max_iters = 200000
    lr_decay_iters = 200000
    grad_clip = 1.0


def test_basic_gpt_trainer() -> None:
    # tokenizer = create_tokenizer("gpt2")
    # print(f"Tokenizer: {tokenizer}")

    torch.cuda.set_device(0)

    config = RetrieverConfig_medium()
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
    vocab_size = 32000  # GPT-3 uses a vocabulary size of 50257
    hidden_size = 768
    num_layers = 12
    num_heads = 12
    # 不过可能是这里的原因，retrieve是 * 2.66
    # 不过也差不多啦
    intermediate_size = hidden_size * 2
    context_length = 1024

    # gpt3 medium: 454166528, 454M
    # vocab_size = 32000  # GPT-3 uses a vocabulary size of 50257
    # hidden_size = 1024
    # num_layers = 24
    # num_heads = 16
    # intermediate_size = hidden_size * 4
    # context_length = 1024
    device = "cuda:0"
    # 最大的batchsize就是8
    # 而且context length只能是1024
    # 显存根本不够啊
    batch_size = 16  # + 8  # 这个batch 非常接近极限了，还是用16吧

    # 13884MiB
    # 4.29it/s
    # 没有我自己实现的好啊
    model = Transformer(
        vocab_size=vocab_size,
        max_seq_len=context_length,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        device=device,  # Pass the device from config
    )
    # 14328Mi
    # 3.8it/s
    # model = Retriever(device=device, ptdtype=torch.float16, config=config)
    model = model.to(device)
    model = torch.compile(model)
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
    total_steps = 100
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
    )

    for _ in tqdm(range(total_steps), desc="Training Steps", total=total_steps):
        batch = {
            "input_ids": torch.randint(0, vocab_size, (batch_size, context_length))
        }
        # batch["input_ids"] = batch["input_ids"].to(device)
        trainer.train_step(batch=batch)

    # state_dict = trainer.get_state_dict(step=0)
    # print(f"Trainer state dict: {state_dict}")
