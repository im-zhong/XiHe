# 2025/7/19
# zhangzhong

from xihe.model import Transformer
from xihe.settings import ModelConfig, load_config
from torch.utils.data import DataLoader

# use pytorch lambdaLR to impl custom learning rate scheduler
from torch.optim import AdamW  # use this optimizer

# https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html
from torch.optim.lr_scheduler import LambdaLR
import torch
import math
from torch import Tensor


def cosine_scheduler_with_warmup(
    warmup_steps, total_steps, initial_lr, max_lr, final_lr
):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return initial_lr + (max_lr - initial_lr) * step / warmup_steps
        # step >= lr
        theta: float = math.pi * (step - warmup_steps) / (total_steps - warmup_steps)
        return final_lr + 0.5 * (max_lr - final_lr) * (1 + math.cos(theta))

    return lr_lambda


# lr_scheduler = LambdaLR(
#     optimizer=AdamW(...),
#     lr_lambda=cosine_scheduler_with_warmup(
#         warmup_steps=2000,
#         total_steps=200000,
#         initial_lr=5e-5,
#         max_lr=1e-4,
#         final_lr=1e-5,
#     ),
# )


# TODO: 更名为 CausalLLMTrainer or GPTTrainer
class TransformerTrainer:
    def __init__(
        self,
        model: Transformer,
        settings: ModelConfig,
        scheduler: LambdaLR,
        dataloader: DataLoader,
        device="cuda",
    ):
        self.config = settings
        # Initialize model, tokenizer, optimizer, etc. based on the config
        self.model = model.to(device)
        # self.tokenizer = model.tokenizer
        self.optimizer = AdamW(model.parameters(), lr=settings.learning_rate)
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.device = device
        self.loss_fn = torch.nn.CrossEntropyLoss()

    # TODO: 我们应该先把trainer给跑起来，在上这些东西
    # https://docs.pytorch.org/docs/stable/amp.html
    def train(self):
        self.model.train()
        for step, batch in enumerate(self.dataloader):
            inputs = batch["input_ids"].to(self.device)
            # labels = batch["labels"].to(self.device)

            labels = inputs.clone()  # For simplicity, using inputs as labels
            # labels.shape = [batch_size, seq_length]
            inputs = inputs.to(self.device)

            logits = self.model(inputs)
            # logits.shape = [batch_size, seq_length, vocab_size]

            # 要在这里构造labels
            # 要使用cross entropy loss
            # 需要手动将logits和labels shift一下
            # logits的最后一个不要，因为那是整个序列的预测 我们没有他的golden
            # labels的第一个不要，因为没有对应的logits预测
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            # https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            loss: Tensor = self.loss_fn(
                shifted_logits.view(-1, self.config.vocab_size),
                shifted_labels.view(-1),
            )

            loss.backward()

            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(
                parameters=self.model.parameters(), max_norm=self.config.grad_clip
            )

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            if step % self.config.logging_steps == 0:
                print(f"Step {step}, Loss: {loss.item()}")

            if step % self.config.save_steps == 0:
                # Save model checkpoint
                self.save_model()

    # TODO: 先不急，等SFT训练完了之后，在eval吧
    def evaluate(self):
        # Implement the evaluation logic
        pass

    def save_model(self):
        # Save the model and tokenizer to the specified paths
        # TODO: 如果训练中断，想要重启训练，是不是还得保存optimizer scheduelr dataloader的状态
        torch.save(self.model.state_dict(), self.config.checkpoint_path)
        pass

    def load_model(self):
        # Load the model and tokenizer from the specified paths
        pass
