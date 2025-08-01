import torch
from torch import Tensor, nn


# https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.normalization.RMSNorm.html
class RMSNorm(nn.Module):
    # 这个东西的实现才是真的可以参考LayerNorm
    # 因为这个东西就直接可以看作是LayerNorm在均值为零的时候的一个特例t
    def __init__(
        self, normalized_shape: int, eps: float = 1e-6, device: str = "cpu"
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        # 这个东西的scale是一个可学习的参数
        self.scale = nn.Parameter(torch.ones(normalized_shape, device=device))

    def forward(self, input_tensor: Tensor) -> Tensor:
        # output.shape == input.shape
        # 要做的事情其实就是先针对于最后一个维度计算
        shape = input_tensor.shape
        dim = input_tensor.shape[-1]
        if dim != self.normalized_shape:
            msg = f"Input shape {shape} does not match normalized shape {self.normalized_shape}!"
            raise ValueError(msg)

        # 确实不需要这样做，因为我们只在最后一个维度上做
        # input = input.view(-1, dim)  # 保证是二维的
        input_tensor = input_tensor * torch.rsqrt(
            input_tensor.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return input_tensor * self.scale  # .view(1, -1)  # shape = (batch_size, dim)
        # output = input.view(shape)  # 恢复原来的形状


# Example usage
rms_norm = RMSNorm(normalized_shape=16)
input_tensor = torch.randn(4, 8, 16)  # batch_size=4, seq_len=8, feature_size=16
output_tensor = rms_norm(input_tensor)
print(output_tensor.shape)  # 应该是 [4, 8, 16]
