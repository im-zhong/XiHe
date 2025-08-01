import torch
from torch import Tensor, nn


def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    # layernorm对input的形状不做假设 他就是对最后一个维度的向量的所有分量之间做标准化
    batch_size, seq_size, normalized_shape = x.shape
    shape = x.shape
    x = torch.reshape(x, shape=(-1, normalized_shape))

    mean = x.mean(dim=-1, keepdim=True)
    # sigma = input(dim=-1)
    # sigma = torch.linalg.vector_norm(input, ord=2, dim=-1)
    var = x.var(dim=-1, keepdim=True)
    assert mean.shape == (batch_size * seq_size, 1)
    assert var.shape == (batch_size * seq_size, 1)

    x = (x - mean) / torch.sqrt(var + eps)
    assert gamma.shape == (normalized_shape,)
    assert beta.shape == (normalized_shape,)
    x = x * gamma + beta

    # input.shape 应该保持不变
    return x.reshape(shape)


class LayerNorm(nn.Module):
    def __init__(self, feature_size: int) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.gamma = torch.nn.Parameter(torch.ones(size=[feature_size]))
        self.beta = torch.nn.Parameter(torch.zeros(size=[feature_size]))

    def forward(self, x: Tensor) -> Tensor:
        return layer_norm(x, self.gamma, self.beta)


norm = LayerNorm(feature_size=16)
input_tensor = torch.randn(4, 8, 16)  # batch_size=4, seq_len=8, feature_size=16
output_tensor = norm(input_tensor)
print(output_tensor.shape)  # 应该是 [4, 8, 16]
