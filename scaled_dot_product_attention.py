import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

seq_len = 8
batch_size = 4
num_heads = 2
seq_len = 8
hidden_size = 16


def generate_causal_mask(seq_len: int) -> Tensor:
    # torch.triu: Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.
    # torch.ones: Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument size
    # boardcast: https://numpy.org/doc/stable/user/basics.broadcasting.html
    return (
        torch.triu(input=torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        .unsqueeze(0)
        .unsqueeze(0)
    )


ones = torch.ones(seq_len, seq_len, dtype=torch.bool)
print(ones.shape)
print(ones)


causal_mask = generate_causal_mask(8)
print(causal_mask.shape)
print(causal_mask)


# https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html
# e^(-inf) = 0
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, float("-inf"), 6.0]])
print(F.softmax(x, dim=-1))

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# 加起来没有啊，只能做mask fill
mask = torch.tensor([[0, 0, 0], [0, 1, 0]], dtype=torch.bool)
x = x.masked_fill(mask, float("-inf"))
print(x)
print(F.softmax(x, dim=-1))


class MaskedDotProductAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        query: (B, H, T_q, d_k)
        key:   (B, H, T_k, d_k)
        value: (B, H, T_k, d_k)
        """
        d_k: int = query.size(-1)
        scores: Tensor = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k)
        )  # (B, H, T_q, T_k)

        # 我记得不是这样的吧，我记得是利用softmax的特性实现的mask
        mask: Tensor = generate_causal_mask(seq_len)  # (1, 1, T_q, d_k)
        scores = scores.masked_fill(mask=mask, value=float("-inf"))

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, value)  # (B, H, T_q, D)

        return output, attn


queries = torch.randn(batch_size, num_heads, seq_len, hidden_size)
keys = torch.randn(batch_size, num_heads, seq_len, hidden_size)
values = torch.randn(batch_size, num_heads, seq_len, hidden_size)


attention = MaskedDotProductAttention()
attention_weights, output = attention(queries, keys, values)
print(attention_weights.shape, output.shape)
