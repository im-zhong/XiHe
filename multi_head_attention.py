import torch
import torch.nn.functional as func
from torch import Tensor, nn


# https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()

        # 参考PaLM 整个模型都没有bias
        if hidden_size % num_heads != 0:
            msg = (
                f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
            )
            raise ValueError(msg)
        self.num_heads: int = num_heads
        self.head_dim: int = hidden_size // num_heads
        # 因为要使用RoPE 所以head dim必须是偶数
        if self.head_dim % 2 != 0:
            msg = f"head_dim {self.head_dim} must be even to apply RoPE!"
            raise ValueError(msg)

        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    # 全流程都没有用dropout哎
    # 大模型过拟合也没什么事情是吧 狠狠的拟合在数据上！！！
    def forward(self, input_tensor: Tensor) -> Tensor:
        # input: [batch_size, seq_len, hidden_size]
        # attention_mask: [batch_size, seq_len, seq_len] or [1, seq_len, seq_len] if all the batch has the same mask
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        # check the attn_mask args in MHA.forward
        # output: [batch_size, seq_len, hidden_size], output should be the same shape as input

        batch_size, seq_len, hidden_size = input_tensor.shape

        # input_tensor -> split
        # each head * proj

        # 多头注意力的映射其实就是直接用一整个参数矩阵去映射，这个我之前的笔记里面应该有画图解释过
        # 而且llama和retriever的代码里都是这么做的，所以这样写就是对的
        # first, proj the input to q, k, v
        query: Tensor = self.query_proj(input_tensor)
        key: Tensor = self.key_proj(input_tensor)
        value: Tensor = self.value_proj(input_tensor)
        # 然后利用view把q, k, v的shape变成
        # batch_size, seq_len, head_dim -> batch_size, seq_len, num_heads, head_dim
        # -> (batch_size, num_heads), seq_len, head_dim
        query = query.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # batch_siea, num_heads, seq_len, head_dim
        key = key.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = value.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # 然后就可以用pytorch的 scaled dot product attention 了

        # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        # flash attention: speed 更快
        output = func.scaled_dot_product_attention(query, key, value, is_causal=True)

        # output.shape = (batch_size, num_heads, seq_len, head_dim)
        # 然后把multihead给concat起来
        # 然后过一个out proj就行了
        # 1. .view() 要求张量是 contiguous 的 PyTorch 的 .view() 操作，是一种“扁平化 + 重组”的重排方式，它要求原始张量在内存中是连续（contiguous）排列的
        # .reshape() 是更通用的操作，它会尝试返回一个 view，如果失败了，就会创建一个 copy。我们不想copy
        #
        output = (
            output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        )
        return self.out_proj(output)


mha = MultiHeadSelfAttention(hidden_size=16, num_heads=2)
input_tensor = torch.randn(4, 8, 16)  # batch_size=4, seq_len=8, hidden_size=16
output_tensor = mha(input_tensor)
print(output_tensor.shape)  # 应该是 [4, 8, 16]
