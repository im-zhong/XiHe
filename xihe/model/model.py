# 2025/7/18
# zhangzhong
# 主要参考两处代码
# https://github.com/zhaibowen/Retriever/blob/main/model.py
# https://github.com/meta-llama/llama/blob/main/llama/model.py

import torch
import torch.nn.functional as func
from torch import Tensor, nn


# 既然如此，不如RMSNorm也自己写了吧
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


# ok！成功了25%了吧，大概
# 接下来就是把这个东西给实现出来
# 我们接受的tensor.shape = (*, dim)
# 然后我们其实就是用一个旋转矩阵乘上这个东西而已啊
# 因为RoPE中的R矩阵有大量的零，所以没必要保存那么多
# 我们只需要保存数个2x2的旋转矩阵就行了
# 然后再做矩阵乘法的时候，也可以用bmm来做，然后再做reshape就行了
# 这样的实现肯定是更加高效的
class RotaryPositionalEmbedding(nn.Module):
    # 为registerbuffer提供类型注释
    cos_theta: Tensor
    sin_theta: Tensor

    # 这里的dim是什么?
    def __init__(self, head_dim: int, max_seq_len: int) -> None:
        super().__init__()
        # dim: 维度
        # max_seq_len: 最大序列长度
        # 这个类的作用是计算RoPE中的R矩阵
        # 然后在forward中应用到输入的tensor上
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer

        # 计算R矩阵
        # shape = (max_seq_len, dim // 2, 2, 2)
        cos_theta, sin_theta = self.calculate_r_matrix(head_dim, max_seq_len)

        # 我在这里显示的改成 to float16 还是对显存和速度没有丝毫的影响
        cos = (
            cos_theta.view(1, 1, max_seq_len, head_dim // 2)
            # .to(torch.float16)
            .contiguous()
        )
        sin = (
            sin_theta.view(1, 1, max_seq_len, head_dim // 2)
            # .to(torch.float16)
            .contiguous()
        )

        # self.cos_threa = cos_threa.to(device)
        # self.sin_threa = sin_threa.to(device)
        # self.cos_threa = cos_threa
        # self.sin_threa = sin_threa
        self.register_buffer(name="cos_theta", tensor=cos, persistent=False)
        self.register_buffer(name="sin_theta", tensor=sin, persistent=False)

    # TODO
    # 卧槽，忘了，还需要设置dtype，因为我们想要使用bfloat16
    # 我们不需要计算出整个矩阵，只需要计算出一个cos 一个sin的矩阵就行了
    # 维度是 max_seq_len, dim // 2
    def calculate_r_matrix(
        self, dim: int, max_seq_len: int, base_freq: float = 10000
    ) -> tuple[Tensor, Tensor]:
        if dim % 2 != 0:
            msg = f"dim {dim} must be even to apply RoPE!"
            raise ValueError(msg)
        # d = dim // 2
        # 计算旋转矩阵
        # dim: 维度
        # max_seq_len: 最大序列长度
        # 返回一个形状为 (max_seq_len, dim) 的张量
        # 这个张量的每一行都是一个2x2的旋转矩阵
        # rotary_matrices = torch.zeros(max_seq_len, d, 2, 2)
        # # 先把串行的写出来吧
        # # 再看看怎么改成并行的
        # # TODO: 确认一下是不是从1开始的
        # Llama代码里面是从0开始的
        # for pos in range(1, max_seq_len + 1):
        #     indices = torch.arange(0, d)  # shape = (d,)
        #     theta = pos / 10000 ** (2 * indices / d)
        #     cos_theta = torch.cos(theta)
        #     sin_theta = torch.sin(theta)  # shape = (d,)
        #     rotary_matrices[pos - 1, :, 0, 0] = cos_theta
        #     rotary_matrices[pos - 1, :, 0, 1] = -sin_theta
        #     rotary_matrices[pos - 1, :, 1, 0] = sin_theta
        #     rotary_matrices[pos - 1, :, 1, 1] = cos_theta

        # return rotary_matrices

        freqs: Tensor = 1.0 / (base_freq ** (2.0 * torch.arange(0, dim // 2) / dim))

        pos: Tensor = torch.arange(0, max_seq_len)

        theta = torch.outer(pos, freqs)  # shape = (max_seq_len, dim // 2)

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)  # shape = (max_seq_len, dim //
        # 我们在内存上组织成和ploar 函数一样的吧
        # 就是shape变成 (max_seq_len, dim // 2, 2)
        # 里面的2 就是 cos, sin 这样的顺序
        # 不方便，我还是不stack起来了
        # 就直接返回cos 和sin吧
        # rotary_matrices = torch.stack([cos_theta, sin_theta], dim=-1)
        # assert rotary_matrices.shape == (max_seq_len, dim // 2, 2)
        # return rotary_matrices
        return cos_theta, sin_theta

    # 然后就是传入一个tensor，我们对其应用rotary positional embedding
    # 我们也不取长度了
    # def forward(self, tensor: Tensor) -> Tensor:
    def apply_rope(self, tensor: Tensor) -> Tensor:
        # 传入的tensor和返回的tensor的shape必须是一样的
        # 我们不在呼tensor的shape
        # 就跟layernorm一样

        # 不对！我们还需要position
        # 所以
        batch_size, num_heads, seq_len, head_dim = tensor.shape
        seq_len: int = tensor.shape[-2]
        # 还要保证传入的seq_len不能超过max_seq_len
        if seq_len > self.max_seq_len:
            msg = f"seq_len {seq_len} must be less than or equal to max_seq_len {self.max_seq_len} to apply RoPE!"
            raise ValueError(msg)

        dim: int = tensor.shape[-1]
        # 因为我们的旋转矩阵是预选计算的，所以我们必须保证传入的tensor的dim和初始化的dim是一致的
        if dim != self.head_dim:
            msg = f"dim {dim} must be equal to the initialized dim {self.head_dim} to apply RoPE!"
            raise ValueError(msg)
        # the last dim must be even
        if dim % 2 != 0:
            msg = f"dim {dim} must be even to apply RoPE!"
            raise ValueError(msg)

        # x = tensor.view(-1, seq_len, dim)  # 保证tensor是三维的
        # x = x.view(-1, seq_len, dim // 2, 2)  # shape = (batch_size, seq_len, d, 2)
        # x = x.unsqueeze(-1)  # shape = (batch_size, seq_len, d, 2, 1)
        # # 我们只需要取rotary matrix中前seq_len就够了
        # # shape = (seq_len, dim // 2, 2, 2)
        # rotary_matrices = self.rotary_matrices[:seq_len, :, :, :].to(tensor.device)
        # rotary_matrices = rotary_matrices.unsqueeze(0)  # shape = (1, seq_len, d, 2, 2)
        # rotary_matrices = rotary_matrices.expand(x.shape[0], -1, -1, -1, -1)

        # # 然后全部都
        # # 没法直接用bmm，需要reshape一下
        # output = torch.bmm(
        #     rotary_matrices.reshape(-1, 2, 2), x.view(-1, 2, 1)
        # )  # shape = (batch_size, seq_len, d, 2)
        # output = output.view(tensor.shape)

        x = tensor.view(*tensor.shape[:-1], head_dim // 2, 2)
        # x.shape = (*, head_dim//2, 2)
        # cos 和 sin 需要做broadcast
        # TIPs: 这里想做broadcast 必须对输入向量的shape进行规定

        # 这个可以提前展开，不需要每次apply都做reshape
        # cos = self.cos_theta.view(1, 1, seq_len, head_dim // 2)
        # sin = self.sin_theta.view(1, 1, seq_len, head_dim // 2)
        # rotate
        # 仍然没有任何的提升。。。
        # cos = self.cos_theta
        # sin = self.sin_theta
        x0 = x[..., 0] * self.cos_theta - x[..., 1] * self.sin_theta
        x1 = x[..., 0] * self.sin_theta + x[..., 1] * self.cos_theta
        # x0, x1 shape = (*, head_dim//2)
        # 然后再concat起来
        output = torch.stack([x0, x1], dim=-1).flatten(start_dim=-2)
        assert output.shape == tensor.shape
        # 和llama里面的实现一样，加上一个typeas 可能传进来的是float16的 可能能降低显存吧
        # 这里llama和retriever都转换了类型，和输入类型一致，不过经过我的测试，对显存占用和速度没有任何影响
        return output.contiguous().type_as(tensor)


class MultiHeadSelfAttention(nn.Module):
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    # F.scaled_dot_product_attention
    # flash attention 就是快！
    # 因为我们要在经过映射的q和k，再进行一个Rotary旋转运算
    #

    # 我们没必要区分query key 和value了
    # 我们就是self attention

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
    def forward(
        self,
        input_rensor: Tensor,
        rope: RotaryPositionalEmbedding | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        # input: [batch_size, seq_len, hidden_size]
        # attention_mask: [batch_size, seq_len, seq_len] or [1, seq_len, seq_len] if all the batch has the same mask
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        # check the attn_mask args in MHA.forward
        # output: [batch_size, seq_len, hidden_size], output should be the same shape as input

        batch_size, seq_len, hidden_size = input_rensor.shape
        # 多头注意力的映射其实就是直接用一整个参数矩阵去映射，这个我之前的笔记里面应该有画图解释过
        # 而且llama和retriever的代码里都是这么做的，所以这样写就是对的
        # first, proj the input to q, k, v
        query: Tensor = self.query_proj(input_rensor)
        key: Tensor = self.key_proj(input_rensor)
        value: Tensor = self.value_proj(input_rensor)
        # 然后利用view把q, k, v的shape变成
        # batch_size, seq_len, head_dim -> batch_size, seq_len, num_heads, head_dim
        # -> (batch_size, num_heads), seq_len, head_dim
        query = query.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = value.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # 然后这个时候就可以使用RoPE了
        # 不过我们有一个问题，RoPE和MHA要如何结合呢
        # 因为使用了多头，hidden size变小了
        # 是在多头之前使用RoPE还是在split head之后的head dim上使用RoPE
        # 这肯定是不一样的呀
        # 看起来llama和retriever的代码应该都是在split之后的head dim上使用的RoPE
        # chatgpt也是这么说的，而且说的很有道理
        # 多头的本质就是每个头都独立运算，就是hidden_size就是一个头一样，所有的head_dim的处理都应该和hidden_size一样
        # 1.	RoPE 只作用于 Q 和 K，不作用于 V。
        # 2.	head_dim 必须是 偶数，因为我们每两个维度为一对做旋转。
        # TODO: 改成和llama一样，在transpose之前apply RoPE
        if rope is not None:
            query = rope.apply_rope(query)
            key = rope.apply_rope(key)
        # 可以用一个参数来控制是否做RoPE
        # 就是从外部传入R矩阵，同样也可以不传，如果不传我们就不做

        # 然后就可以用pytorch的 scaled dot product attention 了

        # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        if attention_mask is None:
            output = func.scaled_dot_product_attention(
                query, key, value, is_causal=True
            )
        else:
            output = func.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask
            )

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


# 这里要应用SwiGLU
# 也就是有三个矩阵
# 两个输入，一个输出
class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # 这里的参数都是没有bias的
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.silu = nn.SiLU()

    def forward(self, inx: Tensor) -> Tensor:
        x1 = self.silu(self.w1(inx))
        x2 = self.w2(inx)
        # 然后x1过silu
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html
        x = x1 * x2
        return self.w3(x)


# 这里要应用Pre-LN Transformer
class TransformerBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, intermediate_size: int
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size

        # TODO: norm应该作用在整个向量上，还是作用在一个头上？
        self.norm1 = RMSNorm(hidden_size)
        # self.norm1 = nn.RMSNorm(hidden_size)
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads)

        self.norm2 = RMSNorm(hidden_size)
        # self.norm2 = nn.RMSNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, intermediate_size)

    def forward(
        self,
        x: Tensor,
        rope: RotaryPositionalEmbedding | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        # input: [batch_size, seq_len, hidden_size]
        # rope: RotaryPositionalEmbedding | None
        # attention_mask: [batch_size, seq_len, seq_len] or [1, seq_len, seq_len] if all the batch has the same mask
        # output: [batch_size, seq_len, hidden_size]

        # 先做自注意力
        h = x + self.attention(self.norm1(x), rope=rope, attention_mask=attention_mask)
        return h + self.ffn(self.norm2(h))
        # return y


class Transformer(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        device: str = "cpu",
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.rope = RotaryPositionalEmbedding(hidden_size // num_heads, max_seq_len)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_heads, intermediate_size)
                for _ in range(num_layers)
            ]
        )

        self.norm = RMSNorm(hidden_size)
        # self.norm = nn.RMSNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)
        # shared embedding and output layer
        # self.token_embedding.weight = self.output.weight
        # 两种写法是等价的
        self.output.weight = self.token_embedding.weight
        assert self.output.weight is self.token_embedding.weight, (
            "Output layer weight must be the same as token embedding weight!"
        )

    # TODO:
    # maybe need specific initialization

    # 这里的输入应该是tokens
    # 而且shape也是 batch_size, seq_len
    # 要经过embedding才能变成 batch_size, seq_len, hidden_size
    # 那问题来了，输出应该是什么呢？
    # 是最后一层的输出？
    # 还是转回token？
    def forward(self, tokens: Tensor) -> Tensor:
        # tokens: [batch_size, seq_len]
        batch_size, seq_len = tokens.shape

        # 先做token embedding
        x = self.token_embedding(tokens)
        # 过layers
        # 不能直接这么过

        # h = self.layers(x, rope=self.rope)
        for layer in self.layers:
            x = layer(x, rope=self.rope)
        # output
        # llama还在模型的最后一层加了一个输出层
        return self.output(self.norm(x))
        # output: [batch_size, seq_len, vocab_size]
        # return output
