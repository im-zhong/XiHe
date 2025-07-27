# 2025/7/19
# zhangzhong


import torch
from torch import Tensor, nn

from xihe.model import RMSNorm as MyRMSNorm
from xihe.model import RotaryPositionalEmbedding, Transformer


# code from llama
# https://github.com/meta-llama/llama/blob/main/llama/model.py
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.




    """
    # freqs = 1.0 / (theta ** ( 2.0*torch.arange(0, dim//2)/dim ))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    # https://docs.pytorch.org/docs/stable/generated/torch.outer.html
    # Outer product of input and vec2, if input is a vector of size n and vec2 is a vector of size m,
    # the output will be a matrix of size (n, m).
    # This funciton does not broadcast
    # and input must be a 1-D vector.
    # TIPs: 在我们自己的实现中，我们是使用了一个外层循环来做的，这里用outer计算的更快！
    freqs = torch.outer(t, freqs).float()  # type: ignore
    print("freqs.shape:", freqs.shape)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim  # noqa: PLR0133
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.



    """
    # https://docs.pytorch.org/docs/stable/generated/torch.view_as_complex.html
    # 不管前面的维度是多少，只要最后一个维度是2 就行了 这样最后两个就会合并成一个复数
    # view_as_complex() is only supported for tensors with torch.dtype torch.float64 and torch.float32. The input is expected to have the last dimension of size 2
    # 那么这里的实现就不适合我们，因为我们要用bfloat16
    # 我还是要看懂他的实现，然后我们自己用矩阵运算实现！
    print("xq.shape:", xq.shape)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    print("xq_.shape:", xq_.shape)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # https://docs.pytorch.org/docs/stable/generated/torch.view_as_real.html
    # https://docs.pytorch.org/docs/stable/generated/torch.flatten.html
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    # https://docs.pytorch.org/docs/stable/generated/torch.Tensor.type_as.html
    return xq_out.type_as(xq), xk_out.type_as(xk)


##########################################################################################


# 或许应该先测试matrix
def test_rotary_matrices() -> None:
    dim = 64
    seq_len = 128
    freqs_cis: Tensor = precompute_freqs_cis(dim=dim, end=seq_len)
    # ([128, 32])
    # seq_len, dim//2
    # 首先使用outer 计算出所有的角度
    # 然后使用polar计算出一个复数 cos, sin
    print("freqs_cis.shape:", freqs_cis.shape)
    # 所以，完全体应该是经过broadcast的结果

    # xq_.shape: torch.Size([4, 128, 8, 32])
    xq = torch.randn(size=(4, seq_len, 8, dim))
    xk = torch.randn(size=(4, seq_len, 8, dim))
    apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    freqs_cis_real = torch.view_as_real(freqs_cis)
    assert freqs_cis_real.shape == (seq_len, dim // 2, 2)

    rope = RotaryPositionalEmbedding(dim=dim, max_seq_len=seq_len)
    rotary_matrices = torch.stack([rope.cos_threa, rope.sin_threa], dim=-1)
    assert rotary_matrices.shape == (seq_len, dim // 2, 2)

    # yes! 过测试啦！！！
    assert torch.allclose(rotary_matrices, freqs_cis_real, atol=1e-5)


def test_rope() -> None:
    head_dim = 64
    max_seq_len = 2048
    rope = RotaryPositionalEmbedding(dim=head_dim, max_seq_len=max_seq_len)

    batch_size = 32
    num_heads = 8
    # 我们测试两种形状，一种是(batch_size, seq_len, dim)
    # 不行，这种没法测，只能测一种情况，就是多头
    xq: Tensor = torch.randn(size=(batch_size, num_heads, max_seq_len, head_dim))
    # 另一种形状是 (batch_size, num_heads, seq_len, head_dim)
    xk: Tensor = torch.randn(size=(batch_size, num_heads, max_seq_len, head_dim))

    # tensor2: Tensor = torch.randn(size=(4, 8, 128, head_dim))

    rotated_tensor1 = rope.apply_rope(xq)
    rotated_tensor2 = rope.apply_rope(xk)
    # 验证旋转后的张量形状
    # 竟然对了！
    assert rotated_tensor1.shape == xq.shape
    assert rotated_tensor2.shape == xk.shape

    # 哦，没有，那么我们用llama的代码吧
    # 他这个函数只能一次计算两个
    # 是不是一次计算两个会快一些？大概是吧
    freqs_cis: Tensor = precompute_freqs_cis(dim=head_dim, end=max_seq_len)

    xqq = xq.transpose(1, 2).contiguous()
    xkk = xk.transpose(1, 2).contiguous()
    xq_out, xk_out = apply_rotary_emb(xqq, xkk, freqs_cis=freqs_cis)
    assert xq_out.shape == xqq.shape
    assert xk_out.shape == xkk.shape

    # 接下来就是验证计算结果了
    # 应该在误差之内
    xqq_out = xq_out.transpose(1, 2).contiguous()
    xkk_out = xk_out.transpose(1, 2).contiguous()

    print("xqq_out.shape:", xqq_out.shape)
    print("rotated_tensor1.shape:", rotated_tensor1.shape)
    # 牛逼！过测试啦！！！
    assert torch.allclose(rotated_tensor1, xqq_out, atol=1e-5)
    assert torch.allclose(rotated_tensor2, xkk_out, atol=1e-5)


def test_rms_norm() -> None:
    # 这个函数是用来测试 rms_norm 的
    # 我们需要一个张量
    x = torch.randn(size=(4, 8, 128, 64))
    # 然后我们计算它的 RMS
    rms = nn.RMSNorm(normalized_shape=x.shape[-1])
    rms.weight = nn.Parameter(torch.randn_like(rms.weight))  # 确保权重是1
    output = rms(x)
    # 验证输出形状
    assert output.shape == x.shape

    # 验证 RMS 是否正确
    myrms = MyRMSNorm(normalized_shape=x.shape[-1])
    myrms.scale = rms.weight
    my_output = myrms(x)
    # 验证输出形状
    assert my_output.shape == x.shape

    # 验证输出是否相等
    assert torch.allclose(output, my_output, atol=1e-5)


def test_transformer() -> None:
    torch.cuda.set_device(0)

    vocab_size = 1024
    hidden_size = 128
    num_layers = 2
    num_heads = 4
    intermediate_size = hidden_size * 4
    max_seq_len = 512

    transformer = Transformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        max_seq_len=max_seq_len,
    )

    # 测试输入
    batch_size = 32
    tokens = torch.randint(low=0, high=vocab_size, size=(batch_size, max_seq_len))
    output: Tensor = transformer(tokens)
    # 验证输出形状
    assert output.shape == (batch_size, max_seq_len, vocab_size)
    assert not torch.any(torch.isnan(output))


def test_transformer_state_dict() -> None:
    torch.cuda.set_device(0)

    vocab_size = 1024
    hidden_size = 128
    num_layers = 2
    num_heads = 4
    intermediate_size = hidden_size * 4
    max_seq_len = 512

    transformer = Transformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        max_seq_len=max_seq_len,
    )

    state_dict = transformer.state_dict()
    print("State dict keys:", state_dict.keys())

    transformer.load_state_dict(state_dict)
