# 2025/7/26
# zhangzhong


from xihe.tokenizer import create_tokenizer


def test_tokenizer() -> None:
    tokenizer = create_tokenizer("gpt2")
    print(tokenizer)


def test_eos_token() -> None:
    tokenizer = create_tokenizer("gpt2")
    eos_token = tokenizer.eos_token
    assert isinstance(eos_token, str), "EOS token should be a string"

    print(f"EOS Token: {eos_token}")
    assert eos_token is not None, "EOS token should not be None"
    assert isinstance(eos_token, str), "EOS token should be a string"
    assert len(eos_token) > 0, "EOS token should not be an empty string"
