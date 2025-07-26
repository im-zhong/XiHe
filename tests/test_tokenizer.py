# 2025/7/26
# zhangzhong


from xihe.tokenizer import create_tokenizer


def test_tokenizer() -> None:
    tokenizer = create_tokenizer("gpt2")
    print(tokenizer)
