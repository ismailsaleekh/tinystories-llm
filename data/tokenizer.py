"""GPT-2 tokenizer wrapper using tiktoken."""
import tiktoken
from typing import List


class Tokenizer:
    """Wrapper around GPT-2 tokenizer."""

    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot_token = self.enc.eot_token  # End of text token (50256)
        self.vocab_size = self.enc.n_vocab   # 50257

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        return self.enc.decode(tokens)

    def __call__(self, text: str) -> List[int]:
        return self.encode(text)


if __name__ == "__main__":
    tokenizer = Tokenizer()

    text = "Once upon a time, there was a little girl named Lily."
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    print(f"Original: {text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"EOT token: {tokenizer.eot_token}")
    print(f"Round-trip OK: {text == decoded}")
