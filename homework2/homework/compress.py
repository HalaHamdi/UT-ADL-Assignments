from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image

from .autoregressive import Autoregressive
from .bsq import Tokenizer


# Used LLM to explain what is arithmetic coding and what are the possible ways of implementing it (including a way that imported constriction library)
# then used these explanations to implement the arithmetic coding compression and decompression functions.


H, W = 20, 30  # token grid: (100 / 5) x (150 / 5)
N = H * W  # 600 tokens per image
BITS = 20  # probabilities become whole-number counts out of 2**BITS


class Compressor:
    def __init__(self, tokenizer: Tokenizer, autoregressive: Autoregressive):
        super().__init__()
        self.tokenizer = tokenizer
        self.autoregressive = autoregressive

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> bytes:
        """
        Compress the image into a bytes stream.

        Use arithmetic coding.
        """
        self.tokenizer.eval()
        self.autoregressive.eval()

        x = x.unsqueeze(0)  # (100, 150, 3) --> (1, 100, 150, 3)
        tokens = self.tokenizer.encode_index(x)  # (1, 100, 150, 3) --> (1, 20, 30)
        flat = tokens.reshape(1, N)  # (1, 600) the real tokens

        # Get the counts for every position the way the decoder will(one token at a time),
        # with only the earlier tokens visible. 
        seen = torch.zeros_like(flat)
        rows = []
        for i in range(N):
            rows.append(self._row_freqs(seen, i))  # (1024,)
            seen[0, i] = flat[0, i]
        freqs = np.stack(rows)  # (600, 1024)
        cums = np.cumsum(freqs, axis=1) - freqs  # where each token's slot starts
        syms = flat.reshape(-1).cpu().numpy()  # (600,)

        # Each token narrows the interval [low, low + rng).
        low, rng = 0, 1
        for i, s in enumerate(syms):
            low = (low << BITS) + rng * int(cums[i, s])
            rng = rng * int(freqs[i, s])

        # Write the shortest number (whole bytes) that lies inside the final interval.
        n_bits = BITS * N
        for k in range(n_bits, -1, -8):
            v = -((-low) >> k) << k
            if v < low + rng:
                break
        return v.to_bytes(n_bits // 8, "big").rstrip(b"\x00")

    @torch.no_grad()
    def decompress(self, x: bytes) -> torch.Tensor:
        """
        Decompress bytes into an image tensor (100, 150, 3).
        You may assume the output image is 150 x 100 pixels.
        """
        self.tokenizer.eval()
        self.autoregressive.eval()
        S = BITS * N
        device = next(self.autoregressive.parameters()).device

        code = int.from_bytes(x.ljust(S // 8, b"\x00"), "big")  # put back the stripped zero bytes

        seen = torch.zeros(1, N, dtype=torch.long, device=device)
        low, rng = 0, 1
        for i in range(N):
            freqs = self._row_freqs(seen, i)  # (1024,)
            cums = np.cumsum(freqs) - freqs

            y = code >> (S - BITS * (i + 1))
            q = (y - (low << BITS)) // rng
            s = int(np.searchsorted(cums, q, side="right")) - 1  # which token's slot holds q

            seen[0, i] = s
            low = (low << BITS) + rng * int(cums[s])
            rng = rng * int(freqs[s])

        return self.tokenizer.decode_index(seen.reshape(1, H, W))[0]  # (100, 150, 3)

    def _row_freqs(self, seen: torch.Tensor, i: int) -> np.ndarray:
        ''' return the expected token at position i given the seen tokens '''
        logits, _ = self.autoregressive(seen.reshape(1, H, W))  # (1, 20, 30, 1024)
        probs = torch.softmax(logits.cpu().double(), dim=-1).reshape(N, -1)[i : i + 1]  # (1, 1024)
        return self._freqs(probs)[0]

    def _freqs(self, probs: torch.Tensor):
        # probs: (M, 1024) probabilities -> (M, 1024) whole numbers, each at least 1
        p = probs.double().cpu().numpy()
        p = p / p.sum(axis=1, keepdims=True)
        return np.floor(p * (2**BITS - p.shape[1])).astype(np.int64) + 1


def compress(tokenizer: Path, autoregressive: Path, image: Path, compressed_image: Path):
    """
    Compress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    images: Path to the image to compress.
    compressed_image: Path to save the compressed image tensor.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    x = torch.tensor(np.array(Image.open(image)), dtype=torch.uint8, device=device)
    cmp_img = cmp.compress(x.float() / 255.0 - 0.5)
    with open(compressed_image, "wb") as f:
        f.write(cmp_img)


def decompress(tokenizer: Path, autoregressive: Path, compressed_image: Path, image: Path):
    """
    Decompress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    compressed_image: Path to the compressed image tensor.
    images: Path to save the image to compress.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    with open(compressed_image, "rb") as f:
        cmp_img = f.read()

    x = cmp.decompress(cmp_img)
    img = Image.fromarray(((x + 0.5) * 255.0).clamp(min=0, max=255).byte().cpu().numpy())
    img.save(image)


if __name__ == "__main__":
    from fire import Fire

    Fire({"compress": compress, "decompress": decompress})