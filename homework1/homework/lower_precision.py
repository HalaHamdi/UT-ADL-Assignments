from pathlib import Path 
import torch
 
from .bignet import BIGNET_DIM, LayerNorm 


def block_quantize_3bit(x: torch.Tensor, group_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 1
    assert x.size(0) % group_size == 0

    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=-1, keepdim=True).values
    q = ((x + normalization) / (2 * normalization) * 7).round().to(torch.int32)  # values in 0..7
    shifts = torch.arange(0, 24, 3, dtype=torch.int32)
    word = (q.view(-1, 8) << shifts).sum(dim=1)  # 8 values x 3 bits = 24 bits
    packed = torch.stack([word & 0xFF, (word >> 8) & 0xFF, (word >> 16) & 0xFF], dim=1).to(torch.uint8)
    return packed, normalization.to(torch.float16)


def block_dequantize_3bit(x_quant_3: torch.Tensor, normalization: torch.Tensor) -> torch.Tensor:
    """The reverse operation of block_quantize_3bit."""
    b = x_quant_3.to(torch.int32)
    word = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    shifts = torch.arange(0, 24, 3, dtype=torch.int32, device=b.device)
    q = ((word[:, None] >> shifts) & 7).reshape(normalization.size(0), -1).to(torch.float32)
    return ((q / 7 * 2 - 1) * normalization.to(torch.float32)).view(-1)



class Linear3Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 32) -> None:
        super().__init__()

        self._shape = (out_features, in_features)
        self._group_size = group_size

        # With 3-bit codes, 8 of them make 8 × 3 = 24 bits, which is exactly 3 bytes.
        # in 4 bits 2 of them make 8 bits, which is exactly 1 byte. so we used to not harcode and divide by 2.
        self.register_buffer(
            "weight_q3",
            torch.zeros(out_features * in_features // 8, 3, dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "weight_norm",
            torch.zeros(out_features * in_features // group_size, 1, dtype=torch.float16),
            persistent=False,
        )
        # Register a hook to load the weights from a checkpoint. This function reaches deep into
        # PyTorch internals. It makes sure that Linear4Bit._load_state_dict_pre_hook is called
        # every time the model is loaded from a checkpoint. We will quantize the weights in that function.
        self._register_load_state_dict_pre_hook(Linear3Bit._load_state_dict_pre_hook, with_module=True)
        # Add in an optional bias
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]  # noqa: F841
            del state_dict[f"{prefix}weight"]
            weight_q3, weight_norm = block_quantize_3bit(weight.flatten(), self._group_size)
            self.weight_q3.copy_(weight_q3)
            self.weight_norm.copy_(weight_norm)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # TODO: Dequantize and call the layer
            # Hint: You can use torch.nn.functional.linear
            dequantized = block_dequantize_3bit(self.weight_q3, self.weight_norm).view(self._shape)
            return torch.nn.functional.linear(x, dequantized, self.bias)


class BigNet3Bit(torch.nn.Module):

    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            # TODO: Implement me (feel free to copy and reuse code from bignet.py)
            self.model = torch.nn.Sequential(
                Linear3Bit(channels, channels),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        # TODO: Implement me (feel free to copy and reuse code from bignet.py)
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def load(path: Path | None)->BigNet3Bit:
    # TODO (extra credit): Implement a BigNet that uses in
    # average less than 4 bits per parameter (<9MB)
    # Make sure the network retains some decent accuracy
    net = BigNet3Bit()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net