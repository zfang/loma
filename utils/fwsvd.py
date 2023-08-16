from typing import Tuple

import torch
from torch import nn
from tqdm import tqdm
from transformers import LlamaModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from loma import LomaDecoderLayer
from loma import LomaModel


def svd_lowrank(input_mat: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    U, S, Vh = torch.linalg.svd(input_mat)
    return U[..., :, :rank] @ torch.diag(S[:rank]), Vh[..., :rank, :]


@torch.no_grad()
def fisher_weighted_svd(
    input_mat: torch.Tensor,
    input_grad: torch.Tensor,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    I_hat = torch.diag((input_grad ** 2).sum(dim=-1) ** 0.5)
    U_S_truncated, Vh_truncated = svd_lowrank(I_hat @ input_mat, rank)
    I_hat_inv = torch.linalg.inv(I_hat)
    A = I_hat_inv @ U_S_truncated
    B = Vh_truncated
    return A, B


def check_copy(src: nn.Linear, dst: nn.Module) -> bool:
    if not (
        isinstance(dst, nn.Sequential) and
        len(dst) == 2 and
        isinstance(dst[0], nn.Linear) and
        isinstance(dst[1], nn.Linear)
    ):
        return False

    assert src.in_features == dst[0].in_features, (src.in_features, dst[0].in_features)
    assert src.out_features == dst[1].out_features, (src.out_features, dst[1].out_features)
    assert (src.bias is None) == (dst[1].bias is None), (src.bias is None, dst[1].bias is None)

    return True


def fwsvd_weight_copy(
    src: nn.Linear,
    dst: nn.Sequential,
    gradient_scale: float = 1.
):
    if gradient_scale != 1:
        src.weight.grad /= gradient_scale

    device = dst[0].weight.device
    dtype = dst[0].weight.dtype
    input_grad = src.weight.grad.T.to(device=device)
    input_mat = src.weight.T.to(device=device)
    a, b = fisher_weighted_svd(
        input_mat=input_mat,
        input_grad=input_grad,
        rank=dst[0].out_features
    )
    dst[0].weight.data = a.T.data.to(dtype=dtype)
    dst[1].weight.data = b.T.data.to(dtype=dtype)

    if src.bias is not None:
        dst[1].bias.data = src.bias.data.to(dtype=dtype)


def fwsvd_decoder_copy(
    llama_decoder_layer: LlamaDecoderLayer,
    loma_decoder_layer: LomaDecoderLayer,
    gradient_scale: float = 1.
):
    for proj_attr in [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]:
        src = getattr(llama_decoder_layer.self_attn, proj_attr)
        dst = getattr(loma_decoder_layer.self_attn, proj_attr)
        if check_copy(src, dst):
            fwsvd_weight_copy(src, dst, gradient_scale)

    for proj_attr in [
        "gate_proj",
        "up_proj",
        "down_proj",
    ]:
        src = getattr(llama_decoder_layer.mlp, proj_attr)
        dst = getattr(loma_decoder_layer.mlp, proj_attr)
        if check_copy(src, dst):
            fwsvd_weight_copy(src, dst, gradient_scale)


def fwsvd_model_copy(
    llama_model: LlamaModel,
    loma_model: LomaModel,
    gradient_scale: float = 1.,
    show_progress: bool = False
):
    assert len(llama_model.layers) == len(loma_model.layers), (len(llama_model.layers), len(loma_model.layers))

    iterable = zip(llama_model.layers, loma_model.layers)
    if show_progress:
        iterable = tqdm(iterable, total=len(llama_model.layers), desc="Running FWSVD")

    for llama_layer, loma_layer in iterable:
        fwsvd_decoder_copy(llama_layer, loma_layer, gradient_scale)
