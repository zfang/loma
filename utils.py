from typing import Tuple

import torch
from torch import nn
from transformers import LlamaModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from loma import LomaDecoderLayer
from loma import LomaModel


def svd_lowrank(input_mat: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    U, S, Vh = torch.linalg.svd(input_mat)
    return U[..., :, :rank] @ torch.diag(S[:rank]), Vh[..., :rank, :]


@torch.no_grad()
def fisher_weighted_svd(input_mat: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    I_hat = torch.diag((input_mat.grad ** 2).sum(dim=-1) ** 0.5)
    U_S_truncated, Vh_truncated = svd_lowrank(I_hat @ input_mat, rank)
    I_hat_inv = 1 / I_hat  # Quick way to compute the inverse of a diagonal matrix is 1/d
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


def fwsvd_weight_copy(src: nn.Linear, dst: nn.Sequential):
    a, b = fisher_weighted_svd(src.weight)
    dst[0].weight.data = a.data
    dst[1].weight.data = b.data

    if src.bias is not None:
        dst[1].bias.data = src.bias.data


def fwsvd_decoder_copy(llama_decoder_layer: LlamaDecoderLayer, loma_decoder_layer: LomaDecoderLayer):
    for proj_attr in [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]:
        src = getattr(llama_decoder_layer.self_attn, proj_attr)
        dst = getattr(loma_decoder_layer.self_attn, proj_attr)
        if check_copy(src, dst):
            fwsvd_weight_copy(src, dst)

    for proj_attr in [
        "gate_proj",
        "up_proj",
        "down_proj",
    ]:
        src = getattr(llama_decoder_layer.mlp, proj_attr)
        dst = getattr(loma_decoder_layer.mlp, proj_attr)
        if check_copy(src, dst):
            fwsvd_weight_copy(src, dst)


def fwsvd_model_copy(llama_model: LlamaModel, loma_model: LomaModel):
    assert len(llama_model.layers) == len(loma_model.layers), (len(llama_model.layers), len(loma_model.layers))

    for llama_layer, loma_layer in zip(llama_model.layers, loma_model.layers):
        fwsvd_decoder_copy(llama_layer, loma_layer)