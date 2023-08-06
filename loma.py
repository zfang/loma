from collections import OrderedDict

from torch import nn
from transformers import LlamaConfig
from transformers import LlamaForCausalLM
from transformers import LlamaModel
from transformers import LlamaPreTrainedModel
from transformers import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.llama.modeling_llama import LlamaRMSNorm


def construct_low_rank_proj(in_features: int, out_features: int, rank: int, bias: bool = False) -> nn.Module:
    return nn.Sequential(OrderedDict([
        ("down_proj", nn.Linear(in_features, rank, bias=False)),
        ("up_proj", nn.Linear(rank, out_features, bias=bias)),
    ]))


def set_low_rank_proj(module: nn.Module, config: PretrainedConfig, proj_attr: str, config_attr: str):
    config_value = getattr(config, config_attr)
    if config_value is not None:
        proj_module = getattr(module, proj_attr)
        low_rank_proj_module = construct_low_rank_proj(
            proj_module.in_features,
            proj_module.out_features,
            config_value,
            proj_module.bias is not None,
        )

        setattr(module, proj_attr, low_rank_proj_module)


class LomaConfig(LlamaConfig):
    def __init__(
        self,
        self_attn_q_proj_r=64,
        self_attn_k_proj_r=64,
        self_attn_v_proj_r=64,
        self_attn_o_proj_r=64,
        mlp_gate_proj_r=64,
        mlp_up_proj_r=64,
        mlp_down_proj_r=64,
        *args,
        **kwargs,
    ):
        self.self_attn_q_proj_r = self_attn_q_proj_r
        self.self_attn_k_proj_r = self_attn_k_proj_r
        self.self_attn_v_proj_r = self_attn_v_proj_r
        self.self_attn_o_proj_r = self_attn_o_proj_r
        self.mlp_gate_proj_r = mlp_gate_proj_r
        self.mlp_up_proj_r = mlp_up_proj_r
        self.mlp_down_proj_r = mlp_down_proj_r

        super().__init__(*args, **kwargs)

    @classmethod
    def from_llama_config(cls, llama_config: LlamaConfig, *args, **kwargs):
        loma_config = cls(*args, **kwargs)
        for key, value in llama_config.__dict__.items():
            loma_config.__dict__[key] = value

        return loma_config


class LomaAttention(LlamaAttention):
    def __init__(self, config: LomaConfig):
        super().__init__(config)

        for proj_attr, config_attr in [
            ("q_proj", "self_attn_q_proj_r"),
            ("k_proj", "self_attn_k_proj_r"),
            ("v_proj", "self_attn_v_proj_r"),
            ("o_proj", "self_attn_o_proj_r"),
        ]:
            set_low_rank_proj(self, config, proj_attr, config_attr)


class LomaMLP(LlamaMLP):
    def __init__(self, config: LomaConfig):
        super().__init__(config)

        for proj_attr, config_attr in [
            ("gate_proj", "mlp_up_proj_r"),
            ("up_proj", "mlp_up_proj_r"),
            ("down_proj", "mlp_down_proj_r"),
        ]:
            set_low_rank_proj(self, config, proj_attr, config_attr)


class LomaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LomaConfig):
        super(LlamaDecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LomaAttention(config=config)
        self.mlp = LomaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class LomaPreTrainedModel(LlamaPreTrainedModel):
    config_class = LomaConfig
    _no_split_modules = ["LomaDecoderLayer"]


class LomaModel(LlamaModel):
    def __init__(self, config: LomaConfig):
        super(LlamaPreTrainedModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LomaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


class LomaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LomaConfig):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LomaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
