from collections import OrderedDict

from torch import nn
from transformers import LlamaConfig
from transformers import LlamaForCausalLM
from transformers import LlamaModel
from transformers import LlamaPreTrainedModel
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.llama.modeling_llama import LlamaRMSNorm


def construct_low_rank_proj(in_features: int, out_features: int, rank: int, bias: bool = False) -> nn.Module:
    return nn.Sequential(OrderedDict([
        ("down_proj", nn.Linear(in_features, rank, bias=False)),
        ("up_proj", nn.Linear(rank, out_features, bias=bias)),
    ]))


def set_low_rank_proj(
    module: nn.Module,
    config: PretrainedConfig,
    proj_attr: str,
    config_attr: str,
    in_features: int,
    out_features: int,
    bias: bool,
):
    config_value = getattr(config, config_attr)
    if config_value is not None:
        low_rank_proj_module = construct_low_rank_proj(
            in_features,
            out_features,
            config_value,
            bias,
        )

        setattr(module, proj_attr, low_rank_proj_module)
    else:
        setattr(module, proj_attr, nn.Linear(in_features, out_features, bias))



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
        attr_config_tuples = [
            ("q_proj", "self_attn_q_proj_r", config.hidden_size, config.hidden_size, False),
            ("k_proj", "self_attn_k_proj_r", config.hidden_size, config.hidden_size, False),
            ("v_proj", "self_attn_v_proj_r", config.hidden_size, config.hidden_size, False),
            ("o_proj", "self_attn_o_proj_r", config.hidden_size, config.hidden_size, False),
        ]
        if any(getattr(config, config_attr) is not None for _, config_attr, _, _, _ in attr_config_tuples):
            super(LlamaAttention, self).__init__()

            self.config = config
            self.hidden_size = config.hidden_size
            self.num_heads = config.num_attention_heads
            self.head_dim = self.hidden_size // self.num_heads
            self.num_key_value_heads = config.num_key_value_heads
            self.num_key_value_groups = self.num_heads // self.num_key_value_heads
            self.pretraining_tp = config.pretraining_tp
            self.max_position_embeddings = config.max_position_embeddings

            if (self.head_dim * self.num_heads) != self.hidden_size:
                raise ValueError(
                    f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                    f" and `num_heads`: {self.num_heads})."
                )

            for proj_attr, config_attr, in_features, out_features, bias in attr_config_tuples:
                set_low_rank_proj(self, config, proj_attr, config_attr, in_features, out_features, bias)

            self._init_rope()
        else:
            super().__init__(config)


class LomaMLP(LlamaMLP):
    def __init__(self, config: LomaConfig):
        attr_config_tuples = [
            ("gate_proj", "mlp_up_proj_r", config.hidden_size, config.intermediate_size, False),
            ("up_proj", "mlp_up_proj_r", config.hidden_size, config.intermediate_size, False),
            ("down_proj", "mlp_down_proj_r", config.intermediate_size, config.hidden_size, False),
        ]

        if any(getattr(config, config_attr) is not None for _, config_attr, _, _, _ in attr_config_tuples):
            super(LlamaMLP, self).__init__()

            self.pretraining_tp = config.pretraining_tp
            self.hidden_size = config.hidden_size
            self.intermediate_size = config.intermediate_size

            for proj_attr, config_attr, in_features, out_features, bias in attr_config_tuples:
                set_low_rank_proj(self, config, proj_attr, config_attr, in_features, out_features, bias)

            self.act_fn = ACT2FN[config.hidden_act]
        else:
            super().__init__(config)


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
