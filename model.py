import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast

# Configure logging
logging.basicConfig(level=logging.INFO)

@dataclass
class Llama4TextConfig:
    vocab_size: int = 128000
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    attention_bias: bool = False
    use_qk_norm: bool = True
    num_local_experts: int = 8
    num_experts_per_tok: int = 2
    moe_layers: List[int] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    attention_chunk_size: Optional[int] = None
    use_flash_attention: bool = False
    pad_token_id: int = 0
    initializer_range: float = 0.02
    hidden_act: str = "silu"

    def __post_init__(self):
        if self.moe_layers is None:
            self.moe_layers = []
        if self.rope_scaling is None:
            self.rope_scaling = {"type": "linear", "factor": 1.0}
        self.no_rope_layers = [i for i in range(self.num_hidden_layers) if i % 2 == 0]

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # print(f"apply_rotary_emb: xq.shape={xq.shape}, xk.shape={xk.shape}, freqs_cis.shape={freqs_cis.shape}")
    batch_size = xq.size(0)
    xq_ = torch.view_as_complex(xq.float().reshape(batch_size, xq.shape[1], xq.shape[2], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(batch_size, xk.shape[1], xk.shape[2], -1, 2))
    freqs_cis_q = freqs_cis[:, None, :, :].expand(batch_size, xq.shape[1], -1, -1)
    freqs_cis_k = freqs_cis[:, None, :, :].expand(batch_size, xk.shape[1], -1, -1)
    xq_out = torch.view_as_real(xq_ * freqs_cis_q).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis_k).flatten(3)
    # print(f"apply_rotary_emb: xq_out.shape={xq_out.shape}, xk_out.shape={xk_out.shape}")
    return xq_out.type_as(xq), xk_out.type_as(xk)

def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: Optional[torch.Tensor],
    sequence_length: int,
    cache_position: torch.LongTensor,
    device: torch.device,
    dtype: torch.dtype,
    past_key_values: Optional['DynamicCache'] = None,
) -> torch.Tensor:
    batch_size = attention_mask.shape[0] if attention_mask is not None else 1
    num_new_tokens = cache_position.shape[-1]
    total_seq_length = sequence_length
    if past_key_values is not None:
        past_seq_length = past_key_values.get_seq_length()
        total_seq_length = past_seq_length + num_new_tokens
    # print(f"_prepare_4d_causal_attention_mask: batch_size={batch_size}, num_new_tokens={num_new_tokens}, total_seq_length={total_seq_length}")
    mask = torch.full(
        (batch_size, 1, num_new_tokens, total_seq_length),
        float("-inf"),
        device=device,
        dtype=dtype,
    )
    for i in range(num_new_tokens):
        current_pos = cache_position[i] if cache_position.dim() == 1 else cache_position[:, i]
        mask[:, :, i, : current_pos + 1] = 0.0
    if attention_mask is not None:
        # print(f"attention_mask.shape before processing: {attention_mask.shape}")
        if attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask[:, None, :, :]
        if attention_mask.shape[-1] != total_seq_length:
            pad_size = total_seq_length - attention_mask.shape[-1]
            attention_mask = F.pad(attention_mask, (0, pad_size), value=0.0)
        # print(f"attention_mask.shape after padding: {attention_mask.shape}")
        mask = mask + attention_mask
    return mask
class DynamicCache:
    def __init__(self):
        self.key_cache = []
        self.value_cache = []

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < len(self.key_cache):
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        else:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self) -> int:
        return self.key_cache[0].shape[-2] if self.key_cache else 0

@dataclass
class BaseModelOutputWithPast:
    last_hidden_state: torch.Tensor
    past_key_values: Optional[DynamicCache] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    router_logits: Optional[Tuple[torch.Tensor]] = None
    logits: Optional[torch.Tensor] = None

class Llama4TextRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) * self.weight

class Llama4TextL2Norm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class Llama4TextRotaryEmbedding(nn.Module):
    def __init__(self, config: Llama4TextConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.rope_type = "llama3" if config.rope_scaling is not None else "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.config = config
        dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> torch.Tensor:
        # logging.info(f"RotaryEmbedding: x.shape={x.shape}, position_ids.shape={position_ids.shape}")
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).squeeze(-1)
        freqs = freqs.transpose(1, 2)
        cos_freqs = torch.cos(freqs)
        sin_freqs = torch.sin(freqs)
        freqs_cis = torch.complex(cos_freqs, sin_freqs)
        # logging.info(f"RotaryEmbedding: freqs_cis.shape={freqs_cis.shape}")
        return freqs_cis * self.attention_scaling

class Llama4TextMLP(nn.Module):
    def __init__(self, config: Llama4TextConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act_fn = F.silu if config.hidden_act == "silu" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        return x

class Llama4TextMoe(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.experts = nn.ModuleList([Llama4TextMLP(config) for _ in range(self.num_experts)])
        self.shared_expert = Llama4TextMLP(config)
        self.router = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=config.initializer_range)
        self.load_balancing_loss = None

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(hidden_states, BaseModelOutputWithPast):
            hidden_states = hidden_states.last_hidden_state
        B, L, H = hidden_states.size()
        N = B * L
        flat = hidden_states.view(N, H)
        router_logits = self.router(flat)
        gates = F.softmax(router_logits, dim=-1)
        topk_vals, topk_inds = torch.topk(gates, self.top_k, dim=-1)
        sparse_gates = torch.zeros_like(gates).scatter_(1, topk_inds, topk_vals)
        sparse_gates = sparse_gates.to(flat.dtype)
        expert_prob = gates.mean(dim=0)
        self.load_balancing_loss = (expert_prob * torch.log(expert_prob * self.num_experts + 1e-9)).sum()
        output = self.shared_expert(flat).to(flat.dtype)
        for idx in range(self.num_experts):
            mask = sparse_gates[:, idx].nonzero(as_tuple=True)[0]
            if mask.numel() == 0:
                continue
            inp = flat[mask]
            w = sparse_gates[mask, idx].unsqueeze(-1)
            out_i = self.experts[idx](inp).to(flat.dtype) * w
            output.index_add_(0, mask, out_i)
        output = output.view(B, L, H)
        return output, sparse_gates.view(B, L, self.num_experts)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights

class Llama4TextAttention(nn.Module):
    def __init__(self, config: Llama4TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.use_rope = layer_idx not in config.no_rope_layers
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.qk_norm = Llama4TextL2Norm(config.rms_norm_eps) if config.use_qk_norm and self.use_rope else None
        self.use_flash_attn = config.use_flash_attention
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=config.initializer_range)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=config.initializer_range)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=config.initializer_range)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=config.initializer_range)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[DynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[DynamicCache]]:
        bsz, seq_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if self.use_rope:
            query_states, key_states = apply_rotary_emb(query_states, key_states, position_embeddings)
        if self.qk_norm:
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, {"cache_position": cache_position})
        if self.use_flash_attn:
            try:
                from flash_attn import flash_attn_func
                attn_output = flash_attn_func(query_states, key_states, value_states, causal=True)
                attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
                attn_weights = None
            except ImportError:
                print("FlashAttention not available, using SDPA.")
                attn_output, attn_weights = F.scaled_dot_product_attention(
                    query_states, key_states, value_states, attn_mask=attention_mask, scale=self.scaling, is_causal=attention_mask is None
                )
                attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        else:
            attn_output, attn_weights = eager_attention_forward(
                self, query_states, key_states, value_states, attention_mask, self.scaling, self.attention_dropout
            )
            attn_output = attn_output.view(bsz, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value

class Llama4TextDecoderLayer(nn.Module):
    def __init__(self, config: Llama4TextConfig, layer_idx: int):
        super().__init__()
        self.attn = Llama4TextAttention(config, layer_idx)
        self.ff = Llama4TextMoe(config) if layer_idx in config.moe_layers else Llama4TextMLP(config)
        self.ln1 = Llama4TextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ln2 = Llama4TextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.use_chunked_attention = config.attention_chunk_size is not None and layer_idx in config.no_rope_layers
        self.layer_idx = layer_idx
        self.is_moe = layer_idx in config.moe_layers

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            chunk_causal_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[DynamicCache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            output_attentions: bool = False,
            output_router_logits: bool = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[DynamicCache]]:
            residual = hidden_states
            hidden_states = self.ln1(hidden_states)
            if self.use_chunked_attention and chunk_causal_mask is not None:
                attention_mask = chunk_causal_mask
            attn_output, attn_weights, past_key_value = self.attn(
                hidden_states, position_embeddings, attention_mask, past_key_value, cache_position, output_attentions
            )
            hidden_states = residual + attn_output
            residual = hidden_states
            hidden_states = self.ln2(hidden_states)
            ff_output = self.ff(hidden_states)
            router_logits = None
            if isinstance(ff_output, tuple) and self.is_moe:
                hidden_states, router_logits = ff_output
            else:
                hidden_states = ff_output
            hidden_states = residual + hidden_states
            return hidden_states, attn_weights, router_logits, past_key_value

class Llama4TextModel(nn.Module):
    def __init__(self, config: Llama4TextConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Llama4TextDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Llama4TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Llama4TextRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, Llama4TextRMSNorm):
                nn.init.ones_(module.weight)

    def _update_causal_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_tensor: torch.Tensor,
        cache_position: torch.LongTensor,
        past_key_values: Optional[DynamicCache],
        output_attentions: bool,
        use_cache: bool,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        dtype, device = input_tensor.dtype, input_tensor.device
        seq_length = input_tensor.shape[1]
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask, seq_length, cache_position, device, dtype, past_key_values
        )
        chunk_causal_mask = None
        if self.config.attention_chunk_size is not None:
            chunk_causal_mask = causal_mask.clone()
        return causal_mask, chunk_causal_mask

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[DynamicCache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            use_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            output_router_logits: bool = False,
        ) -> BaseModelOutputWithPast:
            if self.gradient_checkpointing and self.training:
                forward_fn = torch.utils.checkpoint.checkpoint
            else:
                forward_fn = lambda x, *args, **kwargs: x(*args, **kwargs)
            hidden_states = self.embed_tokens(input_ids)
            batch_size, seq_len = input_ids.shape
            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_len, device=hidden_states.device)
            if position_ids is None:
                position_ids = cache_position[None, :].expand(batch_size, -1)
            # logging.info(f"Model forward: position_ids.shape={position_ids.shape}, cache_position.shape={cache_position.shape}")
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            causal_mask, chunk_causal_mask = self._update_causal_mask(
                attention_mask, hidden_states, cache_position, past_key_values, output_attentions, use_cache
            )
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            all_router_logits = () if output_router_logits else None
            next_cache = DynamicCache() if use_cache else None
            for layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                layer_outputs = forward_fn(
                    layer,
                    hidden_states,
                    position_embeddings,
                    causal_mask,
                    chunk_causal_mask,
                    past_key_value=past_key_values,
                    cache_position=cache_position,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                )
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
                if output_router_logits:
                    all_router_logits += (layer_outputs[2],)
                if use_cache:
                    next_cache = layer_outputs[3]
            hidden_states = self.norm(hidden_states)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
                router_logits=all_router_logits,
            )
class Llama4ForCausalLM(nn.Module):
    def __init__(self, config: Llama4TextConfig):
        super().__init__()
        self.config = config
        self.model = Llama4TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        # مقداردهی اولیه برای lm_head
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_router_logits: bool = False,
    ) -> BaseModelOutputWithPast:
        # فراخوانی مدل اصلی
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
        )
        # print(f"[DEBUG Llama4ForCausalLM] Type of outputs.last_hidden_state: {type(outputs.last_hidden_state)}")
        # if isinstance(outputs.last_hidden_state, torch.Tensor):
        #     print(f"[DEBUG Llama4ForCausalLM] Shape of outputs.last_hidden_state: {outputs.last_hidden_state.shape}")
        #     print(f"[DEBUG Llama4ForCausalLM] Dtype of outputs.last_hidden_state: {outputs.last_hidden_state.dtype}")
        #     print(f"[DEBUG Llama4ForCausalLM] NaNs in last_hidden_state: {torch.isnan(outputs.last_hidden_state).any().item()}")
        #     print(f"[DEBUG Llama4ForCausalLM] Infs in last_hidden_state: {torch.isinf(outputs.last_hidden_state).any().item()}")

        computed_model_logits = self.lm_head(outputs.last_hidden_state)

        # print(f"[DEBUG Llama4ForCausalLM] Type of computed_model_logits after lm_head: {type(computed_model_logits)}")
        # if isinstance(computed_model_logits, torch.Tensor):
        #     print(f"[DEBUG Llama4ForCausalLM] Shape of computed_model_logits: {computed_model_logits.shape}")

                
        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            logits=computed_model_logits  # اضافه کردن logits به خروجی
        )