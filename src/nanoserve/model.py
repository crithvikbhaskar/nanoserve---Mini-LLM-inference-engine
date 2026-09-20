"""Hand-written Llama/Qwen2-style decoder model with RMSNorm, RoPE, GQA, and SwiGLU."""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig


@dataclass
class ModelConfig:
    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    head_dim: int = 64
    rms_norm_eps: float = 1e-5
    rope_theta: float = 100000.0
    tie_word_embeddings: bool = True
    bos_token_id: int = 1
    eos_token_id: int = 2

    @classmethod
    def from_hf_config(cls, hf_config: AutoConfig) -> "ModelConfig":
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=getattr(
                hf_config, "num_key_value_heads", hf_config.num_attention_heads
            ),
            head_dim=getattr(
                hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads
            ),
            rms_norm_eps=getattr(hf_config, "rms_norm_eps", 1e-5),
            rope_theta=getattr(
                hf_config,
                "rope_theta",
                getattr(hf_config, "rope_parameters", {}).get("rope_theta", 10000.0)
                if hasattr(hf_config, "rope_parameters")
                else 10000.0,
            ),
            tie_word_embeddings=getattr(hf_config, "tie_word_embeddings", True),
            bos_token_id=getattr(hf_config, "bos_token_id", 1),
            eos_token_id=getattr(hf_config, "eos_token_id", 2),
        )


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS in float32 for numerical stability, then cast back
        norm_x = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm_x).type_as(x) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding (RoPE) to query and key states."""
    # cos, sin shape: [batch_size, seq_len, 1, head_dim] or broadcastable
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Precomputes rotary position embedding frequencies."""

    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Calculate inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin for given explicit positions tensor.
        
        Args:
            positions: Tensor of shape [batch_size, seq_len]
        Returns:
            cos, sin of shape [batch_size, seq_len, 1, dim]
        """
        # positions: [B, S], inv_freq: [D/2] -> freqs: [B, S, D/2]
        freqs = torch.einsum("bi,j->bij", positions.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [B, S, D]
        cos = emb.cos().unsqueeze(2)  # [B, S, 1, D]
        sin = emb.sin().unsqueeze(2)  # [B, S, 1, D]
        return cos, sin


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward network: down_proj(silu(gate_proj(x)) * up_proj(x))."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    """Grouped-Query Attention with support for naive, contiguous, and paged KV caches."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache=None,
        paged_cache=None,
        block_tables=None,
        seq_lens=None,
    ) -> torch.Tensor:
        """Forward pass for attention.
        
        Args:
            x: [B, S, H]
            cos, sin: [B, S, 1, head_dim]
            mask: Optional causal/padding mask [B, 1, S, total_S]
            kv_cache: Optional contiguous cache object
            paged_cache: Optional PagedKVCache
            block_tables: Optional block tables for paged attention
            seq_lens: Optional list of current sequence lengths per sequence
        """
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Handle Paged KV Cache
        if paged_cache is not None:
            from nanoserve.attention import paged_attention_decode

            if S == 1:
                # Decode step: write 1 token per sequence, run paged attention decode
                for b in range(B):
                    pos = seq_lens[b] - 1
                    block_idx = pos // paged_cache.block_size
                    offset = pos % paged_cache.block_size
                    block_id = block_tables[b][block_idx]
                    paged_cache.write_slot(self.layer_idx, block_id, offset, k[b, 0], v[b, 0])

                context = paged_attention_decode(
                    q=q,
                    paged_cache=paged_cache,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                    layer_idx=self.layer_idx,
                    scale=self.scale,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                )
                return self.o_proj(context)
            else:
                # Prefill step: write full prompt tokens into allocated paged blocks
                for b in range(B):
                    paged_cache.write_slots(
                        self.layer_idx,
                        block_tables[b],
                        start_pos=0,
                        k=k[b],
                        v=v[b],
                    )

        # Standard / Contiguous KV Cache path
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)

        # Repeat KV heads for GQA if needed: [B, total_S, num_heads, head_dim]
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=2)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=2)

        # Transpose to [B, num_heads, S, head_dim] and [B, num_heads, total_S, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores + mask

        probs = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(q)
        output = torch.matmul(probs, v)  # [B, num_heads, S, head_dim]

        # Reshape to [B, S, num_heads * head_dim]
        output = output.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    """Single transformer decoder layer."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache=None,
        paged_cache=None,
        block_tables=None,
        seq_lens=None,
    ) -> torch.Tensor:
        # Attention block with residual
        norm_x = self.input_layernorm(x)
        attn_out = self.self_attn(
            norm_x,
            cos=cos,
            sin=sin,
            mask=mask,
            kv_cache=kv_cache,
            paged_cache=paged_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
        )
        x = x + attn_out

        # MLP block with residual
        norm_x = self.post_attention_layernorm(x)
        mlp_out = self.mlp(norm_x)
        x = x + mlp_out
        return x


class NanoserveModel(nn.Module):
    """Hand-written Llama/Qwen2 decoder architecture loaded from HF safetensors."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            dim=config.head_dim,
            base=config.rope_theta,
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        kv_cache=None,
        paged_cache=None,
        block_tables=None,
        seq_lens=None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input_ids: [batch_size, seq_len]
            positions: [batch_size, seq_len] or None
            mask: Optional attention mask [batch_size, 1, seq_len, total_seq_len]
            kv_cache: Optional contiguous cache
            paged_cache: Optional PagedKVCache
            block_tables: Optional block tables for paged attention
            seq_lens: Optional list of current sequence lengths per sequence
        Returns:
            Logits tensor of shape [batch_size, seq_len, vocab_size]
        """
        B, S = input_ids.shape
        if positions is None:
            if paged_cache is not None and seq_lens is not None:
                if S == 1:
                    positions = torch.tensor(
                        [[sl - 1] for sl in seq_lens],
                        device=input_ids.device,
                        dtype=torch.long,
                    )
                else:
                    positions = (
                        torch.arange(0, S, device=input_ids.device)
                        .unsqueeze(0)
                        .expand(B, -1)
                    )
            else:
                past_len = kv_cache.get_seq_len() if kv_cache is not None else 0
                positions = (
                    torch.arange(past_len, past_len + S, device=input_ids.device)
                    .unsqueeze(0)
                    .expand(B, -1)
                )

        # Compute RoPE cos and sin
        cos, sin = self.rotary_emb(positions)
        cos = cos.to(input_ids.device)
        sin = sin.to(input_ids.device)

        # Causal mask for multi-token sequences (prefill / naive)
        if mask is None and S > 1:
            causal_mask = torch.full((S, S), float("-inf"), device=input_ids.device)
            causal_mask = torch.triu(causal_mask, diagonal=1)
            mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cos=cos,
                sin=sin,
                mask=mask,
                kv_cache=kv_cache,
                paged_cache=paged_cache,
                block_tables=block_tables,
                seq_lens=seq_lens,
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, device: str = "cpu", dtype: torch.dtype = torch.float32
    ) -> "NanoserveModel":
        """Load weights from HuggingFace safetensors into hand-written model."""
        from transformers import AutoConfig, AutoModelForCausalLM

        hf_config = AutoConfig.from_pretrained(model_name_or_path)
        config = ModelConfig.from_hf_config(hf_config)
        model = cls(config)

        # Load weights from HF model
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        hf_state = hf_model.state_dict()

        # Map state dict keys
        my_state = {}
        for k, v in hf_state.items():
            # Strip 'model.' prefix if present
            clean_k = k
            if clean_k.startswith("model."):
                clean_k = clean_k[len("model.") :]

            # Map attention weights
            clean_k = clean_k.replace("self_attn.", "self_attn.")
            clean_k = clean_k.replace("input_layernorm.", "input_layernorm.")
            clean_k = clean_k.replace("post_attention_layernorm.", "post_attention_layernorm.")

            if k == "lm_head.weight":
                my_state["lm_head.weight"] = v.to(dtype)
            elif k == "model.embed_tokens.weight":
                my_state["embed_tokens.weight"] = v.to(dtype)
            elif k == "model.norm.weight":
                my_state["norm.weight"] = v.to(dtype)
            elif clean_k in model.state_dict():
                my_state[clean_k] = v.to(dtype)
            else:
                # Try prefixing with layers
                my_state[clean_k] = v.to(dtype)

        model.load_state_dict(my_state, strict=False)
        model.to(device=device, dtype=dtype)
        model.eval()
        return model

