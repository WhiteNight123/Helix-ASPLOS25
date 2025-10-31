from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import Qwen3Config

from vllm.attention import Attention, AttentionMetadata
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    QKVParallelLinear,
    RowParallelLinear
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput

# Import Qwen3MLP (same as Qwen2MLP)
try:
    from vllm.model_executor.models.qwen2 import Qwen2MLP as Qwen3MLP
except ImportError:
    # Define a placeholder if not available
    Qwen3MLP = None


class Qwen3Attention(nn.Module):
    """
    Qwen3 Attention layer with QK normalization.
    The key difference from Qwen2 is the addition of q_norm and k_norm.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 1000000,
        linear_method: Optional[LinearMethodBase] = None,
        rope_scaling: Optional[Tuple] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            linear_method=linear_method,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads
        )
        
        # QK normalization - the key feature of Qwen3
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # Apply QK normalization
        # Reshape to [batch, seq_len, num_heads, head_dim] for normalization
        q_shape = q.shape
        k_shape = k.shape
        
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        
        # Apply RMSNorm (forward returns normalized tensor)
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Reshape back to original shape
        q = q.view(q_shape)
        k = k.view(k_shape)
        
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):
    """
    Qwen3 Decoder Layer with QK normalized attention.
    """

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)

        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            linear_method=linear_method,
            rope_scaling=rope_scaling,
        )
        
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            linear_method=linear_method,
        )
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3PipelineStage(nn.Module):
    """
    Qwen3 Pipeline Stage for Helix distributed inference.
    Similar to Qwen2PipelineStage but uses Qwen3DecoderLayer with QK normalization.
    """
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(
            self,
            config: Qwen3Config,  # Qwen3 uses Qwen3Config in transformers
            linear_method: Optional[LinearMethodBase] = None,
            lora_config=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.vocab_size = config.vocab_size

        layer_id = config.layer_id
        self.layer_id = layer_id

        # Qwen3DecoderLayer requires layer_idx parameter
        # The key difference with Qwen2 is that Qwen3 has QK normalization in attention
        self.decoder_layer = Qwen3DecoderLayer(config, layer_id, linear_method)
        
        # First layer: handle embedding
        if layer_id == 0:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
            )

        # Last layer: handle output head and normalization
        if layer_id == self.config.num_hidden_layers - 1:
            self.unpadded_vocab_size = config.vocab_size
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE
                # We need bigger padding if using lora for kernel
                # compatibility
                if not lora_config else lora_config.lora_vocab_padding_size,
            )

            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size, logit_scale)
            self.sampler = Sampler()

    def forward(self, **kwargs):
        """
        Forward pass for a single pipeline stage.
        
        Args:
            positions: Token positions
            kv_cache: KV cache tensor
            attn_metadata: Attention metadata
            input_ids: (optional) Input token IDs for first layer
            input_embeds: (optional) Input embeddings for first layer
            hidden_states: (optional) Hidden states from previous layer
        """
        positions = kwargs["positions"]
        kv_cache = kwargs["kv_cache"]
        attn_metadata = kwargs["attn_metadata"]
        
        if self.layer_id == 0:
            # First layer: process input tokens or embeddings
            input_ids = kwargs["input_ids"]
            inputs_embeds = (kwargs["input_embeds"]
                             if "input_embeds" in kwargs else None)
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
            hidden_states, residual = self.decoder_layer(
                positions,
                hidden_states,
                kv_cache,
                attn_metadata,
                residual,
            )
        else:
            # Middle or last layer: process hidden states from previous layer
            hidden_states = kwargs["hidden_states"]
            residual = None
            hidden_states, residual = self.decoder_layer(
                positions,
                hidden_states,
                kv_cache,
                attn_metadata,
                residual,
            )
            # Last layer: apply final normalization
            if self.layer_id == self.config.num_hidden_layers - 1:
                hidden_states, _ = self.norm(hidden_states, residual)
        
        return hidden_states

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get input embeddings for the first layer."""
        return self.embed_tokens(input_ids)

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        """Load weights from HuggingFace model."""
        # FIXME: Implement weight loading for distributed setup
        raise NotImplementedError("Weight loading for Qwen3 pipeline stage not yet implemented")

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        """Compute logits from hidden states (last layer only)."""
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
            self,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        """Sample next tokens from logits (last layer only)."""
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens


# Register Qwen3 models with vLLM
# We need to make Qwen3 available as if it were a vLLM model
from vllm.model_executor.models import _MODELS
import sys
import types

# Create a fake qwen3 module in vllm.model_executor.models namespace
# This is necessary because vLLM's model loader expects to import from vllm.model_executor.models
fake_qwen3_module = types.ModuleType('qwen3')
fake_qwen3_module.Qwen3PipelineStage = Qwen3PipelineStage
fake_qwen3_module.Qwen3Attention = Qwen3Attention
fake_qwen3_module.Qwen3DecoderLayer = Qwen3DecoderLayer
fake_qwen3_module.Qwen3MLP = Qwen3MLP

# Inject it into sys.modules so it can be imported
sys.modules['vllm.model_executor.models.qwen3'] = fake_qwen3_module

# Now register the model with vLLM's model registry
# Use "qwen3" as the module name (not "llm_sys.engine.qwen3")
_MODELS["Qwen3ForCausalLM"] = ("qwen3", "Qwen3PipelineStage")

print("[Qwen3] Successfully registered Qwen3ForCausalLM with vLLM model registry")

