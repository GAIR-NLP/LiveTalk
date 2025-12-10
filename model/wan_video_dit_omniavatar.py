import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from utils.io_utils import hash_state_dict_keys
from .audio_pack import AudioPack
from model.wan_video_dit_omniavatar_helper import (
    WanModelStateDictConverter
)
from safetensors.torch import load_file
from xfuser.core.distributed import (get_sequence_parallel_rank,
                                     get_sequence_parallel_world_size,
                                     get_sp_group)


class DiffAllGather(torch.autograd.Function):
    """Differentiable all-gather with proper gradient flow via reduce-scatter."""
    @staticmethod
    def forward(ctx, x, group, world_size, dim):
        ctx.group = group
        ctx.world_size = world_size
        ctx.dim = dim

        # Handle dim != 0 by moving to 0, gathering, then moving back
        if dim != 0:
            x = x.movedim(dim, 0)

        # Allocate output tensor
        output_size = list(x.size())
        output_size[0] *= world_size
        out = torch.empty(output_size, dtype=x.dtype, device=x.device)

        # All-gather along dim 0
        torch.distributed.all_gather_into_tensor(out, x, group=group)

        # Move back to original dimension
        if dim != 0:
            out = out.movedim(0, dim)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        # Move to dim 0 for reduce-scatter
        if ctx.dim != 0:
            grad_output = grad_output.movedim(ctx.dim, 0)

        # Allocate gradient input tensor
        output_size = list(grad_output.size())
        output_size[0] //= ctx.world_size
        grad_input = torch.empty(output_size, dtype=grad_output.dtype, device=grad_output.device)

        # Reduce-scatter (inverse of all-gather)
        torch.distributed.reduce_scatter_tensor(grad_input, grad_output, group=ctx.group)

        # Move back to original dimension
        if ctx.dim != 0:
            grad_input = grad_input.movedim(0, ctx.dim)

        return grad_input, None, None, None


try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x



class OmniAvatarWanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        use_audio: bool = True,
        audio_hidden_size: int=32,
        sp_size: int = 1  # Sequence parallel size
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.sp_size = sp_size

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
            # nn.LayerNorm(dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim)  # clip_feature_dim = 1280

        self.use_audio = use_audio
        if use_audio:
            audio_input_dim = 10752
            audio_out_dim = dim
            self.audio_proj = AudioPack(audio_input_dim, [4, 1, 1], audio_hidden_size, layernorm=True)
            self.audio_cond_projs = nn.ModuleList()
            for d in range(num_layers // 2 - 1):
                l = nn.Linear(audio_hidden_size, audio_out_dim)
                self.audio_cond_projs.append(l)      

    def patchify(self, x: torch.Tensor):
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                audio_emb: Optional[torch.Tensor] = None,
                use_gradient_checkpointing_offload: bool = False,
                tea_cache = None,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        lat_h, lat_w = x.shape[-2], x.shape[-1]

        if audio_emb != None and self.use_audio: # TODO  cache
            audio_emb = audio_emb.permute(0, 2, 1)[:, :, :, None, None]
            audio_emb = torch.cat([audio_emb[:, :, :1].repeat(1, 1, 3, 1, 1), audio_emb], 2) # 1, 768, 44, 1, 1
            audio_emb = self.audio_proj(audio_emb)

            audio_emb = torch.concat([audio_cond_proj(audio_emb) for audio_cond_proj in self.audio_cond_projs], 0)

        if y is not None: # y is optional
            x = torch.cat([x, y], dim=1)
        x = self.patch_embedding(x)
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        if tea_cache is not None:
            tea_cache_update = tea_cache.check(self, x, t_mod)
        else:
            tea_cache_update = False
        ori_x_len = x.shape[1]
        if tea_cache_update:
            x = tea_cache.update(x)
        else:
            # Sequence Parallel: split sequence along tokens
            pad_size = 0
            if self.sp_size > 1:
                sp_size = self.sp_size
                if ori_x_len % sp_size != 0:
                    pad_size = sp_size - ori_x_len % sp_size
                    x = torch.cat([x, torch.zeros_like(x[:, -1:]).repeat(1, pad_size, 1)], 1)
                    # Pad freqs to match
                    freqs = torch.cat([freqs, freqs[-1:].repeat(pad_size, 1, 1)], dim=0)
                x = torch.chunk(x, sp_size, dim=1)[get_sequence_parallel_rank()]
                # Split freqs the same way as x
                freqs = torch.chunk(freqs, sp_size, dim=0)[get_sequence_parallel_rank()]

            if audio_emb != None and self.use_audio:
                audio_emb = audio_emb.reshape(x.shape[0], audio_emb.shape[0] // x.shape[0], -1, *audio_emb.shape[2:])
                
            for layer_i, block in enumerate(self.blocks):
                # audio cond
                if audio_emb != None and self.use_audio:
                    au_idx = None
                    if (layer_i <= len(self.blocks) // 2 and layer_i > 1): # < len(self.blocks) - 1:
                        au_idx = layer_i - 2
                        audio_emb_tmp = audio_emb[:, au_idx].repeat(1, 1, lat_h // 2, lat_w // 2, 1) # 1, 11, 45, 25, 128
                        audio_cond_tmp = self.patchify(audio_emb_tmp.permute(0, 4, 1, 2, 3))[0]

                        # Apply same padding and chunking as x for sequence parallel
                        if self.sp_size > 1:
                            if pad_size > 0:
                                audio_cond_tmp = torch.cat([audio_cond_tmp, torch.zeros_like(audio_cond_tmp[:, -1:]).repeat(1, pad_size, 1)], 1)
                            audio_cond_tmp = torch.chunk(audio_cond_tmp, sp_size, dim=1)[get_sequence_parallel_rank()]

                        x = audio_cond_tmp + x

                if self.training and use_gradient_checkpointing:
                    if use_gradient_checkpointing_offload:
                        with torch.autograd.graph.save_on_cpu():
                            x = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(block),
                                x, context, t_mod, freqs,
                                use_reentrant=False,
                            )
                    else:
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = block(x, context, t_mod, freqs)
            if tea_cache is not None:
                x_cache = get_sp_group().all_gather(x, dim=1) # TODO: the size should be devided by sp_size
                x_cache = x_cache[:, :ori_x_len]
                tea_cache.store(x_cache)

        x = self.head(x, t)

        # Gather sequence from all sequence parallel ranks
        if self.sp_size > 1:
            x = DiffAllGather.apply(x, get_sp_group().device_group, self.sp_size, 1)
            x = x[:, :ori_x_len]

        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency during training."""
        pass  # This model already supports gradient checkpointing via use_gradient_checkpointing parameter
    
    @classmethod
    def from_pretrained(cls, pretrained_path):
        """
        Load pretrained weights using WanModelStateDictConverter.
        
        Args:
            pretrained_path: Path to the safetensors/pytorch model file
            local_attn_size: Local attention window size
            sink_size: Attention sink size
        """
        print(f"Loading CausalWanModel from {pretrained_path}")
        
        # Load the state dict
        if pretrained_path.endswith('.safetensors'):
            state_dict = load_file(pretrained_path)
        else:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
        
        # Use the converter to get the proper format and config
        converter = WanModelStateDictConverter()
        
        # Try from_civitai (for merged models)
        converted_state_dict, config = converter.from_civitai(state_dict)
        
        # Override/set causal-specific parameters
        config.update({
            'has_image_input': False
        })
        
        # Create the model instance (will be overridden by WanModelStateDictConverter, so no worries)
        model = cls(
            dim=config.get('dim', 1536),
            in_dim=config.get('in_dim', 33),
            ffn_dim=config.get('ffn_dim', 6144),
            out_dim=config.get('out_dim', 16),
            text_dim=config.get('text_dim', 4096),
            freq_dim=config.get('freq_dim', 256),
            eps=config.get('eps', 1e-6),
            patch_size=tuple(config.get('patch_size', [1, 2, 2])),
            num_heads=config.get('num_heads', 24),
            num_layers=config.get('num_layers', 28),
            has_image_input=config.get('has_image_input', False),
            use_audio=config.get('use_audio', False),
            audio_hidden_size=config.get('audio_hidden_size', 32)
        )
        
        # Load the converted state dict
        missing_keys, unexpected_keys = model.load_state_dict(converted_state_dict, strict=False)
        
        print(f"Loaded model with {len(converted_state_dict)} parameters")
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)} (expected for causal-specific components)")
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")
            print(f"Unexpected keys: {unexpected_keys}")
        
        return model
