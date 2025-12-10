from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist
from einops import rearrange
from typing import Tuple, Optional

from .audio_pack import AudioPack
from model.wan_video_dit_omniavatar_helper import (
    RMSNorm,
    rope_apply,
    AttentionModule,
    CrossAttention,
    GateModule,
    modulate,
    precompute_freqs_cis_3d,
    MLP,
    sinusoidal_embedding_1d,
    WanModelStateDictConverter
)
from safetensors.torch import load_file
import pdb

# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

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


class CacheCrossAttention(CrossAttention):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__(dim, num_heads, eps, has_image_input)  # 复用父类的层与属性

    def forward(self, x: torch.Tensor, y: torch.Tensor,crossattn_cache):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
            
        q = self.norm_q(self.q(x))
        
        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(ctx))
                v = self.v(ctx)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"].clone()
                v = crossattn_cache["v"].clone()
        
        else:
        
            k = self.norm_k(self.k(ctx))
            v = self.v(ctx)
            
        x = self.attn(q, k, v)
        
        if self.has_image_input: #We do need to care about k_img. Because even i2v will not go through this code branch
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
            
        return self.o(x)





class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6,local_attn_size=-1,sink_size=0):
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
        
        
        self.local_attn_size=local_attn_size
        self.sink_size=sink_size
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1024
        
    def forward(self, x, freqs,block_mask=None,grid_sizes=None,kv_cache=None,current_start=0,cache_start=None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        
        if kv_cache is None: #We do not consider teacher forcing here
            roped_query = rope_apply(q, freqs,self.num_heads).type_as(v)
            roped_key = rope_apply(k, freqs,self.num_heads).type_as(v)
            
            B, T, D = q.shape
            
        
            q = q.view(B, T, self.num_heads, self.head_dim)   
            roped_query=roped_query.view(B, T, self.num_heads, self.head_dim) 
            k = k.view(B, T, self.num_heads, self.head_dim)      
            roped_key=roped_key.view(B,T,self.num_heads,self.head_dim)
            v = v.view(B, T, self.num_heads, self.head_dim)       
            
            
            #pad至128的倍数
            padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
            padded_roped_query = torch.cat(
                [roped_query,
                    torch.zeros([q.shape[0], padded_length, q.shape[2],q.shape[3]],
                                device=q.device, dtype=v.dtype)],
                dim=1
            )

            padded_roped_key = torch.cat(
                [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2],k.shape[3]],
                                        device=k.device, dtype=v.dtype)],
                dim=1
            )

            padded_v = torch.cat(
                [v, torch.zeros([v.shape[0], padded_length, v.shape[2],v.shape[3]],
                                device=v.device, dtype=v.dtype)],
                dim=1
            )

            x = flex_attention(
                query=padded_roped_query.transpose(2, 1),
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask
            )
            x=x[:, :, :x.shape[2]-padded_length].transpose(2, 1).reshape(B, T, -1) #这里与源代码不同防止pad=0变成空切片
            
        else:
            
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            ##Freqs here are causal.
            roped_query = rope_apply(q, freqs,self.num_heads).type_as(v)
            roped_key = rope_apply(k, freqs,self.num_heads).type_as(v)
            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            
          
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
             
            x = self.attn(
                roped_query,
                kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
                kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
            )
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)
            
        
        x = self.o(x)
        return x

    
    
class CausalDiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6,local_attn_size=-1,sink_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.local_attn_size=local_attn_size
        
        self.self_attn = CausalSelfAttention(dim, num_heads, eps,local_attn_size=local_attn_size,sink_size=sink_size)
        self.cross_attn = CacheCrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs,block_mask=None,grid_sizes=None,kv_cache=None,crossattn_cache=None,current_start=0,cache_start=None):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
         #Note:  t_mod ->（b,f,6,dim）     

         
        num_frames,frame_seqlen=t_mod.shape[1],x.shape[1]//t_mod.shape[1]
         
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.unsqueeze(1).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=2)
        
        input_x = modulate(self.norm1(x).unflatten(dim=1,sizes=(num_frames,frame_seqlen)), shift_msa, scale_msa).flatten(1,2)
        y=self.self_attn(input_x, freqs,grid_sizes=grid_sizes,block_mask=block_mask,kv_cache=kv_cache,current_start=current_start,cache_start=cache_start)
        
        
        x=x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * gate_msa).flatten(1, 2)
        
        x = x + self.cross_attn(self.norm3(x), context,crossattn_cache)
        
        input_x = modulate(self.norm2(x).unflatten(dim=1,sizes=(num_frames,frame_seqlen)), shift_mlp, scale_mlp).flatten(1,2)
        
        x = x+ (self.ffn(input_x).unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * gate_mlp).flatten(1, 2)
        return x    
    

class CausalHead(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        #T_mod (B,F,1,C)
        num_frames, frame_seqlen = t_mod.shape[1], x.shape[1] // t_mod.shape[1]
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device).unsqueeze(1) + t_mod).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + scale) + shift))
        return x


class CausalWanModel(nn.Module):
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
        use_audio: bool,
        model_type: str = 'i2v',
        audio_hidden_size: int=32,
        local_attn_size=-1,  # CausalWanModel新增
        sink_size=0,  # CausalWanModel新增
        freeze_audio: bool = False,  # Control freezing of audio components
        sp_size: int = 1  # Sequence parallel size
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = dim // num_heads
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.model_type = model_type
        self.patch_size = patch_size
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.sp_size = sp_size

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
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
            CausalDiTBlock(has_image_input, dim, num_heads, ffn_dim, eps,local_attn_size=local_attn_size,sink_size=sink_size)
            for _ in range(num_layers)
        ])

        self.head = CausalHead(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim)  # clip_feature_dim = 1280
        
        if use_audio:
            audio_input_dim = 10752
            audio_out_dim = dim
            self.audio_proj = AudioPack(audio_input_dim, [4, 1, 1], audio_hidden_size, layernorm=True)
            self.audio_cond_projs = nn.ModuleList()
            for d in range(num_layers // 2 - 1):
                l = nn.Linear(audio_hidden_size, audio_out_dim)
                self.audio_cond_projs.append(l)
            
            # Conditionally freeze audio components
            if freeze_audio:
                print("AUDIO COMPONENT IS FIXED!!!")
                self.audio_proj.requires_grad_(False)
                self.audio_cond_projs.requires_grad_(False)

        self.use_audio=use_audio
        self.gradient_checkpointing = False
        self.block_mask = None
        self.num_frame_per_block = 1
        self.independent_first_frame = False
    
    def enable_gradient_checkpointing(self):
       """Enable gradient checkpointing for memory efficiency during training."""
       self.gradient_checkpointing = True

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
        
    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)
    
    def _forward_train(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                clean_x: Optional[torch.Tensor]=None,
                use_gradient_checkpointing: bool = False,
                audio_emb: Optional[torch.Tensor] = None,
                use_gradient_checkpointing_offload: bool = False,
                tea_cache = None,
                **kwargs,
                ):
        
        device = self.patch_embedding.weight.device
        
        
        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None:
                pass # Note: We do need teacher forcing now.
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                    
                    
        lat_h, lat_w = x.shape[-2], x.shape[-1]
        x = torch.cat([x, y], dim=1)
        x = self.patch_embedding(x)
        x, (f, h, w) = self.patchify(x)
        grid_sizes = torch.tensor([f, h, w]).unsqueeze(0) 
        # print(f"DEBUG: x.shape = {x.shape}, y.shape = {y.shape if y is not None else None}")
        
  
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).to(self.time_embedding[0].weight.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=timestep.shape)
        #context
        context = self.text_embedding(context)
        

        
        if audio_emb != None: # 
            # print("audio_emb.shape",audio_emb.shape)
            audio_emb = audio_emb.permute(0, 2, 1)[:, :, :, None, None]
            audio_emb = torch.cat([audio_emb[:, :, :1].repeat(1, 1, 3, 1, 1), audio_emb], 2) # 1, 768, 44, 1, 1
            audio_emb = self.audio_proj(audio_emb)

            audio_emb = torch.concat([audio_cond_proj(audio_emb) for audio_cond_proj in self.audio_cond_projs], 0)

        
        # arguments
        kwargs=dict(
            grid_sizes=grid_sizes,
            block_mask=self.block_mask
        )
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        def create_custom_forward(module):
            def custom_forward(*inputs,**kwargs):
                return module(*inputs,**kwargs)
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
                # Pad to ensure each rank's chunk is divisible by num_frames (f)
                # This is needed for unflatten operations in DiTBlocks
                chunk_divisor = sp_size * f
                if ori_x_len % chunk_divisor != 0:
                    pad_size = chunk_divisor - (ori_x_len % chunk_divisor)
                    x = torch.cat([x, torch.zeros_like(x[:, -1:]).repeat(1, pad_size, 1)], 1)
                    # Pad freqs to match
                    freqs = torch.cat([freqs, freqs[-1:].repeat(pad_size, 1, 1)], dim=0)
                x = torch.chunk(x, sp_size, dim=1)[get_sequence_parallel_rank()]
                # Split freqs the same way as x
                freqs = torch.chunk(freqs, sp_size, dim=0)[get_sequence_parallel_rank()]

            audio_emb = audio_emb.reshape(x.shape[0], audio_emb.shape[0] // x.shape[0], -1, *audio_emb.shape[2:])


            for layer_i, block in enumerate(self.blocks):
                # audio cond
                if self.use_audio:
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
                                x, context, t_mod, freqs,**kwargs,
                                use_reentrant=False,
                            )
                    else:
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,**kwargs,
                            use_reentrant=False,
                        )
                else:
                    x = block(x, context, t_mod, freqs,**kwargs)

            if tea_cache is not None:
                x_cache = get_sp_group().all_gather(x, dim=1) # TODO: the size should be devided by sp_size
                x_cache = x_cache[:, :ori_x_len]
                tea_cache.store(x_cache)

        x = self.head(x, t.unflatten(dim=0,sizes=timestep.shape).unsqueeze(2))
        x=x.reshape(x.shape[0],-1,x.shape[-1])

        # Gather sequence from all sequence parallel ranks
        if self.sp_size > 1:
            x = DiffAllGather.apply(x, get_sp_group().device_group, self.sp_size, 1)
            x = x[:, :ori_x_len]

        x = self.unpatchify(x, (f, h, w))
        return x

    def _forward_inference(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                clean_x: Optional[torch.Tensor]=None,
                use_gradient_checkpointing: bool = False,
                audio_emb: Optional[torch.Tensor] = None,
                use_gradient_checkpointing_offload: bool = False,
                tea_cache = None,
                kv_cache: dict = None,
                crossattn_cache: dict = None,
                current_start: int = 0,
                cache_start: int = 0,
                **kwargs,
                ):
    
        device = self.patch_embedding.weight.device
        
        # pdb.set_trace()
        lat_h, lat_w = x.shape[-2], x.shape[-1]
        x = torch.cat([x, y], dim=1)
        x = self.patch_embedding(x)
        x, (f, h, w) = self.patchify(x)
        grid_sizes = torch.tensor([f, h, w]).unsqueeze(0)
        
        #time embeddings
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).to(self.time_embedding[0].weight.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=timestep.shape)
        

        #context
        context = self.text_embedding(context)

        # arguments
        kwargs=dict(
            grid_sizes=grid_sizes,
            block_mask=self.block_mask
        )
        
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward
        
        ##casual rope
        frame_seqlen=math.prod(grid_sizes[0][1:]).item()
        current_start_frame = current_start // frame_seqlen
        freqs = torch.cat([
            self.freqs[0][current_start_frame:current_start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        

        audio_emb = audio_emb.reshape(x.shape[0], audio_emb.shape[0] // x.shape[0], -1, *audio_emb.shape[2:])


        for layer_i, block in enumerate(self.blocks):
            # audio cond
        
            if self.use_audio:
                au_idx = None
                if (layer_i <= len(self.blocks) // 2 and layer_i > 1): # < len(self.blocks) - 1:
                    au_idx = layer_i - 2
                    audio_emb_tmp = audio_emb[:, au_idx].repeat(1, 1, lat_h // 2, lat_w // 2, 1) # 1, 11, 45, 25, 128    
                    audio_cond_tmp = self.patchify(audio_emb_tmp.permute(0, 4, 1, 2, 3))[0]
                    
                    x = audio_cond_tmp + x

            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        kwargs.update(
                            {
                                "kv_cache": kv_cache[layer_i],
                                "current_start": current_start,
                                "cache_start": cache_start
                            }
                        )
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,**kwargs,
                            use_reentrant=False,
                        )
                else:
                    kwargs.update(
                        {
                            "kv_cache": kv_cache[layer_i],
                            "current_start": current_start,
                            "crossattn_cache": crossattn_cache[layer_i],
                            "cache_start": cache_start
                        }
                    )
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,**kwargs,
                        use_reentrant=False,
                    )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[layer_i],
                        "current_start": current_start,
                        "crossattn_cache": crossattn_cache[layer_i],
                        "cache_start": cache_start
                    }
                )
                x = block(x, context, t_mod, freqs,**kwargs)
                    

        x = self.head(x, t.unflatten(dim=0,sizes=timestep.shape).unsqueeze(2))
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        # Gather sequence from all sequence parallel ranks
        if self.sp_size > 1:
            x = DiffAllGather.apply(x, get_sp_group().device_group, self.sp_size, 1)
            x = x[:, :ori_x_len]

        x = self.unpatchify(x, (f, h, w))
        
        return x
    #crate mask.
    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen
        
        # Debug print to understand the massive allocation
        # print(f"DEBUG: num_frames={num_frames}, frame_seqlen={frame_seqlen}")
        # print(f"DEBUG: total_length={total_length}")
        memory_gb = total_length * 8 / (1024**3)
        # print(f"DEBUG: Memory required: {memory_gb:.2f} GB")

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)
        '''
        grid = block_mask.to('cpu').to_dense().to(torch.float32).numpy()[0][0]  # 形状约为(行块数, 列块数)
        print("block grid:", grid.shape)

        # 快速保存为图（块级，而非逐元素）
        import matplotlib.pyplot as plt
        plt.imshow(grid, cmap='gray', interpolation='nearest')
        plt.axis('off'); plt.tight_layout(pad=0)
        plt.savefig("block_mask_grid.png", bbox_inches='tight', pad_inches=0)
        
        import torch.distributed as dist
        '''
        # if not dist.is_initialized() or dist.get_rank() == 0:
            # print(
            #    f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            # print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        #Currently, we do not need teacherforcing/
        raise NotImplementedError()

        return None

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        #Currently, we do not need it
        raise NotImplementedError()

        return None

    @classmethod
    def from_pretrained(cls, pretrained_path, local_attn_size=-1, sink_size=0, freeze_audio=False):
        """
        Load pretrained weights using WanModelStateDictConverter.

        Args:
            pretrained_path: Path to the safetensors/pytorch model file
            local_attn_size: Local attention window size
            sink_size: Attention sink size
            freeze_audio: Whether to freeze audio components
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
            'local_attn_size': local_attn_size,
            'sink_size': sink_size,
            'has_image_input': False
        })
        
        # Create the model instance
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
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            audio_hidden_size=config.get('audio_hidden_size', 32),
            freeze_audio=freeze_audio
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