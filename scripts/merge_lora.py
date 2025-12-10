import os
import sys
sys.path.append("OmniAvatar")
import torch
import torch.nn as nn
import torch.distributed as dist
from safetensors.torch import save_file
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora import LoraLayer
from OmniAvatar.utils.args_config import parse_args
args = parse_args()
from OmniAvatar.utils.io_utils import load_state_dict, hash_state_dict_keys
from OmniAvatar.models.model_manager import ModelManager
from OmniAvatar.wan_video import WanVideoPipeline




class LoRAMerger(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device(f"cuda:{args.rank}")
        if args.dtype == 'bf16':
            self.dtype = torch.bfloat16
        elif args.dtype == 'fp16':
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self.pipe = self.load_model()

    def save_dit_merged_safetensors(self, dit: nn.Module, out_path: str):
        """
        Merge LoRA into dit and save as .safetensors in FP32 precision
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        dit.eval()

        # Check for NaN before merging
        print("=== NaN Detection Before Merging ===")
        nan_count_before = 0
        for name, param in dit.named_parameters():
            if torch.isnan(param).any():
                nan_count_before += 1
                print(f"❌ NaN found before merging: {name}")
        print(f"Number of NaN parameters before merging: {nan_count_before}")
        
        with torch.no_grad():
            # Record original precision, restore after merging to avoid affecting subsequent inference
            orig_dtype = next((p.dtype for p in dit.parameters() if p.is_floating_point()), torch.float32)
            print(f"Original model precision: {orig_dtype}")

            # Execute LoRA merge (merge first, then convert precision)
            print("=== Starting LoRA Merging ===")
            merged_layers = []
            for name, m in dit.named_modules():
                if isinstance(m, LoraLayer) or hasattr(m, "merge"):
                    merged_flag = getattr(m, "merged", False)
                    if hasattr(m, "merge") and not merged_flag:
                        try:
                            m.merge()
                            merged_layers.append(name)
                            print(f"✅ Successfully merged LoRA layer: {name}")
                        except Exception as e:
                            print(f"❌ Failed to merge LoRA layer {name}: {e}")
            print(f"Number of merged LoRA layers: {len(merged_layers)}")

            # Check for NaN after merging, before conversion
            print("=== NaN Detection After Merging, Before Conversion ===")
            nan_count_after_merge = 0
            for name, param in dit.named_parameters():
                if torch.isnan(param).any():
                    nan_count_after_merge += 1
                    print(f"❌ NaN found after merging: {name}")
            print(f"Number of NaN parameters after merging, before conversion: {nan_count_after_merge}")

            # Safely convert to fp32 (check each parameter)
            print("=== Starting Precision Conversion ===")
            print(f"Original model precision: {orig_dtype}")

            conversion_errors = []
            for name, p in dit.named_parameters():
                if torch.is_floating_point(p):
                    try:
                        # Check value range before conversion
                        if torch.isinf(p).any() or torch.isnan(p).any():
                            print(f"❌ Abnormal values found before conversion {name}: inf={torch.isinf(p).sum()}, nan={torch.isnan(p).sum()}")
                            conversion_errors.append(name)
                            continue

                        # Check if value range is reasonable
                        p_abs_max = torch.abs(p).max()
                        if p_abs_max > 1e10:
                            print(f"⚠️  Parameter value too large {name}: max_abs={p_abs_max}")

                        p.data = p.data.float()

                        # Check again after conversion
                        if torch.isnan(p).any():
                            print(f"❌ NaN appeared after conversion {name}")
                            conversion_errors.append(name)
                    except Exception as e:
                        print(f"❌ Failed to convert parameter {name}: {e}")
                        conversion_errors.append(name)

            if conversion_errors:
                print(f"❌ Found {len(conversion_errors)} parameters with conversion errors")
                return

            # Extract state_dict, filter out LoRA-related keys, and ensure saving as fp32
            print("=== Extracting state_dict ===")
            sd = dit.state_dict()
            def is_lora_key(k: str):
                k = k.lower()
                return any(t in k for t in [
                    "lora_", "loraa", "lorab", "lora_up", "lora_down",
                    "lora_embedding_a", "lora_embedding_b"
                ])

            # Filter LoRA keys and remove .base_layer suffix
            pure_sd = {}
            for k, v in sd.items():
                if not is_lora_key(k):
                    # Remove .base_layer suffix to ensure parameter name matching during inference
                    clean_key = k.replace('.base_layer', '') if '.base_layer' in k else k
                    if torch.is_floating_point(v):
                        v_float = v.float()
                        if torch.isnan(v_float).any():
                            print(f"❌ NaN found in state_dict: {k}")
                            continue
                        pure_sd[clean_key] = v_float
                    else:
                        pure_sd[clean_key] = v
                    if clean_key != k:
                        print(f"[Parameter renamed] {k} -> {clean_key}")

            print(f"Number of valid parameter keys: {len(pure_sd)}")

            # Final NaN check
            final_nan_count = 0
            for k, v in pure_sd.items():
                if torch.is_floating_point(v) and torch.isnan(v).any():
                    final_nan_count += 1
                    print(f"❌ NaN in final state_dict: {k}")

            if final_nan_count > 0:
                print(f"❌ Final detection found {final_nan_count} NaN parameters, canceling save")
                return


            save_file(pure_sd, out_path)
            print(f"✅ [LoRA->Merged] Successfully saved to: {out_path}")
                
            print("HASH:",hash_state_dict_keys(pure_sd))

            # Restore to original precision (optional, but may affect subsequent inference using self.dtype)
            if orig_dtype != torch.float32:
                print(f"=== Restoring to original precision {orig_dtype} ===")
                for p in dit.parameters():
                    if torch.is_floating_point(p):
                        p.data = p.data.to(orig_dtype)

        print("✅ LoRA merging completed")
     

    def load_model(self):
        """Load model and merge LoRA"""
     
    

        ckpt_path = f'{args.exp_path}/pytorch_model.pt'
        assert os.path.exists(ckpt_path), f"pytorch_model.pt not found in {args.exp_path}"
        assert args.train_architecture == 'lora', "This script only supports LoRA merging"

        pretrained_lora_path = ckpt_path

        # Load models
        model_manager = ModelManager(device="cpu", infer=True)
        model_manager.load_models(
            [
                args.dit_path.split(",")
            ],
            torch_dtype=self.dtype,
            device='cpu',
        )

        pipe = WanVideoPipeline.from_model_manager(
            model_manager,
            torch_dtype=self.dtype,
            device=f"cuda:0",
            use_usp=True if args.sp_size > 1 else False,
            infer=True
        )

        print(f'Use LoRA: lora rank: {args.lora_rank}, lora alpha: {args.lora_alpha}')
        self.add_lora_to_model(
            pipe.denoising_model(),
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_target_modules=args.lora_target_modules,
            init_lora_weights=args.init_lora_weights,
            pretrained_lora_path=pretrained_lora_path,
        )

        # Save merged model
        output_path = os.path.join(
            args.output_folder,
            "merged_model.safetensors"
        )
        self.save_dit_merged_safetensors(
            dit=pipe.denoising_model(),
            out_path=output_path
        )

        # Verify merged model
        print("=== Verifying Merged Model ===")
        nan_params = []
        for name, param in pipe.denoising_model().named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
        if nan_params:
            print(f"❌ Verification found NaN parameters: {nan_params[:5]}...")
        else:
            print("✅ Verification passed: No NaN in model parameters")


        print("✅ LoRA merging and verification completed, exiting program")
        exit(0)
    
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4,
                          lora_target_modules="q,k,v,o,ffn.0,ffn.2",
                          init_lora_weights="kaiming",
                          pretrained_lora_path=None,
                          state_dict_converter=None):
        """Add LoRA to model and load pretrained weights"""
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)

        # Load pretrained LoRA weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. "
                  f"{num_unexpected_keys} parameters are unexpected.")


def main():
    merger = LoRAMerger(args)


if __name__ == '__main__':
    main()