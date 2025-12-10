from utils.lmdb import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os
import torchaudio
import cv2
from transformers import Wav2Vec2FeatureExtractor
import torchvision.transforms as transforms
import torch.nn.functional as F


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split('_')[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }


def resize_pad(image, ori_size, tgt_size):
    h, w = ori_size
    scale_ratio = max(tgt_size[0] / h, tgt_size[1] / w)
    scale_h = int(h * scale_ratio)
    scale_w = int(w * scale_ratio)

    image = transforms.Resize(size=[scale_h, scale_w])(image)

    padding_h = tgt_size[0] - scale_h
    padding_w = tgt_size[1] - scale_w
    pad_top = padding_h // 2
    pad_bottom = padding_h - pad_top
    pad_left = padding_w // 2
    pad_right = padding_w - pad_left

    image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    return image


class OmniAvatarODEDataset(Dataset):
    """
    Dataset for OmniAvatar ODE distillation training
    Loads data from text@@image@@audio format and corresponding .pt trajectory files
    """
    
    def __init__(self, data_path: str, ode_pairs_dir: str, max_pair: int = int(1e8), wav2vec_model_name: str = None, vae_wrapper=None, num_frame_per_block: int = 3, training_mode: str = "ode", image_or_video_shape: list = None):
        """
        Args:
            data_path: Path to ode_input_small.txt file
            ode_pairs_dir: Directory containing .pt trajectory files
            max_pair: Maximum number of samples to use
            wav2vec_model_name: Path to wav2vec model directory
            vae_wrapper: VAE wrapper for encoding images
            num_frame_per_block: Number of frames per block (should match model config)
            training_mode: Either "ode" or "dmd" to determine reference image frame count
            image_or_video_shape: Shape from config [B, F, C, H_latent, W_latent] to calculate target image size
        """
        self.data_path = data_path
        self.ode_pairs_dir = ode_pairs_dir
        self.max_pair = max_pair
        self.vae_wrapper = vae_wrapper
        self.num_frame_per_block = num_frame_per_block
        self.training_mode = training_mode
        # Calculate target image size from latent dimensions (VAE has 8x spatial compression)
        if image_or_video_shape is not None and len(image_or_video_shape) == 5:
            # image_or_video_shape format: [B, F, C, H_latent, W_latent]
            self.target_image_size = [image_or_video_shape[3] * 8, image_or_video_shape[4] * 8]
            print(f"Dataset: Calculated target image size from config: {self.target_image_size} (latent: {image_or_video_shape[3]}x{image_or_video_shape[4]})")
        else:
            # Fallback to default 400x720 if shape not provided
            self.target_image_size = [400, 720]
            print(f"Warning: image_or_video_shape not provided or invalid, using default size {self.target_image_size}")
        
        # Initialize audio encoder (Wav2Vec2) - following OmniAvatar pattern
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Audio encoder setup (following OmniAvatar scripts/inference.py pattern)
        try:
            from OmniAvatar.models.wav2vec import Wav2VecModel
            # Use provided wav2vec path or fallback to default
            if wav2vec_model_name and os.path.exists(wav2vec_model_name):
                wav2vec_path = wav2vec_model_name
            else:
                wav2vec_path = "./pretrained_models/wav2vec2-base-960h"
                if not os.path.exists(wav2vec_path):
                    wav2vec_path = "facebook/wav2vec2-base-960h"  # fallback to HF
            
            self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_path)
            self.audio_encoder = Wav2VecModel.from_pretrained(wav2vec_path, local_files_only=False).to(device=self.device)
            self.audio_encoder.feature_extractor._freeze_parameters()
            self.audio_encoder.eval()
        except Exception as e:
            print(f"Warning: Could not load Wav2Vec2 audio encoder: {e}")
            print("Falling back to raw audio waveform mode")
            self.audio_encoder = None
            self.wav_feature_extractor = None
        
        # Audio processing parameters (following AudioVideoDataset pattern)
        self.audio_sample_rate = 16000
        self.video_frames = 81  # ODE latent frames
        self.video_fps = 16     # Assumed video FPS 
        self.video_duration = self.video_frames / self.video_fps  # ~1.3 seconds
        self.required_audio_length = int(self.video_duration * self.audio_sample_rate)  # ~21k samples
        
        # Load data entries
        self.samples = []
        with open(data_path, 'r') as f:
            for line in f:
                parts = line.strip().split('@@')
                if len(parts) >= 3:
                    prompt = parts[0]
                    image_path = parts[1]
                    audio_path = parts[2]
                    
                    # Extract base filename for .pt file lookup
                    base_filename = None
                    if os.path.exists(image_path):
                        base_filename = os.path.splitext(os.path.basename(image_path))[0]
                    elif os.path.exists(audio_path):
                        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
                    
                    if base_filename:
                        ode_file = os.path.join(ode_pairs_dir, f"{base_filename}.pt")
                        if os.path.exists(ode_file):
                            self.samples.append({
                                'prompt': prompt,
                                'image_path': image_path if os.path.exists(image_path) else None,
                                'audio_path': audio_path if os.path.exists(audio_path) else None,
                                'ode_file': ode_file,
                                'base_filename': base_filename
                            })
        
        # Limit samples
        self.samples = self.samples[:min(len(self.samples), max_pair)]
        print(f"Loaded {len(self.samples)} ODE samples from {data_path}")
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load audio file and process to specified format (following AudioVideoDataset pattern)
        Returns:
            audio_waveform: [audio_length] audio waveform
        """
        try:
            # Load audio using torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to target sample rate
            if sample_rate != self.audio_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=self.audio_sample_rate
                )
                waveform = resampler(waveform)
            
            # Remove batch dimension: [1, length] -> [length]
            waveform = waveform.squeeze(0)
            
            # Loop audio until it reaches video duration
            target_length = self.required_audio_length
            if waveform.shape[0] >= target_length:
                # If audio is long enough, truncate
                waveform = waveform[:target_length]
            else:
                # If audio is too short, loop it
                original_length = waveform.shape[0]
                repeated_waveform = waveform.clone()
                
                while repeated_waveform.shape[0] < target_length:
                    # Loop original audio
                    repeated_waveform = torch.cat([repeated_waveform, waveform], dim=0)
                
                # Truncate to exact length
                waveform = repeated_waveform[:target_length]
                
            return waveform
            
        except Exception as e:
            raise RuntimeError(f"Failed to load audio {audio_path}: {e}")
    
    def _process_image(self, image_path: str) -> torch.Tensor:
        """
        Process image following OmniAvatar pattern - images are encoded through VAE, not CLIP
        For ODE training, we return the raw image tensor that will be encoded by VAE during training
        Returns:
            image_tensor: [C, H, W] preprocessed image tensor
        """
        image = Image.open(image_path).convert("RGB")
        chained_trainsforms = []
        chained_trainsforms.append(transforms.ToTensor())
        transform = transforms.Compose(chained_trainsforms)
        image = transform(image).unsqueeze(0).to(self.device)
        _, _, h, w = image.shape
        # Use dynamically calculated target size from config
        select_size = self.target_image_size
        image = resize_pad(image, (h, w), select_size)
        image = image * 2.0 - 1.0
        image = image[:, :, None]
        return image
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                "prompts": str,  # Text prompt
                "ode_latent": torch.Tensor [num_denoising_steps, num_frames, num_channels, height, width],
                "audio_emb": torch.Tensor [seq_len, embed_dim],  # Wav2Vec2 audio embeddings (preferred)
                "audio_waveform": torch.Tensor [audio_length],  # Raw audio waveform (fallback)
                "image": torch.Tensor [C, H, W],  # Preprocessed image tensor for VAE encoding
            }
        """
        sample = self.samples[idx]
        
        # Load ODE trajectory from .pt file
        ode_data = torch.load(sample['ode_file'], map_location='cpu')
        
        # Extract trajectory data - the .pt file contains {prompt: trajectory}
        trajectory = None
        for key, value in ode_data.items():
            trajectory = value
            break  # Take the first (and likely only) trajectory
        
        if trajectory is None:
            raise ValueError(f"No trajectory found in {sample['ode_file']}")
        
        # Convert trajectory format if needed
        # Expected format: [num_denoising_steps, num_frames, num_channels, height, width]
        if len(trajectory.shape) == 6:  # [batch, num_denoising_steps, num_channels, num_frames, height, width]
            # Reshape to [num_denoising_steps, num_frames, num_channels, height, width]
            trajectory = trajectory.squeeze(0).permute(0, 2, 1, 3, 4)
        elif len(trajectory.shape) == 5:  # Already correct format
            pass
        else:
            raise ValueError(f"Unexpected trajectory shape: {trajectory.shape}")
        
        # Process audio using Wav2Vec2 encoder (following OmniAvatar pattern)
        audio_embeddings = None
        audio_waveform = None  # Keep raw waveform as fallback
        
        if sample['audio_path'] and os.path.exists(sample['audio_path']):
            audio_waveform = self._load_audio(sample['audio_path'])
            
            # Extract audio embeddings using Wav2Vec2 (following OmniAvatar scripts/inference.py)
            if self.audio_encoder is not None and self.wav_feature_extractor is not None:
                # Convert to numpy for feature extractor
                audio_numpy = audio_waveform.numpy()
                
                # Extract features using Wav2Vec2FeatureExtractor
                input_values = np.squeeze(
                    self.wav_feature_extractor(audio_numpy, sampling_rate=self.audio_sample_rate).input_values
                )
                input_values = torch.from_numpy(input_values).float().unsqueeze(0).to(device=self.device)
                
                # Calculate sequence length (following OmniAvatar pattern)
                audio_len = len(audio_numpy)
                seq_len = int(audio_len / self.audio_sample_rate * 16)  # 16 FPS conversion
                # print("Seq:len",seq_len)
                # Extract embeddings using Wav2Vec2 model
                with torch.no_grad():
                    hidden_states = self.audio_encoder(input_values, seq_len=seq_len, output_hidden_states=True)
                    audio_embeddings = hidden_states.last_hidden_state
                    # Concatenate all hidden states (following OmniAvatar pattern)
                    for mid_hidden_states in hidden_states.hidden_states:
                        audio_embeddings = torch.cat((audio_embeddings, mid_hidden_states), -1)
                    audio_embeddings = audio_embeddings.squeeze(0).cpu()  # Move to CPU and remove batch dim
                        
            
        
        # Process image and create reference image embedding (following inference pattern)
        image_emb_full = None
        image_emb_block = None
        if sample['image_path'] and os.path.exists(sample['image_path']) and self.vae_wrapper is not None:
            # Load and preprocess image
            image_tensor = self._process_image(sample['image_path']) #1,3,1,H,W
            
            # VAE encode the image (following generate_ode_clean.py pattern)
            with torch.no_grad():
                # Add batch and time dimensions: [C, H, W] -> [1, C, 1, H, W] 
                
                img_lat = self.vae_wrapper.encode_to_latent(image_tensor)  # img_lat is [1, 1, C_lat, H_lat, W_lat]
                
                # Create two versions of reference images for different scenarios
                T_full = trajectory.shape[1]  # Full trajectory length (e.g., 21 frames)
                T_block = self.num_frame_per_block  # Block size (e.g., 3 frames)
                
                # Create full trajectory reference image for DMD loss computation
                img_cat_full = img_lat.repeat(1, T_full, 1, 1, 1)
                msk_full = torch.zeros_like(img_cat_full[:, :, :1])  # [1, T_full, 1, H_lat, W_lat]
                msk_full[:, 1:] = 1  # First frame is 0 (reference), others are 1
                image_emb_full = torch.cat([img_cat_full, msk_full], dim=2)  # [1, T_full, C_lat+1, H_lat, W_lat]
                image_emb_full = image_emb_full.squeeze(0).cpu()  # Remove batch dim and move to CPU: [T_full, C_lat+1, H_lat, W_lat]

                # Create block-sized reference image for forward inference/rollout
                img_cat_block = img_lat.repeat(1, T_block, 1, 1, 1)
                msk_block = torch.zeros_like(img_cat_block[:, :, :1])  # [1, T_block, 1, H_lat, W_lat]
                msk_block[:, 1:] = 1  # First frame is 0 (reference), others are 1
                image_emb_block = torch.cat([img_cat_block, msk_block], dim=2)  # [1, T_block, C_lat+1, H_lat, W_lat]
                image_emb_block = image_emb_block.squeeze(0).cpu()  # Remove batch dim and move to CPU: [T_block, C_lat+1, H_lat, W_lat]
                
                
                # print("audio_embeddings.shape:",audio_embeddings.shape)
        
        result = {
            "prompts": sample['prompt'],
            "ode_latent": trajectory.float(),
            "base_filename": sample['base_filename'],
            "ode_file": sample['ode_file']
        }
        
        # Add audio and image data if available (following OmniAvatar pattern)
        if audio_embeddings is not None:
            result["audio_emb"] = audio_embeddings  # Wav2Vec2 embeddings (preferred)
        elif audio_waveform is not None:
            result["audio_waveform"] = audio_waveform  # Raw waveform fallback
        
        if image_emb_full is not None:
            result["ref_image"] = image_emb_full  # Full trajectory reference image [T_full, C_lat+1, H_lat, W_lat]
            result["ref_image_block"] = image_emb_block  # Block-sized reference image [T_block, C_lat+1, H_lat, W_lat]
            
        return result

class AudioVideoDataset(Dataset):
    """
    音频+文本条件的视频生成数据集
    支持同时加载音频、文本提示和视频数据，适配AudioCausalDiffusion训练
    """
    
    def __init__(self, 
                 data_path: str,
                 audio_sample_rate: int = 16000,
                 max_audio_length: int = 48000,  # 3秒*16000Hz (已弃用，音频长度现在基于视频参数计算)
                 video_frames: int = 81,  # 官方标准81帧，VAE会编码为21帧latent
                 video_fps: int = 16,  # 视频帧率，用于计算对应的音频长度
                 video_size: tuple = (480, 832),  # 官方标准分辨率，VAE空间压缩8倍到60x104
                 load_raw_video: bool = True,
                 default_text_prompt: str = "A video with corresponding audio"):
        """
        Args:
            data_path: 数据目录路径，应包含metadata.json
            audio_sample_rate: 音频采样率
            max_audio_length: 最大音频长度（samples，已弃用）
            video_frames: 视频帧数
            video_fps: 视频帧率
            video_size: 视频尺寸 (H, W)
            load_raw_video: 是否加载原始视频（True）还是预计算latent（False）
            default_text_prompt: 默认文本提示（当metadata中没有时使用）
        """
        self.data_path = Path(data_path)
        self.audio_sample_rate = audio_sample_rate
        self.max_audio_length = max_audio_length  # 保留向后兼容
        self.video_frames = video_frames
        self.video_fps = video_fps
        self.video_size = video_size
        self.load_raw_video = load_raw_video
        self.default_text_prompt = default_text_prompt
        
        # 基于视频参数计算对应的音频长度
        self.video_duration = self.video_frames / self.video_fps  # 视频时长（秒）
        self.required_audio_length = int(self.video_duration * self.audio_sample_rate)  # 对应音频长度（samples）
        
        # 加载metadata
        metadata_path = self.data_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
        self.samples = self.metadata["samples"]
        
        # 验证数据文件存在性
        self._validate_files()
        
    def _validate_files(self):
        """验证所有音频和视频文件是否存在"""
        missing_files = []
        for sample in self.samples:
            audio_path = self.data_path / sample["audio_file"]
            video_path = self.data_path / sample["video_file"]
            
            if not audio_path.exists():
                missing_files.append(str(audio_path))
            if not video_path.exists():
                missing_files.append(str(video_path))
                
        if missing_files:
            raise FileNotFoundError(f"Missing files: {missing_files[:5]}...")  # 只显示前5个
    
    def __len__(self):
        return len(self.samples)
        
    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        """
        加载音频文件并处理为指定格式
        Returns:
            audio_waveform: [audio_length] 音频波形
        """
        try:
            # 使用torchaudio加载音频
            waveform, sample_rate = torchaudio.load(str(audio_path))
            
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 重采样到目标采样率
            if sample_rate != self.audio_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=self.audio_sample_rate
                )
                waveform = resampler(waveform)
            
            # 移除batch维度：[1, length] -> [length]
            waveform = waveform.squeeze(0)
            
            # 循环播放音频直到达到视频对应的时长
            target_length = self.required_audio_length
            if waveform.shape[0] >= target_length:
                # 如果音频足够长，直接截断
                waveform = waveform[:target_length]
            else:
                # 如果音频不够长，循环播放
                original_length = waveform.shape[0]
                repeated_waveform = waveform.clone()
                
                while repeated_waveform.shape[0] < target_length:
                    # 循环添加原始音频
                    repeated_waveform = torch.cat([repeated_waveform, waveform], dim=0)
                
                # 截断到精确长度
                waveform = repeated_waveform[:target_length]
                
            return waveform
            
        except Exception as e:
            raise RuntimeError(f"Failed to load audio {audio_path}: {e}")
    
    def _load_video(self, video_path: Path) -> torch.Tensor:
        """
        加载视频文件并处理为指定格式
        Returns:
            frames: [T, C, H, W] 视频帧张量
        """
        try:
            # 使用OpenCV加载视频
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")
            
            frames = []
            frame_count = 0
            
            while len(frames) < self.video_frames:
                ret, frame = cap.read()
                if not ret:
                    # 如果视频帧不够，循环播放
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                
                # BGR转RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 调整尺寸
                frame = cv2.resize(frame, (self.video_size[1], self.video_size[0]))
                
                # 转换为tensor并归一化到[0,1]
                frame = torch.from_numpy(frame).float() / 255.0
                
                # 转换维度: [H, W, C] -> [C, H, W]
                frame = frame.permute(2, 0, 1)
                
                frames.append(frame)
                frame_count += 1
            
            cap.release()
            
            # 堆叠为 [T, C, H, W]
            video_tensor = torch.stack(frames)
            
            return video_tensor
            
        except Exception as e:
            raise RuntimeError(f"Failed to load video {video_path}: {e}")
    
    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                "text_prompts": str,  # 文本提示
                "audio_waveform": torch.Tensor [audio_length],  # 音频波形
                "frames": torch.Tensor [T, C, H, W] (如果load_raw_video=True),
                "ode_latent": torch.Tensor (如果load_raw_video=False, 暂未实现),
                "sample_id": str,
                "duration": float,
                "pattern_type": str
            }
        """
        sample = self.samples[idx]
        
        # 构建文件路径
        audio_path = self.data_path / sample["audio_file"]
        video_path = self.data_path / sample["video_file"]
        
        # 加载音频
        audio_waveform = self._load_audio(audio_path)
        
        # 获取文本提示（从metadata或使用默认值）
        text_prompt = sample.get("text_prompt", None)
        if text_prompt is None:
            # 根据pattern_type生成描述性文本
            pattern_descriptions = {
                "moving_shapes": "A video showing moving geometric shapes synchronized with audio",
                "gradient_waves": "A video with colorful gradient waves flowing with the audio rhythm", 
                "noise_patterns": "A video displaying abstract noise patterns that respond to audio"
            }
            text_prompt = pattern_descriptions.get(
                sample.get("pattern_type", "unknown"), 
                self.default_text_prompt
            )
        
        # 构建返回字典（保持与官方格式兼容）  
        # 注意：单个样本返回字符串，DataLoader会自动batching成字符串列表
        batch = {
            "prompts": text_prompt,  # 单样本：字符串，batch后变为字符串列表
            "audio_waveform": audio_waveform,  # 新增：音频波形 [audio_length]
            "sample_id": sample["sample_id"],
            "duration": sample["duration"],
            "pattern_type": sample["pattern_type"]
        }
        
        # 加载视频数据
        if self.load_raw_video:
            # 加载原始视频帧
            frames = self._load_video(video_path)
            batch["frames"] = frames
        else:
            raise ValueError(f"Error load_raw_vide: {self.load_raw_video}, not be implemented")
            # TODO: 实现预计算latent加载
            # 目前暂时加载原始视频
            frames = self._load_video(video_path)
            batch["frames"] = frames
            # batch["ode_latent"] = self._load_latent(latent_path)
        
        return batch


def cycle(dl):
    """
    Infinite data iterator with proper epoch-based shuffling support.

    For DistributedSampler with shuffle=True, this ensures each epoch gets
    a different shuffle order by calling set_epoch() before each iteration.
    """
    epoch = 0
    while True:
        # Update epoch for DistributedSampler to get different shuffle order
        if hasattr(dl, 'sampler') and hasattr(dl.sampler, 'set_epoch'):
            dl.sampler.set_epoch(epoch)

        for data in dl:
            yield data

        epoch += 1
