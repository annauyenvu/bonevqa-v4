import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger("bonevqa")

BIOMEDCLIP_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CLIP_FALLBACK_ID = "openai/clip-vit-base-patch16"

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _apply_lora(module: nn.Module, target_modules: List[str], rank: int, alpha: int, dropout: float) -> nn.Module:
    from peft import LoraConfig, inject_adapter_in_model

    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_modules, bias="none")
    return inject_adapter_in_model(cfg, module)


def _freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


class VisualEncoder(nn.Module):
    def __init__(self, backbone: str = "biomedclip", d_model: int = 768, lora_rank: int = 4,
                 lora_alpha: int = 8, lora_dropout: float = 0.05, image_size: int = 224,
                 grad_checkpoint: bool = False):
        super().__init__()
        self.image_size = image_size
        self.d_model = d_model
        self.grad_checkpoint = grad_checkpoint
        self.kind = None
        self.shared_clip = None
        if backbone == "biomedclip":
            try:
                self._build_biomedclip()
            except Exception as exc:
                logger.warning("Không tải được BiomedCLIP (%s), chuyển sang CLIP ViT-B/16 của HF", exc)
                self._build_hf_clip()
        else:
            self._build_hf_clip()
        _freeze(self.trunk)
        if lora_rank > 0:
            targets = ["qkv"] if self.kind == "open_clip" else ["q_proj", "v_proj"]
            self.trunk = _apply_lora(self.trunk, targets, lora_rank, lora_alpha, lora_dropout)
        self.proj_tokens = nn.Identity() if self.hidden_dim == d_model else nn.Linear(self.hidden_dim, d_model)
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

    def _build_biomedclip(self):
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(BIOMEDCLIP_ID)
        self.shared_clip = model
        self.trunk = model.visual.trunk
        self.head = model.visual.head
        self.kind = "open_clip"
        self.hidden_dim = self.trunk.embed_dim
        self.embed_dim = 512
        logger.info("VisualEncoder: BiomedCLIP ViT-B/16 (open_clip)")

    def _build_hf_clip(self):
        from transformers import CLIPVisionModelWithProjection

        model = CLIPVisionModelWithProjection.from_pretrained(CLIP_FALLBACK_ID)
        self.trunk = model.vision_model
        self.head = model.visual_projection
        self.kind = "hf_clip"
        self.hidden_dim = model.config.hidden_size
        self.embed_dim = model.config.projection_dim
        logger.info("VisualEncoder: CLIP ViT-B/16 (transformers)")

    @property
    def num_patches(self) -> int:
        return (self.image_size // 16) ** 2

    def preprocess(self, images: List[Image.Image], device: torch.device) -> torch.Tensor:
        import numpy as np

        arrs = []
        for im in images:
            im = im.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
            arrs.append(torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1))
        x = torch.stack(arrs).to(device).float().div_(255.0)
        return (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)

    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "open_clip":
            if self.grad_checkpoint and self.training:
                self.trunk.set_grad_checkpointing(True)
            return self.trunk.forward_features(x)
        out = self.trunk(pixel_values=x)
        return out.last_hidden_state

    def forward(self, pixel_values: torch.Tensor):
        tokens = self._forward_tokens(pixel_values)
        if self.kind == "open_clip":
            pooled_hidden = tokens[:, 0]
            embed = self.head(pooled_hidden)
        else:
            pooled_hidden = self.trunk.post_layernorm(tokens[:, 0])
            embed = self.head(pooled_hidden)
        return {"tokens": self.proj_tokens(tokens), "embed": F.normalize(embed, dim=-1)}


class TextEncoder(nn.Module):
    def __init__(self, visual_encoder: Optional[VisualEncoder] = None, d_model: int = 768,
                 max_len: int = 64, freeze: bool = True):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.kind = None
        if visual_encoder is not None and visual_encoder.kind == "open_clip" and visual_encoder.shared_clip is not None:
            self._build_biomedclip(visual_encoder.shared_clip)
        else:
            self._build_hf_clip()
        if freeze:
            _freeze(self.model)
            if self.proj_head is not None:
                _freeze(self.proj_head)
        self.proj_tokens = nn.Identity() if self.hidden_dim == d_model else nn.Linear(self.hidden_dim, d_model)

    def _build_biomedclip(self, clip_model):
        import open_clip

        text = clip_model.text
        self.model = text.transformer
        self.proj_head = text.proj
        self.pooler_type = getattr(text, "pool_type", "cls_last_hidden_state_pooler")
        tok = open_clip.get_tokenizer(BIOMEDCLIP_ID)
        self.tokenizer = getattr(tok, "tokenizer", tok)
        self.hidden_dim = self.model.config.hidden_size
        self.embed_dim = 512
        self.kind = "open_clip"
        logger.info("TextEncoder: PubMedBERT (BiomedCLIP)")

    def _build_hf_clip(self):
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        model = CLIPTextModelWithProjection.from_pretrained(CLIP_FALLBACK_ID)
        self.model = model.text_model
        self.proj_head = model.text_projection
        self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_FALLBACK_ID)
        self.hidden_dim = model.config.hidden_size
        self.embed_dim = model.config.projection_dim
        self.kind = "hf_clip"
        self.max_len = min(self.max_len, 77)
        logger.info("TextEncoder: CLIP text (transformers)")

    def tokenize(self, texts: List[str], device: torch.device):
        enc = self.tokenizer(list(texts), padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        return {k: v.to(device) for k, v in enc.items()}

    def forward(self, texts: List[str], device: torch.device):
        enc = self.tokenize(texts, device)
        if self.kind == "open_clip":
            out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            hidden = out.last_hidden_state
            pooled = hidden[:, 0]
            embed = self.proj_head(pooled)
        else:
            out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            hidden = out.last_hidden_state
            embed = self.proj_head(out.pooler_output)
        return {
            "tokens": self.proj_tokens(hidden),
            "mask": enc["attention_mask"].bool(),
            "embed": F.normalize(embed, dim=-1),
        }

    @torch.no_grad()
    def encode_frozen(self, texts: List[str], device: torch.device):
        return self.forward(texts, device)
