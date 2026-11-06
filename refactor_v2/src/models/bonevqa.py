import copy
import logging
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..context_controls import lf_context_mode
from PIL import Image

from ..calibration import apply_platt_probability
from ..endpoints import is_canonical_fracture_presence_question
from ..utils import is_closed_question, is_polar_question, normalize_answer
from .encoders import TextEncoder, VisualEncoder
from .fusion import MultimodalFusion
from .heads import AnswerTypeHead, ClosedHead, ContrastiveProjector, FracturePresenceHead
from .latent_prompt import LatentPromptGenerator
from .lens_fusion import LensFusion, QFormerLite
from .losses import (endpoint_masks, lens_alignment_loss, multi_positive_decoupled_loss,
                     multi_positive_info_nce_loss, single_label_cross_entropy)
from .routing import ResidualPromptMixer
from .segment_prompt import SegmentPromptCreator
from ..checkpointing import checkpoint_manifest, compatible_state, validate_manifest

try:
    from ..data.prompts import build_lens, build_visual_prompts, textual_prompt
except Exception:
    from .prompt_draw import build_lens, build_visual_prompts

    def textual_prompt(question, region=None, modality="X-ray", style="clinical"):
        region_text = f" of the {region}" if region else ""
        if style == "clinical":
            return f"This is a bone X-ray image{region_text}. Answer briefly. {question}"
        if style == "clinical_paraphrase":
            return f"Review the provided bone X-ray image{region_text} and answer concisely. {question}"
        if style == "irrelevant":
            return f"This input follows a standard presentation format. Answer briefly. {question}"
        if style == "irrelevant_matched":
            return f"This is a plain text-based format. Answer the question briefly. {question}"
        raise ValueError(f"textual prompt style không hợp lệ: {style}")

logger = logging.getLogger("bonevqa")

DEFAULT_MODEL_CFG: Dict[str, Any] = {
    # Explicit bridge pretraining; zero preserves historical contrastive-only runs.
    "stage1_caption_loss_weight": 0.0,
    # Historical configs remain protocol_version 0.  New causal runs must opt
    # in explicitly and declare B/V/L/P/T in prompt_variant.
    "prompt_protocol_version": 0,
    "prompt_variant": None,
    "d_model": 768,
    "n_heads": 8,
    "dropout": 0.1,
    "visual_backbone": "biomedclip",
    "image_size": 224,
    "visual_lora_rank": 4,
    "grad_checkpoint": True,
    "use_visual_prompt": True,
    "use_latent_prompt": True,
    "use_lens": True,
    "use_textual_prompt": True,
    "textual_prompt_style": "clinical",
    # Architecture-preserving negative controls.  These suppress only the
    # semantic contribution, so parameter count and shared checkpoint tensors
    # remain unchanged during causal/gate-zero evaluation.
    "visual_force_zero": False,
    "lens_force_zero": False,
    "latent_force_zero": False,
    "use_region_in_prompt": True,
    "prompt_kind": "contour",
    "n_masks_max": 3,
    "n_query": 32,
    "qformer_layers": 2,
    "qformer_dropout": None,
    "visual_route": "dual",
    "n_latent": 32,
    "latent_answer_bank_mode": "train",
    "latent_control_seed": 314159,
    "lens_layers": 2,
    "lens_shuffle_r": 2,
    # Lens is a training/alignment view, matching Localization Lens' stated
    # original-image-only inference path.  Its own dropout is disabled so the
    # real-minus-capacity-control delta cannot be manufactured by two random
    # dropout masks.
    "lens_training_only": False,
    "lens_dropout": 0.0,
    "lens_alignment_weight": 0.1,
    "lens_alignment_eval": True,
    # Optional modules are initialized inside isolated RNG scopes.  Therefore
    # enabling V/L/P cannot silently change the initialization of the shared
    # fusion and endpoint heads in matched-seed ablations.
    "module_init_seed": 1729,
    "alpha": 1.0,
    "theta": 0.1,
    "beta": 0.1,
    "eta": 0.01,
    "lam": 0.1,
    "contrastive_mode": "infonce",
    "contrastive_dim": 256,
    "type_loss_weight": 0.1,
    "fracture_positive_weight": 1.0,
    "fracture_yes_threshold": 0.5,
    "fracture_platt_slope": None,
    "fracture_platt_intercept": None,
    # Kept off by default so historical checkpoints remain loadable.  The
    # revised clean protocol explicitly enables and seals this endpoint.
    "use_fracture_head": False,
    "fracture_loss_weight": 1.0,
    "max_question_tokens": 48,
    "max_answer_tokens": 32,
    "max_new_tokens": 32,
    "tta_kinds": None,
    "closed_llm_weight": 0.0,
    "llm_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "llm_dtype": "bfloat16",
    "llm_lora_r": 8,
    "llm_lora_alpha": 16,
    "llm_lora_targets": ["q_proj", "v_proj"],
    "llm_grad_checkpoint": True,
    "use_llm": True,
    "prior_labels": ["hand", "leg", "hip", "shoulder", "mixed", "fracture", "no fracture", "hardware", "normal"],
    "sam": {
        "model_id": "wanglab/medsam-vit-base",
        "sam_med2d_ckpt": None,
        "default_prompt": "grid_boxes",
        "grid": 2,
        "cache_dir": None,
        "enabled": True,
    },
}


def _build_with_isolated_rng(factory, seed: int):
    """Build an optional module without advancing the shared RNG stream."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return factory()


def _dtype_from_str(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}.get(name, torch.float32)


def normalize_batch(batch) -> Dict[str, list]:
    if isinstance(batch, dict):
        out = {}
        for k, v in batch.items():
            key = {"image": "images", "question": "questions", "answer": "answers", "answer_type": "answer_types",
                   "region": "regions", "mask": "masks", "box": "boxes", "image_id": "image_ids"}.get(k, k)
            out[key] = list(v) if not isinstance(v, (str, bytes)) else [v]
        return out
    keys = ["image", "question", "answer", "answer_type", "region", "masks", "boxes", "image_id"]
    out = {f"{k}s" if not k.endswith("s") else k: [s.get(k) for s in batch] for k in keys}
    return out


class BoneVQAModel(nn.Module):
    def __init__(self, cfg: Optional[dict] = None, answer_vocab: Optional[List[str]] = None, device: Optional[str] = None):
        super().__init__()
        self.cfg = copy.deepcopy(DEFAULT_MODEL_CFG)
        for k, v in (cfg or {}).items():
            if isinstance(v, dict) and isinstance(self.cfg.get(k), dict):
                self.cfg[k].update(v)
            else:
                self.cfg[k] = v
        c = self.cfg
        d = c["d_model"]
        vocab = list(answer_vocab or [])
        for i, must in enumerate(("yes", "no")):
            if must not in vocab:
                vocab.insert(i, must)
        self.answer_vocab = vocab
        self.answer_to_idx = {}
        for i, a in enumerate(vocab):
            self.answer_to_idx.setdefault(normalize_answer(a), i)
            self.answer_to_idx.setdefault(a, i)
        self.device_hint = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.visual = VisualEncoder(c["visual_backbone"], d, c["visual_lora_rank"], image_size=c["image_size"],
                                    grad_checkpoint=c["grad_checkpoint"])
        self.text = TextEncoder(self.visual, d, max_len=c["max_question_tokens"])
        need_roi = c["use_visual_prompt"] or c["use_lens"]
        self.segmenter = SegmentPromptCreator(model_id=c["sam"]["model_id"], sam_med2d_ckpt=c["sam"].get("sam_med2d_ckpt"),
                                              device=self.device_hint, max_masks=c["n_masks_max"], grid=c["sam"].get("grid", 2),
                                              default_prompt=c["sam"].get("default_prompt", "grid_boxes"),
                                              cache_dir=c["sam"].get("cache_dir"), enabled=c["sam"].get("enabled", True),
                                              max_area_ratio=c["sam"].get("max_area_ratio", 0.6),
                                              min_mean_intensity=c["sam"].get("min_mean_intensity", 40.0)) if need_roi else None
        qformer_dropout = c.get("qformer_dropout")
        if qformer_dropout is None:
            qformer_dropout = c["dropout"]
        self.qformer = QFormerLite(d, c["n_query"], c["qformer_layers"], c["n_heads"], qformer_dropout)
        grid = c["image_size"] // 16
        module_seed = int(c.get("module_init_seed", 1729))
        self.lens_fusion = (
            _build_with_isolated_rng(
                lambda: LensFusion(
                    d, 3, c["lens_layers"], c["n_heads"], c.get("lens_dropout", 0.0),
                    grid=grid, shuffle_r=c["lens_shuffle_r"],
                ),
                module_seed + 11,
            )
            if c["use_lens"] else None
        )
        self.latent = (
            _build_with_isolated_rng(
                lambda: LatentPromptGenerator(
                    d, c["n_latent"], c["n_heads"], c["dropout"],
                    prior_labels=c["prior_labels"],
                ),
                module_seed + 23,
            )
            if c["use_latent_prompt"] else None
        )
        direct_lens_route = c["use_lens"] and not c.get("lens_training_only", False)
        self.prompt_mixer = _build_with_isolated_rng(
            lambda: ResidualPromptMixer(
                d, c["use_visual_prompt"], direct_lens_route,
                zero_preserving=int(c.get("prompt_protocol_version", 0) or 0) >= 3,
            ),
            module_seed + 37,
        )
        self.fusion = _build_with_isolated_rng(
            lambda: MultimodalFusion(
                d, c["n_heads"], c["dropout"], c["alpha"], c["theta"], c["beta"],
                use_latent=c["use_latent_prompt"],
            ),
            module_seed + 53,
        )
        self.closed_head = ClosedHead(d, len(vocab), c["dropout"])
        self.type_head = AnswerTypeHead(d)
        self.fracture_head = (FracturePresenceHead(d, c["dropout"])
                              if c.get("use_fracture_head", False) else None)
        self.img_proj = ContrastiveProjector(d, c["contrastive_dim"])
        self.txt_proj = ContrastiveProjector(d, c["contrastive_dim"])
        self.answer_bank_ready = False

        self.llm = None
        if c["use_llm"]:
            self._build_llm()
        self._log_trainable()

    def _build_llm(self):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        c = self.cfg
        dtype = _dtype_from_str(c["llm_dtype"])
        self.llm_tokenizer = AutoTokenizer.from_pretrained(c["llm_name"])
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.llm_tokenizer.padding_side = "right"
        llm = AutoModelForCausalLM.from_pretrained(c["llm_name"], torch_dtype=dtype)
        for p in llm.parameters():
            p.requires_grad = False
        if c["llm_lora_r"] > 0:
            lcfg = LoraConfig(r=c["llm_lora_r"], lora_alpha=c["llm_lora_alpha"], lora_dropout=0.05,
                              target_modules=list(c["llm_lora_targets"]), bias="none", task_type="CAUSAL_LM")
            llm = get_peft_model(llm, lcfg)
        if c["llm_grad_checkpoint"]:
            llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            llm.enable_input_require_grads()
        self.llm = llm
        self.llm_dim = llm.config.hidden_size
        self.prefix_proj = nn.Sequential(nn.Linear(self.cfg["d_model"], self.llm_dim), nn.GELU(), nn.Linear(self.llm_dim, self.llm_dim))
        logger.info("LLM: %s (%s) + LoRA r=%d", c["llm_name"], c["llm_dtype"], c["llm_lora_r"])

    def _log_trainable(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("BoneVQAModel: tổng %.1fM tham số, huấn luyện %.2fM (%.2f%%)", total / 1e6, trainable / 1e6, 100 * trainable / max(total, 1))

    @property
    def device(self) -> torch.device:
        return next(self.closed_head.parameters()).device

    @torch.no_grad()
    def refresh_answer_bank(self):
        if not self.cfg["use_latent_prompt"]:
            return
        mode = str(self.cfg.get("latent_answer_bank_mode", "train")).lower()
        if mode not in ("train", "empty", "random"):
            raise ValueError(f"latent_answer_bank_mode không hợp lệ: {mode!r}")
        if mode == "empty":
            self.latent.set_answer_bank(torch.zeros(0, self.cfg["d_model"], device=self.device))
            self.answer_bank_ready = True
            return
        pooled = []
        for i in range(0, len(self.answer_vocab), 128):
            chunk = self.answer_vocab[i:i + 128]
            out = self.text.encode_frozen(chunk, self.device)
            pooled.append(out["tokens"][:, 0].float())
        bank = torch.cat(pooled, dim=0)
        if mode == "random":
            # Distribution/shape-matched placebo: destroy token semantics with
            # a fixed feature permutation and sign flip while preserving each
            # row's L2 norm and the exact answer-bank cardinality.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(self.cfg.get("latent_control_seed", 314159)))
            permutation = torch.randperm(bank.shape[-1], generator=generator, device="cpu").to(bank.device)
            signs = torch.randint(0, 2, (bank.shape[-1],), generator=generator, device="cpu")
            signs = signs.to(bank.device, bank.dtype).mul_(2).sub_(1)
            bank = bank[:, permutation] * signs
        self.latent.set_answer_bank(bank)
        # Never overwrite trainable prior parameters here.  The old code did
        # so on the first P forward after checkpoint loading, destroying the
        # learned checkpoint state.  A real LaPA prior requires a sealed
        # organ-disease graph; plain labels remain an explicitly experimental
        # trainable branch and are restored solely from the checkpoint.
        self.answer_bank_ready = True

    def _prepare_masks(self, images: List[Image.Image], masks: List, boxes: List, image_ids: List) -> List[np.ndarray]:
        need = self.cfg["use_visual_prompt"] or self.cfg["use_lens"]
        out = []
        for i, im in enumerate(images):
            m = masks[i] if masks is not None else None
            if m is not None and len(m) > 0:
                m = np.asarray(m, dtype=np.uint8)
                if m.ndim == 2:
                    m = m[None]
            else:
                b = boxes[i] if boxes is not None else None
                if need and self.cfg["sam"].get("enabled", True):
                    m = self.segmenter(
                        im, boxes=b,
                        cache_key=str(image_ids[i]) if image_ids is not None and image_ids[i] else None,
                    )
                elif need and b is not None and len(b) > 0:
                    # Production detector proposals remain meaningful when SAM
                    # is deliberately disabled/offline.  Previously direct
                    # ``answer(..., boxes=...)`` silently discarded them.
                    proposal_masks = []
                    for box in b:
                        x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
                        x1, x2 = max(0, min(im.width, x1)), max(0, min(im.width, x2))
                        y1, y2 = max(0, min(im.height, y1)), max(0, min(im.height, y2))
                        if x2 > x1 and y2 > y1:
                            mask = np.zeros((im.height, im.width), dtype=np.uint8)
                            mask[y1:y2, x1:x2] = 1
                            proposal_masks.append(mask)
                    m = (np.stack(proposal_masks) if proposal_masks
                         else np.zeros((0, im.height, im.width), dtype=np.uint8))
                else:
                    m = np.zeros((0, im.height, im.width), dtype=np.uint8)
            out.append(m[: self.cfg["n_masks_max"]])
        return out

    def _question_text(self, question: str, region: Optional[str]) -> str:
        if not self.cfg["use_textual_prompt"]:
            return question
        if not self.cfg.get("use_region_in_prompt", True):
            region = None
        return textual_prompt(question, region, style=self.cfg.get("textual_prompt_style", "clinical"))

    def _encode_images(self, images, mask_list, prompt_kind: str, training: bool,
                       include_lens_views: bool = False):
        c = self.cfg
        b = len(images)
        all_imgs: List[Image.Image] = []
        global_idx, local_idx, lens_idx = [], [], []
        for im, ms in zip(images, mask_list):
            global_idx.append(len(all_imgs))
            all_imgs.append(im)
            li = []
            if c["use_visual_prompt"] and len(ms) > 0:
                for pim in build_visual_prompts(im, ms, kind=prompt_kind):
                    li.append(len(all_imgs))
                    all_imgs.append(pim)
            local_idx.append(li)
            if c["use_lens"]:
                # Localization Lens is an alignment/training augmentation by
                # default.  Its paper explicitly supports original-image-only
                # inference, so production evaluation must not silently depend
                # on segmentation-derived views.
                use_semantic_lens = len(ms) > 0 and (
                    training or include_lens_views or not c.get("lens_training_only", True)
                )
                if use_semantic_lens:
                    modes = ["unicolor", "multicolor", "masked"]
                    chosen = random.sample(modes, 2) if training else modes[:2]
                    vi = [global_idx[-1]]
                    for md in chosen:
                        vi.append(len(all_imgs))
                        all_imgs.append(build_lens(im, ms, mode=md))
                else:
                    vi = [global_idx[-1]]
                lens_idx.append(vi)
        pixel = self.visual.preprocess(all_imgs, self.device)
        enc = self.visual(pixel)
        tokens, embeds = enc["tokens"], enc["embed"]
        d = tokens.shape[-1]
        global_tokens = tokens[global_idx]
        global_embed = embeds[global_idx]

        # A stable global prefix exists in every ablation.  FAVP-style V is
        # represented only by the additional information supplied by local
        # prompted views.  With no local ROI, the delta is exactly zero rather
        # than another learned global-capacity branch.
        prefix_base = self.qformer(global_tokens)
        visual_delta = None
        visual_active = torch.tensor(
            [bool(li) for li in local_idx], dtype=torch.bool, device=tokens.device,
        )
        if c["use_visual_prompt"]:
            visual_delta = torch.zeros_like(prefix_base)
        if c["use_visual_prompt"] and bool(visual_active.any()):
            max_local = max(len(li) for li in local_idx)
            n_tok = tokens.shape[1]
            vp_tokens = tokens.new_zeros(b, (1 + max_local) * n_tok, d)
            vp_pad = torch.ones(b, (1 + max_local) * n_tok, dtype=torch.bool, device=tokens.device)
            for i, li in enumerate(local_idx):
                idxs = [global_idx[i]] + li
                seq = tokens[idxs].reshape(-1, d)
                vp_tokens[i, : seq.shape[0]] = seq
                vp_pad[i, : seq.shape[0]] = False
            joint_prefix = self.qformer(vp_tokens, vp_pad)
            active = visual_active[:, None, None].to(joint_prefix.dtype)
            visual_delta = (joint_prefix - prefix_base) * active
        if visual_delta is not None and c.get("visual_force_zero", False):
            visual_delta = torch.zeros_like(visual_delta)

        visual_route = c.get("visual_route", "dual")
        if visual_route not in ("prefix_only", "fusion_only", "dual"):
            raise ValueError(f"visual_route không hợp lệ: {visual_route!r}")
        prefix_vis = prefix_base
        if visual_delta is not None and visual_route in ("prefix_only", "dual"):
            prefix_vis = prefix_base + self.prompt_mixer.gamma_visual * visual_delta

        # L uses a paired, capacity-matched delta:
        #   LensFusion(original + 2 semantic lenses)
        # - LensFusion(original + 2 duplicated originals).
        # The same module/parameters process both sides.  With no lens signal,
        # and at original-only production inference, the delta is exactly zero.
        lens_delta = None
        lens_teacher_tokens = None
        lens_active_rows = torch.zeros(0, dtype=torch.long, device=tokens.device)
        lens_active = torch.tensor(
            [len(vi) > 1 for vi in lens_idx], dtype=torch.bool, device=tokens.device,
        ) if c["use_lens"] else torch.zeros(b, dtype=torch.bool, device=tokens.device)
        if c["use_lens"]:
            lens_delta = tokens.new_zeros(b, self.lens_fusion.n_out_tokens, d)
        if c["use_lens"] and bool(lens_active.any()):
            active_rows = lens_active.nonzero(as_tuple=False).flatten()
            lens_active_rows = active_rows
            active_list = active_rows.detach().cpu().tolist()
            real_views = torch.stack([tokens[lens_idx[i]] for i in active_list], dim=0)
            base_views = global_tokens[active_rows].unsqueeze(1).expand(-1, real_views.shape[1], -1, -1)
            paired = torch.cat([real_views, base_views], dim=0)
            paired_fused = self.lens_fusion(paired)
            n_active = real_views.shape[0]
            lens_teacher_tokens = paired_fused[:n_active]
            semantic_delta = paired_fused[:n_active] - paired_fused[n_active:]
            lens_delta = lens_delta.index_copy(0, active_rows, semantic_delta)
        if lens_delta is not None and c.get("lens_force_zero", False):
            lens_delta = torch.zeros_like(lens_delta)
            if lens_teacher_tokens is not None:
                # Capacity-matched original-only teacher: keep LensFusion and
                # the same ROI-selected samples/optimizer path, but remove the
                # semantic lens pixels.
                lens_teacher_tokens = paired_fused[n_active:]

        # In the paper-aligned mode, L is an auxiliary training teacher.  The
        # answer endpoint always consumes the original-image stream, during
        # both train and inference, eliminating lens-on-train/lens-off-test
        # representation shift.  Legacy/explicit direct-fusion variants must
        # opt out of lens_training_only and be reported separately.
        endpoint_lens_delta = None if c.get("lens_training_only", False) else lens_delta
        endpoint_visual_delta = visual_delta if visual_route in ("fusion_only", "dual") else None
        f_i = self.prompt_mixer(global_tokens, endpoint_visual_delta, endpoint_lens_delta)
        contributions = self.prompt_mixer.contributions(
            global_tokens, endpoint_visual_delta, endpoint_lens_delta,
        )
        return {
            "prefix_vis": prefix_vis,
            "f_i": f_i,
            "global_embed": global_embed,
            "global_tokens": global_tokens,
            "visual_delta": visual_delta,
            "lens_delta": lens_delta,
            "lens_teacher_tokens": lens_teacher_tokens,
            "lens_active_rows": lens_active_rows,
            "visual_active": visual_active,
            "lens_active": lens_active,
            "prompt_contributions": contributions,
        }

    def _encode_core(self, images, questions, regions, mask_list, prompt_kind, training,
                     include_lens_views: bool = False, context_group_ids=None):
        vis = self._encode_images(
            images, mask_list, prompt_kind, training,
            include_lens_views=include_lens_views,
        )
        q_texts = [self._question_text(q, r) for q, r in zip(questions, regions)]
        txt = self.text(q_texts, self.device)
        if self.cfg["use_latent_prompt"]:
            if not self.answer_bank_ready:
                self.refresh_answer_bank()
            context = torch.cat([vis["f_i"], txt["tokens"]], dim=1)
            context_mask = torch.cat([
                torch.ones(vis["f_i"].shape[:2], dtype=torch.bool, device=self.device),
                txt["mask"],
            ], dim=1)
            x_lp, prior = self.latent(context, context_mask)
        else:
            x_lp, prior = None, None
        fusion_x_lp = None if self.cfg.get("latent_force_zero", False) else x_lp
        fusion_prior = None if fusion_x_lp is None else prior
        fus = self.fusion(vis["f_i"], txt["tokens"], fusion_x_lp, txt["mask"], fusion_prior,
                          latent_context_control=lf_context_mode(self.cfg, training),
                          context_group_ids=context_group_ids)
        logits_closed = self.closed_head(fus["x_f"])
        type_logit = self.type_head(fus["x_f"])
        fracture_logit = self.fracture_head(fus["x_f"]) if self.fracture_head is not None else None
        return {"vis": vis, "txt": txt, "x_lp": x_lp, "fus": fus, "logits_closed": logits_closed,
                "type_logit": type_logit, "fracture_logit": fracture_logit, "q_texts": q_texts}

    @staticmethod
    def _format_llm_prompt(question: str) -> str:
        return f"Question: {question} Answer:"

    def _build_llm_inputs(self, prefix: torch.Tensor, questions: List[str], answers: Optional[List[str]]):
        tok = self.llm_tokenizer
        emb_layer = self.llm.get_input_embeddings()
        b = prefix.shape[0]
        prompts = [self._format_llm_prompt(q) for q in questions]
        p_enc = tok(prompts, add_special_tokens=False, padding=False, truncation=True, max_length=self.cfg["max_question_tokens"] + 8)
        seqs, labels = [], []
        for i in range(b):
            ids = list(p_enc["input_ids"][i])
            lab = [-100] * len(ids)
            if answers is not None:
                a_ids = tok(" " + answers[i], add_special_tokens=False, truncation=True, max_length=self.cfg["max_answer_tokens"])["input_ids"]
                a_ids = a_ids + [tok.eos_token_id]
                ids += a_ids
                lab += a_ids
            seqs.append(ids)
            labels.append(lab)
        max_len = max(len(s) for s in seqs)
        pad_id = tok.pad_token_id
        ids_t = torch.full((b, max_len), pad_id, dtype=torch.long, device=prefix.device)
        lab_t = torch.full((b, max_len), -100, dtype=torch.long, device=prefix.device)
        att_t = torch.zeros((b, max_len), dtype=torch.long, device=prefix.device)
        left_pad = answers is None
        for i, (s, l) in enumerate(zip(seqs, labels)):
            if left_pad:
                ids_t[i, max_len - len(s):] = torch.tensor(s, device=prefix.device)
                att_t[i, max_len - len(s):] = 1
            else:
                ids_t[i, : len(s)] = torch.tensor(s, device=prefix.device)
                lab_t[i, : len(l)] = torch.tensor(l, device=prefix.device)
                att_t[i, : len(s)] = 1
        text_emb = emb_layer(ids_t)
        n_prefix = prefix.shape[1]
        inputs_embeds = torch.cat([prefix.to(text_emb.dtype), text_emb], dim=1)
        attention = torch.cat([torch.ones(b, n_prefix, dtype=torch.long, device=prefix.device), att_t], dim=1)
        full_labels = torch.cat([torch.full((b, n_prefix), -100, dtype=torch.long, device=prefix.device), lab_t], dim=1)
        return inputs_embeds, attention, full_labels

    def _llm_prefix(self, core) -> torch.Tensor:
        # x_f is shared with the closed head, so L and P also reach the LLM.
        parts = [core["vis"]["prefix_vis"], core["fus"]["x_f"].unsqueeze(1)]
        return self.prefix_proj(torch.cat(parts, dim=1))

    def forward(self, batch, stage: int = 2) -> Dict[str, torch.Tensor]:
        c = self.cfg
        nb = normalize_batch(batch)
        images, questions, answers = nb["images"], nb["questions"], nb["answers"]
        answer_types = nb.get("answer_types") or ["OPEN"] * len(images)
        regions = nb.get("regions") or [None] * len(images)
        prompt_kind = c["prompt_kind"]
        if self.training and prompt_kind == "random":
            prompt_kind = random.choice(["contour", "box", "circle", "mask"])
        mask_list = self._prepare_masks(images, nb.get("masks"), nb.get("boxes"), nb.get("image_ids"))
        core = self._encode_core(
            images, questions, regions, mask_list, prompt_kind, self.training,
            include_lens_views=(
                stage == 1 and c.get("use_lens", False) and c.get("lens_alignment_eval", True)
            ),
            context_group_ids=nb.get("component_ids"),
        )
        dev = self.device
        b = len(images)
        zero = torch.zeros((), device=dev)

        target_idx = torch.tensor([self.answer_to_idx.get(normalize_answer(a), -1) for a in answers], device=dev)
        masks = endpoint_masks(answer_types, target_idx)
        is_closed = masks["is_closed"]
        valid_cls = masks["closed"]
        loss_cls = (zero if stage == 1 else
                    single_label_cross_entropy(core["logits_closed"], target_idx.clamp(min=0), valid_cls))
        loss_type = (zero if stage == 1 else
                     F.binary_cross_entropy_with_logits(core["type_logit"].float(), is_closed.float()))

        loss_fracture = zero
        if stage != 1 and core["fracture_logit"] is not None:
            fracture_idx = [
                i for i, (question, answer) in enumerate(zip(questions, answers))
                if is_canonical_fracture_presence_question(question)
                and normalize_answer(answer) in ("yes", "no")
            ]
            if fracture_idx:
                idx = torch.tensor(fracture_idx, dtype=torch.long, device=dev)
                target = torch.tensor(
                    [1.0 if normalize_answer(answers[i]) == "yes" else 0.0 for i in fracture_idx],
                    dtype=torch.float32, device=dev,
                )
                pos_weight = torch.tensor(float(c.get("fracture_positive_weight", 1.0)), device=dev)
                loss_fracture = F.binary_cross_entropy_with_logits(
                    core["fracture_logit"][idx].float(), target, pos_weight=pos_weight,
                )

        loss_cs = zero
        if c["use_latent_prompt"] and stage != 1:
            # LaPA consistency is supervised only during training: the latent
            # prompt is pulled toward the *ground-truth answer embedding*, not
            # toward the image feature.  Answer text never enters inference.
            answer_target = self.text.encode_frozen(answers, dev)
            loss_cs = LatentPromptGenerator.consistency_loss(
                core["x_lp"], answer_target["tokens"], answer_target["mask"],
            )

        con_texts = []
        for q, a, r in zip(questions, answers, regions):
            if stage == 1:
                con_texts.append(a if c.get("stage1_caption_loss_weight", 0.0) > 0 else (r if r else a))
            else:
                con_texts.append(f"{q} {a}".strip())
        con_txt = self.text.encode_frozen(con_texts, dev)
        img_vec = self.img_proj(core["fus"]["f_i"].mean(dim=1))
        txt_vec = self.txt_proj(con_txt["tokens"][:, 0])
        mode = "dcl" if stage == 1 or c["contrastive_mode"] == "dcl" else "infonce"
        image_groups = nb.get("image_ids") or list(range(b))
        loss_con = (multi_positive_decoupled_loss(img_vec, txt_vec, image_groups) if mode == "dcl"
                    else multi_positive_info_nce_loss(img_vec, txt_vec, image_groups))

        # Paper-aligned Localization Lens supervision.  Lens views are a
        # teacher/alignment path only; all answer losses above consumed the
        # original-image representation.  Phase/stage 1 uses DCL and VQA
        # stages use InfoNCE, matching the intended two-phase protocol.
        loss_lens_align = zero
        lens_teacher = core["vis"].get("lens_teacher_tokens")
        lens_rows = core["vis"].get("lens_active_rows")
        if lens_teacher is not None and lens_rows is not None and lens_rows.numel() > 0:
            lens_img_vec = self.img_proj(lens_teacher.mean(dim=1))
            lens_txt_vec = txt_vec[lens_rows]
            row_list = lens_rows.detach().cpu().tolist()
            lens_groups = [image_groups[i] for i in row_list]
            loss_lens_align = lens_alignment_loss(lens_img_vec, lens_txt_vec, lens_groups, mode)

        loss_lm = zero
        open_idx = masks["open"].nonzero(as_tuple=False).flatten().tolist()
        lm_weight = float(c.get("stage1_caption_loss_weight", 0.0)) if stage == 1 else 1.0
        if not np.isfinite(lm_weight) or lm_weight < 0:
            raise ValueError("stage1_caption_loss_weight must be finite and nonnegative")
        if stage == 1 and lm_weight > 0 and (self.llm is None or len(open_idx) != b):
            raise ValueError("Caption bridge pretraining requires an LLM and all-OPEN caption targets")
        if lm_weight > 0 and self.llm is not None and open_idx:
            open_core = self._subset_core(core, open_idx)
            prefix = self._llm_prefix(open_core)
            # Use the same question representation for both the biomedical
            # text encoder and the LLM.  Otherwise T affected only one branch
            # and its ablation had no coherent semantic interpretation.
            open_questions = [core["q_texts"][i] for i in open_idx]
            open_answers = [answers[i] for i in open_idx]
            inputs_embeds, attention, labels = self._build_llm_inputs(prefix, open_questions, open_answers)
            out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention, labels=labels)
            loss_lm = out.loss.float()

        loss_answer = loss_cls + lm_weight * loss_lm + c.get("fracture_loss_weight", 1.0) * loss_fracture
        loss = (loss_answer + c["eta"] * loss_cs + c["lam"] * loss_con
                + c.get("lens_alignment_weight", 0.1) * loss_lens_align
                + c["type_loss_weight"] * loss_type)
        prompt_contrib = core["vis"]["prompt_contributions"]
        visual_contribution_norm = prompt_contrib["visual"].float().norm(dim=-1).mean()
        lens_contribution_norm = prompt_contrib["lens"].float().norm(dim=-1).mean()
        latent_contribution_norm = core["fus"]["latent_delta"].float().norm(dim=-1).mean()
        return {"loss": loss, "loss_answer": loss_answer,
                "loss_cls": loss_cls.detach(), "loss_lm": loss_lm.detach(), "loss_cs": loss_cs.detach(),
                "loss_con": loss_con.detach(), "loss_lens_align": loss_lens_align.detach(),
                "loss_lens_alignment": loss_lens_align,
                "loss_type": loss_type.detach(),
                "loss_fracture": loss_fracture.detach(), "logits_closed": core["logits_closed"].detach(),
                "type_logit": core["type_logit"].detach(),
                "visual_contribution_norm": visual_contribution_norm.detach(),
                "lens_contribution_norm": lens_contribution_norm.detach(),
                "latent_contribution_norm": latent_contribution_norm.detach()}

    @torch.no_grad()
    def generate_open(self, core, questions: List[str]) -> List[Dict[str, Any]]:
        if self.llm is None:
            return [{"text": "", "confidence": 0.0} for _ in questions]
        prefix = self._llm_prefix(core)
        inputs_embeds, attention, _ = self._build_llm_inputs(prefix, questions, None)
        gen = self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention, max_new_tokens=self.cfg["max_new_tokens"],
                                do_sample=False, num_beams=1, use_cache=True, pad_token_id=self.llm_tokenizer.pad_token_id,
                                eos_token_id=self.llm_tokenizer.eos_token_id, output_scores=True, return_dict_in_generate=True)
        seqs = gen.sequences
        results = []
        for i in range(seqs.shape[0]):
            ids = seqs[i].tolist()
            probs = []
            for step, sc in enumerate(gen.scores):
                if step >= len(ids):
                    break
                tid = ids[step]
                if tid == self.llm_tokenizer.eos_token_id or tid == self.llm_tokenizer.pad_token_id:
                    break
                probs.append(torch.softmax(sc[i].float(), dim=-1)[tid].item())
            text = self.llm_tokenizer.decode(ids, skip_special_tokens=True).strip()
            for stop in ("\nQuestion:", "Question:", "\n\n"):
                if stop in text:
                    text = text.split(stop)[0].strip()
            results.append({"text": text, "confidence": float(np.mean(probs)) if probs else 0.0})
        return results

    @torch.no_grad()
    def _yes_no_prob_llm(self, core, questions: List[str]) -> Optional[torch.Tensor]:
        if self.llm is None:
            return None
        if not hasattr(self, "_yes_no_ids"):
            tok = self.llm_tokenizer
            self._yes_no_ids = ([tok(t, add_special_tokens=False)["input_ids"][0] for t in (" yes", " Yes", " YES")],
                                [tok(t, add_special_tokens=False)["input_ids"][0] for t in (" no", " No", " NO")])
        prefix = self._llm_prefix(core)
        inputs_embeds, attention, _ = self._build_llm_inputs(prefix, questions, None)
        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention)
        logits = out.logits[:, -1].float()
        yes_ids, no_ids = self._yes_no_ids
        p_yes = torch.logsumexp(logits[:, yes_ids], dim=-1)
        p_no = torch.logsumexp(logits[:, no_ids], dim=-1)
        return torch.sigmoid(p_yes - p_no)

    @torch.no_grad()
    def predict_batch(self, batch, force_type: Optional[str] = None, prompt_kind: Optional[str] = None,
                      tta_kinds: Optional[List[str]] = None, tra_chi_tiet: bool = False) -> List[Dict[str, Any]]:
        self.eval()
        nb = normalize_batch(batch)
        images, questions = nb["images"], nb["questions"]
        regions = nb.get("regions") or [None] * len(images)
        mask_list = self._prepare_masks(images, nb.get("masks"), nb.get("boxes"), nb.get("image_ids"))
        kinds = list(tta_kinds or self.cfg.get("tta_kinds") or [prompt_kind or self.cfg["prompt_kind"]])
        cores = [self._encode_core(images, questions, regions, mask_list, k, False,
                                  context_group_ids=nb.get("component_ids")) for k in kinds]
        core = cores[0]
        # Match the single-label cross-entropy objective used by the closed
        # head. Independent sigmoid scores are not calibrated categorical
        # probabilities for mutually exclusive answer classes.
        probs_closed = torch.stack([
            torch.softmax(c["logits_closed"].float(), dim=-1) for c in cores
        ]).mean(dim=0)
        fracture_probs = None
        if self.fracture_head is not None:
            fracture_probs = torch.stack([
                torch.sigmoid(c["fracture_logit"].float()) for c in cores
            ]).mean(dim=0)
        type_prob = torch.stack([torch.sigmoid(c["type_logit"].float()) for c in cores]).mean(dim=0)
        w_llm = float(self.cfg.get("closed_llm_weight", 0.0))
        yn_llm = self._yes_no_prob_llm(core, core["q_texts"]) if w_llm > 0 else None
        results = []
        need_open = []
        for i, q in enumerate(questions):
            if force_type:
                atype = force_type
            elif is_closed_question(q):
                atype = "CLOSED"
            else:
                atype = "CLOSED" if type_prob[i].item() > 0.7 else "OPEN"
            if atype == "CLOSED":
                j_full = int(probs_closed[i].argmax().item())
                if is_polar_question(q) or self.answer_vocab[j_full] in ("yes", "no"):
                    fracture_presence = is_canonical_fracture_presence_question(q)
                    if fracture_presence and fracture_probs is not None:
                        p_yes_raw = float(fracture_probs[i].item())
                        slope = self.cfg.get("fracture_platt_slope")
                        intercept = self.cfg.get("fracture_platt_intercept")
                        p_yes = (
                            apply_platt_probability(p_yes_raw, slope, intercept)
                            if slope is not None and intercept is not None else p_yes_raw
                        )
                    else:
                        yn = probs_closed[i, [self.answer_to_idx["yes"], self.answer_to_idx["no"]]]
                        p_yes = float(yn[0].item()) / max(float(yn[0].item() + yn[1].item()), 1e-6)
                        if yn_llm is not None:
                            p_yes = (1.0 - w_llm) * p_yes + w_llm * float(yn_llm[i].item())
                    threshold = (float(self.cfg.get("fracture_yes_threshold", 0.5))
                                 if fracture_presence else 0.5)
                    ans = "yes" if p_yes >= threshold else "no"
                    conf = p_yes if ans == "yes" else 1.0 - p_yes
                else:
                    ans, conf = self.answer_vocab[j_full], float(probs_closed[i, j_full].item())
                results.append({"answer": ans, "answer_type": "CLOSED", "confidence": conf, "masks": mask_list[i]})
                if is_polar_question(q):
                    results[-1]["p_yes"] = p_yes
                    if fracture_presence and fracture_probs is not None:
                        results[-1]["p_yes_raw"] = p_yes_raw
                if tra_chi_tiet:
                    # Preserve the dedicated fracture probability (and any
                    # configured LLM blend for other polar questions); detailed
                    # diagnostics must never overwrite the decision score.
                    if not is_polar_question(q):
                        yn = probs_closed[i, [self.answer_to_idx["yes"], self.answer_to_idx["no"]]]
                        results[-1]["p_yes"] = float(yn[0].item()) / max(float(yn[0].item() + yn[1].item()), 1e-6)
                    results[-1]["vocab_top1"] = self.answer_vocab[j_full]
                    results[-1]["vocab_p"] = float(probs_closed[i, j_full].item())
                if self.cfg.get("closed_via_llm", False):
                    need_open.append(i)
            else:
                need_open.append(i)
                results.append({"answer": "", "answer_type": "OPEN", "confidence": 0.0, "masks": mask_list[i]})
                if tra_chi_tiet:
                    j_full = int(probs_closed[i].argmax().item())
                    yn = probs_closed[i, [self.answer_to_idx["yes"], self.answer_to_idx["no"]]]
                    results[-1]["p_yes"] = float(yn[0].item()) / max(float(yn[0].item() + yn[1].item()), 1e-6)
                    results[-1]["vocab_top1"] = self.answer_vocab[j_full]
                    results[-1]["vocab_p"] = float(probs_closed[i, j_full].item())
        if need_open:
            sub_core = self._subset_core(core, need_open)
            gens = self.generate_open(sub_core, [core["q_texts"][i] for i in need_open])
            for i, g in zip(need_open, gens):
                text = g["text"].strip()
                if results[i]["answer_type"] == "CLOSED":
                    low = text.lower()
                    if is_polar_question(questions[i]):
                        if low.startswith("yes"):
                            results[i]["answer"], results[i]["confidence"] = "yes", g["confidence"]
                        elif low.startswith("no"):
                            results[i]["answer"], results[i]["confidence"] = "no", g["confidence"]
                    elif text:
                        results[i]["answer"], results[i]["confidence"] = text, g["confidence"]
                else:
                    results[i]["answer"] = text
                    results[i]["confidence"] = g["confidence"]
        return results

    def _subset_core(self, core, idxs: List[int]):
        idx = torch.tensor(idxs, device=self.device)
        sub = {"vis": {"prefix_vis": core["vis"]["prefix_vis"][idx]},
               "fus": {"x_ii": core["fus"]["x_ii"][idx], "x_f": core["fus"]["x_f"][idx]}}
        return sub

    @torch.no_grad()
    def answer(self, image: Image.Image, question: str, boxes=None, prompt_kind: str = "contour",
               region: Optional[str] = None, masks: Optional[np.ndarray] = None) -> Dict[str, Any]:
        batch = {"images": [image], "questions": [question], "answers": [""], "answer_types": ["OPEN"],
                 "regions": [region], "masks": [masks], "boxes": [boxes], "image_ids": [None]}
        res = self.predict_batch(batch, prompt_kind=prompt_kind)[0]
        ms = res["masks"]
        prompt_image = image.convert("RGB")
        lens_image = image.convert("RGB")
        if ms is not None and len(ms) > 0:
            from .prompt_draw import PALETTE, draw_prompt

            for i in range(len(ms)):
                prompt_image = draw_prompt(prompt_image, ms[i], kind=prompt_kind, color=PALETTE[i % len(PALETTE)])
            lens_image = build_lens(image, ms, mode="multicolor", alpha=0.5)
        res["prompt_image"] = prompt_image
        res["lens_image"] = lens_image
        res["question"] = question
        return res

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        trainable_names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in trainable_names}

    def save(self, path: str, extra: Optional[dict] = None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = self.trainable_state_dict()
        payload = {"state": state, "config": self.cfg, "answer_vocab": self.answer_vocab, "extra": extra or {},
                   "manifest": checkpoint_manifest(self.cfg, self.answer_vocab, state)}
        torch.save(payload, path)
        logger.info("Đã lưu checkpoint (%d tensor huấn luyện) vào %s", len(payload["state"]), path)

    def load_trainable(self, path: str, allowed_mismatch_prefixes=()):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_manifest(payload, self.cfg, self.answer_vocab)
        expected = self.trainable_state_dict()
        state = compatible_state(expected, payload["state"], allowed_mismatch_prefixes)
        self.load_state_dict(state, strict=False)
        logger.info("Đã nạp checkpoint strict %s (%d tensor)", path, len(state))
        self.answer_bank_ready = False
        return payload.get("extra", {})

    @classmethod
    def load(cls, path: str, device: Optional[str] = None, cfg_override: Optional[dict] = None) -> "BoneVQAModel":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        saved_cfg = payload["config"]
        validate_manifest(payload, saved_cfg, payload["answer_vocab"])
        cfg = copy.deepcopy(saved_cfg)
        # PyTorch MPS does not implement bfloat16 tensors.  The Qwen backbone
        # can be loaded in fp16 without changing checkpoint semantics.
        if device == "mps" and cfg.get("llm_dtype") in ("bfloat16", "bf16"):
            cfg = copy.deepcopy(cfg)
            cfg["llm_dtype"] = "float16"
        if cfg_override:
            cfg = copy.deepcopy(cfg)
            for k, v in cfg_override.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
        model = cls(cfg, payload["answer_vocab"], device=device)
        state = compatible_state(model.trainable_state_dict(), payload["state"])
        model.load_state_dict(state, strict=False)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(dev)
        model.eval()
        return model
