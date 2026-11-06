import copy
import logging
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ..utils import is_closed_question, is_polar_question, normalize_answer
from .encoders import TextEncoder, VisualEncoder
from .fusion import MultimodalFusion
from .heads import AnswerTypeHead, ClosedHead, ContrastiveProjector
from .latent_prompt import LatentPromptGenerator
from .lens_fusion import LensFusion, QFormerLite
from .losses import bce_multi_label, decoupled_contrastive_loss, info_nce_loss
from .segment_prompt import SegmentPromptCreator

try:
    from ..data.prompts import build_lens, build_visual_prompts, textual_prompt
except Exception:
    from .prompt_draw import build_lens, build_visual_prompts

    def textual_prompt(question, region=None, modality="X-ray"):
        region_text = f" of the {region}" if region else ""
        return f"This is a bone X-ray image{region_text}. Question: {question}"

logger = logging.getLogger("bonevqa")

DEFAULT_MODEL_CFG: Dict[str, Any] = {
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
    "use_region_in_prompt": True,
    "prompt_kind": "contour",
    "n_masks_max": 3,
    "n_query": 32,
    "qformer_layers": 2,
    "n_latent": 32,
    "lens_layers": 2,
    "lens_shuffle_r": 2,
    "alpha": 1.0,
    "theta": 0.1,
    "beta": 0.1,
    "eta": 0.01,
    "lam": 0.1,
    "contrastive_mode": "infonce",
    "contrastive_dim": 256,
    "type_loss_weight": 0.1,
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
        self.segmenter = SegmentPromptCreator(model_id=c["sam"]["model_id"], sam_med2d_ckpt=c["sam"].get("sam_med2d_ckpt"),
                                              device=self.device_hint, max_masks=c["n_masks_max"], grid=c["sam"].get("grid", 2),
                                              default_prompt=c["sam"].get("default_prompt", "grid_boxes"),
                                              cache_dir=c["sam"].get("cache_dir"), enabled=c["sam"].get("enabled", True),
                                              max_area_ratio=c["sam"].get("max_area_ratio", 0.6),
                                              min_mean_intensity=c["sam"].get("min_mean_intensity", 40.0))
        self.qformer = QFormerLite(d, c["n_query"], c["qformer_layers"], c["n_heads"], c["dropout"])
        grid = c["image_size"] // 16
        self.lens_fusion = LensFusion(d, 3, c["lens_layers"], c["n_heads"], c["dropout"], grid=grid, shuffle_r=c["lens_shuffle_r"])
        self.latent = LatentPromptGenerator(d, c["n_latent"], c["n_heads"], c["dropout"],
                                            prior_labels=c["prior_labels"] if c["use_latent_prompt"] else None)
        self.fusion = MultimodalFusion(d, c["n_heads"], c["dropout"], c["alpha"], c["theta"], c["beta"])
        self.closed_head = ClosedHead(d, len(vocab), c["dropout"])
        self.type_head = AnswerTypeHead(d)
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
        pooled = []
        for i in range(0, len(self.answer_vocab), 128):
            chunk = self.answer_vocab[i:i + 128]
            out = self.text.encode_frozen(chunk, self.device)
            pooled.append(out["tokens"][:, 0].float())
        self.latent.set_answer_bank(torch.cat(pooled, dim=0))
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
            elif need and self.cfg["sam"].get("enabled", True):
                b = boxes[i] if boxes is not None else None
                m = self.segmenter(im, boxes=b, cache_key=str(image_ids[i]) if image_ids is not None and image_ids[i] else None)
            else:
                m = np.zeros((0, im.height, im.width), dtype=np.uint8)
            out.append(m[: self.cfg["n_masks_max"]])
        return out

    def _question_text(self, question: str, region: Optional[str]) -> str:
        if not self.cfg["use_textual_prompt"]:
            return question
        if not self.cfg.get("use_region_in_prompt", True):
            region = None
        return textual_prompt(question, region)

    def _encode_images(self, images, mask_list, prompt_kind: str, training: bool):
        c = self.cfg
        b = len(images)
        all_imgs: List[Image.Image] = []
        global_idx, local_idx, lens_idx = [], [], []
        rng = random  # EXP4: seeded process RNG controls lens-view augmentation
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
                if len(ms) > 0:
                    modes = ["unicolor", "multicolor", "masked"]
                    chosen = rng.sample(modes, 2) if training else modes[:2]
                    vi = [global_idx[-1]]
                    for md in chosen:
                        vi.append(len(all_imgs))
                        all_imgs.append(build_lens(im, ms, mode=md, alpha=None if training else 0.5))
                else:
                    vi = [global_idx[-1]] * 3
                lens_idx.append(vi)
        pixel = self.visual.preprocess(all_imgs, self.device)
        enc = self.visual(pixel)
        tokens, embeds = enc["tokens"], enc["embed"]
        d = tokens.shape[-1]
        global_tokens = tokens[global_idx]
        global_embed = embeds[global_idx]
        if c["use_visual_prompt"]:
            max_local = max(len(li) for li in local_idx)
            n_tok = tokens.shape[1]
            vp_tokens = tokens.new_zeros(b, (1 + max_local) * n_tok, d)
            vp_pad = torch.ones(b, (1 + max_local) * n_tok, dtype=torch.bool, device=tokens.device)
            for i, li in enumerate(local_idx):
                idxs = [global_idx[i]] + li
                seq = tokens[idxs].reshape(-1, d)
                vp_tokens[i, : seq.shape[0]] = seq
                vp_pad[i, : seq.shape[0]] = False
            prefix_vis = self.qformer(vp_tokens, vp_pad)
        else:
            prefix_vis = self.qformer(global_tokens)
        if c["use_lens"]:
            view_tokens = torch.stack([tokens[vi] for vi in lens_idx], dim=0)
            f_i = self.lens_fusion(view_tokens)
        else:
            f_i = global_tokens
        return {"prefix_vis": prefix_vis, "f_i": f_i, "global_embed": global_embed, "global_tokens": global_tokens}

    def _encode_core(self, images, questions, regions, mask_list, prompt_kind, training):
        vis = self._encode_images(images, mask_list, prompt_kind, training)
        q_texts = [self._question_text(q, r) for q, r in zip(questions, regions)]
        txt = self.text(q_texts, self.device)
        b = len(images)
        if self.cfg["use_latent_prompt"]:
            if not self.answer_bank_ready:
                self.refresh_answer_bank()
            x_lp, prior = self.latent(b)
        else:
            x_lp, prior = None, None
        fus = self.fusion(vis["f_i"], txt["tokens"], x_lp, txt["mask"], prior)
        logits_closed = self.closed_head(fus["x_f"])
        type_logit = self.type_head(fus["x_f"])
        return {"vis": vis, "txt": txt, "x_lp": x_lp, "fus": fus, "logits_closed": logits_closed,
                "type_logit": type_logit, "q_texts": q_texts}

    def _build_llm_inputs(self, prefix: torch.Tensor, questions: List[str], answers: Optional[List[str]]):
        tok = self.llm_tokenizer
        emb_layer = self.llm.get_input_embeddings()
        b = prefix.shape[0]
        prompts = [f"Question: {q} Answer:" for q in questions]
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
        parts = [core["vis"]["prefix_vis"]]
        if self.cfg["use_latent_prompt"]:
            parts.append(core["fus"]["x_ii"])
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
        core = self._encode_core(images, questions, regions, mask_list, prompt_kind, self.training)
        dev = self.device
        b = len(images)
        zero = torch.zeros((), device=dev)

        target_idx = torch.tensor([self.answer_to_idx.get(normalize_answer(a), -1) for a in answers], device=dev)
        is_closed = torch.tensor([t == "CLOSED" for t in answer_types], device=dev)
        valid_cls = target_idx >= 0
        loss_cls = zero if stage == 1 else bce_multi_label(core["logits_closed"], target_idx.clamp(min=0), valid_cls)
        loss_type = F.binary_cross_entropy_with_logits(core["type_logit"].float(), is_closed.float())

        loss_cs = zero
        if c["use_latent_prompt"]:
            ans_enc = self.text.encode_frozen([a if a else "none" for a in answers], dev)
            loss_cs = LatentPromptGenerator.consistency_loss(core["x_lp"], ans_enc["tokens"], ans_enc["mask"])

        con_texts = []
        for q, a, r in zip(questions, answers, regions):
            if stage == 1:
                con_texts.append(r if r else a)
            else:
                con_texts.append(f"{q} {a}".strip())
        con_txt = self.text.encode_frozen(con_texts, dev)
        img_vec = self.img_proj(core["fus"]["f_i"].mean(dim=1))
        txt_vec = self.txt_proj(con_txt["tokens"][:, 0])
        mode = "dcl" if stage == 1 or c["contrastive_mode"] == "dcl" else "infonce"
        loss_con = decoupled_contrastive_loss(img_vec, txt_vec) if mode == "dcl" else info_nce_loss(img_vec, txt_vec)

        loss_lm = zero
        if self.llm is not None:
            prefix = self._llm_prefix(core)
            inputs_embeds, attention, labels = self._build_llm_inputs(prefix, questions, answers)
            out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention, labels=labels)
            loss_lm = out.loss.float()

        loss = loss_cls + loss_lm + c["eta"] * loss_cs + c["lam"] * loss_con + c["type_loss_weight"] * loss_type
        return {"loss": loss, "loss_cls": loss_cls.detach(), "loss_lm": loss_lm.detach(), "loss_cs": loss_cs.detach(),
                "loss_con": loss_con.detach(), "loss_type": loss_type.detach(), "logits_closed": core["logits_closed"].detach(),
                "type_logit": core["type_logit"].detach()}

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
        cores = [self._encode_core(images, questions, regions, mask_list, k, False) for k in kinds]
        core = cores[0]
        probs_closed = torch.stack([torch.sigmoid(c["logits_closed"].float()) for c in cores]).mean(dim=0)
        type_prob = torch.stack([torch.sigmoid(c["type_logit"].float()) for c in cores]).mean(dim=0)
        w_llm = float(self.cfg.get("closed_llm_weight", 0.0))
        yn_llm = self._yes_no_prob_llm(core, questions) if w_llm > 0 else None
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
                    yn = probs_closed[i, [self.answer_to_idx["yes"], self.answer_to_idx["no"]]]
                    p_yes = float(yn[0].item()) / max(float(yn[0].item() + yn[1].item()), 1e-6)
                    if yn_llm is not None:
                        p_yes = (1.0 - w_llm) * p_yes + w_llm * float(yn_llm[i].item())
                    ans = "yes" if p_yes >= 0.5 else "no"
                    conf = p_yes if ans == "yes" else 1.0 - p_yes
                else:
                    ans, conf = self.answer_vocab[j_full], float(probs_closed[i, j_full].item())
                results.append({"answer": ans, "answer_type": "CLOSED", "confidence": conf, "masks": mask_list[i]})
                if tra_chi_tiet:
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
            gens = self.generate_open(sub_core, [questions[i] for i in need_open])
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
        sub = {"vis": {"prefix_vis": core["vis"]["prefix_vis"][idx]}, "fus": {"x_ii": core["fus"]["x_ii"][idx]}}
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
        payload = {"state": self.trainable_state_dict(), "config": self.cfg, "answer_vocab": self.answer_vocab, "extra": extra or {}}
        torch.save(payload, path)
        logger.info("Đã lưu checkpoint (%d tensor huấn luyện) vào %s", len(payload["state"]), path)

    def load_trainable(self, path: str):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        missing, unexpected = self.load_state_dict(payload["state"], strict=False)
        unexpected = [u for u in unexpected]
        logger.info("Đã nạp checkpoint %s (%d tensor, %d thừa)", path, len(payload["state"]), len(unexpected))
        self.answer_bank_ready = False
        return payload.get("extra", {})

    @classmethod
    def load(cls, path: str, device: Optional[str] = None, cfg_override: Optional[dict] = None) -> "BoneVQAModel":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = payload["config"]
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
        model.load_state_dict(payload["state"], strict=False)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(dev)
        model.eval()
        return model
