"""Backbone·tokenizer 로드, chat prompt 생성, greedy 생성, backbone 무결성 확인함."""
import hashlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_backbone(cfg):
    """backbone을 GPU에 올리고 전체 parameter를 freeze함."""
    m = cfg["model"]
    tok = AutoTokenizer.from_pretrained(m["id"], revision=m["revision"])
    tok.padding_side = "left"  # batch 생성 시 left padding 필요함
    model = AutoModelForCausalLM.from_pretrained(
        m["id"], revision=m["revision"], dtype=getattr(torch, m["dtype"]), device_map="cuda")
    model.eval()
    model.requires_grad_(False)
    return model, tok


def build_prompt(tok, template, input_text, target_language):
    """기본 template을 Qwen3 chat template(enable_thinking=False)으로 감쌈."""
    content = template.format(target_language=target_language, input_text=input_text)
    return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def generate(model, tok, prompts, batch_size, max_new_tokens):
    """greedy 생성함. (번역문, 생성 token 수, max_new_tokens 도달 여부) 리스트 반환함."""
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    outs = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        seqs = model.generate(**enc, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                              pad_token_id=tok.pad_token_id)
        for s in seqs[:, enc["input_ids"].shape[1]:].tolist():
            n = next((j for j, t in enumerate(s) if t in eos), len(s))  # EOS 앞까지만 사용함
            outs.append((tok.decode(s[:n], skip_special_tokens=True).strip(), n, n >= max_new_tokens))
    return outs


def backbone_sha256(model):
    """LoRA 제외 parameter 전체의 sha256 계산함. 실행 전후 비교로 weight 불변 확인함."""
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters(), key=lambda x: x[0]):
        if "lora_" in name:
            continue
        h.update(name.encode())
        h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy())
    return h.hexdigest()
