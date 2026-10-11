"""chrF++ 계산(group별·공식 가중 점수), metric 검증, 평가 실행함."""
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime

import peft
import sacrebleu
import torch
import transformers
from sacrebleu.metrics import CHRF

from src.data import GROUPS, check_eval_rows, load_jsonl, save_jsonl, take_per_group
from src.model import backbone_sha256, build_prompt, generate, load_backbone

# 과제 slide 11 예시 (reference 1개, prediction별 공식 chrF++)
SLIDE11_REF = "Although the train arrived late, we held the scheduled meeting online instead of canceling it."
SLIDE11 = [
    ("Exact match", SLIDE11_REF, 100.0),
    ("Paraphrase", "Even though the train was late, we conducted the planned meeting online rather than canceling it.", 54.2),
    ("Omission", "Although the train arrived late, we held the meeting online.", 64.1),
    ("Wrong meaning", "Although the train arrived late, we canceled the scheduled meeting.", 65.3),
    ("Untranslated", "Aunque el tren llegó tarde, we held the scheduled meeting online instead of canceling it.", 70.6),
]
SCORE_KEYS = ["single_en", "single_ko", "mixed_en", "mixed_ko", "S_single", "S_mixed", "S"]
CSV_COLS = ["time", "exp", "split", "n", "trainable_params", "dtype", "batch_size", "max_new_tokens",
            *SCORE_KEYS, "S_sentence_avg", "output_dir"]


def check_metric():
    """slide 11 예시값과 sentence-level chrF++ 비교함. 소수 첫째 자리까지 같으면 OK."""
    chrf = CHRF(word_order=2)
    ok = True
    for name, hyp, expected in SLIDE11:
        got = chrf.sentence_score(hyp, [SLIDE11_REF]).score
        match = round(got, 1) == expected
        ok &= match
        print(f"{name:14s} slide {expected:5.1f} | ours {got:7.3f} | {'OK' if match else 'MISMATCH'}")
    print("signature:", chrf.get_signature())
    return ok


def score(rows, preds):
    """group별 chrF++(corpus-level, 문장 평균)와 공식 가중 점수 계산함."""
    chrf = CHRF(word_order=2)
    res = {"n": {}, "corpus": {}, "sentence_avg": {}}
    for t, l in GROUPS:
        idx = [i for i, r in enumerate(rows) if r["input_type"] == t and r["target_language"] == l]
        if not idx:
            continue
        k = f"{t}_{l}"
        hyps, refs = [preds[i] for i in idx], [rows[i]["answer"] for i in idx]
        res["n"][k] = len(idx)
        res["corpus"][k] = chrf.corpus_score(hyps, [refs]).score
        res["sentence_avg"][k] = sum(chrf.sentence_score(h, [r]).score for h, r in zip(hyps, refs)) / len(idx)
    for agg in ("corpus", "sentence_avg"):
        s = res[agg]
        if len(s) == len(GROUPS):
            s["S_single"] = (s["single_en"] + s["single_ko"]) / 2
            s["S_mixed"] = (s["mixed_en"] + s["mixed_ko"]) / 2
            s["S"] = 0.3 * s["S_single"] + 0.7 * s["S_mixed"]
    res["signature"] = str(chrf.get_signature())
    return res


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def env_info():
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__,
            "sacrebleu": sacrebleu.__version__, "gpu": torch.cuda.get_device_name(0),
            "fp32_matmul_precision": torch.get_float32_matmul_precision(),  # highest면 TF32 미사용
            "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain"))}


def append_results(path, meta):
    """results.csv에 실행 1건을 한 줄로 누적 기록함 (corpus-level 점수 기준)."""
    s = meta["scores"]["corpus"]
    row = {"time": meta["time"], "exp": meta["exp"], "split": meta["split"], "n": meta["n"],
           "trainable_params": meta["trainable_params"], "dtype": meta["model"]["dtype"],
           "batch_size": meta["decoding"]["batch_size"], "max_new_tokens": meta["decoding"]["max_new_tokens"],
           **{k: round(s[k], 2) for k in SCORE_KEYS},
           "S_sentence_avg": round(meta["scores"]["sentence_avg"]["S"], 2), "output_dir": meta["output_dir"]}
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if new:
            w.writeheader()
        w.writerow(row)


def run_eval(cfg, exp, split="validation", per_group=None, overrides=None):
    """예측 생성 후 predictions.jsonl·metrics.json 저장함. 전체 평가일 때만 results.csv에 기록함."""
    dec = {**cfg["decoding"], **(overrides or {})}
    if dec.get("max_new_tokens") is None or dec.get("batch_size") is None:
        raise ValueError("decoding.max_new_tokens / batch_size 미확정임. config나 CLI로 지정해야 함")
    path = cfg["data"][split]
    rows = load_jsonl(path)
    print("data groups:", dict(check_eval_rows(rows, need_answer=True)))
    if per_group:
        rows = take_per_group(rows, per_group)
    out_dir = cfg["experiments"][exp]["output_dir"] + ("_smoke" if per_group else "")
    os.makedirs(out_dir, exist_ok=True)

    t_start = time.time()
    model, tok = load_backbone(cfg)
    sha_before = backbone_sha256(model)
    prompts = [build_prompt(tok, cfg["prompt_template"], r["input"], r["target_language"]) for r in rows]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    gens = generate(model, tok, prompts, dec["batch_size"], dec["max_new_tokens"])
    gen_sec = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30
    if backbone_sha256(model) != sha_before:
        raise RuntimeError("평가 중 backbone weight가 바뀌었음")

    preds = [g[0] for g in gens]
    save_jsonl([{**r, "prediction": p} for r, p in zip(rows, preds)], f"{out_dir}/predictions.jsonl")
    scores = score(rows, preds)
    n_tok = [g[1] for g in gens]
    meta = {
        "exp": exp, "split": split, "n": len(rows), "smoke": bool(per_group),
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": cfg["model"],
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "backbone_sha256": sha_before, "backbone_unchanged": True,
        "prompt_template": cfg["prompt_template"],
        "decoding": {"do_sample": False, "num_beams": 1, "enable_thinking": False, **dec},
        "seed": cfg["seed"],
        "data": {"path": path, "sha256": file_sha256(path)},
        "chrf_signature": scores["signature"],
        "scores": {k: scores[k] for k in ("n", "corpus", "sentence_avg")},
        "gen_tokens": {"mean": round(sum(n_tok) / len(n_tok), 1), "max": max(n_tok),
                       "hit_max_new_tokens": sum(g[2] for g in gens)},
        "runtime_sec": {"generate": round(gen_sec, 1), "total": round(time.time() - t_start, 1)},
        "peak_vram_gib": {"allocated": round(peak[0], 2), "reserved": round(peak[1], 2)},
        "env": env_info(),
        "output_dir": out_dir,
    }
    with open(f"{out_dir}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    if not per_group:
        append_results(cfg["results_csv"], meta)

    c, s = scores["corpus"], scores["sentence_avg"]
    print(f"{'group':10s} {'n':>4s} {'chrF++':>8s} {'sent_avg':>9s}")
    for k in [*scores["n"], *(x for x in ("S_single", "S_mixed", "S") if x in c)]:
        print(f"{k:10s} {scores['n'].get(k, ''):>4} {c[k]:8.2f} {s[k]:9.2f}")
    print(f"backbone unchanged: True | trainable params: {meta['trainable_params']} | "
          f"generate {meta['runtime_sec']['generate']}s | peak VRAM {meta['peak_vram_gib']} | "
          f"gen tokens {meta['gen_tokens']}")
    print("saved:", out_dir)
    return meta
