"""실험 진입점. chrF++ 구현 검증, 학습 데이터 생성, baseline 평가를 실행함."""
import argparse
import os
import sys
from pathlib import Path

import yaml
from transformers import set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # config의 상대 경로 기준을 repo root로 맞춤
from src.evaluate import check_metric, run_eval  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--exp", default="baseline")
    ap.add_argument("--check-metric", action="store_true", help="slide 11 예시값으로 chrF++ 구현 확인함")
    ap.add_argument("--build-data", action="store_true", help="config의 train_data 설정으로 train/dev JSONL 생성함")
    ap.add_argument("--per-group", type=int, help="group별 앞 N개만 평가함 (smoke test, results.csv 기록 안 함)")
    ap.add_argument("--max-new-tokens", type=int)
    ap.add_argument("--batch-size", type=int)
    args = ap.parse_args()

    if args.check_metric:
        sys.exit(0 if check_metric() else 1)
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    if args.build_data:
        from src.data import build_train_data
        print(build_train_data(cfg))
        return
    overrides = {k: v for k, v in [("max_new_tokens", args.max_new_tokens), ("batch_size", args.batch_size)]
                 if v is not None}
    run_eval(cfg, args.exp, per_group=args.per_group, overrides=overrides)


if __name__ == "__main__":
    main()
