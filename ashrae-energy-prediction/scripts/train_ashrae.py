"""ASHRAE 학습 실행 (CLI).

  python scripts/train_ashrae.py --trials 30 --timeout 2400

PostgreSQL 접속은 PGHOST/PGUSER/PGPASSWORD, MLflow 는 MLFLOW_TRACKING_URI(기본 http://localhost:5000).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# 프로젝트 루트의 .env (git 에 올리지 않음) 를 환경변수로 읽는다. 이미 설정된 값은 덮어쓰지 않는다.
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# 학습 중 CPU/메모리 부하로 MLflow 응답이 늦어질 수 있어 기본 120초 → 300초
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "300")

# Windows 콘솔/리다이렉트 인코딩(cp949)에서 MLflow 의 이모지 출력이 실패하지 않도록 UTF-8 로 고정
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import ashrae  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=None, help="튜닝 제한 시간(초)")
    ap.add_argument("--tune-frac", type=float, default=0.1, help="튜닝에 쓸 샘플 비율")
    ap.add_argument("--refresh-data", action="store_true", help="DB 에서 데이터셋 다시 추출")
    ap.add_argument("--per-meter", action="store_true", help="미터(전기/냉수/증기/온수)별 모델 4개 학습")
    ap.add_argument("--max-trees", type=int, default=None, help="최종 학습 트리 상한 (기본 3000)")
    ap.add_argument("--cv-per-meter", action="store_true",
                    help="1~12월 전체 학습: 미터별 반기 교차검증 튜닝·평가 → 전체 학습 → test 예측 CSV")
    ap.add_argument("--full-submit", action="store_true",
                    help="2016년 전체 학습 → test 예측 → 제출 파일 (building_level·month 피처, 0값 유지)")
    ap.add_argument("--no-floor", action="store_true", help="--full-submit 에서 floor_count 피처 제외")
    ap.add_argument("--with-ids", action="store_true", help="--full-submit 에 building_id·site_id 피처 추가 (실험 A)")
    ap.add_argument("--train-clean", action="store_true",
                    help="학습 기간 정답값 정리 (계절 운휴 0 유지, 값 고착·위로 튄 값 제거)")
    ap.add_argument("--tune-only", action="store_true", help="튜닝 단계까지만 실행 (최종 학습·저장 생략)")
    ap.add_argument("--no-save", action="store_true",
                    help="모델 등록·ONNX·파일 업로드·검증 생략 (파일 저장소 없이 튜닝→학습만)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    if args.no_save:
        ashrae.SAVE_ARTIFACTS = False
    if args.train_clean:
        ashrae.TRAIN_CLEAN = True
    if args.tune_only:
        ashrae.TUNE_ONLY = True
        ashrae.SAVE_ARTIFACTS = False
    if args.max_trees:
        ashrae.BASE_PARAMS["n_estimators"] = args.max_trees
    if args.full_submit:
        if args.with_ids:
            res = ashrae.run_full_submit(extra_features=["building_id", "site_id"], tag="A-ids")
        elif args.no_floor:
            res = ashrae.run_full_submit(drop_features=["floor_count"], tag="no-floor")
        else:
            res = ashrae.run_full_submit()
        print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
        sys.exit(0)
    fn = (ashrae.run_cv_per_meter if args.cv_per_meter
          else ashrae.run_per_meter if args.per_meter else ashrae.run)
    result = fn(args.trials, args.timeout, args.tune_frac, args.refresh_data)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
