import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEADERBOARD_PATH = os.path.join(ROOT_DIR, "data/leaderboard.json")
EVAL_RESULTS_PATH = os.path.join(ROOT_DIR, "data/eval_results.json")
MODEL_INFO_PATH = os.path.join(ROOT_DIR, "data/models.json")
MODEL_DIR = "/home/llm_models"
DATASET_DIR = "/home/data"
LOG_DIR = os.path.join(ROOT_DIR, "logs")
SAVE_DATASET_DIR = os.path.join(DATASET_DIR, "llm-tuning-dataset")
SAVE_MODEL_DIR = os.path.join(MODEL_DIR, "llm-tuning-model")


def ensure_paths_exist():
    """필요한 모든 경로가 존재하는지 확인하고 없으면 생성합니다."""
    # 데이터 디렉토리 생성
    data_dir = os.path.join(ROOT_DIR, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 로그 디렉토리 생성
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # 모델 디렉토리 생성
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 데이터셋 디렉토리 생성
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    # 저장 디렉토리 생성
    if not os.path.exists(SAVE_DATASET_DIR):
        os.makedirs(SAVE_DATASET_DIR)

    if not os.path.exists(SAVE_MODEL_DIR):
        os.makedirs(SAVE_MODEL_DIR)

    # JSON 파일들 생성 (빈 파일로)
    for json_path in [LEADERBOARD_PATH, EVAL_RESULTS_PATH, MODEL_INFO_PATH]:
        if not os.path.exists(json_path):
            with open(json_path, "w", encoding="utf-8") as f:
                f.write("{}")
