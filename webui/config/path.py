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
