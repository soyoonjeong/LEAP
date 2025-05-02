import os
from .path import SAVE_DATASET_DIR


EVAL_DATASET_METRIC = {
    "klue_nli": "acc",
    "klue_sts": "acc",
    "klue_ynat": "acc",
    "kobest_boolq": "acc",
    "kobest_copa": "acc",
    "kobest_sentineg": "acc",
    "kobest_wic": "acc",
    "kobest_hellaswag": "acc_norm",
}


def list_json_files(directory):
    """
    주어진 디렉토리 아래의 모든 .json 파일의 경로를 반환합니다.

    Args:
        directory (str): 검색할 디렉토리 경로

    Returns:
        list: .json 파일 경로의 리스트
    """
    json_files = {}
    try:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".json"):
                    json_files[os.path.join(root.split("/")[-1], file)] = os.path.join(
                        root, file
                    )
    except:
        pass
    return json_files


def get_eval_datasets():
    EVAL_DATASETS = {
        "klue_ynat": {
            "explain": "뉴스 기사의 주제를 분류하는 능력 평가",
            "metrics": ["acc, macro_f1"],
        },
        "klue_sts": {
            "explain": "두 문장의 의미적 유사도를 측정하는 능력 평가",
            "metrics": ["acc, f1"],
        },
        "klue_nli": {
            "explain": "두 문장 간의 함의 관계를 추론하는 능력 평가",
            "metrics": ["acc"],
        },
        "kobest_sentineg": {
            "explain": "문장의 긍정 또는 부정을 정확히 분류하는 능력을 평가",
            "metrics": ["acc, macro_f1"],
        },
        "kobest_wic": {
            "explain": "같은 단어가 서로 다른 문맥에서 같은 의미로 사용되는지 판단",
            "metrics": ["acc, macro_f1"],
        },
        "kobest_boolq": {
            "explain": "문단에서 주어진 질문에 대해 참 또는 거짓을 판단",
            "metrics": ["acc, macro_f1"],
        },
        "kobest_copa": {
            "explain": "주어진 상황에 가장 적합한 결과를 두 가지 대안 중에서 선택",
            "metrics": ["acc, macro_f1"],
        },
        "kobest_hellaswag": {
            "explain": "주어진 맥락 다음 올 가능성이 높은 문장을 4개 중에서 선택하는 추론 능력 평가",
            "metrics": ["acc, normalized acc, macro_f1"],
        },
        # "korquad": {
        #     "explain": "질문에 대해 주어진 문단에서 정답을 찾아내는 능력을 평가",
        #     "metrics": ["exact match, f1"],
        # },
        # "korquad_fewshot": {
        #     "explain": "질문에 대해 주어진 문단에서 정답을 찾아내는 능력을 평가",
        #     "metrics": ["exact match, f1"],
        # },
    }
    return EVAL_DATASETS


def get_tuning_datasets():
    # 사용 예시
    search_directory = "/home/data/1_Tuning_models/Runpod-LoRA/LoRA-Max-seq-len-3k/1_instruction_dataset"
    TUNING_DATASETS = list_json_files(search_directory)
    TUNING_DATASETS.update(list_json_files(SAVE_DATASET_DIR))

    return TUNING_DATASETS
