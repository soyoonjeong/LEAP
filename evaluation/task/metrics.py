import re
import string
from collections import Counter
from collections.abc import Iterable

import torch
import numpy as np
import sacrebleu
import sklearn.metrics
from rouge import Rouge
from bert_score import score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def mean(arr):
    """평균 반환"""
    return np.mean(arr)


def median(arr):
    """중앙값 반환"""
    return np.median(arr)


def f1_score(items):
    """F1 점수 반환"""
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = sklearn.metrics.f1_score(golds, preds)

    return np.max(fscore)


def macro_f1_score(items):
    """ "매크로 F1 점수 반환"""
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = sklearn.metrics.f1_score(golds, preds, average="macro")

    return fscore


def normalize_answer(s):
    def remove_(text):
        """불필요한 기호 제거"""
        text = re.sub("'", " ", text)
        text = re.sub('"', " ", text)
        text = re.sub("《", " ", text)
        text = re.sub("》", " ", text)
        text = re.sub("<", " ", text)
        text = re.sub(">", " ", text)
        text = re.sub("〈", " ", text)
        text = re.sub("〉", " ", text)
        text = re.sub("\(", " ", text)
        text = re.sub("\)", " ", text)
        text = re.sub("‘", " ", text)
        text = re.sub("’", " ", text)
        return text

    def white_space_fix(text):
        """공백 제거"""
        return " ".join(text.split())

    def remove_punc(text):
        """구두점 제거"""
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        """소문자로 변환"""
        return text.lower()

    return white_space_fix(remove_punc(lower(remove_(s))))


def f1_score_sentence(prediction, ground_truth):
    """예측값과 정답값의 F1 점수 계산"""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    # 문자 단위로 토큰화하여 F1 점수 계산
    prediction_Char = []
    for tok in prediction_tokens:
        now = [a for a in tok]
        prediction_Char.extend(now)

    ground_truth_Char = []
    for tok in ground_truth_tokens:
        now = [a for a in tok]
        ground_truth_Char.extend(now)

    common = Counter(prediction_Char) & Counter(ground_truth_Char)
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_Char)
    recall = 1.0 * num_same / len(ground_truth_Char)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def exact_match_score(prediction, ground_truth):
    """예측값과 정답값이 완전히 일치하는지 여부 반환"""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def bleu(items):
    """BLEU score (높을수록 좋음)"""
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_bleu(preds, refs).score


def is_non_str_iterable(obj):
    """문자열이 아닌 반복 가능한 객체인지 확인"""
    return isinstance(obj, Iterable) and not isinstance(obj, str)


def _sacreformat(refs, preds):
    """sacrebleu 계산을 위한 참조 및 예측 포맷을 조정"""
    if not is_non_str_iterable(refs):
        refs = list(refs)
    if not is_non_str_iterable(refs[0]):
        refs = [[ref] for ref in refs]
    refs = list(zip(*refs))

    if not is_non_str_iterable(preds):
        preds = list(preds)
    if is_non_str_iterable(preds[0]):
        assert len(preds[0]) == 1, f"Pred must be a str, was {preds[0]}"
        preds = [pred[0] for pred in preds]

    return refs, preds


def rouge(items):
    """ROUGE score (높을수록 좋음)"""
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]

    rouge_scorer = Rouge()
    rouge_score = rouge_scorer.get_scores(preds, refs, avg=True)

    return rouge_score["rouge-1"]["r"]


def bleurt(items):
    """BLEURT score (높을수록 좋음)"""
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]

    tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-base-128")
    model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-128")

    model.eval()
    batch_size = 16
    all_scores = []

    for i in range(0, len(refs), batch_size):
        refs_batch = refs[i : i + batch_size]
        preds_batch = preds[i : i + batch_size]

        with torch.no_grad():
            inputs = tokenizer(
                refs_batch,
                preds_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            scores = model(**inputs)[0].squeeze().tolist()

        if isinstance(
            scores, float
        ):  # If only one item in batch, scores will be a single float
            all_scores.append(scores)
        else:
            all_scores.extend(scores)

    return mean(all_scores)


def bert_score(items):
    """BERTScore (높을수록 좋음)"""
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]

    P, R, F1 = score(preds, refs, lang="ko")
    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1_score": float(F1.mean()),
    }


def kobert_score(items):
    """KoBERTScore (높을수록 좋음)"""
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]

    model_type = "beomi/kcbert-base"
    P, R, F1 = score(preds, refs, model_type=model_type, num_layers=12)

    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1_score": float(F1.mean()),
    }
