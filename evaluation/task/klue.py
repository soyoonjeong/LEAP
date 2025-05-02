import numpy as np

from .base import Task, rf
from .utils import general_detokenize
from .metrics import mean, f1_score, macro_f1_score


class STS(Task):
    VERSION = 0
    IN_HF_HUB = True
    DATASET_PATH = "bm_klue"
    DATASET_NAME = "sts"

    def n_shot(self):
        return 2

    def request_type(self):
        return "loglikelihood"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return "문장1: {}\n문장2: {}\n정답:".format(
            general_detokenize(doc["sentence1"]), general_detokenize(doc["sentence2"])
        )

    def doc_to_target(self, doc):
        return " {}".format({0: "다름", 1: "같음"}[doc["labels"]["binary-label"]])

    def idx_to_target(self, idx):
        return " {}".format({0: "다름", 1: "같음"}[idx])

    def construct_requests(self, doc, ctx):
        ll_negative, _ = rf.loglikelihood(ctx, " 다름")
        ll_positive, _ = rf.loglikelihood(ctx, " 같음")
        return ll_negative, ll_positive

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["labels"]["binary-label"]
        return {"acc": pred == gold, "f1": (gold, pred)}

    def higher_is_better(self):
        return {"acc": True, "f1": True}

    def aggregation(self):
        return {"acc": mean, "f1": f1_score}


class NLI(Task):
    VERSION = 0
    IN_HF_HUB = True
    DATASET_PATH = "bm_klue"
    DATASET_NAME = "nli"

    def n_shot(self):
        return 3

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return "{}:{}".format(doc["premise"], doc["hypothesis"].strip())

    def doc_to_target(self, doc):
        """
        참 = entailment
        모순 = contradiction
        중립 = neutral
        """
        return " ({})".format({0: "참", 1: "중립", 2: "모순"}[doc["label"]])

    def idx_to_target(self, idx):
        return " {}".format({0: "참", 1: "중립", 2: "모순"}[idx])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " (참)")
        ll_neither, _ = rf.loglikelihood(ctx, " (중립)")
        ll_false, _ = rf.loglikelihood(ctx, " (모순)")
        return ll_true, ll_neither, ll_false

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        return {
            "acc": pred == gold,
        }

    def higher_is_better(self):
        return {
            "acc": True,
        }

    def aggregation(self):
        return {
            "acc": mean,
        }


class YNAT(Task):
    VERSION = 0
    IN_HF_HUB = True
    DATASET_PATH = "bm_klue"
    DATASET_NAME = "ynat"

    def n_shot(self):
        return 5

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc):
        out_doc = {
            "title": doc["title"],
            "choices": [
                "(과학)",
                "(경제)",
                "(사회)",
                "(생활)",
                "(세계)",
                "(스포츠)",
                "(정치)",
            ],
            "gold": doc["label"],
        }
        return out_doc

    def doc_to_text(self, doc):
        return "{}".format(doc["title"])

    def doc_to_target(self, doc):
        return " {}".format(
            {i: choice for i, choice in enumerate(doc["choices"])}[doc["gold"]]
        )

    def construct_requests(self, doc, ctx):
        lls = [
            rf.loglikelihood(ctx, " {}".format(choice))[0] for choice in doc["choices"]
        ]

        return lls

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["gold"]

        acc = 1.0 if np.argmax(results) == gold else 0.0

        return {"acc": acc, "macro_f1": (gold, pred)}

    def higher_is_better(self):
        return {
            "acc": True,
            "macro_f1": True,
        }

    def aggregation(self):
        return {
            "acc": mean,
            "macro_f1": macro_f1_score,
        }


class RE(Task):
    VERSION = 0
    IN_HF_HUB = True
    DATASET_PATH = "bm_klue"
    DATASET_NAME = "re"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc):
        out_doc = {
            "sentence": doc["sentence"],
            "word1": doc["subject_entity"]["word"],
            "word2": doc["object_entity"]["word"],
            "choices": [
                "무관",
                "종료일",
                "설립일",
                "본부소재지",
                "별명",
                "구성원",
                "모임",
                "종교단체",
                "회사생산품",
                "설립자",
                "대표",
                "회원수",
                "출생일",
                "사망일",
                "출생한곳",
                "사망한곳",
                "거주지",
                "출신",
                "회사원",
                "학교학생",
                "별명",
                "부모",
                "자식",
                "형제",
                "배우자",
                "가족",
                "동료",
                "개인생산품",
                "종교",
                "직위",
            ],
            "gold": doc["label"],
        }
        return out_doc

    def doc_to_text(self, doc):
        return "문장: {}\n단어1: {}\n단어2: {}\n관계: ".format(
            general_detokenize(doc["sentence"]), doc["word1"], doc["word2"]
        )

    def doc_to_target(self, doc):
        return " ({})".format(
            {i: choice for i, choice in enumerate(doc["choices"])}[doc["gold"]]
        )

    def construct_requests(self, doc, ctx):
        lls = [
            rf.loglikelihood(ctx, " {}".format(choice))[0] for choice in doc["choices"]
        ]

        return lls

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["gold"]

        acc = 1.0 if pred == gold else 0.0

        return {"acc": acc, "macro_f1": (gold, pred)}

    def higher_is_better(self):
        return {
            "acc": True,
            "macro_f1": True,
        }

    def aggregation(self):
        return {
            "acc": mean,
            "macro_f1": macro_f1_score,
        }
