import numpy as np

from .base import Task, rf
from .utils import general_detokenize
from .metrics import mean, f1_score


class KorNLI(Task):
    VERSION = 0
    IN_HF_HUB = False
    DATASET_PATH = "bm_kor-nlu-datasets"
    DATASET_NAME = "KorNLI"
    DATA_FILES = data_files = {
        "train": "multinli.train.ko.tsv",
        "validation": "xnli.dev.ko.tsv",
        "test": "xnli.test.ko.tsv",
    }
    DATASET_TYPE = "csv"
    DOWNLOAD_OPTIONS = {"delimiter": "\t", "quoting": 3}

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
        return "{}:{}".format(doc["sentence1"], doc["sentence2"].strip())

    def doc_to_target(self, doc):
        """
        참 = entailment
        모순 = contradiction
        중립 = neutral
        """
        return " ({})".format(
            {"entailment": "참", "neutral": "중립", "contradiction": "모순"}[
                doc["gold_label"]
            ]
        )

    def idx_to_target(self, idx):
        return " {}".format({0: "참", 1: "중립", 2: "모순"}[idx])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " (참)")
        ll_neither, _ = rf.loglikelihood(ctx, " (중립)")
        ll_false, _ = rf.loglikelihood(ctx, " (모순)")
        return ll_true, ll_neither, ll_false

    def process_results(self, doc, results):
        gold = {"entailment": 0, "neutral": 1, "contradiction": 2}[doc["gold_label"]]
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


class KorSTS(Task):
    VERSION = 0
    IN_HF_HUB = False
    DATASET_PATH = "bm_kor-nlu-datasets"
    DATASET_NAME = "KorSTS"
    DATA_FILES = data_files = {
        "train": "sts-train.tsv",
        "validation": "sts-dev.tsv",
        "test": "sts-test.tsv",
    }
    DATASET_TYPE = "csv"
    DOWNLOAD_OPTIONS = {"delimiter": "\t", "quoting": 3}

    def n_shot(self):
        return 2

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
        return " {}".format({0: "다름", 1: "같음"}[1 if doc["score"] >= 3 else 0])

    def idx_to_target(self, idx):
        return " {}".format({0: "다름", 1: "같음"}[idx])

    def construct_requests(self, doc, ctx):
        ll_negative, _ = rf.loglikelihood(ctx, " 다름")
        ll_positive, _ = rf.loglikelihood(ctx, " 같음")
        return ll_negative, ll_positive

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = 1 if doc["score"] >= 3 else 0
        return {"acc": pred == gold, "f1": (gold, pred)}

    def higher_is_better(self):
        return {"acc": True, "f1": True}

    def aggregation(self):
        return {"acc": mean, "f1": f1_score}
