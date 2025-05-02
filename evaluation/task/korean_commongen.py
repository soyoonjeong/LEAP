from .base import rf, Task

from .metrics import bleu, bleurt, rouge, bert_score


class KoreanCommonGen(Task):
    VERSION = 1
    IN_HF_HUB = False
    DATASET_PATH = "hf_Korean_CommonGen"
    DATASET_NAME = "dataset"
    DATA_FILES = data_files = {
        "train": "korean_commongen_official_train.json",
        "test": "korean_commongen_converted_test.json",
    }
    DATASET_TYPE = "json"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return (
            "\n####\n"
            + "단어들: "
            + ",".join(doc["concept-set"].split("#"))
            + "\n####\n"
            + "조합한 문장:"
        )

    def doc_to_target(self, doc):
        return " " + doc["scene"]

    def construct_requests(self, doc, ctx):
        continuation = rf.greedy_until(ctx, {"until": ["\n", "\u200b", "##"]})
        return continuation

    def process_results(self, doc, results):
        predicted_sentence = results
        reference_sentence = self.doc_to_target(doc)

        inputs = (reference_sentence, predicted_sentence)

        return {"bleu": inputs, "rouge": inputs, "bleurt": inputs, "bert": inputs}

    def aggregation(self):
        return {
            "bleu": bleu,
            "rouge": rouge,
            "bleurt": bleurt,
            "bert": bert_score,
        }

    def higher_is_better(self):
        return {
            "bleu": True,
            "rouge": True,
            "bleurt": True,
            "bert": True,
        }
