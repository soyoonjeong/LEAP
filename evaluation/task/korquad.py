from .base import rf, Task
from .metrics import mean, exact_match_score, f1_score_sentence


class KorQuad(Task):
    VERSION = 1
    IN_HF_HUB = False
    DATASET_PATH = "korquad_dev"
    DATASET_NAME = ""
    DATA_FILES = {"validation": "converted_dev2.json"}
    DATASET_TYPE = "json"

    def n_shot(self):
        return 0

    def request_type(self):
        return "greedy_until"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return None

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        # return f"""<|begin_of_text|>아래 문맥을 읽고 질문에 답변하십시오. 답변은 \"답변 : [정답]\" 형식으로 작성해 주십시오.\n문맥 : {doc["context"]}\n질문 : {doc["question"]}\n"""
        return f"""<start_of_turn>user\n아래 문맥을 읽고 질문에 답변하십시오. 답변은 \"답변 : [정답]\" 형식으로 작성해 주십시오.\n문맥 : {doc["context"]}\n질문 : {doc["question"]}\n<end_of_turn>\n<start_of_turn>model\n"""
        # return f"""\n####\n제목: {doc["title"]}\n####\n내용: {doc["context"]}\n####\n문제: {doc["question"]}\n####\n"""

        # return ("<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|> 제목과 문서를 참고하여 사용자 질문에 답변해주세요. <출력 폼 예시> \n 답변 : \n <질문>\n {data["question"]} \n </질문> \n <문서> \n {data["context"]} </문서> \n <|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")

    def doc_to_target(self, doc):
        return doc["answer"]

    def construct_requests(self, doc, ctx):
        continuation = rf.greedy_until(ctx, {"until": ["\n", "##"]})
        return continuation

    def process_results(self, doc, results):
        pred = results[0]
        ref = self.doc_to_target(doc)

        return {
            "exact_match": exact_match_score(pred, ref),
            "f1": f1_score_sentence(pred, ref),
        }

    def aggregation(self):
        return {
            "exact_match": mean,  # Exact match (the normalized answer exactly match the gold answer)
            "f1": mean,
        }

    def higher_is_better(self):
        return {
            "exact_match": True,  # Exact match (the normalized answer exactly match the gold answer)
            "f1": True,  #  The F-score of predicted tokens versus the gold answer
        }


class KorQuadFewShot(KorQuad):
    def n_shot(self):
        return 3

    def doc_to_text(self, doc):
        return f"""\n####\n제목: {doc["title"]}\n####\n내용: {doc["context"]}\n####\n문제: {doc["question"]}\n####\n"""
