from .base import Task


class DATASET(Task):
    # 객체 변수, 모두 정의할 필요 없음
    VERSION = 0  # 데이터셋 버전
    IN_HF_HUB = True  # (필수) 허깅페이스 허브에 업로드되어 있는 데이터셋인지 여부
    DATASET_PATH = ""  # (필수)데이터셋 폴더 경로  (e.g., "bm_kobest_v1", "bm_klue")
    DATASET_NAME = ""  # (필수) 데이터셋 이름  (e.g., "boolq", "nli")
    DATA_FILES = data_files = (
        {  # (허깅페이스 허브에 업로드되어 있지 않을 시) 데이터셋 파일 이름 (e.g., {"train": "train.jsonl", "validation": "validation.jsonl"})
            "train": "",
            "validation": "",
            "test": "",
        }
    )
    DATASET_TYPE = ""  # (허깅페이스 허브에 업로드되어 있지 않을 시) 파일 타입 (e.g., "json", "csv")
    DOWNLOAD_OPTIONS = {
        "": ""
    }  # (허깅페이스 허브에 업로드되어 있지 않을 시) 데이터셋 다운로드 옵션 (e.g., {"delimiter": "\t", "quoting": 3})

    def n_shot(self):
        """
        데이터셋마다 적합한 fewshot 수 반환

        - 1~5 사이 권장
        - OOM 에러 발생 시 fewshot 수 낮출 것)
        """
        return 0

    def request_type(self):
        """
        데이터셋 유형 반환

        - loglikelihood: 모델이 계산한 선택지들의 로그 확률을 정답 선택지와 비교해 평가
        - greedy_until: 모델이 생성한 문장을 정답 문장과 비교해 평가
        """
        return "loglikelihood"

    def description(self):
        """fewshot 예제에 앞서 추가될 작업(Task)의 설명"""
        return None

    def has_training_docs(self):
        """학습 세트를 가지고 있는지 여부"""
        return True

    def has_validation_docs(self):
        """검증 세트를 가지고 있는지 여부"""
        return True

    def has_test_docs(self):
        """테스트 세트를 가지고 있는지 여부"""
        return True

    def training_docs(self):
        """
        학습 데이터셋 반환
        has_training_docs()가 True일 때만 호출됨
        Returns:
            Iterable[obj]: doc_to_text에서 처리할 수 있는 모든 객체의 iterable
        """
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        """
        검증 데이터셋 반환
        has_validation_docs()가 True일 때만 호출됨
        Returns:
            Iterable[obj]: doc_to_text에서 처리할 수 있는 모든 객체의 iterable
        """
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        """
        테스트 데이터셋 반환
        has_test_docs()가 True일 때만 호출됨
        Returns:
            Iterable[obj]: doc_to_text에서 처리할 수 있는 모든 객체의 iterable
        """
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        """
        개별 데이터 샘플을 처리하는 함수 (detokenize, strip, replace 등)
        분할된 데이터에 대해 map 연산으로 적용 가능
        E.g. `map(self._process_doc, self.dataset["validation"])`

        Returns:
            Dict: 처리된 데이터 샘플
        """
        pass

    def doc_to_text(self, doc):
        """
        json 형태의 문서를 텍스트 형식으로 변환
        doc의 형식을 파악한 후 작성
        """
        pass

    def doc_to_target(self, doc):
        """
        json 형태의 문서에서 정답 추출
        doc의 형식을 파악한 후 작성
        """
        pass

    def construct_requests(self, doc, ctx):
        """
        RequestFactory를 사용해서 Request 객체를 생성하고, 모델에 입력될 Request 객체의 Iterable를 반환
        Args:
            doc: training_docs, validation_docs 또는 test_docs에서 반환된 데이터 샘플
            ctx: fewshot_context에 의해 생성된 컨텍스트 문자열
                (몇 가지 샘플 예시, doc의 질문 부분 포함)
        """
        pass

    def process_results(self, doc, results):
        """
        단일 데이터 샘플과 모델의 예측 결과를 받아 평가 지표를 계산하는 함수
        Args:
            doc: training_docs, validation_docs 또는 test_docs에서 반환된 데이터 샘플
            results: 모델의 예측 결과
        Returns:
            Dict[str, Any]: 평가 지표 이름을 key로 하고, 평가 지표 값을 value로 하는 딕셔너리
        """
        pass

    def aggregation(self):
        """
        Returns:
            Dict[str, Callable]: 평가 지표 이름을 key로 하고, 평가 지표 값 리스트를 집계하는 함수를 value로 하는 딕셔너리
        """
        return {"acc": "...", "f1": "..."}

    def higher_is_better(self):
        """
        지표에 대해 값이 높을수록 좋은지 여부를 정의
        Returns:
            Dict[str, bool]: 평가 지표 이름을 key로 하고, 높은 값이 더 좋은 평가 지표인지 여부를 value로 하는 딕셔너리
        """
        return {"acc": True, "f1": True}
