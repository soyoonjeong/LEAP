## 데이터셋 추가 방법

### STEP 1. 데이터셋 저장
[DATASET_DIR](api/config/path.py) 경로에 데이터셋을 저장합니다. 

### STEP 2. leap/evaluation/task/{dataset_name}.py 파일 생성
1. [evaluation/task/template.py](evaluation/task/template.py) 파일을 복사하여 새로운 데이터셋 파일을 생성합니다.<br>

2. 데이터셋을 로드하기 위한 클래스 변수를 작성합니다. <br>
`os.path.join(args.task_dir, DATASET_PATH, DATASET_NAME)`을 데이터셋 경로로 인식합니다. (참고: [datasets/load_dataset](https://huggingface.co/docs/datasets/loading))
- 추가한 데이터셋으로 평가를 실행하기 전, 데이터셋 로드 코드를 따로 실행해보시는 것을 권장합니다. 
- 허깅페이스에 업로드되어 있는 데이터셋의 경우, 아래와 같이 클래스 변수를 작성합니다.
    ```python
    # 예시: 
    class STS(Task):
        VERSION = 0
        IN_HF_HUB = True
        DATASET_PATH = "bm_klue"
        DATASET_NAME = "sts"

    # 데이터셋 로드 코드
    data_dir = "/home/data/0_origin" # default args.task_dir 
    self.dataset = datasets.load_dataset(
        os.path.join(data_dir, self.DATASET_PATH),
        self.DATASET_NAME,
    )  # 로컬 폴더명만 지정해도 load_dataset 가능
    ```
- 허깅페이스에 업로드되어 있지 않은 데이터셋의 경우, 아래와 같이 클래스 변수를 작성합니다.
    
    ```python
    # 예시
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

    #  데이터셋 로드 코드
    data_dir = "/home/data/0_origin" # default args.task_dir 
    base_path = os.path.join(data_dir, self.DATASET_PATH, self.DATASET_NAME)
    data_files = {k: base_path + "/" + v for k, v in self.DATA_FILES.items()}
    self.dataset = datasets.load_dataset(
        self.DATASET_TYPE, data_files=data_files, **self.DOWNLOAD_OPTIONS
    )  # json 파일 경로 직접 지정

    ```
3. 데이터셋 클래스를 생성했으면 모델 입력 형식으로 전처리하는 함수들의 내용을 작성합니다.  <br>
[evaluation/task/template.py](evaluation/task/template.py) 파일의 주석과 [evaluation/task/base.py](evaluation/task/base.py)의 `Task` 클래스 참고하여 작성 
- [ ] n_shot
- [ ] request_type
- [ ] description
- [ ] has_training_docs, has_validation_docs, has_test_docs
- [ ] _process_doc <br>
    참고: [klue_ynat](evaluation/task/klue.py)
    ```python
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
    ```
- [ ] training_docs, validation_docs, test_docs
- [ ] doc_to_text, doc_to_target <br>
참고: [fewshot_context](evaluation/task/base.py), [klue_sts](evaluation/task/klue.py)
    ```python 
    # doc_to_text, doc_to_target으로 fewshot_context 생성 
    labeled_examples = (
        "\n\n".join(
            [
                self.doc_to_text(doc) + self.doc_to_target(doc)
                for doc in fewshotex
            ]
        )
        + "\n\n"
    )

    # klue_sts 예시
    def doc_to_text(self, doc):
        return "문장1: {}\n문장2: {}\n정답:".format(
            general_detokenize(doc["sentence1"]), general_detokenize(doc["sentence2"])
        )

    def doc_to_target(self, doc):
        return " {}".format({0: "다름", 1: "같음"}[doc["labels"]["binary-label"]])

    """
    예시
    doc = {'guid': 'klue-sts-v1_dev_00001',
            'source': 'airbnb-sampled',
            'sentence1': '주요 관광지 모두 걸어서 이동가능합니다.',
            'sentence2': '위치는 피렌체 중심가까지 걸어서 이동 가능합니다.',
            'labels': {'label': 1.4,
            'real-label': 1.428571428571429,
            'binary-label': 0}

    doc_to_text(doc) returns "문장1: 잘때 팝송말고 클래식곡 들어보지 그래?
                              문장2: 당신 국악말고 클래식곡 들으면서 자요 국내 첫 출시
                              정답: "

    doc_to_target(doc) returns "다름"
    """
    ```
- [ ] construct_requests<br>
  request_type에 따른 request 생성 함수를 사용합니다. 
  - loglikelihood
    - 모델이 추론한 값이 정답 선택지와 명확하게 일치하는지 확인하여 평가
    - 모델로 하여금 선택지들의 로그 확률을 계산하도록 함
    - 참고: [klue_ynat](evaluation/task/klue.py), [klue_sts](evaluation/task/klue.py)
    ```python
    def construct_requests(self, doc, ctx):
        ll_negative, _ = rf.loglikelihood(ctx, " 다름")
        ll_positive, _ = rf.loglikelihood(ctx, " 같음")
        return ll_negative, ll_positive
    ```
  - greedy_until
    - 모델이 생성한 문장이 정답 문장과 맥락상 유사한지를 평가
    - 모델로 하여금 프롬프트에 따른 문장을 생성하도록 함 
    - 참고: [korquad](evaluation/task/korquad.py)
    ```python
    def construct_requests(self, doc, ctx):
        continuation = rf.greedy_until(ctx, {"until": ["\n", "##"]})
        return continuation
    ```
- [ ] process_results, aggregation<br>
    참고: [get_result](evaluation/eval/evaluator.py)
    ```python
    # 각 response에 대해 평가 지표 계산 및 기록
        for (task_name, doc_id), response in response_dict.items():
            response = list(set(response))
            response.sort(key=lambda x: x[0])
            response = [x[1] for x in response]

            task = self.task_dict[task_name]
            doc = docs[(task_name, doc_id)]

            # 결과 계산
            metrics = task.process_results(doc, response)
            for metric, value in metrics.items():
                vals[(task_name, metric)].append(value)

        # 결과 집계
        for (task_name, metric), items in vals.items():
            task = self.task_dict[task_name]
            metric_dict[task_name][metric] = round(task.aggregation()[metric](items), 4)
    ```
- [ ] higher_is_better


### STEP 3: 데이터셋 정보 추가 
[api/config/dataset.py](api/config/dataset.py) 파일에서 아래의 내용을 추가하세요.
- get_eval_datasets(): 추가한 데이터셋이 평가하는 능력 범위, 평가 지표를 작성합니다. 
    ```python
    def get_eval_datasets():
        EVAL_DATASETS = {
            "klue_ynat": {
                "explain": "뉴스 기사의 주제를 분류하는 능력 평가",
                "metrics": ["acc, macro_f1"],
            },
            ...
            "new_dataset": {
                "explain": "새로운 데이터셋 설명",
                "metrics": ["새로운 평가지표"],
            }
        }
        return EVAL_DATASETS
    ```
- EVAL_DATASET_METRIC: 리더보드에 표시할 주요 평가지표 1개를 지정합니다. 
    ```python
    EVAL_DATASET_METRIC = {
        "klue_nli": "acc",
        ...
        "new_dataset": "새로운 평가지표",
    }
    ```

### 데이터셋 추가를 완료하였습니다!