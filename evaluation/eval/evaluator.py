import os
import time
import json
import random
import datetime
import itertools
import collections
from typing import Any, Dict, Tuple, List

from task import get_task_dict
from task.base import Request
from hparams import get_eval_args
from model import get_engine
from .utils import save_results_to_json, renew_leaderboard


class Evaluator:
    def __init__(self, task_id: str, args: Dict[str, Any], logger) -> None:
        """
        Evaluator 초기화

        Args:
            task_id (str): scheduler에서 받아온 task_id
            args (Dict[str, Any]): 사용자가 입력한 파라미터
            logger (logging.Logger): 평가 작업 수행 로그 기록
        """
        self.task_id = task_id
        self.args = args
        self.write_out = args.pop("write_out")

        # arguments 초기화 및 사용자 입력 적용
        self.model_args, self.eval_args, self.generation_args = get_eval_args(args)

        # 기록용
        self.write_out_info = {}
        self.task_dict = None
        self.logger = logger

    def preprocess_data(
        self,
    ) -> Tuple[Dict[Tuple[str, int], Dict], Dict[str, Tuple[str, Request]]]:
        """
        데이터셋 로드 및 전처리
        Returns:
            Dict[Tuple[str, int], Dict]: 원본 데이터셋(docs)
            Dict[str, Tuple[str, Request]]: 모델 입력 형식에 맞게 변환한 데이터셋(request_dict)
        """
        # 데이터셋 로드
        self.logger.info(f"데이터셋 {self.eval_args.task} 로드 시작")
        self.task_dict = get_task_dict(self.eval_args.task_dir, self.eval_args.task)
        self.logger.info(f"데이터셋 {self.eval_args.task} 로드 완료")

        # 필요한 딕셔너리 및 리스트 초기화
        docs = {}
        request_dict = collections.defaultdict(
            list
        )  # request_type -> [(request_id, request), ...]

        # 데이터셋 전처리
        for task_name, task in self.task_dict.items():
            # validation doc 준비
            if task.has_test_docs():
                task_doc_func = task.test_docs()
                task.set = "test"
            elif task.has_validation_docs():
                task_doc_func = task.validation_docs()
                task.set = "val"
            else:
                raise RuntimeError("Validation 또는 Test 데이터셋이 없습니다.")

            task_docs = list(task_doc_func)
            rnd = random.Random()
            rnd.seed(42)
            rnd.shuffle(task_docs)
            self.logger.info(f"데이터셋: {task_name} ({len(task_docs)})")

            prompt_details = []  # write_out
            # doc마다 request 생성
            for doc_id, doc in enumerate(
                itertools.islice(task_docs, 0, self.eval_args.max_samples)
            ):
                ctx = task.fewshot_context(
                    doc=doc,
                    num_fewshot=(
                        task.n_shot()
                        if self.eval_args.num_fewshot == -1
                        else self.eval_args.num_fewshot
                    ),
                    rnd=rnd,
                    description=task.description(),
                )
                requests = task.construct_requests(doc, ctx)  # request 생성
                prompt_details.append({"doc_id": doc_id})
                docs[(task_name, doc_id)] = doc

                # 첫 번째 문서만 request 출력
                if doc_id < 1:
                    self.logger.info(
                        f"Task: {task_name}; document {doc_id}; context prompt (starting on next line):\n{ctx}\n(end of prompt on previous line)"
                    )
                    self.logger.info(f"Requests: {requests}")

                if not isinstance(requests, (list, tuple)):
                    requests = [requests]

                # request_id 생성 및 request_dict에 저장
                for i, request in enumerate(requests):
                    request_id = str(task_name) + "/" + str(doc_id) + "/" + str(i)
                    request_dict[task_name].append((request_id, request))

                    prompt_details[-1][f"prompt_{i}"] = "".join(
                        (map(lambda x: "".join(x), request.args))
                    )  # write out
            self.write_out_info[task_name] = prompt_details
        return docs, request_dict

    def inference(
        self,
        request_dict: List[Tuple[str, Request]],
    ) -> Tuple[Dict[str, str], Dict[Tuple[str, int], List[Tuple[int, Any]]]]:
        """
        모델 로드 및 추론 수행
        Args:
            request_list (List[Tuple[str, Any]]): request_id와 request 쌍으로 이루어진 리스트
        Returns:
            Dict[str, str]: 각 task_name(str)에 대한 추론 시간(time_dict)
            Dict[Tuple[str, int], List[Tuple[int, Any]]]: 각 task_name(str), doc_id(int)에 대한 추론 결과 (response_dict)
            Dict[str, Dict[str, str]]: 각 task_name(str)에 따른 에러 메시지(error_dict)
        """
        # 모델 로드
        self.logger.info(f"모델 {self.model_args.model} 로드 시작")
        engine = get_engine(
            self.task_id,
            self.model_args,
            self.eval_args,
            self.generation_args,
            self.logger,
        )
        self.logger.info(f"모델 {self.model_args.model} 로드 완료")

        # 추론 수행 및 수행 시간 측정
        time_dict = {}
        response_dict = {}
        error_dict = {}
        for task_name, requests in request_dict.items():
            reqtype = self.task_dict[
                task_name
            ].request_type()  # loglikelihood or greedy_until

            self.logger.info(f"추론 시작 ({task_name})")
            start = time.time()
            return_dict = getattr(engine, reqtype)(requests)  # 추론 수행
            end = time.time()
            times = str(datetime.timedelta(seconds=end - start))
            time_taken = times.split(".")[0]  # 시간 측정

            if "error" not in return_dict.keys():
                # 추론 성공
                response_dict.update(return_dict)
                time_dict[task_name] = time_taken
                self.logger.info(f"추론 완료 ({task_name}: {time_taken})")
            else:
                # 추론 실패
                error_dict[task_name] = return_dict
                self.logger.error(f"추론 실패 ({task_name})")

        return time_dict, response_dict, error_dict

    def get_result(
        self,
        docs: Dict[Tuple[str, int], Dict],
        time_dict: Dict[str, str],
        response_dict: Dict[Tuple[str, int], List[Tuple[int, Any]]],
    ) -> Dict[Tuple[str, str], Any]:
        """
        추론 결과 지표 계산
        Args:
            docs (List[Dict[str, Any]]): 원본 데이터셋
            time_dict (Dict[str, str]): 각 task_name에 대한 추론 시간
            response_dict (Dict[Tuple[str, int], List[Tuple[int, Any]]]): 각 task_name(str), doc_id(int)에 대한 추론 결과
        Returns:
            Dict[Tuple[str, str], Any]: 각 task_name, metric에 대한 결과값(metric_dict)
        """

        vals = collections.defaultdict(list)
        metric_dict = collections.defaultdict(dict)

        # 각 response에 대해 평가 지표 계산 및 기록
        for (task_name, doc_id), response in response_dict.items():
            response = list(set(response))
            response.sort(key=lambda x: x[0])
            response = [x[1] for x in response]

            task = self.task_dict[task_name]
            doc = docs[(task_name, doc_id)]

            # write out
            for i, logprob in enumerate(response):
                self.write_out_info[task_name][doc_id][f"logit_{i}"] = logprob
            self.write_out_info[task_name][doc_id]["truth"] = task.doc_to_target(doc)

            # 결과 계산
            metrics = task.process_results(doc, response)
            for metric, value in metrics.items():
                vals[(task_name, metric)].append(value)
                self.write_out_info[task_name][doc_id][metric] = str(value)

        # 결과 집계
        for task_name, time_taken in time_dict.items():
            metric_dict[task_name]["time"] = time_taken

        for (task_name, metric), items in vals.items():
            task = self.task_dict[task_name]
            metric_dict[task_name][metric] = round(task.aggregation()[metric](items), 4)

        return metric_dict

    def eval(self) -> None:
        """
        평가 수행 함수
        """
        # STEP 1: 데이터셋 준비
        docs, request_list = self.preprocess_data()
        # STEP 2: 추론 수행
        time_dict, response_dict, error_dict = self.inference(request_list)
        # STEP 3: 결과 계산
        metric_dict = self.get_result(docs, time_dict, response_dict)
        metric_dict.update(error_dict)

        result_dict = {}
        result_dict["configs"] = self.args
        result_dict["results"] = metric_dict
        self.logger.info(json.dumps(result_dict, indent=2))

        # STEP 4: 결과 저장
        save_dir = os.path.join(self.eval_args.save_dir, self.task_id)
        save_results_to_json(self.logger, "result", result_dict, save_dir)
        for task_name, _ in self.task_dict.items():
            if task_name not in error_dict.keys():
                save_results_to_json(
                    self.logger,
                    task_name,
                    self.write_out_info[task_name],
                    save_dir,
                )

        # 리더보드 갱신
        if self.write_out:
            renew_leaderboard(self.logger, self.args["model"], result_dict["results"])

        return metric_dict
