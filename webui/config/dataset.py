import requests
from .url import API_URL

get_eval_datasets = lambda: requests.get(API_URL + "/list/eval/dataset").json()
get_tuning_datasets = lambda: requests.get(API_URL + "/list/tuning/dataset").json()
DATASET_PATH = {
    "klue": "klue/klue",
    "kobest": "skt/kobest_v1",
}
