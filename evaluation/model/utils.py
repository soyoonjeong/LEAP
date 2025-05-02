import os
import shutil
import collections
from multiprocessing import Process
from huggingface_hub import snapshot_download, login


def chunks(iter, n):
    arr = []
    for x in iter:
        arr.append(x)
        if len(arr) == n:
            yield arr
            arr = []

    if arr:
        yield arr


def group(arr, fn):
    res = collections.defaultdict(list)

    for ob in arr:
        res[fn(ob)].append(ob)

    return list(res.values())


class Reorderer:
    def __init__(self, arr, fn):
        self.size = len(arr)
        arr = list(enumerate(arr))
        arr = group(arr, lambda x: fn(x[1]))
        arr = [([y[0] for y in x], x[0][1]) for x in arr]
        arr.sort(key=lambda x: fn(x[1]))

        self.arr = arr

    def get_reordered(self):
        return [x[1] for x in self.arr]

    def get_original(self, newarr):
        res = [None] * self.size
        cov = [False] * self.size

        for (inds, _), v in zip(self.arr, newarr):
            for ind in inds:
                res[ind] = v
                cov[ind] = True

        assert all(cov)

        return res


def hf_down(repo, save_path):
    snapshot_download(repo_id=repo, local_dir=save_path, local_dir_use_symlinks=False)


def download_model(model_dir, model_name):
    path = os.path.join(model_dir, model_name)
    try:
        # login(token=os.getenv("HF_TOKEN", HF_TOKEN))
        login(token=os.getenv("HF_TOKEN", "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"))
        processes = [
            Process(
                target=hf_down,
                args=(
                    model_name,
                    path,
                ),
            ),
        ]

        for process in processes:
            process.start()

        for process in processes:
            process.join()
    except Exception as e:
        print(e)
        if os.path.exists(path):
            shutil.rmtree(os.path.join(model_dir, model_name))
