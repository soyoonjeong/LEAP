import GPUtil
import collections


def get_gputil_info():
    try:
        gpus = GPUtil.getGPUs()
    except Exception as e:
        print(f"Error getting GPU information: {e}")
        return {}

    if not gpus:
        return {}

    gpu_info = collections.defaultdict(list)
    for gpu in gpus:
        gpu_info["name"].append(gpu.name)
        gpu_info["memory_total"].append(f"{gpu.memoryTotal} MB")
        gpu_info["memory_used"].append(f"{gpu.memoryUsed} MB")
        gpu_info["load"].append(round(gpu.load * 100, 2))  # GPU 사용률
    return {"gpu_status": dict(gpu_info)}
