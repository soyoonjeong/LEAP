import os

API_URL = f"http://{os.getenv('HOST', '0.0.0.0')}:{os.getenv('BACKEND_PORT', 11181)}"
TYPE_URL = {
    "evaluate": f"http://{os.getenv('HOST', '0.0.0.0')}:{os.getenv('EVAL_PORT', 11183)}/evaluate",
    "tuning": f"http://{os.getenv('HOST', '0.0.0.0')}:{os.getenv('TUNING_PORT', 11182)}/tuning",
}
