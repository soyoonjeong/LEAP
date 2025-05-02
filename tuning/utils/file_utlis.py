import json
from typing import Any


def read_json_file(file_path: str) -> Any:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        data = json.load(file)
    return data


def save_json(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
