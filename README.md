# leap
LLM의 튜닝부터 평가, 결과 분석까지 전 과정을 통합적으로 제공하는 시스템

## Requirements
> Eval Server: 모델 평가 서버 <br>
> Tuning Server: 모델 PEFT 서버 <br>
> Backend Server: 작업 관리 서버 <br>
> Frontend Server: 웹 UI 서버 <br>

### Eval Server 
| Component       | Version  |
|-----------------|----------|
| NVIDIA Driver   | >= 450   |
| CUDA            | 12.1     |
| Python          | 3.10     |
| vLLM            | 0.6.0    |
| Torch           | 2.4.0    |
| Transformers    | 4.45.1   |
| Datasets        | 3.0.1    |
| Multiprocess    | 0.70.16  |
| FastAPI         | 0.115.5  |

---

### Tuning Server
| Component       | Version  |
|-----------------|----------|
| NVIDIA Driver   | >= 450   |
| CUDA            | 12.1     |
| Python          | 3.10     |
| Torch           | 2.4.0    |
| Transformers    | 4.45.1   |
| Datasets        | 3.0.1    |
| FastAPI         | 0.115.5  |
| DeepSpeed       | 0.15     |
| Flash-Attn      | 2.6.3    |
| Accelerate      | 1.0.1    |

---

### Backend Server
| Component       | Version  |
|-----------------|----------|
| Python          | 3.10     |
| Multiprocess    | 0.70.16  |
| FastAPI         | 0.115.5  |
| GPUtil          | 1.4.0    |

---

### Frontend Server
| Component              | Version  |
|------------------------|----------|
| Python                | 3.10     |
| Gradio                | 5.5.0    |
| Gradio Leaderboard    | 0.0.13   |
| Gradio Modal          | 0.0.4    |
| Transformers          | 4.45.1   |





## Getting Started

### Installation
```
git clone https://github.com/surromind/leap.git
cd leap
```

### Shell Script Version (without Docker)
#### Evaluate 
[evaluate.sh](evaluate.sh) 파일에 shell script 예시를 작성하였으니 참고하여 실행해주세요. <br>
각 arguments에 대한 정보는 [evaluation/GUIDE.md](evaluation/GUIDE.md) 를 참고해주세요. 

#### Tuning
[tuning.sh](tuning.sh) 파일에 shell script 예시를 작성하였으니 참고하여 실행해주세요. <br>
각 arguments에 대한 정보는 [tuning/GUIDE.md](tuning/GUIDE.md) 를 참고해주세요. 

### UI Version (with Docker)
> ⭐**실행 전 확인 사항**⭐
- 실행 서버에서 **seaweedfs 마운트**가 되어있는지 확인 
    - /mnt/seaweedfs/llm_models 에 모델이 저장되어 있는지, /mnt/seaweedfs/llm_data 에 데이터셋이 저장되어 있는지 확인 
    - seaweedfs 마운트 경로가 다를 경우 [docker-compose.yml](docker/docker-compose.yml) 파일의 마운트 경로를 수정해주세요.
- **사용 가능 포트** 확인
    - [docker-compose.yml](docker/docker-compose.yml), [Dockerfile.eval](docker/eval/Dockerfile.eval), [Dockerfile.tuning](docker/tuning/Dockerfile.tuning), [Dockerfile.frontend](docker/frontend/Dockerfile.frontend), [Dockerfile.backend](docker/backend/Dockerfile.backend) 파일의 포트 번호를 수정해주세요.
- **허깅페이스 토큰** 확인
    - 허깅페이스 토큰 발급 후 [Dockerfile.eval](docker/eval/Dockerfile.eval) 파일의 허깅페이스 토큰을 수정해주세요.


<details><summary>Build with Docker Compose</summary>

<br>

```
docker compose -f docker/docker-compose.yml up -d
docker logs leap-frontend
```
</details>

<details><summary>Build without Docker Compose</summary>

<br>

**Backend**
```bash
docker build -t leap-api -f docker/backend/Dockerfile.backend .

docker run -d \
    --gpus all \
    --name leap-backend \
    -p 11188:11188 \
    -v /mnt/seaweedfs/llm_models/:/home/llm_models/ \  
    -v /mnt/seaweedfs/llm_data/:/home/data/ \
    -v $(pwd)/data/:/home/leap/data/ \
    -it leap-api \
    python api/app_api.py
```
**Eval**
```bash
docker build -t leap-eval -f docker/eval/Dockerfile.eval .

docker run -d \
    --gpus all \
    --name leap-eval \
    -p 11189:11189 \
    -v /mnt/seaweedfs/llm_models/:/home/llm_models/ \
    -v /mnt/seaweedfs/llm_data/:/home/data/ \
    -v $(pwd)/../logs/:/home/leap/logs/ \
    -it leap-eval \
    --restart unless-stopped
    /bin/bash -c "source activate vllm && python evaluation/app_eval.py"
```
**Tuning** 
```bash
docker build -t leap-tuning -f docker/tuning/Dockerfile.tuning .

docker run -d \
    --gpus all \
    --name leap-tuning \
    -p 11190:11190 \
    -v /mnt/seaweedfs/llm_models/:/home/llm_models/ \
    -v /mnt/seaweedfs/llm_data/:/home/data/ \
    -v $(pwd)/../logs/:/home/leap/logs/ \
    -it leap-tuning \
    --restart unless-stopped
    /bin/bash -c "source activate vllm && python tuning/app_tuning.py"
```
**Frontend**
```bash
docker build -t leap-gradio -f docker/frontend/Dockerfile.frontend .  

docker run \
    --name leap-frontend \
    -v $(pwd)/logs/:/home/leap/logs/ \
    -p 11191:11191 \
    -it leap-gradio \
    python webui/app_gui.py 
```
</details>

## Model

**평가(evaluation)**
- 지원 모델
    <br>아래의 조건을 모두 만족하는 모델을 지원합니다. 
    - huggingface hub에 업로드되어 있거나 로컬에 저장되어 있는 모델(default: `/home/llm_models/` 경로에 저장되어 있어야 함)
    - vllm 지원 모델 (참고: [vllm/supported_models](https://docs.vllm.ai/en/latest/models/supported_models.html))
- 모델 추가 방법 
    - GUI 사용 시, Model Type으로 "custom" 선택 후 Custom Model 텍스트 박스에 원하는 모델 이름을 입력하세요.
        - 예시: `google/gemma-1.1-2b-it`
        - Custom 모델 로드 실패 시 다음 항목을 확인하세요:
            1. Hugging Face Hub 토큰이 유효한지 확인하세요.
            2. 해당 토큰을 발급받은 계정으로 해당 모델에 대한 접근 권한이 있는지 확인하세요.
            3. 모델이 vLLM에서 지원되는지 확인하세요. 원하는 모델이 지원 목록에 없을 경우, vLLM을 업그레이드 해보세요.
    - Shell Script 사용 시, `--model` 옵션에 원하는 모델 이름을 입력하세요. 
        - 예시: `--model google/gemma-1.1-2b-it`
        - 참고: [evaluate.sh](evaluate.sh)


**학습(tuning)**
- 지원 모델<br>
참고: [api/config/model.py](api/config/model.py)의 `get_tuning_models` 
- 모델 추가 방법 
    - 모델 저장 경로(default: `/home/llm_models/`)에 원하는 모델 다운로드 
    - GUI 사용 시, [api/config/models.py](api/config/models.py) 파일의 `get_tuning_models` 함수에 다운로드받은 모델 이름 추가 
    - Shell Script 사용 시, `--model` 옵션에 원하는 모델 이름을 입력하세요. 
        - 예시: `--model google/gemma-1.1-2b-it`
        - 참고: [tuning.sh](tuning.sh)



## Dataset
**평가(evaluation)**
- 지원 데이터셋 <br>
[api/config/datasets.py](api/config/dataset.py)의 `get_eval_datasets` 참고
- 데이터셋 추가 방법<br>
[evaluation/task/README.md](evaluation/task/README.md) 참고


**학습(tuning)**
- 지원 데이터셋<br>
[api/config/datasets.py](api/config/dataset.py)의 `get_tuning_datasets` 참고 
- 데이터셋 추가 방법
    - GUI 사용 시, `Tuning Dataset` 탭에서 원하는 데이터셋 업로드
    - Shell Script 사용 시, [api/config/path.py](api/config/path.py)의 `SAVE_DATASET_DIR` 경로에 데이터셋 저장 후 `--instruction_data_path` 옵션에 데이터셋 경로 입력 
        - 아래의 조건을 만족하는 데이터셋을 추가해야 합니다. 
        - json 파일 형식
        - {"instruction": "...", "output": "..."} 형식
     

## 📄 License

This repository is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).

You may share this work non-commercially, without modification, and must provide attribution.
