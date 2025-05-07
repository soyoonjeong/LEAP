# 🚀 LEAP (Language model Evaluation And PEFT)

<br>

**LEAP**는 LLM의 튜닝, 평가, 결과 분석까지 전 과정을 통합적으로 제공하는 시스템입니다.

<br>

https://github.com/user-attachments/assets/a9cba88a-5d42-4f1d-a0c6-01c17c62b52b

https://github.com/user-attachments/assets/63ff2544-6b33-4361-8e91-c941766d48c6


---

## 📦 구성 요소 및 요구 사항

LEAP 시스템은 4개의 주요 서버로 구성되어 있으며, 각 서버의 필수 환경은 다음과 같습니다.

### 🧪 Eval Server (모델 평가 서버)
| Component     | Version  |
|---------------|----------|
| NVIDIA Driver | >= 450   |
| CUDA          | 12.1     |
| Python        | 3.10     |
| vLLM          | 0.6.0    |
| Torch         | 2.4.0    |
| Transformers  | 4.45.1   |
| Datasets      | 3.0.1    |
| Multiprocess  | 0.70.16  |
| FastAPI       | 0.115.5  |

---

### 🧬 Tuning Server (모델 PEFT 서버)
| Component     | Version  |
|---------------|----------|
| NVIDIA Driver | >= 450   |
| CUDA          | 12.1     |
| Python        | 3.10     |
| Torch         | 2.4.0    |
| Transformers  | 4.45.1   |
| Datasets      | 3.0.1    |
| FastAPI       | 0.115.5  |
| DeepSpeed     | 0.15     |
| Flash-Attn    | 2.6.3    |
| Accelerate    | 1.0.1    |

---

### 🧠 Backend Server (작업 관리 서버)
| Component     | Version  |
|---------------|----------|
| Python        | 3.10     |
| Multiprocess  | 0.70.16  |
| FastAPI       | 0.115.5  |
| GPUtil        | 1.4.0    |

---

### 💻 Frontend Server (웹 UI 서버)
| Component           | Version  |
|---------------------|----------|
| Python              | 3.10     |
| Gradio              | 5.5.0    |
| Gradio Leaderboard  | 0.0.13   |
| Gradio Modal        | 0.0.4    |
| Transformers        | 4.45.1   |

---

## 🚀 Getting Started

### ✅ 설치

```bash
git clone https://github.com/surromind/leap.git
cd leap
```

<br>

### 🐚 Shell Script Version (without Docker)

#### 평가 (Evaluation)
- shell script 예시: [`evaluate.sh`](evaluate.sh)
- 인자 설명: [`evaluation/GUIDE.md`](evaluation/GUIDE.md)

#### 튜닝 (Tuning)
- shell script 예시: [`tuning.sh`](tuning.sh)
- 인자 설명: [`tuning/GUIDE.md`](tuning/GUIDE.md)

<br>

### 🐳 UI Version (with Docker)

#### 🔎 사전 체크리스트
- 모델 및 데이터 경로 확인하여 [docker-compose.yml](docker/docker-compose.yml) 파일의 마운트 경로를 수정
  - 모델: `/mnt/seaweedfs/llm_models/`
  - 데이터: `/mnt/seaweedfs/llm_data/`
- **포트 충돌 방지**  
  - [docker-compose.yml](docker/docker-compose.yml), [Dockerfile.eval](docker/eval/Dockerfile.eval), [Dockerfile.tuning](docker/tuning/Dockerfile.tuning), [Dockerfile.frontend](docker/frontend/Dockerfile.frontend), [Dockerfile.backend](docker/backend/Dockerfile.backend) 파일의 포트 번호를 수정해주세요.
- **HuggingFace 토큰 유효성 확인**
  - 허깅페이스 토큰 발급 후 [Dockerfile.eval](docker/eval/Dockerfile.eval) 파일의 허깅페이스 토큰을 수정해주세요.

<br>

#### Docker Compose로 실행

```bash
docker compose -f docker/docker-compose.yml up -d
docker logs leap-frontend
```

<br>

<details>
<summary><strong>Docker Compose 없이 개별 실행</strong></summary>

#### Backend
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

#### Eval
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

#### Tuning
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

#### Frontend
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

<br>

---

## 🧠 모델

### 평가 (Evaluation)

- **지원 조건**
  - HuggingFace에 업로드되었거나 로컬 모델 경로(e.g. `/home/llm_models/`)에 저장된 모델
  - vLLM에서 지원하는 모델 ([지원 목록](https://docs.vllm.ai/en/latest/models/supported_models.html))

- **모델 등록**
  - **GUI**: Model Type → `custom`, 모델 이름 직접 입력 (e.g. `google/gemma-1.1-2b-it`)
  - **Shell Script**: `--model` 인자에 모델명 입력(e.g. `--model google/gemma-1.1-2b-it`, 참고: [evaluate.sh](evaluate.sh))

<br>

> 모델 로드 오류 시 확인 사항:
> 1. HuggingFace 토큰 유효성
> 2. 해당 모델 접근 권한
> 3. vLLM 호환 여부

<br>

### 튜닝 (Tuning)

- **지원 모델 목록**: [`api/config/model.py`](api/config/model.py) 의 `get_tuning_models`

- **모델 등록**
  1. 로컬 모델 경로(e.g. `/home/llm_models/`)에 모델 다운로드
  2. - **GUI**: `get_tuning_models` 함수에 모델명 추가
     - **Shell Script**: `--model` 인자에 모델명 입력

<br>

---

## 📚 데이터셋

### 평가 (Evaluation)

- **지원 목록**: [`api/config/datasets.py`](api/config/dataset.py)의 `get_eval_datasets`
- **데이터셋 추가**: [`evaluation/task/README.md`](evaluation/task/README.md)

<br>

### 튜닝 (Tuning)

- **지원 목록**: `get_tuning_datasets` 참조
- **데이터셋 추가**
  - **GUI**: `Tuning Dataset` 탭에서 업로드
  - **Shell Script**:
    - 경로: `SAVE_DATASET_DIR` ([path.py](api/config/path.py))
    - 형식: JSON (예: `{"instruction": "...", "output": "..."}`)
