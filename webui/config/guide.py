EVAL_USAGE_GUIDE = {
    "Model": """**Tip**: Hugging Face Hub에서 제공하며, vLLM이 지원하는 LLM을 사용할 수 있습니다.

- **Baseline**: 베이스라인 모델 목록을 확인할 수 있습니다.
- **Finetuned**: 해당 시스템에서 튜닝한 모델 목록을 확인할 수 있습니다.
- **Custom**: 선택 목록에 없는 모델을 사용하려면 **Custom Model** 텍스트 박스에 원하는 모델 이름을 입력하세요.
    - 예시: `google/gemma-1.1-2b-it`
    - Custom 모델 로드 실패 시 다음 항목을 확인하세요:
        1. Hugging Face Hub 토큰이 유효한지 확인하세요.
        2. 해당 토큰을 발급받은 계정으로 해당 모델에 대한 접근 권한이 있는지 확인하세요.
        3. 모델이 vLLM에서 지원되는지 확인하세요. 원하는 모델이 지원 목록에 없을 경우, vLLM을 업그레이드 해보세요.
        [지원 모델 목록 확인하기](https://docs.vllm.ai/en/v0.6.2/models/supported_models.html) """,
    "Dataset": """**Tip**: 모델 성능 평가를 위한 1개 이상의 데이터셋을 선택합니다. 
- **Evaluation Dataset** 탭에서 자세한 데이터셋 정보를 확인할 수 있습니다.""",
    "Device": """
**Tip**: 사용 중이지 않은 GPU를 우선적으로 선택하세요. 
- **사용 가능한 GPU의 개수, 종류, 및 현재 상태는 Device 체크박스와 그 아래 상태 표시 바를 통해 확인할 수 있습니다.** 
- Tensor Parallel Size와 동일한 개수의 GPU를 선택해야 최적의 성능을 낼 수 있습니다.
- 최신 GPU는 더 큰 메모리 용량과 높은 연산 성능을 제공하므로, 고사양 GPU를 우선적으로 선택하는 것이 효율적입니다.""",
    "Tensor Parallel Size": """
**Tip**: tensor_parallel_size = **GPU 수**로 설정하는 것이 좋습니다.
- infer backend가 vllm인 경우만 2 이상 설정 가능합니다. 
- vllm 특성상 16의 약수 (1, 2, 4, 8, 16)로 설정해야 합니다.""",
    "GPU Memory Utilization": """**Tip**: 권장 값은 **0.8~0.9** 사이입니다.
- 모델 실행에 사용할 GPU 메모리 비율을 설정하며, 0에서 1 사이 값으로 지정할 수 있습니다.
- 다른 프로세스에서 이미 GPU 메모리를 사용 중이라면, 남은 메모리 비율에 따라 실제 활용 가능한 비율이 줄어들 수 있습니다.
- 사용 중인 프로세스를 고려하여 GPU 메모리 활용률을 조정하고, **너무 높은 값을 설정하지 않아 OOM(Out of Memory) 오류를 방지하세요.**""",
    "Max Model Length": """**Tip**: 현재 설정 상, 모델 구성에서 자동으로 파생된 값을 **최댓값**으로 사용하고 있으므로 최댓값을 사용하는 것이 좋습니다.  
- 사용할 모델이 지원하는 최대 토큰 길이를 확인한 뒤, 입력 시퀀스 길이를 고려하여 설정합니다. 
- 예: GPT-3 기반 모델은 보통 2048~4096 토큰까지 지원. 
- 작업에 필요한 맥락의 크기를 고려해 적절히 설정하면 메모리 사용량을 줄이고 효율성을 높일 수 있습니다.""",
    "Max New Tokens": """**Tip**: 생성할 토큰 수를 작업의 요구사항에 맞게 설정하세요.
- 필요 이상으로 토큰을 생성하지 않도록 설정하면 시간과 자원을 절약할 수 있습니다.
- 추천 값
    - 짧은 답변: 20~50
    - 중간 길이 답변: 100~200
    - 긴 생성 작업: 500~1000 """,
    "Num Fewshot": """**Tip**: Few-shot 학습 예제 수는 메모리 및 성능과 균형을 맞춥니다. 
- 작은 GPU 메모리: 1~2개
- 큰 GPU 메모리 및 높은 정확도 필요: 4~5개""",
    "Top K": """**Tip**: 생성 다양성을 조절하며, 낮은 값을 사용할수록 보수적으로 생성합니다.
- **권장 값: 40~100 (일반적으로 50이 적절)**
- 작은 작업(단답형 질문)은 낮은 값(10~20), 창의적인 작업은 높은 값(100 이상)""",
    "Top P": """Top P (Nucleus Sampling)
**Tip**: 확률 기반으로 토큰 선택 범위를 제한합니다.
- **권장 값: 0.8~0.95 (안정적이며 다양성 유지)**
- 생성이 불안정할 경우 값을 낮추고, 더 창의적이고 자유로운 생성을 원할 경우 값을 높임.""",
    "Temperature": """**Tip**: 모델의 출력을 더 무작위로 만들거나, 더 결정론적으로 만듭니다.
- 결정론적 생성: 0.1~0.3
- 균형 잡힌 설정: 0.7
- 창의적 생성: 1.0 이상""",
}

TUNING_USAGE_GUIDE = {
    "Model": """- 지원 모델 목록 (8개)
    - Qwen/Qwen2.5-1.5B
    - Qwen/Qwen2.5-3B
    - Qwen/Qwen2.5-7B
    - Qwen/Qwen2.5-14B
    - meta-llama/Llama-3.1-8B
    - google/gemma-1.1-2b-it
    - google/gemma-1.1-7b-it
    - google/gemma-2-2b
    - google/gemma-2-9b""",
    "Instruction Data Path": """- 학습 데이터셋 경로를 지정합니다. 
- **Tuning Dataset** 탭에서 데이터셋 예시를 확인할 수 있습니다. 
- **Tuning Dataset** 탭에서 데이터셋을 추가할 수 있습니다. 
  1. 추가 데이터셋의 이름을 지정합니다.
  2. {\"instruction\":\"\", \"output\":\"\"} 형식의 json 파일 업로드하면 지정된 이름으로 데이터셋이 저장됩니다.  
  3. 추가된 데이터셋은 **Tuning** 탭의 instruction data path에서 확인할 수 있습니다.""",
    "Device": """**Tip**: 사용 중이지 않은 GPU를 우선적으로 선택하세요. 
- **사용 가능한 GPU의 개수, 종류, 및 현재 상태는 Device 체크박스와 그 아래 상태 표시 바를 통해 확인할 수 있습니다.** 
- Tensor Parallel Size와 동일한 개수의 GPU를 선택해야 최적의 성능을 낼 수 있습니다.
- 최신 GPU는 더 큰 메모리 용량과 높은 연산 성능을 제공하므로, 고사양 GPU를 우선적으로 선택하는 것이 효율적입니다.""",
    "Chat Template": """- LLM의 입력 형식을 정의합니다. 
- 현재 모델의 타입마다 Chat Template이 자동 입력됩니다. 
- 예시
    - **Gemma2**: `<bos><start_of_turn>user \n {instruction} \n <end_of_turn><start_of_turn>model`
    - **Llama3.1**: `<|begin_of_text|><|start_header_id|>system<|end_header_id|> {system_prompt} <|eot_id|><|start_header_id|>user<|end_header_id|> {instruction} <|eot_id|><|start_header_id|>assistant<|end_header_id|>`
    - **Qwen2.5**: `<|im_start|> system\n {system_prompt} <|im_end|>\n<|im_start|> user\n {instruction} <|im_end|>\n<|im_start|>assistant`""",
    "System Prompt": """- 모델 초기화를 위한 기본 지침(prompt)를 설정합니다. 
- 이 프롬프트는 모델의 역할, 말투, 행동 방식을 정의합니다. 
- 예시
    - `You are a helpful assistant.`""",
    "Optimization Configurations": """### Learning Rate
**Tip**: 일반적으로 1e-5 ~ 5e-5 사이를 권장합니다. 
- 모델 학습 시 가중치를 조정하는 비율입니다. 
- 너무 크면 학습이 불안정해지고, 너무 작으면 학습이 느려질 수 있습니다.
### Learning Rate Scheduler
- 학습 속도를 조정하는 방식입니다. 
- 학습 초기, 중기, 말기에 따라 적절한 학습률을 자동으로 설정합니다.
- 주요 옵션:
    - linear: 일정 비율로 학습률 감소
    - cosine: 코사인 곡선을 따라 감소
    - constant: 고정된 학습률 유지
    - constant_with_warmup: 학습 초기에는 워밍업 단계로 학습률을 점진적으로 증가시키고, 이후 고정된 학습률을 유지
### Warmup Ratio
**Tip**: 일반적으로 0.0~0.1 사이를 권장합니다.
- 학습 초기 단계에 학습률을 점진적으로 증가시키는 비율입니다. 
- 모델이 안정적으로 학습을 시작할 수 있도록 합니다. 
### Weight Decay
**Tip**: 일반적으로 0.0~0.1 사이를 권장합니다.
- 과적합을 방지하기 위해서 가중치에 패널티를 부여하는 비율입니다. 
### Gradient Accumulation Steps
- 작은 GPU 메모리를 가진 환경에서 미니 배치를 쌓아, 여러 단계동안 기울기를 누적 후 업데이트를 수행하는 설정입니다. 
- 예시
  - `gradient_accumulation_steps=5` 설정 시, 5번의 기울기 계산 후 업데이트 수행
### Gradient Checkpointing
- `Gradient Checkpointing`에 체크를 하면 메모리를 절약하기 위해 중간 계산 값(예: 활성화 값)을 저장하지 않고 필요 시 다시 계산하겠다는 의미입니다. 
- 중간 값을 저장하지 않기에 큰 모델을 더 작은 GPU 메모리로 학습할 수 있습니다. 
- 중간 값이 필요할 때마다 다시 계산하므로, 연산 시간이 증가할 수 있습니다.""",
    "Tuning Configurations": """### Max Sequence Length
- 모델이 처리할 수 있는 최대 입력 시퀀스 길이입니다. 
- 튜닝시킬 베이스스 모델이 지원하는 최대 토큰 길이를 확인한 뒤, 입력 시퀀스 길이를 고려하여 설정합니다. 
### Max Gradient Norm
**Tip**: 일반적으로 1.0~5.0 사이를 권장합니다.
- Gradient Explosion을 방지하기 위해 기울기의 최대 값을 제한합니다. 
### Epochs
**Tip**: 일반적으로 3~10 사이를 권장합니다.(데이터 크기 및 목적에 따라 조정정)
- 데이터셋 전체를 몇 번 반복하여 학습할 것인지 설정합니다. 
### Per Device Batch Size
**Tip**: 일반적으로 1을 권장합니다.
- 각 GPU 에서 처리할 데이터 배치 크기입니다. 
- GPU 메모리 용량에 따라 조정합니다. 
### Compute Type
**Tip**: 일반적으로 bf16을 권장합니다.
- 모델 학습에 사용되는 연산 유형을 설정합니다. 
- 주요 옵션:
    - fp16: 절반 정밀도 연산을 사용하여 메모리 사용량을 줄입니다.
    - bf16: 더 높은 정확도를 제공하는 반정밀도 연산을 사용합니다.
    - fp32: 전체 정밀도 연산을 사용합니다.
### Deepspeed Config
- 대규모 모델 학습을 위한 최적화 프레임워크 DeepSpeed의 설정입니다. 
- 학습 속도와 효율성을 높입니다. 
### Use Flash Attention 
- 속도와 메모리 효율성을 개선하기 위해 Flash Attention을 활성화합니다. """,
    "LoRA Configurations": """### Use LoRA
- LoRA(Low-Rank Adaptation) 기법을 활성화합니다.
- 대규모 모델을 효율적으로 미세 조정하기 위한 방법입니다. 
- 저차원 행렬로 모델의 특정 가중치 계층을 업데이트하여 메모리와 계산 비용을 절약합니다.
### Use DoRA
- DoRA(Dynamic Offset Rank Adaptation) 기법을 활성화합니다. 
- 동적 오프셋을 활용해 적응형으로 학습을 진행하여 더 나은 성능을 제공합니다.
### Use RSLoRA
- RSLoRA(Random Sparse Low-Rank Adaptation) 기법을 활성화합니다. 
- 무작위로 선택된 가중치에 대해 희소성과 저차원 특성을 결합하여 메모리 효율성을 더욱 높입니다.
### LoRA R
**Tip**: 일반적으로 LoRA R=16, Alpha=64 조합 또는 LoRA R=64, Alpha=128 조합을 권장합니다.
- LoRA에서 학습 가능한 저차원 표현의 rank입니다.
- 모델 크기에 따라 조정해야 합니다.

### LoRA Alpha
**Tip**: 일반적으로 LoRA R=16, Alpha=64 조합 또는 LoRA R=64, Alpha=128 조합을 권장합니다.
- LoRA에서 학습률을 조정하는 스케일링 하이퍼파라미터입니다.

### LoRA Dropout
- **Tip**: 일반적으로 0.1 ~ 0.3를 권장합니다. 
- 학습 과정에서 일부 노드를 무작위로 비활성화하여 과적합을 방지합니다.""",
}


MEMORY_GUIDE = """
## Inference Memory
총 추론 메모리는 모델 가중치를 저장하는 데 필요한 메모리와 순전파 중 약간의 추가 오버헤드를 고려하여 다음과 같이 계산됩니다.

$$\\text{Total Memory}_{\\text{Inference}} \\approx 1.2 \\times \\text{Model Memory}.$$

## Training Memory
총 학습 메모리는 모델 파라미터 메모리, 옵티마이저 상태 메모리, 활성화 메모리, 그리고 그래디언트 메모리의 합으로 계산됩니다.

$$\\text{Total Memory}_{\\text{Training}} = \\text{Model Memory} + \\text{Optimizer Memory} + \\text{Activation Memory} + \\text{Gradient Memory}$$

각 구성 요소는 데이터 유형(fp32, fp16), 옵티마이저 유형(예: AdamW, 8비트), 활성화 재계산 전략(No/Selective/Full Recomputation) 등 여러 변수에 따라 다릅니다.
아래 계산된 훈련 메모리는 해당 설정을 기반으로 계산하였습니다.

- 데이터 유형: fp32
- 옵티마이저 유형: AdamW
- 활성화 재계산 전략: Full Recomputation
- 배치 사이즈: 1
- 텐서 병렬화 사이즈: 1
<br><br>
출처: [Transformer Math 101](https://blog.eleuther.ai/transformer-math/)
"""
