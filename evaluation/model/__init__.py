from .vllm import Vllm
from .vllm_engine import VllmEngine
from .transformers import TransformerEngine
from hparams import (
    ModelArguments,
    GenerationArguments,
    EvaluationArguments,
)

ENGINE_REGISTRY = {
    "vllm": Vllm,
    "vllm_engine": VllmEngine,
    "transformers": TransformerEngine,
}


def get_engine(
    task_id: str,
    model_args: ModelArguments,
    eval_args: EvaluationArguments,
    generation_args: GenerationArguments,
    logger,
):
    infer_backend = model_args.infer_backend
    return ENGINE_REGISTRY[infer_backend](
        task_id, model_args, eval_args, generation_args, logger
    )
