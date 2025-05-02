from typing import Any, Dict, Optional, Tuple

import transformers

from .model_args import ModelArguments
from .evaluation_args import EvaluationArguments
from .generation_args import GenerationArguments


_EVAL_ARGS = [ModelArguments, EvaluationArguments, GenerationArguments]
_EVAL_CLS = Tuple[ModelArguments, EvaluationArguments, GenerationArguments]


def get_eval_args(args: Optional[Dict[str, Any]] = None) -> _EVAL_CLS:
    """
    Retrieves evaluation arguments by parsing the provided arguments.

    Args:
        args (Optional[Dict[str, Any]]): A dictionary of arguments to parse.

    Returns:
        _EVAL_CLS: A tuple containing model arguments, data arguments, evaluation arguments, and generation arguments.

    Raises:
        ValueError: If there are unknown or deprecated arguments that are not recognized by the HfArgumentParser.
    """
    parser = transformers.HfArgumentParser(_EVAL_ARGS)
    if args is not None:
        return parser.parse_dict(args)

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    if unknown_args:
        print(parser.format_help())
        print(
            "Got unknown args, potentially deprecated arguments: {}".format(
                unknown_args
            )
        )
        raise ValueError(
            "Some specified arguments are not used by the HfArgumentParser: {}".format(
                unknown_args
            )
        )

    model_args, eval_args, generation_args = (*parsed_args,)

    transformers.set_seed(eval_args.seed)

    return model_args, eval_args, generation_args
