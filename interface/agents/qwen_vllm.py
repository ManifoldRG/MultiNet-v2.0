"""Qwen chat agent via vLLM offline inference."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List

from interface.agents.runner_messages import ContentPart, parse_runner_content

DEFAULT_QWEN_VLLM_MODEL = "Qwen/Qwen3.6-27B"
_AGENT_NAME = "Qwen vLLM agent"


def _part_to_openai_block(part: ContentPart) -> dict[str, object]:
    if part.kind == "text":
        return {"type": "text", "text": part.text}
    if part.image is not None:
        url = f"data:{part.image.media_type};base64,{part.image.data_b64}"
        return {"type": "image_url", "image_url": {"url": url}}
    return {"type": "text", "text": ""}


def _to_openai_content(content: object) -> object:
    parsed = parse_runner_content(content)
    if isinstance(parsed, str):
        return parsed
    blocks = [_part_to_openai_block(part) for part in parsed]
    if not blocks:
        return ""
    if len(blocks) == 1 and blocks[0].get("type") == "text":
        return str(blocks[0].get("text", ""))
    return blocks


def _to_openai_messages(messages: List[dict]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for message in messages:
        role = message.get("role")
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"Unsupported message role for {_AGENT_NAME}: {role!r}")
        content = message.get("content", "")
        if role == "system":
            content = str(content)
        elif role == "assistant":
            content = content if isinstance(content, str) else str(content)
        else:
            content = _to_openai_content(content)
        out.append({"role": role, "content": content})
    return out


def _count_ids(value: object) -> int | None:
    if value is None:
        return None
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return None


def _usage_from_output(output: object, completion: object) -> dict[str, int] | None:
    prompt_tokens = _count_ids(getattr(output, "prompt_token_ids", None))
    completion_tokens = _count_ids(getattr(completion, "token_ids", None))
    if prompt_tokens is None and hasattr(output, "num_prompt_tokens"):
        prompt_tokens = int(getattr(output, "num_prompt_tokens"))
    if completion_tokens is None and hasattr(completion, "num_generated_tokens"):
        completion_tokens = int(getattr(completion, "num_generated_tokens"))
    if prompt_tokens is None and completion_tokens is None:
        return None
    prompt_tokens = int(prompt_tokens or 0)
    completion_tokens = int(completion_tokens or 0)
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


@dataclass
class QwenVLLMConfig:
    model: str = DEFAULT_QWEN_VLLM_MODEL
    temperature: float = 0.0
    max_tokens: int = 4096
    max_model_len: int | None = 8192
    gpu_memory_utilization: float | None = 0.88
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    quantization: str | None = None
    trust_remote_code: bool = False
    enforce_eager: bool = True
    max_num_seqs: int | None = None
    enable_prefix_caching: bool = True
    enable_thinking: bool = False
    seed: int | None = None
    download_dir: str | None = None
    local_files_only: bool = False
    use_tqdm: bool = False
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    sampling_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class QwenVLLMAgent:
    """Qwen via in-process vLLM offline inference.

    This keeps the pipeline's synchronous ``messages -> str`` contract while
    using vLLM's scheduler and kernels inside the worker process. Run only one
    such agent per GPU unless a separate serving process owns the engine.
    """

    config: QwenVLLMConfig = field(default_factory=QwenVLLMConfig)
    llm: Any = None
    last_usage: dict[str, int] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.config.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        if self.llm is None:
            try:
                from vllm import LLM
            except ImportError as exc:
                raise ImportError(
                    "QwenVLLMAgent requires vLLM. Install the optional vLLM environment "
                    "or run deploy/setup_qwen_vllm_vm.sh on a GPU worker."
                ) from exc
            self.llm = LLM(**self._engine_kwargs())

    def reset_usage(self) -> None:
        self.last_usage = None

    def _engine_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "trust_remote_code": self.config.trust_remote_code,
            "dtype": self.config.dtype,
            "enforce_eager": self.config.enforce_eager,
            "enable_prefix_caching": self.config.enable_prefix_caching,
        }
        optional = {
            "max_model_len": self.config.max_model_len,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "quantization": self.config.quantization,
            "max_num_seqs": self.config.max_num_seqs,
            "seed": self.config.seed,
            "download_dir": self.config.download_dir,
        }
        kwargs.update({key: value for key, value in optional.items() if value is not None})
        kwargs.update(self.config.engine_kwargs)
        return kwargs

    def _sampling_params(self):
        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise ImportError("QwenVLLMAgent requires vLLM SamplingParams.") from exc
        kwargs = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.temperature <= 0:
            kwargs["temperature"] = 0.0
        kwargs.update(self.config.sampling_kwargs)
        return SamplingParams(**kwargs)

    def __call__(self, messages: List[dict]) -> str:
        qwen_messages = _to_openai_messages(messages)
        sampling_params = self._sampling_params()
        chat_kwargs = {
            "sampling_params": sampling_params,
            "use_tqdm": self.config.use_tqdm,
        }
        if self.config.enable_thinking is not None:
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": self.config.enable_thinking}
        try:
            outputs = self.llm.chat(qwen_messages, **chat_kwargs)
        except TypeError as exc:
            if "chat_template_kwargs" not in str(exc):
                raise
            chat_kwargs.pop("chat_template_kwargs", None)
            outputs = self.llm.chat(qwen_messages, **chat_kwargs)
        output = outputs[0]
        completion = output.outputs[0]
        text = str(getattr(completion, "text", "")).strip()
        self.last_usage = _usage_from_output(output, completion)
        return text
