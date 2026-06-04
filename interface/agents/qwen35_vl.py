"""Qwen 3.5 vision-language agent via Hugging Face Transformers."""

from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass, field
from typing import Any, List, Union

from PIL import Image

from interface.agents.runner_messages import ContentPart, parse_runner_content

logger = logging.getLogger(__name__)

# Qwen3.5 dense checkpoints are native vision-language models (no separate "-VL" suffix).
DEFAULT_QWEN35_VL_MODEL = "Qwen/Qwen3.5-4B"
_AGENT_NAME = "Qwen agent"


def _parts_to_qwen_blocks(parts: List[ContentPart]) -> Union[str, List[dict]]:
    blocks: List[dict] = []
    for part in parts:
        if part.kind == "text":
            blocks.append({"type": "text", "text": part.text})
        elif part.image is not None:
            raw = base64.b64decode(part.image.data_b64)
            blocks.append({"type": "image", "image": Image.open(io.BytesIO(raw))})
    if not blocks:
        return ""
    if len(blocks) == 1 and blocks[0].get("type") == "text":
        return str(blocks[0].get("text", ""))
    return blocks


def _to_qwen_content(content: object) -> Union[str, List[dict]]:
    parsed = parse_runner_content(content)
    if isinstance(parsed, str):
        return parsed
    return _parts_to_qwen_blocks(parsed)


def _to_qwen_messages(messages: List[dict]) -> List[dict]:
    out: List[dict] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "system":
            out.append({"role": "system", "content": str(content)})
        elif role == "user":
            out.append({"role": "user", "content": _to_qwen_content(content)})
        elif role == "assistant":
            out.append(
                {
                    "role": "assistant",
                    "content": content if isinstance(content, str) else str(content),
                }
            )
        else:
            raise ValueError(f"Unsupported message role for {_AGENT_NAME}: {role!r}")
    return out


@dataclass
class Qwen35VLConfig:
    model: str = DEFAULT_QWEN35_VL_MODEL
    temperature: float = 0.0
    max_new_tokens: int = 1024
    device_map: str = "auto"


@dataclass
class Qwen35VLAgent:
    """Qwen 3.5 vision-language model via Hugging Face Transformers (local, no API credits)."""

    config: Qwen35VLConfig = field(default_factory=Qwen35VLConfig)
    processor: Any = None
    model: Any = None

    def __post_init__(self) -> None:
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.config.model)
        if self.model is None:
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                self.config.model,
                device_map=self.config.device_map,
            )

    def __call__(self, messages: List[dict]) -> str:
        qwen_messages = _to_qwen_messages(messages)
        inputs = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        t0 = time.perf_counter()
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.temperature > 0,
        )
        prompt_len = inputs["input_ids"].shape[1]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Qwen35VL generate: model=%s elapsed=%.2fs prompt_tokens=%d",
                self.config.model,
                time.perf_counter() - t0,
                prompt_len,
            )

        new_tokens = generated[0][prompt_len:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()
