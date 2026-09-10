"""
OpenAI GPT vision adapter for MultiNet-v2.0.

Uses OpenAI's Chat Completions-compatible endpoint with GPT-5.x vision models.

Usage:
    # Set OPENAI_API_KEY before starting Python.
    adapter = GPTAstraAdapter(model="gpt-5.4")
    output = adapter.predict(model_input)
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

import numpy as np
from PIL import Image

try:
    from ..model_interface import ModelInterface, ModelInput, ModelOutput
except ImportError:
    from model_interface import ModelInterface, ModelInput, ModelOutput


class GPTAstraAdapter(ModelInterface):
    """Model adapter for GPT-5.x vision models served by OpenAI's API."""

    def __init__(
        self,
        model: str = "gpt-5.4",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        max_tokens: int = 256,
        reasoning_effort: str | None = "low",
        min_image_size: int = 1024,
        max_prior_images: int = 2,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.min_image_size = min_image_size
        self.max_prior_images = max_prior_images

    @property
    def model_name(self) -> str:
        return f"gpt_astra_{self.model}"

    def predict(self, input: ModelInput) -> ModelOutput:
        """Send the maze observation to GPT and return its next action."""
        if not self.api_key:
            return ModelOutput(
                action=6,
                confidence=0.0,
                reasoning="API error: OPENAI_API_KEY is not set.",
                raw_output=None,
            )

        try:
            raw_output = self._predict_once(input)
            action, confidence, reasoning = self._parse_response(
                raw_output, input.action_space
            )
            return ModelOutput(
                action=action,
                confidence=confidence,
                reasoning=reasoning,
                raw_output=raw_output,
            )
        except (
            TimeoutError,
            urllib.error.URLError,
            urllib.error.HTTPError,
            ConnectionError,
            KeyError,
            ValueError,
        ) as exc:
            return ModelOutput(
                action=6,
                confidence=0.0,
                reasoning=f"API error: {self._format_request_error(exc)}",
                raw_output=str(exc),
            )

    def _predict_once(self, input: ModelInput) -> str:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": self._build_prompt(input)}
        ]
        prior_images = list(input.prior_images or [])[-self.max_prior_images:]
        for index, prior_image in enumerate(prior_images, start=1):
            content.append({
                "type": "text",
                "text": f"Previous image {index} of {len(prior_images)} (earlier timestep).",
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": self._to_data_url(
                        prior_image, min_size=max(512, self.min_image_size // 2)
                    )
                },
            })

        content.append({"type": "text", "text": "Current image (choose the action from this image)."})
        content.append({
            "type": "image_url",
            "image_url": {"url": self._to_data_url(input.image)},
        })

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]

    def _build_prompt(self, input: ModelInput) -> str:
        action_lines = "\n".join(
            f"  {action_id}: {action_name}"
            for action_id, action_name in sorted(input.action_space.items())
        )
        additional_context = (
            f"\n\nText memory:\n{input.additional_context}"
            if input.additional_context
            else ""
        )
        return (
            "You are controlling a top-down gridworld maze agent.\n"
            f"Mission: {input.text_prompt}\n"
            f"Step: {input.step_number}/{input.max_steps}\n\n"
            "Visual facts:\n"
            "- The blue triangle is the agent; its point shows its facing direction.\n"
            "- The green square is the goal.\n"
            "- Walls and dark cells block movement.\n"
            "- Earlier images are short-term memory; the final image is the current observation.\n\n"
            "Choose the action that advances a valid route to the goal. Check whether "
            "forward is open and useful; turn when the agent faces the wrong direction.\n\n"
            "Action list:\n"
            f"{action_lines}\n\n"
            "Respond with exactly one action number from 0 to 6 on the first line. "
            "You may add a short reason on a second line."
            f"{additional_context}"
        )

    def _prepare_image(self, image: np.ndarray, min_size: int | None = None) -> Image.Image:
        """Convert an observation to RGB and upscale small maze renders."""
        prepared = Image.fromarray(image).convert("RGB")
        target = min_size or self.min_image_size
        if min(prepared.width, prepared.height) >= target:
            return prepared
        scale = max(1, int(np.ceil(target / min(prepared.width, prepared.height))))
        return prepared.resize(
            (prepared.width * scale, prepared.height * scale), Image.Resampling.NEAREST
        )

    def _to_data_url(self, image: np.ndarray, min_size: int | None = None) -> str:
        buffer = io.BytesIO()
        self._prepare_image(image, min_size).save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def _parse_response(
        self, text: str, action_space: dict[int, str]
    ) -> tuple[int, float | None, str | None]:
        """Parse an action ID or action name from the model's response."""
        valid_actions = set(action_space)
        text = (text or "").strip()

        first_line = text.split("\n", 1)[0].strip()
        first_line_match = re.search(r"\b([0-6])\b", first_line)
        if first_line_match:
            action = int(first_line_match.group(1))
            if action in valid_actions:
                return action, None, text[first_line_match.end():].strip() or None

        for match in re.finditer(r"\b([0-6])\b", text):
            action = int(match.group(1))
            if action in valid_actions:
                return action, None, text

        text_lower = text.lower()
        for action_id, action_name in action_space.items():
            if action_name.lower() in text_lower:
                return action_id, None, text

        return 6, 0.0, f"Could not parse action from: {text[:200]}"

    def _format_request_error(self, error: Exception) -> str:
        """Include OpenAI's HTTP response body when it is available."""
        details = str(error)
        if isinstance(error, urllib.error.HTTPError):
            try:
                body = error.read().decode("utf-8", errors="replace").strip()
            except OSError:
                body = ""
            if body:
                details = f"{details} | body={body}"
        return details
