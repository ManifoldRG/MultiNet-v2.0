"""Dedicated OpenAI agent for MultiNet v2.0.

Two interfaces:
- ``generate(messages) -> Reply`` — the primary interface the runner uses.
- ``act(...)`` / ``__call__(...)`` — legacy keyword-argument interface for
  compatibility with existing calls that pass system_prompt/observation
  separately.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, List, Optional

from interface.agents.reply import Reply, detect_token_truncated
from interface.telemetry import normalize_token_usage


@dataclass
class OpenAIConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: float = 60.0
    max_attempts: int = 3
    base_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None


def _clean_str(val: Any) -> str:
    if val is None:
        return ""
    if hasattr(val, "text"):
        return str(val.text)
    if isinstance(val, dict):
        return json.dumps(val) if val else ""
    return str(val)


class OpenAIAgent:
    """Agent for official OpenAI endpoints.

    The pipeline calls ``generate(messages) -> Reply``; the legacy ``act()``
    path is kept for callers that pass structured keyword arguments.
    """

    def __init__(self, config: Optional[OpenAIConfig] = None):
        self.config = config or OpenAIConfig()
        self.last_usage: Optional[dict] = None
        self.last_thinking: Optional[str] = None

    # ------------------------------------------------------------------
    # Primary interface (used by the pipeline runner)
    # ------------------------------------------------------------------
    def generate(self, messages: List[dict]) -> Reply:
        """Send chat messages to OpenAI and return a ``Reply``."""
        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

        body = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        payload = self._post_chat_completions(body, api_key)

        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = str(message.get("content") or "").strip()
        usage = normalize_token_usage(payload.get("usage"))
        if usage is None or int(usage.get("total_tokens", 0) or 0) <= 0:
            raise ValueError(
                "OpenAI response missing positive token telemetry; "
                "usage must include a positive total_tokens value."
            )
        stop_reason = choice.get("finish_reason")

        self.last_usage = usage

        return Reply(
            text=text,
            usage=usage,
            thinking=None,
            stop_reason=stop_reason,
            token_truncated=detect_token_truncated(
                stop_reason, usage, self.config.max_tokens
            ),
        )

    # ------------------------------------------------------------------
    # Legacy interface (kept for any existing keyword-arg callers)
    # ------------------------------------------------------------------
    def act(
        self,
        system_prompt: str = "",
        current_observation: str = "",
        recent_history: Optional[List[Any]] = None,
        context_window: str = "full",
        step_budget_remaining: Optional[int] = None,
        **kwargs: Any,
    ) -> Reply:
        if not system_prompt and "system_prompt" in kwargs:
            system_prompt = kwargs.pop("system_prompt")
        if not current_observation and "current_observation" in kwargs:
            current_observation = kwargs.pop("current_observation")
        if recent_history is None and "recent_history" in kwargs:
            recent_history = kwargs.pop("recent_history")
        if step_budget_remaining is None and "step_budget_remaining" in kwargs:
            step_budget_remaining = kwargs.pop("step_budget_remaining")

        messages: List[dict] = []

        sys_str = _clean_str(system_prompt)
        if sys_str:
            messages.append({"role": "system", "content": sys_str})

        if recent_history:
            for step in recent_history:
                obs = getattr(step, "observation", None) or (
                    step.get("observation") if isinstance(step, dict) else step
                )
                act_text = getattr(step, "action", None) or (
                    step.get("action") if isinstance(step, dict) else ""
                )
                obs_str = _clean_str(obs)
                act_str = _clean_str(act_text)
                if obs_str:
                    messages.append({"role": "user", "content": obs_str})
                if act_str:
                    messages.append({"role": "assistant", "content": act_str})

        user_content = _clean_str(current_observation)
        if step_budget_remaining is not None:
            user_content += f"\n[Step budget remaining: {step_budget_remaining}]"
        if user_content:
            messages.append({"role": "user", "content": user_content})

        return self.generate(messages)

    def __call__(self, *args: Any, **kwargs: Any) -> Reply:
        return self.act(*args, **kwargs)

    # ------------------------------------------------------------------
    # HTTP layer
    # ------------------------------------------------------------------
    def _post_chat_completions(self, body: dict, api_key: str) -> dict:
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        raw_data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        req = urllib.request.Request(url, data=raw_data, headers=headers, method="POST")

        last_error: Optional[str] = None

        for attempt in range(self.config.max_attempts):
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                    res_bytes = resp.read()
                    return json.loads(res_bytes.decode("utf-8"))
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")
                last_error = f"HTTP {e.code}: {err_body}"
                print(f"\n================ [OPENAI API HTTP ERROR {e.code}] ================")
                print(err_body)
                print("=================================================================\n")
                # Auth errors and bad requests won't fix themselves; bail.
                if e.code in (400, 401, 403, 404):
                    break
            except Exception as e:
                last_error = str(e)
                print(f"\n[OpenAIAgent EXCEPTION]: {e}\n")
                time.sleep(1.0 * (attempt + 1))

        raise RuntimeError(f"OpenAI API call failed: {last_error}")