"""Shared parsing for chat messages produced by the interface runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

PartKind = Literal["text", "image"]


@dataclass(frozen=True)
class ImagePayload:
    media_type: str
    data_b64: str


@dataclass(frozen=True)
class ContentPart:
    kind: PartKind
    text: str = ""
    image: Optional[ImagePayload] = None
    cache_breakpoint: bool = False


def parse_data_image_url(url: str) -> ImagePayload:
    """Split ``data:<mime>;base64,<data>`` into media type and raw base64 payload."""
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ValueError("Expected a data: URL with base64 image payload.")
    rest = url[5:]
    if ";base64," not in rest:
        raise ValueError("Expected ';base64,' in image data URL.")
    meta, _, b64 = rest.partition(";base64,")
    media_type = (meta.strip() or "image/png").split(";")[0].strip()
    return ImagePayload(media_type=media_type, data_b64=b64.strip())


def _image_url_from_block(block: dict) -> Optional[str]:
    url_holder = block.get("image_url")
    if isinstance(url_holder, dict):
        url = url_holder.get("url")
    else:
        url = url_holder
    return url if isinstance(url, str) else None


def parse_runner_content(content: object) -> Union[str, List[ContentPart]]:
    """Normalize runner message ``content`` to plain text or structured parts."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: List[ContentPart] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        cache_breakpoint = bool(block.get("cache_breakpoint"))
        if block_type == "text":
            parts.append(
                ContentPart(
                    kind="text",
                    text=str(block.get("text", "")),
                    cache_breakpoint=cache_breakpoint,
                )
            )
        elif block_type == "image_url":
            url = _image_url_from_block(block)
            if url and url.startswith("data:"):
                payload = parse_data_image_url(url)
                parts.append(
                    ContentPart(
                        kind="image",
                        image=payload,
                        cache_breakpoint=cache_breakpoint,
                    )
                )
    return parts


def join_split_text(content: object) -> object:
    """Rejoin text the runner split into blocks only to place cache breakpoints.

    A text block flagged ``text_continues`` continues into the next block (the
    text_only full history, one block per step). Agents without block-level
    breakpoints send the text as the runner would have unsplit: one string when
    nothing else is in the message.
    """
    if not isinstance(content, list) or not any(
        isinstance(block, dict) and block.get("text_continues") for block in content
    ):
        return content
    out: List[object] = []
    pending = ""
    for block in content:
        is_text = isinstance(block, dict) and block.get("type") == "text"
        if is_text and block.get("text_continues"):
            pending += str(block.get("text", ""))
            continue
        if pending:
            if is_text:
                block = {**block, "text": pending + str(block.get("text", ""))}
            else:
                out.append({"type": "text", "text": pending})
            pending = ""
        out.append(block)
    if pending:
        out.append({"type": "text", "text": pending})
    if len(out) == 1 and isinstance(out[0], dict) and out[0].get("type") == "text":
        return str(out[0].get("text", ""))
    return out


def strip_cache_markers(content: object) -> object:
    """Remove runner-only cache metadata before forwarding raw content blocks."""
    content = join_split_text(content)
    if not isinstance(content, list):
        return content
    return [
        {key: value for key, value in block.items() if key != "cache_breakpoint"}
        if isinstance(block, dict)
        else block
        for block in content
    ]


def split_system_prompt(messages: List[dict]) -> Tuple[Optional[str], List[dict]]:
    """Extract concatenated system text; return remaining user/assistant messages."""
    system_parts: List[str] = []
    turns: List[dict] = []
    for message in messages:
        msg_role = message.get("role")
        content = message.get("content", "")
        if msg_role == "system":
            system_parts.append(str(content))
        else:
            turns.append(message)
    system = "\n\n".join(system_parts) if system_parts else None
    return system, turns
