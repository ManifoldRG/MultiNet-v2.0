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
        if block_type == "text":
            parts.append(ContentPart(kind="text", text=str(block.get("text", ""))))
        elif block_type == "image_url":
            url = _image_url_from_block(block)
            if url and url.startswith("data:"):
                payload = parse_data_image_url(url)
                parts.append(ContentPart(kind="image", image=payload))
    return parts


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
