"""Condition set 2: observation format."""

from __future__ import annotations

from .core import ConditionSet, Variant


CONDITION_SET = ConditionSet(
    name="Observation format",
    comparisons=(
        "Text only",
        "Image + text",
        "Image only",
    ),
    decision="Measure whether text adds meaningful signal beyond image input.",
    variants={
        "text_only": Variant(
            name="text_only",
            description="Natural-language current observation, no image blocks.",
            config_overrides={
                "observation": "text_only",
                "include_current_observation_description": True,
                "observation_text_includes_facing": True,
            },
        ),
        "image_text": Variant(
            name="image_text",
            description="Image block plus natural-language observation.",
            config_overrides={
                "observation": "image_text",
                "include_current_observation_description": True,
                "observation_text_includes_facing": True,
            },
        ),
        "image_only": Variant(
            name="image_only",
            description="Image block with no initial natural-language maze map.",
            config_overrides={"observation": "image_only"},
        ),
    },
)
