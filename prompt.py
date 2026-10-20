# Copyright 2026 Bria AI and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified 2026-10-20 for ComfyUI-Fibo2; see NOTICE and CHANGELOG.md.
# Source: diffusers PR #14991, commit ab96e2b58c89adb9df418af74d81cc8b1e1be074.
# Changes: extracted pipeline prompt methods into a standalone local Prompt helper.
import json
import math

VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"


class Prompt:
    @staticmethod
    def _parse_prompt(prompt: str) -> tuple[dict | str | None, str | None]:
        # A prompt's caption, in the form Fibo 2.0 was trained with, and its edit instruction. A prompt that isn't a JSON
        # object comes back as it is, with no instruction: a short description, or an edit's instruction alone. An edit
        # prompt's caption is its other keys or its "structured_caption" key, a JSON caption or a short description
        try:
            caption = json.loads(prompt)
        except json.JSONDecodeError:
            return prompt, None
        if not isinstance(caption, dict):
            return prompt, None
        instruction = caption.pop("edit_instruction", None)
        if "structured_caption" in caption:
            caption = caption.pop("structured_caption")
            if not isinstance(caption, dict):
                return caption or None, instruction

        # Training captions hold no empty values (None, "", {}, [] or NaN), and no aesthetic_score or preference_score:
        # training removes them from the caption and from its sections
        def empty(value):
            if isinstance(value, float):
                return math.isnan(value)
            return value is None or (isinstance(value, (str, dict, list)) and not value)

        def clean(value):
            if isinstance(value, dict):
                value = {k: clean(v) for k, v in value.items()}
                return {k: v for k, v in value.items() if not empty(v)}
            if isinstance(value, list):
                return [v for v in map(clean, value) if not empty(v)]
            return value

        keys = set(caption)
        scores = ("aesthetic_score", "preference_score")
        caption = {
            key: {k: v for k, v in value.items() if k not in scores} if isinstance(value, dict) else value
            for key, value in clean(caption).items()
            if key not in scores
        }
        caption = {key: value for key, value in caption.items() if value != {}}
        # A caption of exactly a "short_description" and a "text_render" list keeps those keys in that order, and the
        # list even when it's empty: it then says the image has no text
        if keys == {"short_description", "text_render"}:
            caption = {
                "short_description": caption.get("short_description") or "",
                "text_render": caption.get("text_render") or [],
            }
        return caption or None, instruction

    @staticmethod
    def _mark_text_images(caption: dict, marker: str) -> dict:
        # Marks, in place, each text_render entry that has text with an "image_tokens" entry: the vision marker of the
        # image of its text, or empty where the image isn't shown
        for entry in caption.get("text_render") or []:
            if isinstance(entry, dict) and isinstance(entry.get("text"), str) and entry["text"]:
                entry["image_tokens"] = marker
        return caption

    @classmethod
    def _get_edit_prompt(cls, prompt: str, num_images: int, text_images: bool = False) -> str:
        # An edit prompt is one JSON object: a vision marker per image, the caption of the result when the prompt has
        # one, then the edit instruction. With images of the caption's text, each text entry gets the marker of its
        # image, which follows the images to edit
        caption, instruction = cls._parse_prompt(prompt)
        if instruction is None:
            # The prompt is the instruction alone
            caption, instruction = None, prompt
        if text_images and isinstance(caption, dict):
            cls._mark_text_images(caption, VISION_MARKER)

        marker = VISION_MARKER
        if num_images == 1:
            edit = {"image": marker}
        else:
            edit = {f"image_{k}": marker for k in range(1, num_images + 1)}
        if caption:
            edit["structured_caption"] = caption
        edit["edit_instruction"] = instruction
        return json.dumps(edit, separators=(",", ":"), ensure_ascii=False)

