# Deterministic event-centric prompt expansion for WMReward/MAGI-1.
# This module intentionally does not call an LLM or any external model.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptExpansionConfig:
    mode: str = "coect"
    max_events: int = 4
    include_physical_constraints: bool = True


def _clean_prompt(prompt: str) -> str:
    return " ".join(str(prompt).strip().split())


def _sentence(text: str) -> str:
    text = _clean_prompt(text)
    if not text:
        return text
    return text if text[-1] in ".!?" else text + "."


def _event_templates(prompt: str, max_events: int) -> list[str]:
    p = prompt.lower()

    if any(word in p for word in ("fall", "falls", "drop", "drops", "dropped", "released", "let go", "lets go")):
        events = [
            "the object starts from its initial support, grip, or suspended position",
            "gravity pulls it downward with continuous acceleration",
            "it keeps the same identity, color, and approximate shape while moving",
            "on contact, it bounces, compresses, or settles according to the surface",
        ]
    elif any(word in p for word in ("roll", "rolls", "rolling")):
        events = [
            "the object begins from a visible pipe, hand, ramp, or resting position",
            "it rolls along the supporting surface with continuous motion",
            "its direction and speed change only through contact, slope, friction, or collision",
            "after contact with another object or boundary, it deflects, slows, or settles plausibly",
        ]
    elif any(word in p for word in ("rotate", "rotates", "rotating", "spin", "spins", "spinning")):
        events = [
            "the rotating object or platform starts with a clear axis and orientation",
            "rotation proceeds smoothly without sudden jumps or identity changes",
            "objects in contact with the rotating surface follow or resist the motion consistently",
            "contacts and occlusions remain spatially coherent as the rotation continues",
        ]
    elif any(word in p for word in ("lower", "lowers", "lowered", "place", "places", "placed")):
        events = [
            "the held object begins from a visible hand, tool, or support",
            "it moves toward the target surface along a continuous path",
            "contact occurs only when the object reaches the surface or another object",
            "after contact, it remains supported, compresses, slides, or settles consistently",
        ]
    elif any(word in p for word in ("burn", "burns", "smoke", "water", "liquid", "pour", "dispensing")):
        events = [
            "the initial material state is visible and stable",
            "the visible cause produces a continuous physical change",
            "the changed material remains spatially attached to its source or container",
            "the final motion or deformation follows contact, gravity, heat, or fluid flow",
        ]
    else:
        events = [
            "the scene begins from a stable, physically coherent initial state",
            "each object motion is caused by a visible force, support change, contact, or gravity",
            "objects preserve identity, scale, color, and approximate shape over time",
            "collisions, occlusions, and settling behavior remain consistent with the scene geometry",
        ]

    return events[: max(1, int(max_events))]



def _key_action_directive(prompt: str) -> str:
    p = prompt.lower()

    if "rubber duck" in p and any(word in p for word in ("roll", "rolls", "rolling")):
        return (
            "Keep the same objects and static camera. Clearly show the tennis ball rolling out from the pipe, "
            "moving across the table toward the yellow rubber duck, and reaching or contacting the duck."
        )

    if "ball" in p and "block" in p and any(word in p for word in ("let go", "lets go", "released", "suspended")):
        return (
            "Keep the same objects and static camera. Clearly show the grabber tools releasing the tennis ball and block, "
            "then show both objects falling downward under gravity while their identities remain visible."
        )

    if "cardstock" in p and any(word in p for word in ("rotate", "rotates", "rotating")):
        return (
            "Keep the same objects and static camera. Clearly show the platform and cardstock rotating clockwise, "
            "while the grabber lowers the tennis ball onto the table and the occlusion remains spatially consistent."
        )

    if any(word in p for word in ("fall", "falls", "drop", "drops", "dropped", "released", "let go", "lets go")):
        return (
            "Keep the same objects and camera. Clearly show the released object moving downward under gravity "
            "from its initial support or suspended position to the lower surface."
        )

    if any(word in p for word in ("roll", "rolls", "rolling")):
        return (
            "Keep the same objects and camera. Clearly show the rolling object starting from its source, "
            "moving continuously along the supporting surface, and reaching the described target or contact."
        )

    if any(word in p for word in ("rotate", "rotates", "rotating", "spin", "spins", "spinning")):
        return (
            "Keep the same objects and camera. Clearly show the rotating object or platform turning around a stable axis, "
            "with contacted objects and occlusions remaining coherent."
        )

    return (
        "Keep the same objects and camera. Clearly show the main action described in the prompt from start to finish, "
        "including the visible cause, motion path, contact, and final state."
    )


def expand_prompt(prompt: str, config: PromptExpansionConfig | None = None) -> str:
    """Expand a prompt with a short event-centric physical causal chain."""
    config = config or PromptExpansionConfig()
    prompt = _clean_prompt(prompt)
    if not prompt:
        return prompt

    # Avoid duplicating the template when rerunning an expanded prompt.
    if (
        "Event-centric causal chain:" in prompt
        or "Soft stage event guidance:" in prompt
        or "Physical plausibility constraints:" in prompt
        or "Key action directive:" in prompt
    ):
        return prompt

    mode = (config.mode or "coect").lower().replace("-", "_")
    if mode in {"action_focus", "key_action", "action", "focused_action"}:
        return " ".join(
            [
                _sentence(prompt),
                "Key action directive:",
                _key_action_directive(prompt),
            ]
        )

    events = _event_templates(prompt, config.max_events)
    event_text = " ".join(f"Event {i + 1}: {event}." for i, event in enumerate(events))

    parts = [
        _sentence(prompt),
        "Event-centric causal chain:",
        event_text,
    ]

    if mode in {"soft_stage", "softstage", "stage_soft"}:
        parts.extend(
            [
                "Soft stage event guidance:",
                (
                    "Early stage, use weak event pressure: establish the subject, scene layout, visible support, "
                    "and camera stability while keeping the original prompt semantics dominant."
                ),
                (
                    "Middle stage, use strong event pressure: follow the causal event chain continuously, "
                    "with coherent gravity, contact, collision, rolling, rotation, or support changes."
                ),
                (
                    "Late stage, use semantic repair pressure: preserve object identity, material, color, scale, "
                    "and prompt details while avoiding over-corrected or abrupt physical motion."
                ),
                (
                    "Treat these stages as soft overlapping tendencies rather than hard temporal cuts; "
                    "transitions should be gradual and physically plausible."
                ),
            ]
        )

    if config.include_physical_constraints:
        parts.extend(
            [
                "Physical plausibility constraints:",
                (
                    "Preserve a static and coherent camera unless the prompt explicitly says otherwise; "
                    "avoid teleportation, sudden scale changes, object identity swaps, and impossible intersections; "
                    "make every motion caused by a visible force, contact, support change, gravity, material change, or fluid flow."
                ),
            ]
        )

    return " ".join(parts)
