"""Checks on the Osler training configs before a run starts (ADVISORY O6-O8 and O10; PROGRAM D22, D23).

The O10 runs (Osler-9B and Osler-0.8B) are one-shot and long, so each config is checked before the model loads:
the base and its revision are the pinned ones; the head is the one O9 chose; and the recipe is arm L's recipe, changed
only in the base, the output directory and the head. The matched arms (L, P, N) are checked the same way in the tests.
"""

from __future__ import annotations

from meddecide.train.config import StudentConfig

# base model -> pinned revision (ADVISORY, "Model ladder"; the same pins as the O2 registry and the O6-O8 arms)
PINNED_BASES: dict[str, str] = {
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
}

# arm -> (readout, bidirectional_full_attention), as configs/osler_v0/arm_<arm>.yaml sets them (D23)
HEAD_SETTINGS: dict[str, tuple[str, bool]] = {
    "L": ("option_code", False),
    "P": ("pointer", False),
    "N": ("option_code", True),
}

# the keys that may differ from arm L's recipe: the base, its revision, the output directory and the head
BASE_AND_HEAD_KEYS = frozenset({"base_model", "revision", "output_dir", "readout", "bidirectional_full_attention"})


def check_pinned_base(config: StudentConfig) -> None:
    """Raise unless the config's base is an O10 base and its revision is the pinned one."""
    expected = PINNED_BASES.get(config.base_model)
    if expected is None:
        raise ValueError(f"base {config.base_model!r} is not an O10 base; expected one of {sorted(PINNED_BASES)}")
    if config.revision != expected:
        raise ValueError(f"{config.base_model}: revision {config.revision!r} is not the pinned {expected!r}")


def check_head_matches(config: StudentConfig, chosen: str) -> None:
    """Raise unless the config's head is the one O9 chose (``chosen`` is 'L', 'P' or 'N')."""
    if chosen not in HEAD_SETTINGS:
        raise ValueError(f"O9 chose {chosen!r}; expected one of {sorted(HEAD_SETTINGS)}")
    readout, bidirectional = HEAD_SETTINGS[chosen]
    if config.readout != readout or config.bidirectional_full_attention != bidirectional:
        raise ValueError(
            f"the config's head (readout {config.readout!r}, bidirectional {config.bidirectional_full_attention}) "
            f"is not O9's choice {chosen} (readout {readout!r}, bidirectional {bidirectional})"
        )


def recipe_differences(config: StudentConfig, reference: StudentConfig) -> list[str]:
    """The sorted keys, other than the base, output directory and head, on which the two configs differ."""
    a, b = config.to_dict(), reference.to_dict()
    return sorted(k for k in set(a) | set(b) if k not in BASE_AND_HEAD_KEYS and a.get(k) != b.get(k))
