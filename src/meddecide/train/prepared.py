"""Prepared training rows: encode the mix once and replay the token ids per batch.

The 9B final cached its prepared rows because a crash at step 1 threw away a 66-minute preparation (T8,
``outputs/osler_v7/T8/arm/prepared.pt``). The 4B path re-encodes each batch on the fly, which is cheap per step, but it
cannot answer the question the token-capped plan (``meddecide.train.token_plan``) asks: how long is every row? One full
pass over the mix gives both — the lengths the plan sorts on, and a cache that makes a resumed or repeated run
independent of the tokeniser and identical row for row.

What is stored (one ``.npz``, gitignored) is exactly what ``ModelBatch`` needs:

* the concatenated token ids, per-row offsets, the readout marker, the offered option count and the gold index;
* the template id (the KL-anchor mask and the step log read it), the qtype and the source;
* the teacher distribution over the offered options (``None`` when the row has none);
* the item id, so a cached row can be traced back to its mix row.

Option keys are **not** stored. No training code reads them, and the pplx encoder writes the offered codes
(``EncodedItem.option_keys is optcode_names[:n]``), so ``encoded()`` rebuilds them from the cache's ``option_codes``
exactly as the encoder wrote them. The cache is a training artifact only: the evaluation path scores ``Item`` objects.

The cache is versioned (``CACHE_VERSION``) and refuses to load a file written by a different layout.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from meddecide.model.meddecide_model import EncodedItem

CACHE_VERSION = 1
PROGRESS_EVERY = 100_000


@dataclass
class PreparedRows:
    """A compact, replayable cache of encoded training rows (see the module docstring)."""

    lengths: np.ndarray  # (n,) int32: token count of every row
    ids: np.ndarray  # (sum(lengths),) int32: concatenated token ids
    offsets: np.ndarray  # (n + 1,) int64
    markers: np.ndarray  # (n,) int32: the readout marker position inside the row
    n_options: np.ndarray  # (n,) int16
    gold: np.ndarray  # (n,) int16
    truncated: np.ndarray  # (n,) bool
    template_index: np.ndarray  # (n,) int32
    templates: list[str] = field(default_factory=list)
    qtype_index: np.ndarray | None = None
    qtypes: list[str] = field(default_factory=list)
    source_index: np.ndarray | None = None
    sources: list[str] = field(default_factory=list)
    item_ids: np.ndarray | None = None  # (n,) str
    teacher_offsets: np.ndarray | None = None  # (n + 1,) int64: empty span = no teacher distribution
    teacher_values: np.ndarray | None = None  # (sum teacher options,) float32
    option_codes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = int(self.lengths.shape[0])
        if self.offsets.shape != (n + 1,):
            raise ValueError(f"offsets has shape {self.offsets.shape}, expected ({n + 1},)")
        if int(self.offsets[-1]) != int(self.ids.shape[0]):
            raise ValueError("the offsets do not close on the token id array")
        for name in ("markers", "n_options", "gold", "truncated", "template_index"):
            array = getattr(self, name)
            if array.shape != (n,):
                raise ValueError(f"{name} has shape {array.shape}, expected ({n},)")
        if self.teacher_offsets is not None and self.teacher_offsets.shape != (n + 1,):
            raise ValueError(f"teacher_offsets has shape {self.teacher_offsets.shape}, expected ({n + 1},)")

    def __len__(self) -> int:
        return int(self.lengths.shape[0])

    def encoded(self, index: int) -> EncodedItem:
        """The ``EncodedItem`` of a cached row, rebuilt exactly as the encoder wrote it."""
        from meddecide.model.meddecide_model import EncodedItem

        if not 0 <= index < len(self):
            raise IndexError(f"row {index} outside 0..{len(self) - 1}")
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        n = int(self.n_options[index])
        marker = int(self.markers[index])
        codes = list(self.option_codes[:n])
        if len(codes) != n:
            raise ValueError(f"row {index} offers {n} options, the cache lists {len(codes)} codes")
        teacher = None
        if self.teacher_offsets is not None and self.teacher_values is not None:
            t_start, t_end = int(self.teacher_offsets[index]), int(self.teacher_offsets[index + 1])
            if t_end > t_start:
                teacher = [float(value) for value in self.teacher_values[t_start:t_end]]
        return EncodedItem(
            item_id=str(self.item_ids[index]) if self.item_ids is not None else f"row#{index:08d}",
            qtype=str(self.qtypes[int(self.qtype_index[index])]) if self.qtype_index is not None else "?",
            template_id=str(self.templates[int(self.template_index[index])]),
            source=str(self.sources[int(self.source_index[index])]) if self.source_index is not None else "?",
            option_keys=codes,
            original_option_keys=list(codes),
            gold_index=int(self.gold[index]),
            input_ids=self.ids[start:end].tolist(),
            marker_positions=[marker] * n,
            option_end_positions=[marker] * n,
            answer_position=marker,
            prompt_tokens=int(self.lengths[index]),
            truncated=bool(self.truncated[index]),
            teacher_probs=teacher,
        )

    def __getitem__(self, index: int) -> EncodedItem:
        """The ``EncodedItem`` of a cached row (the trainer's batch plan indexes the cache directly)."""
        return self.encoded(index)

    def labelled(self) -> np.ndarray:
        """Boolean mask of the rows that carry a teacher distribution (D57's labelled stratum)."""
        if self.teacher_offsets is None:
            return np.zeros(len(self), dtype=bool)
        spans = np.diff(self.teacher_offsets)
        return spans > 0

    def to_meta(self) -> dict[str, Any]:
        return {"version": CACHE_VERSION, "rows": len(self), "total_tokens": int(self.ids.shape[0]),
                "templates": self.templates, "qtypes": self.qtypes, "sources": self.sources,
                "option_codes": self.option_codes}

    def save(self, path: str | Path) -> Path:
        """Write the cache as one ``.npz`` (uncompressed: token ids do not compress, and loading stays fast)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "lengths": self.lengths.astype(np.int32), "ids": self.ids.astype(np.int32),
            "offsets": self.offsets.astype(np.int64), "markers": self.markers.astype(np.int32),
            "n_options": self.n_options.astype(np.int16), "gold": self.gold.astype(np.int16),
            "truncated": self.truncated.astype(bool), "template_index": self.template_index.astype(np.int32),
            "meta": np.array(json.dumps(self.to_meta())),
        }
        if self.qtype_index is not None:
            payload["qtype_index"] = self.qtype_index.astype(np.int32)
        if self.source_index is not None:
            payload["source_index"] = self.source_index.astype(np.int32)
        if self.item_ids is not None:
            payload["item_ids"] = np.asarray(self.item_ids, dtype=str)
        if self.teacher_offsets is not None and self.teacher_values is not None:
            payload["teacher_offsets"] = self.teacher_offsets.astype(np.int64)
            payload["teacher_values"] = self.teacher_values.astype(np.float32)
        np.savez(out, **payload)
        return out

    @classmethod
    def load(cls, path: str | Path) -> PreparedRows:
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if int(meta.get("version", -1)) != CACHE_VERSION:
                raise ValueError(f"{path}: cache version {meta.get('version')} != {CACHE_VERSION}")
            rows = cls(
                lengths=data["lengths"], ids=data["ids"], offsets=data["offsets"], markers=data["markers"],
                n_options=data["n_options"], gold=data["gold"], truncated=data["truncated"],
                template_index=data["template_index"], templates=[str(t) for t in meta["templates"]],
                qtype_index=data.get("qtype_index", None),
                qtypes=[str(t) for t in meta.get("qtypes", [])],
                source_index=data.get("source_index", None),
                sources=[str(t) for t in meta.get("sources", [])],
                item_ids=data.get("item_ids", None),
                teacher_offsets=data.get("teacher_offsets", None),
                teacher_values=data.get("teacher_values", None),
                option_codes=[str(c) for c in meta.get("option_codes", [])],
            )
        if len(rows) != int(meta["rows"]):
            raise ValueError(f"{path}: the cache holds {len(rows)} rows, its meta {meta['rows']}")
        return rows


def _vocabulary(values: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """First-seen order vocabulary and the index of every value (a stable, replayable encoding)."""
    order: dict[str, int] = {}
    index = np.empty(len(values), dtype=np.int32)
    for row, value in enumerate(values):
        position = order.get(value)
        if position is None:
            position = len(order)
            order[value] = position
        index[row] = position
    return list(order), index


def build_prepared(
    n_rows: int,
    encode: Callable[[int], EncodedItem],
    *,
    option_codes: Sequence[str],
    progress_every: int = PROGRESS_EVERY,
) -> PreparedRows:
    """Encode rows ``0..n_rows-1`` once through ``encode(position)`` and pack the result into a :class:`PreparedRows`.

    ``encode`` is the model's own ``encode_item`` (with the option permutation the caller wants); nothing here re-renders
    a prompt, so a cached row is the model's encoding of that item, not a reimplementation of it. The row count rather
    than a list of items is the interface because the training mix holds 1.5M items and no caller needs them all in memory
    at once: the encoder converts one mix row at a time.
    """
    if n_rows < 1:
        raise ValueError("nothing to prepare")
    lengths = np.empty(n_rows, dtype=np.int32)
    markers = np.empty(n_rows, dtype=np.int32)
    n_options = np.empty(n_rows, dtype=np.int16)
    gold = np.empty(n_rows, dtype=np.int16)
    truncated = np.empty(n_rows, dtype=bool)
    templates: list[str] = []
    qtypes: list[str] = []
    sources: list[str] = []
    item_ids: list[str] = []
    ids_chunks: list[np.ndarray] = []
    offsets = np.zeros(n_rows + 1, dtype=np.int64)
    teacher_offsets = np.zeros(n_rows + 1, dtype=np.int64)
    teacher_values: list[np.ndarray] = []
    started = time.time()
    for position in range(n_rows):
        encoded = encode(position)
        row_ids = np.asarray(encoded.input_ids, dtype=np.int32)
        if row_ids.size < 1:
            raise ValueError(f"row {position}: the encoder produced an empty row")
        if len(set(encoded.marker_positions)) != 1 or len(set(encoded.option_end_positions)) != 1:
            raise ValueError(f"row {position}: the cache assumes one readout marker position for every offered option")
        if len(encoded.marker_positions) != len(encoded.option_keys):
            raise ValueError(f"row {position}: {len(encoded.marker_positions)} markers for "
                             f"{len(encoded.option_keys)} option keys")
        lengths[position] = row_ids.size
        markers[position] = int(encoded.marker_positions[0])
        n_options[position] = len(encoded.marker_positions)
        gold[position] = int(encoded.gold_index)
        truncated[position] = bool(encoded.truncated)
        ids_chunks.append(row_ids)
        offsets[position + 1] = offsets[position] + row_ids.size
        templates.append(str(encoded.template_id))
        qtypes.append(str(encoded.qtype))
        sources.append(str(encoded.source))
        item_ids.append(str(encoded.item_id))
        if encoded.teacher_probs is not None:
            teacher_values.append(np.asarray(encoded.teacher_probs, dtype=np.float32))
            teacher_offsets[position + 1] = teacher_offsets[position] + len(encoded.teacher_probs)
        else:
            teacher_offsets[position + 1] = teacher_offsets[position]
        if progress_every and (position + 1) % progress_every == 0:
            elapsed = time.time() - started
            share = (position + 1) / n_rows
            print(f"[prepared] {position + 1}/{n_rows} rows ({share:.1%}) in {elapsed:.0f}s; "
                  f"{int(offsets[position + 1]) / 1e6:.1f}M tokens", flush=True)
    template_vocab, template_index = _vocabulary(templates)
    qtype_vocab, qtype_index = _vocabulary(qtypes)
    source_vocab, source_index = _vocabulary(sources)
    teacher_flat = np.concatenate(teacher_values) if teacher_values else np.zeros(0, dtype=np.float32)
    if teacher_flat.size != int(teacher_offsets[-1]):
        raise ValueError("the teacher distribution offsets do not close")
    ids = np.concatenate(ids_chunks) if ids_chunks else np.zeros(0, dtype=np.int32)
    print(f"[prepared] {n_rows} rows, {int(ids.shape[0]) / 1e6:.2f}M tokens, "
          f"{int((lengths >= 1).sum())} non-empty, in {time.time() - started:.0f}s", flush=True)
    return PreparedRows(
        lengths=lengths, ids=ids, offsets=offsets, markers=markers, n_options=n_options, gold=gold,
        truncated=truncated, template_index=template_index, templates=template_vocab,
        qtype_index=qtype_index, qtypes=qtype_vocab, source_index=source_index, sources=source_vocab,
        item_ids=np.asarray(item_ids, dtype=str), teacher_offsets=teacher_offsets, teacher_values=teacher_flat,
        option_codes=[str(code) for code in option_codes],
    )


def _same_teacher(left: Sequence[float] | None, right: Sequence[float] | None, tol: float = 1e-6) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return len(left) == len(right) and all(abs(a - b) <= tol for a, b in zip(left, right, strict=True))


def verify_rows(prepared: PreparedRows, live: Sequence[EncodedItem], positions: Sequence[int]) -> dict[str, Any]:
    """Compare cached rows against live encodings of the same positions (the cache-is-the-model's-encoding check)."""
    mismatch: list[int] = []
    for position, encoded in zip(positions, live, strict=True):
        cached = prepared.encoded(int(position))
        same = (cached.input_ids == encoded.input_ids and cached.marker_positions == encoded.marker_positions
                and cached.gold_index == encoded.gold_index
                and _same_teacher(cached.teacher_probs, encoded.teacher_probs))
        if not same:
            mismatch.append(int(position))
    return {"checked": len(positions), "mismatched": mismatch, "identical": not mismatch}
