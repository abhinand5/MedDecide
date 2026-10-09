"""S7 acceptance tests for the MedDecide architecture (frozen base + LoRA + pointer head).

The four required tests are:

* :func:`test_output_is_a_proper_distribution_over_offered_options` — non-negative, sums to 1,
  over *exactly* the offered option keys, for `noul`, `choice` and `score`, plus synthetic
  2-option and 64-option items;
* :func:`test_option_permutation_permutes_the_output` — permuting an item's options permutes the
  output, checked by option **content**;
* :func:`test_adapter_disabled_generation_is_byte_identical` — greedy generation with the
  adapter disabled equals an untouched base loaded separately (token ids and decoded text),
  with a control proving the comparison can detect a changed adapter;
* :func:`test_overfit_64_items` — 64 training items reach >= 0.95 training accuracy.

The rest are fast CPU checks of the pieces the GPU tests depend on (segment softmax locality,
the loss, the temperature fit, marker location, the config file). GPU tests are marked ``gpu``
and skip when no CUDA device is present.

Test-time overrides of the shipped config (smaller prompt cap and batches) exist to keep the
suite quick; the shipped values are validated by :func:`test_config_file_matches_the_plan`.
"""

from __future__ import annotations

import dataclasses
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import torch

from meddecide.bench.schema import Item, Option, QuestionType, Tier, make_item
from meddecide.eval.readout import canonicalise_options, key_token_id, render_prompt
from meddecide.model.head import HeadSettings, PointerHead, segment_softmax
from meddecide.model.markers import locate_option_markers
from meddecide.model.meddecide_model import MedDecideModel, reorder_options
from meddecide.train.config import load_config
from meddecide.train.data import read_items
from meddecide.train.losses import ce_plus_brier
from meddecide.train.temperature import fit_temperature
from meddecide.train.trainer import Trainer

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs" / "student_v0.yaml"
DEV_PATH = REPO / "data" / "train" / "student_v0" / "dev.jsonl"
TRAIN_PATH = REPO / "data" / "train" / "student_v0" / "train.jsonl"

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
GPU_MARK = pytest.mark.gpu


# --------------------------------------------------------------------------- helpers
def synthetic_item(
    *,
    qtype: QuestionType = QuestionType.CHOICE,
    n_options: int = 4,
    seed: int = 0,
    source: str = "synthetic_pointer",
) -> Item:
    """A validated synthetic item with the requested option count."""
    if qtype is QuestionType.NOUL:
        options = [Option(key="yes", label="Yes"), Option(key="no", label="No")]
        gold = "yes" if seed % 2 == 0 else "no"
    elif qtype is QuestionType.SCORE:
        options = [Option(key=str(i + 1), label=f"Level {i + 1} of 5") for i in range(n_options)]
        gold = str(1 + seed % n_options)
    else:
        from meddecide.eval.readout import option_letter

        options = [Option(key=option_letter(i), label=f"Candidate answer {i + 1}") for i in range(n_options)]
        gold = options[seed % n_options].key
    return make_item(
        tier=Tier.FRESH,
        source=source,
        source_record_id=f"r{seed}-{n_options}",
        source_url="https://example.org/synthetic",
        source_license="synthetic",
        record_date=date(2026, 1, 1),
        split="dev",
        template_id=f"synthetic_{qtype}_v1",
        skill="synthetic",
        qtype=qtype,
        state=(
            "A 61-year-old patient with hypertension presents with a two-day history of "
            "headache and blurred vision. Blood pressure is 178/104 mmHg."
        ),
        question="Which of the following is the most appropriate next step?",
        options=options,
        gold=gold,
        option_order_seed=seed,
    )


@pytest.fixture(scope="session")
def dev_items() -> list[Item]:
    """A deterministic, source-spread slice of the dev split, one item per qtype minimum."""
    if not DEV_PATH.exists():
        return [
            synthetic_item(qtype=QuestionType.NOUL, seed=1),
            synthetic_item(qtype=QuestionType.CHOICE, n_options=4, seed=2),
            synthetic_item(qtype=QuestionType.SCORE, n_options=5, seed=3),
        ]
    # the whole dev file: `score` items are rare (33 of 16,454) and a short head-of-file slice
    # can miss them entirely (it did, and the distribution test then failed on its own qtype
    # coverage assertion rather than on the property it tests)
    items = read_items(DEV_PATH)
    by_qtype: dict[str, list[Item]] = {}
    for item in items:
        by_qtype.setdefault(str(item.qtype), []).append(item)
    picked: list[Item] = []
    for qtype in sorted(by_qtype):
        picked.extend(by_qtype[qtype][:4])
    assert {"noul", "choice", "score"} <= set(by_qtype), sorted(by_qtype)
    return picked


@pytest.fixture(scope="session")
def test_config():
    """Shipped config with test-sized limits (smaller caps, tiny dev evaluation)."""
    config = load_config(CONFIG_PATH)
    return dataclasses.replace(
        config,
        max_prompt_tokens=2048,
        max_batch_tokens=4096,
        batch_size=4,
        eval_every=0,
        eval_items=32,
        log_every=0,
        fit_temperature=False,
    )


@pytest.fixture(scope="session")
def fresh_model(test_config):
    """Untrained MedDecide model (base + LoRA + fresh head), never trained by this suite."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    model = MedDecideModel(
        test_config.base_model,
        lora=test_config.lora,
        head_settings=test_config.head,
        dtype=test_config.dtype,
        device=test_config.device,
        variant=test_config.variant,
        max_prompt_tokens=test_config.max_prompt_tokens,
    )
    yield model
    del model
    torch.cuda.empty_cache()


# ------------------------------------------------------- CPU checks of the pieces
def test_segment_softmax_is_a_proper_local_distribution() -> None:
    """Probabilities are non-negative, sum to 1 per item, and never mix items."""
    logits = torch.tensor([0.5, -1.0, 2.0, 1.0, 0.0, -0.5, 3.0, 0.25])
    item_index = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2])
    probs = segment_softmax(logits, item_index, n_items=3)
    assert torch.all(probs >= 0)
    for item in range(3):
        mask = item_index == item
        assert abs(float(probs[mask].sum()) - 1.0) < 1e-6
    # shifting one item's logits by a constant cannot change its distribution
    shifted = logits.clone()
    shifted[item_index == 1] += 7.0
    assert torch.allclose(segment_softmax(shifted, item_index, 3), probs, atol=1e-6)
    # a huge logit in item 2 must not leak into item 0
    spiked = logits.clone()
    spiked[item_index == 2] += 100.0
    after = segment_softmax(spiked, item_index, 3)
    assert torch.allclose(after[item_index == 0], probs[item_index == 0], atol=1e-6)


@pytest.mark.parametrize("option_state", ["key", "key_end", "end"])
def test_pointer_head_scores_each_option_from_its_own_hidden_state(option_state: str) -> None:
    """An option's logit depends on that option's own states and the answer position only.

    Each option has two candidate states (its key token and its line end); which of them the
    head consumes is decided by ``option_state``, and the test checks that unused ones really
    are unused.
    """
    torch.manual_seed(0)
    d_model = 8
    head = PointerHead(d_model, HeadSettings(hidden=16, option_state=option_state))
    head.eval()
    hidden = torch.randn(2, 10, d_model)
    # item 0: keys 1,2 — line ends 3,4 ; item 1: keys 5,6 — line ends 7,8 ; answer 9
    marker_positions = torch.tensor([1, 2, 5, 6])
    end_positions = torch.tensor([3, 4, 7, 8])
    marker_item = torch.tensor([0, 0, 1, 1])
    answer_positions = torch.tensor([9, 9])
    base = head(hidden, marker_positions, marker_item, answer_positions, end_positions)
    assert base.shape == (4,)

    key_matters = option_state in ("key", "key_end")
    end_matters = option_state in ("end", "key_end")
    # (tok position, owning item, owning option index, kind)
    probes = [(1, 0, 0, "key"), (3, 0, 0, "end"), (5, 1, 2, "key"), (7, 1, 2, "end")]
    for position, row, option_index, kind in probes:
        matters = key_matters if kind == "key" else end_matters
        changed = hidden.clone()
        changed[row, position] += 5.0
        after = head(changed, marker_positions, marker_item, answer_positions, end_positions)
        for other in range(4):
            if other == option_index and matters:
                assert not torch.allclose(after[other], base[other]), (option_state, position)
            else:
                assert torch.allclose(after[other], base[other], atol=1e-6), (option_state, position)

    changed = hidden.clone()
    changed[1, 9] += 5.0  # item 1's answer position
    after = head(changed, marker_positions, marker_item, answer_positions, end_positions)
    assert not torch.allclose(after[2:], base[2:])
    assert torch.allclose(after[:2], base[:2], atol=1e-6)


def test_ce_plus_brier_matches_hand_computation() -> None:
    """Hand-computed CE and Brier for one item, and the lambda weighting."""
    probs = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)
    gold = torch.tensor([0])
    item_index = torch.tensor([0, 0, 0])
    out = ce_plus_brier(probs, gold, item_index, 1, lam=1.0)
    assert abs(float(out.ce) - (-np.log(0.7))) < 1e-9
    assert abs(float(out.brier) - (0.3**2 + 0.2**2 + 0.1**2)) < 1e-9
    assert abs(float(out.total) - (float(out.ce) + float(out.brier))) < 1e-9
    assert out.accuracy == 1.0
    out2 = ce_plus_brier(probs, gold, item_index, 1, lam=0.5)
    assert abs(float(out2.total) - (float(out.ce) + 0.5 * float(out.brier))) < 1e-9


def test_temperature_fit_recovers_a_known_scale() -> None:
    """Labels drawn from softmax(3z) must come back with a fitted T ~ 1/3.

    Labels are *sampled*, so the objective has a finite optimum; perfectly separable labels
    would have none (the NLL keeps falling as T -> 0), which is why the earlier version of this
    test hit the search's lower bound.
    """
    rng = np.random.default_rng(0)
    z = rng.normal(size=(20000, 4))
    true_temperature = 1.0 / 3.0
    scaled = (z / true_temperature)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    p = np.exp(scaled)
    p /= p.sum(axis=1, keepdims=True)
    gold = np.array([rng.choice(4, p=row) for row in p])
    assert 0 < (gold != np.argmax(z, axis=1)).mean() < 0.9, "labels must be noisy but learnable"
    fit = fit_temperature(list(z), list(gold), qtype="choice")
    assert abs(fit.temperature - true_temperature) < 0.05, fit
    assert fit.nll_after <= fit.nll_before


def test_temperature_fit_accepts_items_with_different_option_counts() -> None:
    """One qtype holds items with different option counts (dev `choice`: 3 to 17 options).

    The first implementation stacked the logits into one rectangular array and crashed on the
    S7 smoke run; the fit must be per item.
    """
    rng = np.random.default_rng(1)
    rows = [rng.normal(size=2), rng.normal(size=4), rng.normal(size=17), rng.normal(size=3)]
    gold = [0, 2, 5, 1]
    fit = fit_temperature(rows, gold, qtype="choice")
    assert fit.n_items == 4
    assert fit.temperature > 0
    assert fit.nll_after <= fit.nll_before
    with pytest.raises(ValueError):
        fit_temperature([rng.normal(size=4)], [7], qtype="choice")


def test_reorder_options_keeps_gold_on_its_content() -> None:
    item = synthetic_item(qtype=QuestionType.CHOICE, n_options=4, seed=1)
    canonical, _ = canonicalise_options(item)
    permutation = [2, 0, 3, 1]
    moved = reorder_options(canonical, permutation)
    assert [o.label for o in moved.options] == [
        canonical.options[i].label for i in permutation
    ]
    expected_gold_label = canonical.options[canonical.gold_index].label
    assert moved.options[moved.gold_index].label == expected_gold_label
    # keys travel with their option; canonicalisation later re-letters by display position
    assert moved.option_keys == [canonical.options[i].key for i in permutation]
    with pytest.raises(ValueError):
        reorder_options(canonical, [0, 0, 1, 2])


def test_permuted_item_canonicalises_to_lettered_options() -> None:
    """The augmentation path keeps the prompt's A/B/C… lettering and the gold on its content."""
    item = synthetic_item(qtype=QuestionType.CHOICE, n_options=4, seed=3)
    permutation = [3, 1, 0, 2]
    permuted = reorder_options(item, permutation)
    canonical, original_keys = canonicalise_options(permuted)
    assert canonical.option_keys == ["A", "B", "C", "D"]
    gold_label = item.options[item.gold_index].label
    assert canonical.options[canonical.gold_index].label == gold_label
    assert [o.label for o in canonical.options] == [item.options[i].label for i in permutation]
    assert original_keys == [item.options[i].key for i in permutation]


@pytest.fixture(scope="session")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    except OSError:  # pragma: no cover - no local cache
        pytest.skip("tokenizer not available locally")
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def test_markers_match_the_harness_rendering(tokenizer, dev_items) -> None:
    """The located marker tokens are the tokens the prompt shows, for every qtype."""
    for item in dev_items[:6]:
        canonical, _ = canonicalise_options(item)
        markers = locate_option_markers(canonical, tokenizer, variant="bare")
        assert markers.prompt == render_prompt(canonical, tokenizer, "bare")
        assert markers.answer_position == len(markers.input_ids) - 1
        assert markers.matches_harness_key_tokens, markers.mismatches
        for i, key in enumerate(markers.option_keys):
            assert markers.marker_token_ids[i] == key_token_id(key, tokenizer, "bare")
            assert 0 <= markers.positions[i] < len(markers.input_ids)
        assert len(set(markers.positions)) == len(markers.positions)


def test_markers_survive_labels_that_contain_other_option_keys(tokenizer) -> None:
    """A label like "C. difficile colitis" must not be mistaken for option C's marker.

    The first implementation searched the prompt for ``"C. "`` from a running cursor and landed
    inside option B's *label*. The S7 smoke run caught it (it scored token ``" C"`` instead of
    the key token ``"C"`` and the marker check refused the item). This is the regression test.
    """
    item = make_item(
        tier=Tier.FRESH,
        source="synthetic_pointer",
        source_record_id="c-difficile",
        source_url="https://example.org/synthetic",
        source_license="synthetic",
        record_date=date(2026, 1, 1),
        split="dev",
        template_id="synthetic_trap_v1",
        skill="synthetic",
        qtype=QuestionType.CHOICE,
        state="A patient presents with bloody diarrhoea after antibiotics.",
        question="Which organism is the most likely cause?",
        options=[
            Option(key="A", label="Rotavirus infection"),
            Option(key="B", label="C. difficile colitis"),
            Option(key="C", label="A. baumannii infection"),
            Option(key="D", label="B. cereus infection"),
        ],
        gold="B",
        option_order_seed=0,
    )
    canonical, _ = canonicalise_options(item)
    markers = locate_option_markers(canonical, tokenizer, variant="bare")
    assert markers.matches_harness_key_tokens, markers.mismatches
    assert not any(markers.merged_with_prefix)
    for i, opt in enumerate(canonical.options):
        assert tokenizer.decode([markers.marker_token_ids[i]]).strip() == opt.key
        assert markers.end_positions[i] >= markers.positions[i]
    # the span [marker, line end] really is that option's own line, in full
    for i, opt in enumerate(canonical.options):
        span = markers.input_ids[markers.positions[i] : markers.end_positions[i] + 1]
        assert tokenizer.decode(span) == f"{opt.key}. {opt.label}", (i, tokenizer.decode(span))

    if DEV_PATH.exists():
        for real in read_items(DEV_PATH, limit=512, stride=32):
            canon, _ = canonicalise_options(real)
            got = locate_option_markers(canon, tokenizer, variant="bare")
            assert got.matches_harness_key_tokens, (real.item_id, got.mismatches)


def test_markers_extend_to_a_64_option_item(tokenizer) -> None:
    """64 options work: every key is located once, even where a key is multi-token."""
    item = synthetic_item(qtype=QuestionType.CHOICE, n_options=64, seed=0)
    canonical, _ = canonicalise_options(item)
    markers = locate_option_markers(canonical, tokenizer, variant="bare")
    assert markers.n_options == 64
    assert len(set(markers.positions)) == 64
    multi = [k for k, single in zip(markers.option_keys, markers.single_token_key, strict=True)
             if not single]
    # the repo's option_letter() emits a few keys this tokenizer splits (BQ, BZ, CJ at 64)
    assert len(multi) < 8, multi
    for i, key in enumerate(markers.option_keys):
        if markers.single_token_key[i]:
            assert markers.marker_token_ids[i] == key_token_id(key, tokenizer, "bare")


def test_config_file_matches_the_plan() -> None:
    """The shipped config carries the values the plan fixes."""
    config = load_config(CONFIG_PATH)
    assert config.base_model == "Qwen/Qwen3.5-0.8B"
    assert config.lora.r == 16
    targets = set(config.lora.target_modules)
    assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= targets
    assert {"gate_proj", "up_proj", "down_proj"} <= targets
    assert config.lambda_brier == 1.0
    assert config.fit_temperature is True
    assert config.temperature_per_qtype is True
    assert config.train_path == "data/train/student_v0/train.jsonl"
    assert config.dev_path == "data/train/student_v0/dev.jsonl"
    assert config.max_prompt_tokens > 0 and config.seed >= 0
    assert config.shuffle_options is True


# ------------------------------------------------------------------ GPU acceptance
@GPU_MARK
@GPU
def test_output_is_a_proper_distribution_over_offered_options(fresh_model, dev_items) -> None:
    """(1) Non-negative, sums to 1, over exactly the offered options — 2 to 64 options."""
    items: list[Item] = []
    for qtype in (QuestionType.NOUL, QuestionType.CHOICE, QuestionType.SCORE):
        matching = [item for item in dev_items if item.qtype is qtype]
        assert matching, f"no dev item of qtype {qtype}"
        items.extend(matching[:2])
    items.append(synthetic_item(qtype=QuestionType.CHOICE, n_options=2, seed=11))
    items.append(synthetic_item(qtype=QuestionType.CHOICE, n_options=64, seed=12))
    items.append(synthetic_item(qtype=QuestionType.CHOICE, n_options=17, seed=13))

    decisions = fresh_model.predict(
        items, batch_size=4, max_batch_tokens=4096, temperature={}
    )
    assert len(decisions) == len(items)
    seen_qtypes = set()
    seen_counts = set()
    worst_deviation = 0.0
    for item, decision in zip(items, decisions, strict=True):
        canonical, original_keys = canonicalise_options(item)
        seen_qtypes.add(str(item.qtype))
        seen_counts.add(canonical.n_options)
        probs = np.asarray(decision.probs, dtype=np.float64)
        assert decision.item_id == item.item_id
        assert decision.option_keys == canonical.option_keys
        assert decision.original_option_keys == original_keys
        assert len(probs) == canonical.n_options
        assert np.all(probs >= 0.0)
        # the head is float32, so a row sums to 1 within ~1e-7; the exact sum is printed below
        deviation = abs(float(probs.sum()) - 1.0)
        worst_deviation = max(worst_deviation, deviation)
        assert deviation < 1e-6, f"{item.item_id}: option probs sum to {probs.sum()!r}"
        assert decision.argmax_index == int(np.argmax(probs))
        assert decision.gold_key == canonical.gold
        assert decision.gold_key in decision.option_keys
        if item.qtype is QuestionType.SCORE:
            assert decision.expected_level is not None
            assert 1.0 <= decision.expected_level <= canonical.n_options
        else:
            assert decision.expected_level is None
        assert decision.prompt_tokens > 0
        assert decision.latency_s >= 0.0
    assert {"noul", "choice", "score"} <= seen_qtypes
    assert {2, 64} <= seen_counts
    print(f"[distribution] worst |sum(p) - 1| over {len(items)} items: {worst_deviation:.2e}")


@GPU_MARK
@GPU
def test_option_permutation_permutes_the_output(fresh_model, dev_items) -> None:
    """(2) Permuting an item's options permutes the distribution, mapped by content."""
    candidates = [i for i in dev_items if i.qtype is QuestionType.CHOICE and i.n_options >= 4]
    item = candidates[0] if candidates else synthetic_item(n_options=4, seed=5)
    canonical, _ = canonicalise_options(item)
    n = canonical.n_options
    reference = fresh_model.predict([canonical], temperature={})[0]
    base_probs = np.asarray(reference.probs, dtype=np.float64)

    permutations = [[(i + 2) % n for i in range(n)], list(reversed(range(n)))]
    deviations: list[float] = []
    moved_any = False
    for permutation in permutations:
        permuted = reorder_options(canonical, permutation)
        # the gold follows its content
        assert permuted.options[permuted.gold_index].label == (
            canonical.options[canonical.gold_index].label
        )
        got = fresh_model.predict([permuted], temperature={})[0]
        got_probs = np.asarray(got.probs, dtype=np.float64)
        for new_index, source_index in enumerate(permutation):
            # mapping by *content*: the label at the new position is the old position's label
            assert got.option_keys[new_index] == canonical.options[new_index].key
            assert permuted.options[new_index].label == canonical.options[source_index].label
            deviations.append(abs(float(got_probs[new_index]) - float(base_probs[source_index])))
        moved_any = moved_any or not np.allclose(got_probs, base_probs, atol=1e-6)
    assert moved_any, "permuting the options left the output unchanged — the head ignores content"
    worst = max(deviations)
    print(f"[permutation] n={n} permutations={len(permutations)} worst |dp|={worst:.4f}")
    assert worst <= 0.25, f"permutation changed the distribution by {worst:.4f} (tolerance 0.25)"


@GPU_MARK
@GPU
def test_adapter_disabled_generation_is_byte_identical(fresh_model, test_config) -> None:
    """(3) Adapter off == an untouched base, on 20 fixed prompts, greedily."""
    transformers = pytest.importorskip("transformers")

    tokenizer = fresh_model.tokenizer
    if DEV_PATH.exists():
        items = read_items(DEV_PATH, limit=64, stride=11)
        items.sort(key=lambda i: i.item_id)
        prompt_items = items[:20]
    else:  # pragma: no cover - data tree absent
        prompt_items = [
            synthetic_item(n_options=2 + (i % 5), seed=i) for i in range(20)
        ]
    prompts = []
    for item in prompt_items:
        canonical, _ = canonicalise_options(item)
        prompts.append(render_prompt(canonical, tokenizer, fresh_model.variant))
    assert len(prompts) == 20

    # Reference first: the untouched base, loaded separately from the same checkpoint.
    pristine = transformers.AutoModelForCausalLM.from_pretrained(
        test_config.base_model, dtype=torch.bfloat16, device_map=test_config.device
    )
    pristine.eval()
    reference: list[list[int]] = []
    with torch.inference_mode():
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {k: v.to(test_config.device) for k, v in encoded.items()}
            generated = pristine.generate(**encoded, max_new_tokens=8, do_sample=False)
            reference.append(
                [int(t) for t in generated[0, encoded["input_ids"].shape[1] :]]
            )

    # Control: a *changed* adapter must change generation, or this test cannot detect anything.
    # lora_B is zero at initialisation, so perturb it and restore the exact tensors afterwards.
    lora_b = [
        (name, param)
        for name, param in fresh_model.peft_model.named_parameters()
        if name.endswith("lora_B.default.weight")
    ]
    assert lora_b, "no LoRA B matrices found"
    saved = [param.detach().clone() for _, param in lora_b]
    with torch.no_grad():
        for _, param in lora_b:
            param.normal_(0.0, 0.05)
    with torch.inference_mode():
        perturbed = fresh_model.generate_greedy(prompts, max_new_tokens=8, seed=0)
    with torch.no_grad():
        for (_, param), before in zip(lora_b, saved, strict=True):
            param.copy_(before)
    assert perturbed != reference, "a perturbed adapter produced the base's tokens — check is void"

    with fresh_model.adapter_disabled():
        ours = fresh_model.generate_greedy(prompts, max_new_tokens=8, seed=0)

    del pristine
    torch.cuda.empty_cache()

    assert ours == reference, "adapter-off generation differs from the untouched base"
    for token_ids, ref_ids in zip(ours, reference, strict=True):
        assert fresh_model.decode(token_ids) == tokenizer.decode(ref_ids)


@GPU_MARK
@GPU
def test_overfit_64_items(test_config) -> None:
    """(4) 64 training items reach >= 0.95 training accuracy (a learning-path smoke test)."""
    if TRAIN_PATH.exists():
        items = read_items(TRAIN_PATH, limit=64, stride=3000)
        items.sort(key=lambda i: i.item_id)
        items = items[:64]
    else:  # pragma: no cover - data tree absent
        items = [synthetic_item(n_options=4, seed=i) for i in range(64)]
    assert len(items) == 64

    # the shipped learning rates; 800 steps is what the S7 measurements showed is needed for
    # 64 items (400 steps reached 60/64, 800 steps 64/64 at the same seed and rates)
    config = dataclasses.replace(test_config, batch_size=8, max_batch_tokens=8192)
    steps = 800
    model = MedDecideModel(
        config.base_model,
        lora=config.lora,
        head_settings=config.head,
        dtype=config.dtype,
        device=config.device,
        variant=config.variant,
        max_prompt_tokens=config.max_prompt_tokens,
    )
    try:
        trainer = Trainer(model, config, log_path=None)
        result = trainer.train(items, steps=steps)
        assert result.steps == steps
        scored = model.score_items(
            items,
            batch_size=config.batch_size,
            max_batch_tokens=config.max_batch_tokens,
            max_prompt_tokens=config.max_prompt_tokens,
        )
        metrics = scored.metrics()
        first = result.loss_at(1)
        last = result.loss_at(steps)
        assert first is not None and last is not None
        assert last < first, f"training loss did not decrease: {first:.4f} -> {last:.4f}"
        assert metrics["accuracy"] >= 0.95, (
            f"training accuracy {metrics['accuracy']:.4f} "
            f"({metrics['n_correct']}/{metrics['n']}) after {result.steps} steps; "
            f"loss {first:.3f} -> {last:.3f}"
        )
        # an extra, reported-not-required check: after training, permuting a memorised item's
        # options should permute its distribution tightly (order-invariant content scoring)
        canonical, _ = canonicalise_options(items[0])
        base = model.predict([canonical], temperature={})[0]
        permutation = list(reversed(range(canonical.n_options)))
        moved = reorder_options(canonical, permutation)
        got = model.predict([moved], temperature={})[0]
        worst = max(
            abs(got.probs[j] - base.probs[src]) for j, src in enumerate(permutation)
        )
        print(f"[overfit] accuracy={metrics['accuracy']:.4f} permuted-item max |dp|={worst:.4f}")
    finally:
        del model
        torch.cuda.empty_cache()
