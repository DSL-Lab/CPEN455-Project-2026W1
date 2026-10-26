"""Regressions for the distance used by the F1 accuracy analysis."""
import json
from pathlib import Path

import pytest

from cpen455kv.data import answer_distance, encoder, load_item


def test_nag_distance_uses_last_complete_mention_in_cbt_test_0784():
    path = Path(__file__).resolve().parents[1] / "data" / "cbt_ne_test.jsonl"
    with path.open() as source:
        row = next(row for line in source
                   if (row := json.loads(line))["id"] == "cbt-ne-test-0784")
    question = row["questions"][0]
    assert question["answer"] == "Nag"
    # Later Nagaina occurrences must not move the last Nag mention forward.
    end = row["context"].index("Nag waved") + len("Nag")
    assert row["context"].rfind("Nagaina") > end
    expected = len(encoder().encode(row["context"][end:]))
    assert expected == 128
    assert load_item(row)["questions"][0]["distance"] == expected


@pytest.mark.parametrize("text,answer", [
    ("xNag Nagaina Nag_1 Nag2 Nagé", "Nag"),
    ("alice waited. ALICE left.", "Alice"),
    ("Zoe\u0308 waited.", "Zoë"),
    ("Nothing here.", ""),
    ("", "Alice"),
])
def test_answer_distance_requires_an_exact_whole_word(text, answer):
    assert answer_distance(encoder().encode(text), answer) is None


def test_answer_distance_maps_unicode_offsets_to_original_tokens():
    enc = encoder()
    text = "你好, Zoë met Zoë."
    # Several GPT-2 tokens contain only part of a UTF-8 character.
    assert any(len(enc.decode_single_token_bytes(token)) < 3
               for token in enc.encode("你"))
    assert answer_distance(enc.encode(text), "Zoë") == 1


def test_answer_distance_excludes_token_overlapping_mention_end():
    enc = encoder()
    text = "Alice waited... Bob arrived."
    assert len(enc.encode("...")) == 1
    # The first period belongs to the mention; the whole '...' token overlaps it.
    # Re-encoding the remaining '.. Bob arrived.' would incorrectly count 4.
    assert answer_distance(enc.encode(text), "Alice waited.") == 3


def test_answer_distance_accepts_possessive_word_boundary():
    enc = encoder()
    assert answer_distance(enc.encode("Nag's head moved."), "Nag") == 4
