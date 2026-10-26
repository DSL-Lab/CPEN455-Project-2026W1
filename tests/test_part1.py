"""Part 1: format of student/part1_answers.yaml. The values are checked by hidden tests."""
import os

import yaml

PATH = os.path.join(os.path.dirname(__file__), "..", "student", "part1_answers.yaml")


def test_answers_file_is_complete():
    a = yaml.safe_load(open(PATH))
    shape = a["q1_key_cache_shape"]["shape"]
    assert isinstance(shape, list) and len(shape) == 4 and all(isinstance(x, int) for x in shape)
    assert isinstance(a["q2_kv_memory"]["bytes"], int)
    assert isinstance(a["q2_kv_memory"]["break_even_budget"], int)
    for q in ("q3_why_not_queries", "q4_attention_flops"):
        assert isinstance(a[q], str) and len(a[q].split()) >= 10, f"{q} needs a written answer"
