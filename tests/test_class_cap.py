"""Per-epoch class cap: balanced passes that neither repeat nor discard files."""

import random
from collections import Counter

from birdnet_stm32.data.generator import ClassCappedPasses


def paths(counts):
    return [f"/d/{c}/{c}_{i}.wav" for c, n in counts.items() for i in range(n)]


def per_class(files):
    return Counter(p.split("/")[-2] for p in files)


def test_caps_large_classes_and_keeps_small_ones_whole():
    passes = ClassCappedPasses(paths({"big": 10, "small": 3}), cap=4, rng=random.Random(0))
    assert len(passes) == 7
    assert per_class(passes.next_pass()) == {"big": 4, "small": 3}


def test_a_pass_never_repeats_a_file():
    passes = ClassCappedPasses(paths({"big": 10, "small": 3}), cap=4, rng=random.Random(0))
    for _ in range(5):
        p = passes.next_pass()
        assert len(p) == len(set(p))


def test_a_large_class_uses_every_file_before_reusing_any():
    files = paths({"big": 10})
    passes = ClassCappedPasses(files, cap=4, rng=random.Random(0))
    seen = passes.next_pass() + passes.next_pass()  # 8 of 10, all distinct
    assert len(set(seen)) == 8
    third = passes.next_pass()  # the two not yet drawn, then two from a fresh cycle
    assert set(files) - set(seen) <= set(third)


def test_cap_larger_than_every_class_is_a_plain_shuffle():
    files = paths({"a": 3, "b": 2})
    passes = ClassCappedPasses(files, cap=100, rng=random.Random(0))
    assert sorted(passes.next_pass()) == sorted(files)


def test_noise_folders_are_never_capped():
    passes = ClassCappedPasses(paths({"big": 10, "noise": 9}), cap=4, rng=random.Random(0))
    assert len(passes) == 13
    assert per_class(passes.next_pass()) == {"big": 4, "noise": 9}
