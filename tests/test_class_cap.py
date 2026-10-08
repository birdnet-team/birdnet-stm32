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


def field_dataset(focal, sites):
    """Focal files plus field clips: sites = {site: {recording: n_segments}} for class 'c'."""
    files = [f"/d/c/f{i}.wav" for i in range(focal)]
    table = {}
    for site, recs in sites.items():
        for rec, n in recs.items():
            for j in range(n):
                stem = f"{site}_{rec}_{j}"
                files.append(f"/d/c/{stem}.wav")
                table[stem] = (site, rec)
    return files, table


def split(files, table):
    stems = [f.split("/")[-1][:-4] for f in files]
    return [s for s in stems if s not in table], [s for s in stems if s in table]


def test_field_clips_fill_at_most_the_share_with_two_per_recording():
    files, table = field_dataset(20, {"s1": {f"r{i}": 3 for i in range(10)}})
    passes = ClassCappedPasses(files, cap=10, rng=random.Random(0), field=table, field_share=0.5)
    out = passes.next_pass()
    focal, fld = split(out, table)
    assert len(passes) == len(out) == 10 and len(fld) == 5 and len(focal) == 5
    assert max(Counter(table[s][1] for s in fld).values()) <= 2


def test_field_clips_alternate_sites_so_one_site_cannot_dominate():
    files, table = field_dataset(20, {"big": {f"r{i}": 5 for i in range(10)}, "small": {"only": 10}})
    out = ClassCappedPasses(files, cap=8, rng=random.Random(1), field=table, field_share=0.5).next_pass()
    _, fld = split(out, table)
    assert Counter(table[s][0] for s in fld) == {"big": 2, "small": 2}


def test_field_fills_what_focal_cannot_but_keeps_the_recording_limit():
    files, table = field_dataset(1, {"s1": {f"r{i}": 4 for i in range(3)}})
    passes = ClassCappedPasses(files, cap=10, rng=random.Random(0), field=table, field_share=0.5)
    focal, fld = split(passes.next_pass(), table)
    assert len(focal) == 1 and len(fld) == 6  # 3 recordings x 2, short of the 9 the quota would allow
    assert len(passes) == 7


def test_every_field_clip_is_drawn_over_passes():
    files, table = field_dataset(10, {"s1": {"r0": 6, "r1": 6}})
    passes = ClassCappedPasses(files, cap=8, rng=random.Random(0), field=table, field_share=0.5)
    seen = set()
    for _ in range(3):  # 4 field clips per pass (2 per recording), 12 in all
        seen |= set(split(passes.next_pass(), table)[1])
    assert seen == set(table)


def test_noise_folders_ignore_the_field_table():
    files = [f"/d/noise/n{i}.wav" for i in range(5)]
    table = {f"n{i}": ("s", "r") for i in range(5)}
    passes = ClassCappedPasses(files, cap=2, rng=random.Random(0), field=table)
    assert len(passes) == 5 and len(passes.next_pass()) == 5
