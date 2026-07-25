from dataclasses import asdict
import json

from quran_transcript import Aya, quran_phonetizer, MoshafAttributes
import torch
import pytest
from librosa.core import load

from quran_muaalem.decode import (
    ctc_decode,
    CTCDecodeOut,
    multilevel_greedy_decode,
    align_sequence,
    align_predicted_sequence,
)
from quran_muaalem.muaalem_typing import Unit, Sifa, SingleUnit
from quran_muaalem.modeling.multi_level_tokenizer import MultiLevelTokenizer
from quran_muaalem.inference import format_sifat, Muaalem
from quran_muaalem.explain import explain_for_terminal


@pytest.mark.parametrize(
    f"ref, pred, exp_out, exp_mask",
    [
        (
            [0, 1, 2, 3],
            [3, 2, 1, 0],
            [3, 2, 1, 0],
            [1, 1, 1, 1],
        ),
        (
            [0, 1, 2, 3],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3],
            [1, 1, 1, 1, 0],
        ),
        (
            [0, 1, 2, 3],
            [2, 3, 4],
            [-100, -100, 2, 3],
            [1, 1, 0],
        ),
        (
            [0, 1, 2, 3],
            [-1, 0, 1, 2, 4],
            [0, 1, 2, 4],
            [0, 1, 1, 1, 1],
        ),
        (
            [0, 1, 2, 3],
            [1, 2, 3],
            [-100, 1, 2, 3],
            [1, 1, 1],
        ),
        (
            [0, 1, 2, 3],
            [],
            [-100, -100, -100, -100],
            [],
        ),
    ],
)
def test_align_predicted_sequence(ref, pred, exp_out, exp_mask):
    exp_mask = [bool(i) for i in exp_mask]
    out, mask = align_predicted_sequence(ref, pred)
    print(f"Ref: {ref}")
    print(f"Pred: {pred}")
    print(f"Out: {out}")
    print(f"EXP: {exp_out}")
    print(f"Mask: {mask}")
    print(f"EXP Mask: {exp_mask}")

    assert out == exp_out
    assert mask == exp_mask


@pytest.mark.parametrize(
    "batch_ids, batch_probs, ex_batch_ids, ex_batch_probs",
    [
        (
            [
                [1],
            ],
            [
                [1.0],
            ],
            [
                [1],
            ],
            [
                [1.0],
            ],
        ),
        (
            [
                [1, 1, 0, 2, 2],
            ],
            [
                [0.3, 0.2, 0.9, 0.6, 0.8],
            ],
            [
                [1, 2],
            ],
            [
                [0.25, 0.7],
            ],
        ),
        # all diffrent
        (
            [
                [1, 2, 3, 4, 5],
            ],
            [
                [0.3, 0.2, 0.9, 0.6, 0.8],
            ],
            [
                [1, 2, 3, 4, 5],
            ],
            [
                [0.3, 0.2, 0.9, 0.6, 0.8],
            ],
        ),
        # all diffrent with blank
        (
            [
                [0, 0, 0, 1, 2, 3, 4, 5, 0, 0, 0],
            ],
            [
                [0.9, 0.9, 0.8, 0.3, 0.2, 0.9, 0.6, 0.8, 0.9, 0.9, 0.8],
            ],
            [
                [1, 2, 3, 4, 5],
            ],
            [
                [0.3, 0.2, 0.9, 0.6, 0.8],
            ],
        ),
        # all diffrent with blank in between
        (
            [
                [0, 0, 0, 1, 2, 0, 3, 4, 5, 0, 0, 0],
            ],
            [
                [0.9, 0.9, 0.8, 0.3, 0.2, 1.0, 0.9, 0.6, 0.8, 0.9, 0.9, 0.8],
            ],
            [
                [1, 2, 3, 4, 5],
            ],
            [
                [0.3, 0.2, 0.9, 0.6, 0.8],
            ],
        ),
        # all diffrent with blank in between
        (
            [
                [0, 0, 0, 1, 1, 1, 2, 0, 3, 4, 5, 0, 0, 0],
            ],
            [
                [0.9, 0.9, 0.8, 0.3, 0.2, 0.7, 0.8, 1.0, 0.9, 0.6, 0.8, 0.9, 0.9, 0.8],
            ],
            [
                [1, 2, 3, 4, 5],
            ],
            [
                [0.4, 0.8, 0.9, 0.6, 0.8],
            ],
        ),
        # complete example
        (
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2],
                [0, 0, 0, 1, 1, 1, 2, 0, 3, 4, 5, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.2, 0.9, 0.6, 0.8],
                [0.9, 0.9, 0.8, 0.3, 0.2, 0.7, 0.8, 1.0, 0.9, 0.6, 0.8, 0.9, 0.9, 0.8],
            ],
            [
                [1, 2],
                [1, 2, 3, 4, 5],
            ],
            [
                [0.25, 0.7],
                [0.4, 0.8, 0.9, 0.6, 0.8],
            ],
        ),
    ],
)
def test_ctc_decode(batch_ids, batch_probs, ex_batch_ids, ex_batch_probs):
    outs = ctc_decode(
        torch.LongTensor(batch_ids),
        torch.FloatTensor(batch_probs),
        collapse_consecutive=True,
    )
    for idx in range(len(batch_ids)):
        print(f"IDS: {outs[idx].ids}")
        print(f"EXP IDS: {torch.LongTensor(ex_batch_ids[idx])}")
        print(f"Probs: {outs[idx].p}")
        print(f"EXP Probs: {torch.FloatTensor(ex_batch_probs[idx])}")
        torch.testing.assert_close(outs[idx].ids, torch.LongTensor(ex_batch_ids[idx]))
        torch.testing.assert_close(outs[idx].p, torch.FloatTensor(ex_batch_probs[idx]))


# @pytest.mark.parametrize(
#     "level_to_probs, level_to_ref_ids, ex_level_to_units",
#     [
#         (
#             {
#                 "phonemes": [
#                     [
#                         [0, 1, 0, 0, 0],
#                     ],
#                 ]
#             },
#             {
#                 "phonemes": [
#                     [
#                         [0, 1, 0, 0, 0, 3, 4],
#                     ],
#                 ]
#             },
#             {
#                 "phonemes": [
#                     Unit(
#                         text="a",
#                         probs=torch.FloatTensor([1.0]),
#                         ids=torch.LongTensor([1]),
#                     )
#                 ]
#             },
#         ),
#         # biger example
#         (
#             {
#                 "phonemes": [
#                     [
#                         [0, 1, 0, 0, 0],
#                         [0, 0, 1, 0, 0],
#                         [0, 0, 0, 1, 0],
#                         [0, 0, 0, 0, 1],
#                     ],
#                 ]
#             },
#             {
#                 "phonemes": [
#                     [
#                         [0, 1, 0, 0, 0, 4],
#                         [0, 0, 1, 0, 0, 3],
#                         [0, 0, 0, 1, 0, 2],
#                         [0, 0, 0, 0, 1, 4],
#                     ],
#                 ]
#             },
#             {
#                 "phonemes": [
#                     Unit(
#                         text="abcd",
#                         probs=torch.FloatTensor([1.0, 1.0, 1.0, 1.0]),
#                         ids=torch.LongTensor([1, 2, 3, 4]),
#                     )
#                 ]
#             },
#         ),
#         # diffrent probs
#         (
#             {
#                 "phonemes": [
#                     [
#                         [0, 1, 0, 0, 0],
#                         [0, 0, 1, 0, 0],
#                         [0, 0, 0.6, 0.4, 0],
#                         [0, 0, 0, 0, 1],
#                     ],
#                 ]
#             },
#             {
#                 "phonemes": [
#                     [
#                         [0, 1, 0, 0, 0],
#                         [0, 0, 1, 0, 0],
#                         [0, 0, 0.6, 0.4, 0],
#                         [0, 0, 0, 0, 1],
#                     ],
#                 ]
#             },
#             {
#                 "phonemes": [
#                     Unit(
#                         text="abd",
#                         probs=torch.FloatTensor([1.0, 0.8, 1.0]),
#                         ids=torch.LongTensor([1, 2, 4]),
#                     ),
#                 ]
#             },
#         ),
#         # diffrent probs and levels
#         (
#             {
#                 "phonemes": [
#                     [
#                         [0, 1, 0, 0, 0],
#                         [0, 0, 1, 0, 0],
#                         [0, 0, 0.6, 0.4, 0],
#                         [0, 0, 0, 0, 1],
#                     ],
#                 ],
#                 "hams": [
#                     [
#                         [0, 1, 0],
#                         [0, 0.8, 0.2],
#                     ],
#                 ],
#             },
#             {
#                 "phonemes": [
#                     [
#                         [0, 1, 0, 0, 0],
#                         [0, 0, 1, 0, 0],
#                         [0, 0, 0.6, 0.4, 0],
#                         [0, 0, 0, 0, 1],
#                     ],
#                 ],
#                 "hams": [
#                     [
#                         [0, 1, 0],
#                         [0, 0.8, 0.2],
#                     ],
#                 ],
#             },
#             {
#                 "phonemes": [
#                     Unit(
#                         text="abd",
#                         probs=torch.FloatTensor([1.0, 0.8, 1.0]),
#                         ids=torch.LongTensor([1, 2, 4]),
#                     )
#                 ],
#                 "hams": [
#                     Unit(
#                         text="A",
#                         probs=torch.FloatTensor([0.9]),
#                         ids=torch.LongTensor([1]),
#                     )
#                 ],
#             },
#         ),
#         # # diffrent probs and multiple sequences
#         # (
#         #     {
#         #         "phonemes": [
#         #             [
#         #                 [0, 1, 0, 0, 0],
#         #                 [0, 0, 1, 0, 0],
#         #                 [0, 0, 0.6, 0.4, 0],
#         #                 [0, 0, 0, 0, 1],
#         #             ],
#         #             [
#         #                 [0, 1, 0, 0, 0],
#         #                 [0, 1, 0, 0, 0],
#         #                 [0, 1, 0, 0, 0],
#         #                 [0, 1, 0, 0, 0],
#         #             ],
#         #         ],
#         #         "hams": [
#         #             [
#         #                 [0, 1, 0],
#         #                 [0, 0.8, 0.2],
#         #             ],
#         #         ],
#         #     },
#         #     {
#         #         "phonemes": [
#         #             Unit(
#         #                 text="abd",
#         #                 probs=torch.FloatTensor([1.0, 0.8, 1.0]),
#         #                 ids=torch.LongTensor([1, 2, 4]),
#         #             ),
#         #             Unit(
#         #                 text="a",
#         #                 probs=torch.FloatTensor([1.0]),
#         #                 ids=torch.LongTensor([1]),
#         #             ),
#         #         ],
#         #         "hams": [
#         #             Unit(
#         #                 text="A",
#         #                 probs=torch.FloatTensor([0.9]),
#         #                 ids=torch.LongTensor([1]),
#         #             )
#         #         ],
#         #     },
#         # ),
#     ],
# )
# def test_multilevel_greedy_decode(level_to_probs, level_to_ref_ids, ex_level_to_units):
#     level_to_vocab = {
#         "phonemes": {1: "a", 2: "b", 3: "c", 4: "d"},
#         "hams": {1: "A", 2: "B"},
#     }
#     level_to_probs = {l: torch.FloatTensor(p) for l, p in level_to_probs.items()}
#     level_to_ref_ids = {l: torch.LongTensor(i) for l, i in level_to_ref_ids.items()}
#     for level in level_to_probs:
#         print(level_to_probs[level].shape)
#
#     level_to_units = multilevel_greedy_decode(
#         level_to_probs,
#         level_to_vocab,
#         level_to_ref_ids,
#     )
#     assert len(ex_level_to_units) == len(level_to_units)
#     assert set(ex_level_to_units.keys()) == set(level_to_units.keys())
#
#     for level in ex_level_to_units:
#         assert len(ex_level_to_units[level]) == len(level_to_units[level])
#         for unit, ex_unit in zip(level_to_units[level], ex_level_to_units[level]):
#             print(f"OUT UNIT: {unit}")
#             print(f"EXP UNIT: {ex_unit}")
#             assert unit.text == ex_unit.text
#             torch.testing.assert_close(unit.probs, ex_unit.probs)


@pytest.mark.parametrize(
    "seq, target_len, min_repeat, ex_out",
    [
        (
            [1, 1, 1, 1, 1],
            4,
            3,
            [0],
        ),
        (
            [1, 1, 1, 1, 1],
            3,
            3,
            [0, 1],
        ),
        (
            [1, 1, 1, 1, 1],
            3,
            5,
            [0, 1],
        ),
        (
            [1, 1, 1, 1, 1],
            3,
            6,
            [],
        ),
        (
            [1, 0, 1, 1, 0],
            3,
            1,
            [],
        ),
        (
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
            10,
            5,
            [3, 4],
        ),
    ],
)
def test_alilgn_sequence(seq, target_len, min_repeat, ex_out):
    out = align_sequence(seq, target_len, min_repeat=min_repeat)
    assert out == ex_out


@pytest.mark.slow
@pytest.mark.parametrize(
    "level_to_units, chunked_phonemes_batch, ex_sifat_batch",
    [
        (
            {
                "hams_or_jahr": [Unit(text="", probs=[0.1], ids=[1])],
                "shidda_or_rakhawa": [Unit(text="", probs=[0.2], ids=[1])],
                "tafkheem_or_taqeeq": [Unit(text="", probs=[0.3], ids=[1])],
                "itbaq": [Unit(text="", probs=[0.4], ids=[1])],
                "safeer": [Unit(text="", probs=[0.5], ids=[1])],
                "qalqla": [Unit(text="", probs=[0.6], ids=[1])],
                "tikraar": [Unit(text="", probs=[0.7], ids=[1])],
                "tafashie": [Unit(text="", probs=[0.8], ids=[1])],
                "istitala": [Unit(text="", probs=[0.9], ids=[1])],
                "ghonna": [Unit(text="", probs=[0.1], ids=[1])],
            },
            [
                ["a"],
            ],
            [
                [
                    Sifa(
                        phonemes_group="a",
                        hams_or_jahr=SingleUnit(text="hams", prob=0.1, idx=1),
                        shidda_or_rakhawa=SingleUnit(text="shadeed", prob=0.2, idx=1),
                        tafkheem_or_taqeeq=SingleUnit(text="mofakham", prob=0.3, idx=1),
                        itbaq=SingleUnit(text="monfateh", prob=0.4, idx=1),
                        safeer=SingleUnit(text="safeer", prob=0.5, idx=1),
                        qalqla=SingleUnit(text="moqalqal", prob=0.6, idx=1),
                        tikraar=SingleUnit(text="mokarar", prob=0.7, idx=1),
                        tafashie=SingleUnit(text="motafashie", prob=0.8, idx=1),
                        istitala=SingleUnit(text="mostateel", prob=0.9, idx=1),
                        ghonna=SingleUnit(text="maghnoon", prob=0.1, idx=1),
                    )
                ]
            ],
        ),
        # seq len of 2
        (
            {
                "hams_or_jahr": [Unit(text="", probs=[0.1, 0.9], ids=[1, 2])],
                "shidda_or_rakhawa": [Unit(text="", probs=[0.2, 0.8], ids=[1, 2])],
                "tafkheem_or_taqeeq": [Unit(text="", probs=[0.3, 0.7], ids=[1, 2])],
                "itbaq": [Unit(text="", probs=[0.4, 0.6], ids=[1, 2])],
                "safeer": [Unit(text="", probs=[0.5, 0.5], ids=[1, 2])],
                "qalqla": [Unit(text="", probs=[0.6, 0.4], ids=[1, 2])],
                "tikraar": [Unit(text="", probs=[0.7, 0.3], ids=[1, 2])],
                "tafashie": [Unit(text="", probs=[0.8, 0.2], ids=[1, 2])],
                "istitala": [Unit(text="", probs=[0.9, 0.1], ids=[1, 2])],
                "ghonna": [Unit(text="", probs=[0.1, 0.9], ids=[1, 2])],
            },
            [
                ["a", "b"],
            ],
            [
                [
                    Sifa(
                        phonemes_group="a",
                        hams_or_jahr=SingleUnit(text="hams", prob=0.1, idx=1),
                        shidda_or_rakhawa=SingleUnit(text="shadeed", prob=0.2, idx=1),
                        tafkheem_or_taqeeq=SingleUnit(text="mofakham", prob=0.3, idx=1),
                        itbaq=SingleUnit(text="monfateh", prob=0.4, idx=1),
                        safeer=SingleUnit(text="safeer", prob=0.5, idx=1),
                        qalqla=SingleUnit(text="moqalqal", prob=0.6, idx=1),
                        tikraar=SingleUnit(text="mokarar", prob=0.7, idx=1),
                        tafashie=SingleUnit(text="motafashie", prob=0.8, idx=1),
                        istitala=SingleUnit(text="mostateel", prob=0.9, idx=1),
                        ghonna=SingleUnit(text="maghnoon", prob=0.1, idx=1),
                    ),
                    Sifa(
                        phonemes_group="b",
                        hams_or_jahr=SingleUnit(text="jahr", prob=0.9, idx=2),
                        shidda_or_rakhawa=SingleUnit(text="between", prob=0.8, idx=2),
                        tafkheem_or_taqeeq=SingleUnit(text="moraqaq", prob=0.7, idx=2),
                        itbaq=SingleUnit(text="motbaq", prob=0.6, idx=2),
                        safeer=SingleUnit(text="no_safeer", prob=0.5, idx=2),
                        qalqla=SingleUnit(text="not_moqalqal", prob=0.4, idx=2),
                        tikraar=SingleUnit(text="not_mokarar", prob=0.3, idx=2),
                        tafashie=SingleUnit(text="not_motafashie", prob=0.2, idx=2),
                        istitala=SingleUnit(text="not_mostateel", prob=0.1, idx=2),
                        ghonna=SingleUnit(text="not_maghnoon", prob=0.9, idx=2),
                    ),
                ]
            ],
        ),
        # seq len of 2 with None
        (
            {
                "hams_or_jahr": [Unit(text="", probs=[0.1, 0.9], ids=[1, 2])],
                "shidda_or_rakhawa": [Unit(text="", probs=[0.2, 0.8], ids=[1, 2])],
                "tafkheem_or_taqeeq": [Unit(text="", probs=[0.3, 0.7], ids=[1, 2])],
                "itbaq": [Unit(text="", probs=[0.4, 0.6], ids=[1, 2])],
                "safeer": [Unit(text="", probs=[0.5, 0.5], ids=[1, 2])],
                "qalqla": [Unit(text="", probs=[0.6, 0.4], ids=[1, 2])],
                "tikraar": [Unit(text="", probs=[0.7, 0.3], ids=[1, 2])],
                "tafashie": [Unit(text="", probs=[0.8, 0.2], ids=[1, 2])],
                "istitala": [Unit(text="", probs=[0.9, 0.1], ids=[1, 2])],
                "ghonna": [Unit(text="", probs=[0.1], ids=[1])],
            },
            [
                ["a", "b"],
            ],
            [
                [
                    Sifa(
                        phonemes_group="a",
                        hams_or_jahr=SingleUnit(text="hams", prob=0.1, idx=1),
                        shidda_or_rakhawa=SingleUnit(text="shadeed", prob=0.2, idx=1),
                        tafkheem_or_taqeeq=SingleUnit(text="mofakham", prob=0.3, idx=1),
                        itbaq=SingleUnit(text="monfateh", prob=0.4, idx=1),
                        safeer=SingleUnit(text="safeer", prob=0.5, idx=1),
                        qalqla=SingleUnit(text="moqalqal", prob=0.6, idx=1),
                        tikraar=SingleUnit(text="mokarar", prob=0.7, idx=1),
                        tafashie=SingleUnit(text="motafashie", prob=0.8, idx=1),
                        istitala=SingleUnit(text="mostateel", prob=0.9, idx=1),
                        ghonna=SingleUnit(text="maghnoon", prob=0.1, idx=1),
                    ),
                    Sifa(
                        phonemes_group="b",
                        hams_or_jahr=SingleUnit(text="jahr", prob=0.9, idx=2),
                        shidda_or_rakhawa=SingleUnit(text="between", prob=0.8, idx=2),
                        tafkheem_or_taqeeq=SingleUnit(text="moraqaq", prob=0.7, idx=2),
                        itbaq=SingleUnit(text="motbaq", prob=0.6, idx=2),
                        safeer=SingleUnit(text="no_safeer", prob=0.5, idx=2),
                        qalqla=SingleUnit(text="not_moqalqal", prob=0.4, idx=2),
                        tikraar=SingleUnit(text="not_mokarar", prob=0.3, idx=2),
                        tafashie=SingleUnit(text="not_motafashie", prob=0.2, idx=2),
                        istitala=SingleUnit(text="not_mostateel", prob=0.1, idx=2),
                        ghonna=None,
                    ),
                ]
            ],
        ),
        # seq len of 2 with None with bathing
        (
            {
                "hams_or_jahr": [
                    Unit(text="", probs=[0.1], ids=[1]),
                    Unit(text="", probs=[0.1, 0.9], ids=[1, 2]),
                ],
                "shidda_or_rakhawa": [
                    Unit(text="", probs=[0.2], ids=[1]),
                    Unit(text="", probs=[0.2, 0.8], ids=[1, 2]),
                ],
                "tafkheem_or_taqeeq": [
                    Unit(text="", probs=[0.3], ids=[1]),
                    Unit(text="", probs=[0.3, 0.7], ids=[1, 2]),
                ],
                "itbaq": [
                    Unit(text="", probs=[0.4], ids=[1]),
                    Unit(text="", probs=[0.4, 0.6], ids=[1, 2]),
                ],
                "safeer": [
                    Unit(text="", probs=[0.5], ids=[1]),
                    Unit(text="", probs=[0.5, 0.5], ids=[1, 2]),
                ],
                "qalqla": [
                    Unit(text="", probs=[0.6], ids=[1]),
                    Unit(text="", probs=[0.6, 0.4], ids=[1, 2]),
                ],
                "tikraar": [
                    Unit(text="", probs=[0.7], ids=[1]),
                    Unit(text="", probs=[0.7, 0.3], ids=[1, 2]),
                ],
                "tafashie": [
                    Unit(text="", probs=[0.8], ids=[1]),
                    Unit(text="", probs=[0.8, 0.2], ids=[1, 2]),
                ],
                "istitala": [
                    Unit(text="", probs=[0.9], ids=[1]),
                    Unit(text="", probs=[0.9, 0.1], ids=[1, 2]),
                ],
                "ghonna": [
                    Unit(text="", probs=[0.1], ids=[1]),
                    Unit(text="", probs=[0.1], ids=[1]),
                ],
            },
            [
                ["a"],
                ["a", "b"],
            ],
            [
                [
                    Sifa(
                        phonemes_group="a",
                        hams_or_jahr=SingleUnit(text="hams", prob=0.1, idx=1),
                        shidda_or_rakhawa=SingleUnit(text="shadeed", prob=0.2, idx=1),
                        tafkheem_or_taqeeq=SingleUnit(text="mofakham", prob=0.3, idx=1),
                        itbaq=SingleUnit(text="monfateh", prob=0.4, idx=1),
                        safeer=SingleUnit(text="safeer", prob=0.5, idx=1),
                        qalqla=SingleUnit(text="moqalqal", prob=0.6, idx=1),
                        tikraar=SingleUnit(text="mokarar", prob=0.7, idx=1),
                        tafashie=SingleUnit(text="motafashie", prob=0.8, idx=1),
                        istitala=SingleUnit(text="mostateel", prob=0.9, idx=1),
                        ghonna=SingleUnit(text="maghnoon", prob=0.1, idx=1),
                    )
                ],
                [
                    Sifa(
                        phonemes_group="a",
                        hams_or_jahr=SingleUnit(text="hams", prob=0.1, idx=1),
                        shidda_or_rakhawa=SingleUnit(text="shadeed", prob=0.2, idx=1),
                        tafkheem_or_taqeeq=SingleUnit(text="mofakham", prob=0.3, idx=1),
                        itbaq=SingleUnit(text="monfateh", prob=0.4, idx=1),
                        safeer=SingleUnit(text="safeer", prob=0.5, idx=1),
                        qalqla=SingleUnit(text="moqalqal", prob=0.6, idx=1),
                        tikraar=SingleUnit(text="mokarar", prob=0.7, idx=1),
                        tafashie=SingleUnit(text="motafashie", prob=0.8, idx=1),
                        istitala=SingleUnit(text="mostateel", prob=0.9, idx=1),
                        ghonna=SingleUnit(text="maghnoon", prob=0.1, idx=1),
                    ),
                    Sifa(
                        phonemes_group="b",
                        hams_or_jahr=SingleUnit(text="jahr", prob=0.9, idx=2),
                        shidda_or_rakhawa=SingleUnit(text="between", prob=0.8, idx=2),
                        tafkheem_or_taqeeq=SingleUnit(text="moraqaq", prob=0.7, idx=2),
                        itbaq=SingleUnit(text="motbaq", prob=0.6, idx=2),
                        safeer=SingleUnit(text="no_safeer", prob=0.5, idx=2),
                        qalqla=SingleUnit(text="not_moqalqal", prob=0.4, idx=2),
                        tikraar=SingleUnit(text="not_mokarar", prob=0.3, idx=2),
                        tafashie=SingleUnit(text="not_motafashie", prob=0.2, idx=2),
                        istitala=SingleUnit(text="not_mostateel", prob=0.1, idx=2),
                        ghonna=None,
                    ),
                ],
            ],
        ),
    ],
)
def test_fromat_sifat(level_to_units, chunked_phonemes_batch, ex_sifat_batch):
    multi_level_tokenizer = MultiLevelTokenizer("obadx/muaalem-model-v2_1")
    sifat_batch = format_sifat(
        level_to_units, chunked_phonemes_batch, multi_level_tokenizer
    )
    for level in level_to_units:
        for idx in range(len(level_to_units[level])):
            level_to_units[level][idx].probs = torch.FloatTensor(
                level_to_units[level][idx].probs
            )
            level_to_units[level][idx].ids = torch.LongTensor(
                level_to_units[level][idx].ids
            )

    assert len(sifat_batch) == len(ex_sifat_batch)
    for sifat, ex_sifat in zip(sifat_batch, ex_sifat_batch):
        assert len(sifat) == len(ex_sifat)
        for sifa, ex_sifa in zip(sifat, ex_sifat):
            print(f"OUPUT: {json.dumps(asdict(sifa), indent=2)}")
            print(f"EX OUPUT: {json.dumps(asdict(ex_sifa), indent=2)}")
            print("-" * 50)
            assert sifa == ex_sifa


@pytest.mark.slow
def test_inference():
    cache_dir = "./assets/test_cache"
    sampling_rate = 16000
    audio_path = "./assets/test.wav"
    device = "cpu"

    uthmani_ref = Aya(8, 75).get_by_imlaey_words(17, 9).uthmani
    moshaf = MoshafAttributes(
        rewaya="hafs",
        madd_monfasel_len=2,
        madd_mottasel_len=4,
        madd_mottasel_waqf=4,
        madd_aared_len=2,
    )
    phonetizer_out = quran_phonetizer(uthmani_ref, moshaf, remove_spaces=True)

    muaalem = Muaalem(device=device)
    wave, _ = load(audio_path, sr=sampling_rate, mono=True)
    outs = muaalem(
        [wave],
        [phonetizer_out],
        sampling_rate=sampling_rate,
    )

    for out in outs:
        print(out.phonemes)
        for sifa in out.sifat:
            print(json.dumps(asdict(sifa), indent=2, ensure_ascii=False))
            print("*" * 30)
        print("-" * 40)

    # Explaining Results
    explain_for_terminal(
        outs[0].phonemes.text,
        phonetizer_out.phonemes,
        outs[0].sifat,
        phonetizer_out.sifat,
    )
