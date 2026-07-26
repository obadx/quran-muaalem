import logging
from typing import Sequence, Any
from dataclasses import dataclass
import torch
import numpy as np
from numpy.typing import NDArray

from .modeling.vocab import PAD_TOKEN_IDX
from .muaalem_typing import Unit


# def align_predicted_sequence(
#     ref: Sequence[Any], predicted: Sequence[Any]
# ) -> Sequence[Any]:
#     """Aligns the preficeted sequence to the ref sequnce
#
#     Example (1): `predicted` length > `ref` length
#         ref: abcde
#         predicted: abcdef
#         Returns: abcde
#
#     Example (2): `predicted` length <`ref` length
#         ref: abcde
#         predicted: abcd
#         Returns: abcde
#
#     Returns:
#         Sequnce[Any]: new precicted sequence that best matches the ref sequence
#     """
#     n = len(ref)
#     m = len(predicted)
#     if n == m:
#         return predicted
#     if n == 0:
#         return []
#     if m == 0:
#         return ref
#
#     dp = [[0] * (m + 1) for _ in range(n + 1)]
#
#     for i in range(1, n + 1):
#         dp[i][0] = 0
#     for j in range(1, m + 1):
#         dp[0][j] = 0
#
#     for i in range(1, n + 1):
#         for j in range(1, m + 1):
#             insertion = dp[i - 1][j]
#             deletion = dp[i][j - 1]
#             match_cost = dp[i - 1][j - 1] + (1 if ref[i - 1] != predicted[j - 1] else 0)
#             dp[i][j] = min(insertion, deletion, match_cost)
#
#     i, j = n, m
#     output_chars = []
#     while i > 0 or j > 0:
#         if (
#             i > 0
#             and j > 0
#             and ref[i - 1] == predicted[j - 1]
#             and dp[i][j] == dp[i - 1][j - 1]
#         ):
#             output_chars.append(predicted[j - 1])
#             i -= 1
#             j -= 1
#         elif i > 0 and dp[i][j] == dp[i - 1][j]:
#             output_chars.append(ref[i - 1])
#             i -= 1
#         elif j > 0 and dp[i][j] == dp[i][j - 1]:
#             j -= 1
#         else:
#             output_chars.append(predicted[j - 1])
#             i -= 1
#             j -= 1
#
#     # return "".join(output_chars[::-1])
#     return output_chars[::-1]


# def align_predicted_sequence(ref, predicted):
#     n = len(ref)
#     m = len(predicted)
#     if m == n:
#         return predicted
#
#     INF = 10**9
#     dp = [[0] * (m + 1) for _ in range(n + 1)]
#     choice = [[0] * (m + 1) for _ in range(n + 1)]
#
#     for j in range(m + 1):
#         dp[0][j] = 0
#
#     for i in range(1, n + 1):
#         dp[i][0] = INF
#
#     for i in range(1, n + 1):
#         for j in range(1, m + 1):
#             # above
#             option1 = dp[i][j - 1]
#             # adjacent
#             if j >= i:
#                 cost = 0 if predicted[j - 1] == ref[i - 1] else 1
#                 option2 = dp[i - 1][j - 1] + cost
#             else:
#                 option2 = INF
#
#             if option2 <= option1:
#                 dp[i][j] = option2
#                 choice[i][j] = 1
#             else:
#                 dp[i][j] = option1
#                 choice[i][j] = 0
#
#     print(np.array(dp))
#     print(np.array(choice))
#
#     res_chars = []
#     i, j = n, m
#     while i > 0 and j > 0:
#         if choice[i][j] == 1:
#             res_chars.append(predicted[j - 1])
#             i -= 1
#             j -= 1
#         else:
#             j -= 1
#
#     return res_chars[::-1]


def align_chunked_phonemes_sequence(
    ref: list[list[str]],
    predicted: list[list[str]],
) -> list[bool]:
    """Aligns phonemes level to get mask that descripts what is missing

    Returns the mask for the `ref` inputs that best matches the `predicted`
    Note element wise comparison but retuns mask for best seqence (even with errors)

    Example (1): `predicted` length > `ref` length
        ref: abcde
        predicted: abcdef
        Returns: [T, T, T, T]

    Example (2): `predicted` length <`ref` length
        ref: abcde
        predicted: abcd
        Returns: [T, T, T, T, F]

    Example (2): `predicted` length <`ref` length
        ref: afcde
        predicted: abcd
        Returns: [T, T, T, T, F]


    Len(mask] == Len(ref)

    """

    n = len(predicted)
    m = len(ref)

    if len(predicted) == len(ref):
        return [True] * len(predicted)

    if m == 0:
        raise ValueError("`ref` length must not be zero length")

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    choice = [[0] * (m + 1) for _ in range(n + 1)]

    for j in range(m + 1):
        dp[0][j] = 0

    for i in range(1, n + 1):
        dp[i][0] = i

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            option1 = dp[i][j - 1]
            option2 = dp[i - 1][j] + 1
            cost = 0 if predicted[i - 1][0] == ref[j - 1][0] else 1
            option3 = dp[i - 1][j - 1] + cost

            if option3 <= option1 and option3 <= option2:
                dp[i][j] = option3
                choice[i][j] = 3
            elif option1 <= option2:
                dp[i][j] = option1
                choice[i][j] = 1
            else:
                dp[i][j] = option2
                choice[i][j] = 2

    i = n
    j = m
    mask = []
    # res_chars = []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            if choice[i][j] == 3:
                # res_chars.append(ref[j - 1])
                mask.append(True)
                i -= 1
                j -= 1
            elif choice[i][j] == 2:
                # res_chars.append(missing_placeholder)
                i -= 1
            else:
                j -= 1
                mask.append(False)
        elif i > 0:
            # res_chars.append(missing_placeholder)
            i -= 1
        else:
            j -= 1
            mask.append(False)

    return mask[::-1]


def align_predicted_sequence(
    ref: Sequence[Any] | torch.LongTensor,
    predicted: Sequence[Any] | torch.LongTensor,
    missing_placeholder=-100,
) -> Sequence[Any] | torch.LongTensor:
    """Aligns the preficeted sequence to the ref sequnce

    Example (1): `predicted` length > `ref` length
        ref: abcde
        predicted: abcdef
        Returns: abcde

    Example (2): `predicted` length <`ref` length
        ref: abcde
        predicted: abcd
        Returns: abcd(missing_placeholder)

    Returns:
        Sequnce[Any]: new precicted sequence that best matches the ref sequence
    """

    n = len(ref)
    m = len(predicted)

    if len(ref) == len(predicted):
        return predicted, [True] * len(ref)

    if m == 0:
        return [missing_placeholder] * n, []

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    choice = [[0] * (m + 1) for _ in range(n + 1)]

    for j in range(m + 1):
        dp[0][j] = 0

    for i in range(1, n + 1):
        dp[i][0] = i

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            option1 = dp[i][j - 1]
            option2 = dp[i - 1][j] + 1
            cost = 0 if ref[i - 1] == predicted[j - 1] else 1
            option3 = dp[i - 1][j - 1] + cost

            if option3 <= option1 and option3 <= option2:
                dp[i][j] = option3
                choice[i][j] = 3
            elif option1 <= option2:
                dp[i][j] = option1
                choice[i][j] = 1
            else:
                dp[i][j] = option2
                choice[i][j] = 2

    i = n
    j = m
    mask = []
    res_chars = []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            if choice[i][j] == 3:
                res_chars.append(predicted[j - 1])
                mask.append(True)
                i -= 1
                j -= 1
            elif choice[i][j] == 2:
                res_chars.append(missing_placeholder)
                i -= 1
            else:
                j -= 1
                mask.append(False)
        elif i > 0:
            res_chars.append(missing_placeholder)
            i -= 1
        else:
            j -= 1
            mask.append(False)

    return res_chars[::-1], mask[::-1]


@dataclass
class CTCDecodeOut:
    """
    Both are 1D Tensors
    """

    ids: torch.LongTensor
    p: torch.FloatTensor


# BUG: trailing zero id at the end
def ctc_decode(
    batch_ids: torch.LongTensor,
    batch_probs: torch.FloatTensor,
    blank_id=PAD_TOKEN_IDX,
    collapse_consecutive=True,
) -> list[CTCDecodeOut]:
    """
    batch_ids (torch.LongTensor): batch on integer ids of shape: batch, sequecne_len
    batch_probs (torch.LongTensor): batch on float32 ids of shape: batch, sequecne_len

    Return:
        list[tuple[list[int], float]]]:


    """
    outs = []
    assert batch_ids.shape == batch_probs.shape, (
        f"Input Shape mismatch: `batch_ids.shape={batch_ids.shape}`, batch_probs.shape={batch_probs.shape}"
    )
    for seq_idx, seq in enumerate(batch_ids):
        if collapse_consecutive:
            tokens = []
            probs = []
            start = 0
            end = 0
            if len(seq) == 1 and seq[0] != blank_id:
                tokens.append(seq[0])
                probs.append(batch_probs[seq_idx][0])

            for idx in range(len(seq) - 1):
                curr = seq[idx]
                next = seq[idx + 1]
                # Last Item
                if idx == len(seq) - 2 and curr != blank_id:
                    if curr == next:
                        end = idx + 2
                        tokens.append(curr)
                        probs.append(
                            batch_probs[seq_idx][start:end].sum() / (end - start)
                        )
                    elif curr != next:
                        end = idx + 1
                        tokens.append(curr)
                        probs.append(
                            batch_probs[seq_idx][start:end].sum() / (end - start)
                        )
                        # the last item is only a token if it is not a blank
                        if next != blank_id:
                            tokens.append(next)
                            probs.append(batch_probs[seq_idx][idx + 1])
                # Normal Case
                elif curr != next and curr != blank_id:
                    end = idx + 1
                    tokens.append(curr)
                    probs.append(batch_probs[seq_idx][start:end].sum() / (end - start))
                    start = end
                elif curr == blank_id:
                    start = idx + 1

            outs.append(
                CTCDecodeOut(
                    ids=torch.LongTensor(tokens),
                    p=torch.FloatTensor(probs),
                )
            )
        else:
            mask = seq != blank_id
            tokens = seq[mask]
            probs = batch_probs[seq_idx][mask]
            outs.append(CTCDecodeOut(ids=tokens, p=probs))
    return outs


# def multilevel_greedy_decode(
#     level_to_probs: dict[str, torch.FloatTensor],
#     level_to_id_to_vocab: dict[str, dict[int, str]],
#     level_to_ref_ids: dict[str, torch.LongTensor],
#     missing_placeholder=-100,
#     pad_idx=PAD_TOKEN_IDX,
# ) -> dict[str, list[Unit]]:
#     level_to_units = {}
#     for level in level_to_probs:
#         batch_probs, batch_ids = level_to_probs[level].topk(1, dim=-1)
#         decode_outs = ctc_decode(
#             batch_ids.squeeze(-1), batch_probs.squeeze(-1), collapse_consecutive=True
#         )
#         level_to_units[level] = []
#         for seq_idx, decode_out in enumerate(decode_outs):
#             # Trying to align Ids of the sifat levels
#             if level != "phonemes":
#                 aligned_ids, mask = align_predicted_sequence(
#                     level_to_ref_ids[level][seq_idx],
#                     decode_out.ids,
#                     missing_placeholder=missing_placeholder,
#                 )
#             else:
#                 aligned_ids = decode_out.ids
#
#             probs = decode_out.p
#             if len(aligned_ids) != len(decode_out.ids):
#                 aligned_ids = torch.LongTensor(aligned_ids)
#                 mask = torch.BoolTensor(mask)
#
#                 new_probs = torch.zeros(len(aligned_ids), dtype=torch.float32)
#                 new_probs[aligned_ids != missing_placeholder] = probs[mask]
#
#                 aligned_ids[aligned_ids == missing_placeholder] = pad_idx
#                 probs = new_probs
#
#             probs = decode_out.p.clone()
#             text = ""
#             for idx in aligned_ids:
#                 text += level_to_id_to_vocab[level][int(idx)]
#             level_to_units[level].append(
#                 Unit(
#                     text=text,
#                     probs=probs,
#                     ids=aligned_ids,
#                 ),
#             )
#
#     return level_to_units


def phonemes_level_greedy_decode(
    probs: torch.FloatTensor,
    phonemes_level_vocab: dict[int, str],
) -> list[Unit]:
    """Decodes only phonemes level

    Args:
        probs (torch.FloatTensor) of shape batch, seq_len, num_classes
        phonmes_level_vocab (dict[int, str]): mapping ids of phonemes to the
            acutial string represnetation
    """
    batch_probs, batch_ids = probs.topk(1, dim=-1)
    decode_outs = ctc_decode(
        batch_ids.squeeze(-1), batch_probs.squeeze(-1), collapse_consecutive=True
    )
    units = []
    for seq_idx, decode_out in enumerate(decode_outs):
        text = ""
        for idx in decode_out.ids:
            text += phonemes_level_vocab[int(idx)]
        units.append(
            Unit(
                text=text,
                probs=decode_out.p,
                ids=decode_out.ids,
            ),
        )
    return units


def multilevel_greedy_decode(
    level_to_probs: dict[str, torch.FloatTensor],
    level_to_id_to_vocab: dict[str, dict[int, str]],
    level_to_ref_ids: dict[str, torch.LongTensor],
    chunked_phonemes_batch: list[list[str]],
    ref_chuncked_phonemes_batch: list[list[str]],
    phonemes_units: list[Unit],
    missing_placeholder=-100,
    pad_idx=PAD_TOKEN_IDX,
) -> dict[str, list[Unit]]:
    level_to_units = {}
    for level in level_to_probs:
        if level == "phonemes":
            continue
        batch_probs, batch_ids = level_to_probs[level].topk(1, dim=-1)
        decode_outs = ctc_decode(
            batch_ids.squeeze(-1), batch_probs.squeeze(-1), collapse_consecutive=True
        )
        level_to_units[level] = []
        for seq_idx, decode_out in enumerate(decode_outs):
            # Trying to align Ids of the sifat levels
            phonemes_mask = align_chunked_phonemes_sequence(
                ref=ref_chuncked_phonemes_batch[seq_idx],
                predicted=chunked_phonemes_batch[seq_idx],
            )
            phonemes_mask = torch.BoolTensor(phonemes_mask)

            # NOTE:
            # We want to align every level with predited phonme, but
            # in some cases the length of every sifa level is > or < the
            # length for the predited phonemes
            # we slove this by two steps
            # 1. Align the sifa level with length mismatch to the refrence sifa level
            # 2. align the alinged sifa level back to the the length of prediced phonmes
            if len(decode_out.ids) != len(chunked_phonemes_batch[seq_idx]) and (
                len(chunked_phonemes_batch[seq_idx])
                <= len(ref_chuncked_phonemes_batch[seq_idx])
            ):
                logging.info(f"Sequence: `{seq_idx}` has mismatch Level: {level}")
                # 1. Align sifa level to the reference sifa level
                ref_aligned_ids, mask = align_predicted_sequence(
                    level_to_ref_ids[level][seq_idx],
                    decode_out.ids,
                    missing_placeholder=missing_placeholder,
                )

                probs = decode_out.p
                ref_aligned_ids = torch.LongTensor(ref_aligned_ids)
                mask = torch.BoolTensor(mask)

                new_probs = torch.zeros(len(ref_aligned_ids), dtype=torch.float32)
                new_probs[ref_aligned_ids != missing_placeholder] = probs[mask]

                ref_aligned_ids[ref_aligned_ids == missing_placeholder] = pad_idx

                # 2. Align the predicted aligned to the ref back to the predicted seqence
                aligned_ids = ref_aligned_ids[phonemes_mask]
                new_probs = ref_aligned_ids[phonemes_mask]

                probs = new_probs
            else:
                aligned_ids = decode_out.ids
                probs = decode_out.p

            text = ""
            for idx in aligned_ids:
                text += level_to_id_to_vocab[level][int(idx)]
            level_to_units[level].append(
                Unit(
                    text=text,
                    probs=probs,
                    ids=aligned_ids,
                ),
            )
    level_to_units["phonemes"] = phonemes_units

    return level_to_units


def align_sequence(
    seq: Sequence[int] | torch.LongTensor, target_len: int, min_repeat: int = 3
) -> list[int]:
    """Aligns a sequence by removing items from the longest repateted items

    Returns:
        list[int]: the ids which are goning to be deleted if longest_repeat > len(seq) - target_len

    Example:
                seq = [1, 0, 1, 0, 0, 0, 0, 1], target_len = 7
                                ^  ^  ^
    Longest Repeat              ^  ^  ^
    Ouput: [3]
    """

    if len(seq) <= target_len:
        return []

    longest_start = 0
    longest_repeat = 0
    curr_repeat = 1
    curr_start = 0
    for idx in range(len(seq) - 1):
        curr = seq[idx]
        next = seq[idx + 1]
        if curr == next:
            curr_repeat += 1
        if (curr != next) or (idx == len(seq) - 2):
            if curr_repeat > longest_repeat and curr_repeat >= min_repeat:
                longest_repeat = curr_repeat
                longest_start = curr_start
            curr_start = idx + 1
            curr_repeat = 1

    # logical case to remote only from the longest repeat
    if longest_repeat > len(seq) - target_len:
        return list(range(longest_start, longest_start + len(seq) - target_len))
    else:
        return []
