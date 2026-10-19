import pytest
from quran_transcript import Aya, MoshafAttributes, quran_phonetizer
from quran_transcript.phonetics.search import NoPhonemesSearchResult, PhoneticSearch

from quran_muaalem.app.serve import search_near

MOSHAF = MoshafAttributes(
    rewaya="hafs", madd_monfasel_len=4, madd_mottasel_len=4, madd_mottasel_waqf=4, madd_aared_len=4
)


@pytest.fixture(scope="module")
def ph_search():
    return PhoneticSearch()


def phonemes(sura, aya):
    return quran_phonetizer(Aya(sura, aya).get().uthmani, MOSHAF, remove_spaces=True).phonemes


def test_whole_quran_search_prefers_earlier_repeat(ph_search):
    # 1:3 is also the end of the basmala: the unhinted search lands on 1:1 first
    assert ph_search.search(phonemes(1, 3))[0].start.aya_idx == 1


def test_hint_finds_the_recited_aya(ph_search):
    r = search_near(ph_search, phonemes(1, 3), 0.1, 1, 3)[0]
    assert (r.start.sura_idx, r.start.aya_idx, r.end.aya_idx) == (1, 3, 3)


def test_hint_window_crosses_sura_boundary(ph_search):
    # hinted at the last aya of al-Fatiha, recited the start of al-Baqara
    r = search_near(ph_search, phonemes(2, 1), 0.1, 1, 7)[0]
    assert (r.start.sura_idx, r.start.aya_idx) == (2, 1)


def test_closest_match_first(ph_search):
    # hinted at 78:5 but 78:4 recited (identical except the leading "thumma")
    r = search_near(ph_search, phonemes(78, 4), 0.2, 78, 5)[0]
    assert (r.start.aya_idx, r.end.aya_idx) == (4, 4)


def test_one_breath_over_several_short_ayat(ph_search):
    # hint lags behind: still on 112:2 while 112:2-4 were recited in one breath
    q = "".join(phonemes(112, a) for a in (2, 3, 4))
    r = search_near(ph_search, q, 0.1, 112, 2)[0]
    assert (r.start.aya_idx, r.end.aya_idx) == (2, 4)


def test_unknown_aya_is_no_result(ph_search):
    with pytest.raises(NoPhonemesSearchResult):
        search_near(ph_search, phonemes(1, 1), 0.1, 1, 50)


def test_empty_query_is_no_result(ph_search):
    with pytest.raises(NoPhonemesSearchResult):
        search_near(ph_search, "", 0.2, 78, 3)
