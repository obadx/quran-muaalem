# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.19.0",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.19.6"
app = marimo.App()


@app.cell
def _():
    # alignment_verify.py
    import marimo as mo
    import numpy as np
    import polars as pl
    import altair as alt
    import json



    from quran_muaalem import Muaalem
    from quran_muaalem.forced_alignment import align_sample,extract_audio_segment,get_moshaf,prepare_waveform_data

    muaalem = Muaalem(model_name_or_path="obadx/muaalem-model-v3_2")


    moshaf = get_moshaf("moshaf_0.0")
    sample = next(iter(moshaf))

    # Run alignment
    result = align_sample(muaalem.model, muaalem.processor, muaalem.multi_level_tokenizer.get_tokenizer(), sample)
    timestamps = json.loads(result)["time_stamps"]
    audio = sample["audio"]["array"]
    audio_df, boundaries_df = prepare_waveform_data(audio, timestamps)

    # Get sifat from sample
    sifat_data = sample["sifat"]
    return (
        alt,
        audio,
        extract_audio_segment,
        mo,
        pl,
        sample,
        sifat_data,
        timestamps,
    )


@app.cell
def _(audio, mo, sample):
    # Title and full audio
    mo.md("# Phoneme Alignment Verification Tool")
    mo.md(f"**Expected phonemes:** `{sample['phonemes']}`")
    mo.audio(audio, rate=16000)
    return


@app.cell
def _(mo, pl, sifat_data):
    # 1. Multi-level view: phonemes + sifat attributes table
    sifat_df = pl.DataFrame(sifat_data)
    mo.md("### 1. Multi-Level View (Phonemes + Sifat)")
    mo.ui.table(sifat_df.to_pandas(), label="Phonemes with Sifat Attributes")
    return


@app.function
# 2. Tajweed rule highlighting
def get_tajweed_color(sifat):
    if sifat.get("ghonna") == "maghnoon":
        return "Ghunna"
    elif (
        "ۥ" in sifat.get("phonemes", "")
        or "ۦ" in sifat.get("phonemes", "")
        or "اا" in sifat.get("phonemes", "")
    ):
        return "Madd"
    elif sifat.get("qalqla") == "moqalqal":
        return "Qalqala"
    elif sifat.get("tafkheem_or_taqeeq") == "mofakham":
        return "Tafkheem"
    else:
        return "Normal"


@app.cell
def _(sifat_data, timestamps):
    tajweed_data = []
    for i, ts in enumerate(timestamps):
        rule = "Normal"
        if i < len(sifat_data):
            rule = get_tajweed_color(sifat_data[i])
        tajweed_data.append(
            {
                "phoneme": ts["phoneme"],
                "start_frame": ts["start_frame"],
                "end_frame": ts["end_frame"],
                "tajweed_rule": rule,
            }
        )
    return (tajweed_data,)


@app.cell
def _(alt, mo, pl, tajweed_data):
    tajweed_df = pl.DataFrame(tajweed_data)

    mo.md("### 2. Tajweed Rule Highlighting")
    color_scale = alt.Scale(
        domain=["Normal", "Ghunna", "Madd", "Qalqala", "Tafkheem"],
        range=["gray", "green", "blue", "orange", "red"],
    )

    timeline_chart = (
        alt.Chart(tajweed_df.to_pandas())
        .mark_bar(height=20)
        .encode(
            x=alt.X("start_frame:Q", title="Frame"),
            x2="end_frame:Q",
            y=alt.Y("phoneme:N", sort=None),
            color=alt.Color("tajweed_rule:N", scale=color_scale, title="Tajweed Rule"),
            tooltip=["phoneme", "start_frame", "end_frame", "tajweed_rule"],
        )
        .properties(width=700, height=400)
    )

    timeline_chart
    return (tajweed_df,)


@app.cell
def _(mo, sample, timestamps):
    # 3. Side-by-side: Expected vs Predicted
    expected_phonemes = sample["phonemes"].replace(" ", "")
    predicted_phonemes = "".join([ts["phoneme"] for ts in timestamps])

    mo.md("### 3. Expected vs Predicted Comparison")
    mo.hstack(
        [
            mo.vstack([mo.md("**Expected:**"), mo.md(f"`{expected_phonemes}`")]),
            mo.vstack([mo.md("**Predicted:**"), mo.md(f"`{predicted_phonemes}`")]),
        ]
    )
    return


@app.cell
def _(mo):
    # Interactive phoneme player with group slider
    group_slider = mo.ui.slider(start=1, stop=5, value=1, label="Phonemes to play")
    group_slider
    return (group_slider,)


@app.cell
def _(mo, tajweed_df):
    # Phoneme table with selection
    table = mo.ui.table(
        tajweed_df.to_pandas(), selection="single", label="Select phoneme to play"
    )
    table
    return (table,)


@app.cell
def _(
    audio,
    extract_audio_segment,
    group_slider,
    mo,
    pl,
    sample,
    sifat_data,
    table,
    timestamps,
):
    selected_idx = table.value.index[0]
    group_size = group_slider.value
    end_idx = min(selected_idx + group_size, len(timestamps))

    start_frame = timestamps[selected_idx]["start_frame"]
    end_frame = timestamps[min(end_idx - 1, len(timestamps) - 1)]["end_frame"]

    segment, _, _ = extract_audio_segment(audio, start_frame, end_frame)
    phonemes_str = "".join(
        [timestamps[i]["phoneme"] for i in range(selected_idx, end_idx)]
    )

    # Get sifat for selected phonemes
    selected_sifat = (
        sifat_data[selected_idx:end_idx] if selected_idx < len(sifat_data) else []
    )
    sifat_display = pl.DataFrame(selected_sifat).to_pandas() if selected_sifat else None

    mo.vstack(
        [
            mo.md(f"### Uthmani: `{sample['uthmani']}`"),
            mo.md(f"**Playing:** {phonemes_str} | Frames: {start_frame} - {end_frame}"),
            mo.audio(segment, rate=16000),
            mo.md("**Sifat Attributes:**"),
            mo.ui.table(sifat_display)
            if sifat_display is not None
            else mo.md("No sifat data"),
            group_slider,
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
