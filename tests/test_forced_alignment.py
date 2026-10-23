import importlib.util
import wave
from pathlib import Path

from forced_alignment.alignment import (
    discover_alignment_pairs,
    parse_textgrid_intervals,
    prepare_alignment_inputs,
    prepare_mfa_inputs,
    write_recombined_textgrid,
)


def load_split_utterances_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "forced_alignment" / "split_utterances.py"
    spec = importlib.util.spec_from_file_location("split_utterances_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_configured_inputs_uses_project_config_directories(tmp_path):
    module = load_split_utterances_module()

    config = {
        "raw_waves_dir": str(tmp_path / "raw_waves"),
        "normalised_transcripts_dir": str(tmp_path / "normalised_transcripts"),
        "mfa_output_dir": str(tmp_path / "mfa_output"),
    }

    audio_dir, transcript_dir, output_dir = module.resolve_configured_inputs(config, root=tmp_path)

    assert audio_dir == (tmp_path / "raw_waves").resolve()
    assert transcript_dir == (tmp_path / "normalised_transcripts").resolve()
    assert output_dir == (tmp_path / "mfa_output").resolve()


def test_discover_alignment_pairs_matches_audio_and_text(tmp_path):
    audio_dir = tmp_path / "audio"
    transcript_dir = tmp_path / "transcripts"
    audio_dir.mkdir(parents=True)
    transcript_dir.mkdir(parents=True)

    (audio_dir / "sample1.wav").write_bytes(b"wav")
    (audio_dir / "sample2.wav").write_bytes(b"wav")
    (transcript_dir / "sample1.txt").write_text("hello world", encoding="utf-8")
    (transcript_dir / "sample2.txt").write_text("another line", encoding="utf-8")

    pairs = discover_alignment_pairs(audio_dir, transcript_dir)

    assert [(pair[0].name, pair[1].name) for pair in pairs] == [
        ("sample1.wav", "sample1.txt"),
        ("sample2.wav", "sample2.txt"),
    ]


def test_prepare_alignment_inputs_creates_audio_and_text_directories(tmp_path):
    audio_dir = tmp_path / "audio"
    transcript_dir = tmp_path / "transcripts"
    output_dir = tmp_path / "prepared"
    audio_dir.mkdir(parents=True)
    transcript_dir.mkdir(parents=True)

    (audio_dir / "sample.wav").write_bytes(b"wav")
    (transcript_dir / "sample.txt").write_text("hello world", encoding="utf-8")

    prepared = prepare_alignment_inputs(audio_dir, transcript_dir, output_dir)

    assert prepared["audio_dir"].exists()
    assert prepared["text_dir"].exists()
    assert (prepared["audio_dir"] / "sample.wav").exists()
    assert (prepared["text_dir"] / "sample.txt").exists()


def test_parse_textgrid_intervals_reads_utterance_tier(tmp_path):
    textgrid_path = tmp_path / "sample.TextGrid"
    textgrid_path.write_text(
        'File type = "ooTextFile"\n'
        'Object class = "TextGrid"\n\n'
        'xmin = 0\n'
        'xmax = 2.0\n'
        'tiers? <exists>\n'
        'size = 1\n'
        'item []:\n'
        '\titem [1]:\n'
        '\t\tclass = "IntervalTier"\n'
        '\t\tname = "utterance"\n'
        '\t\txmin = 0\n'
        '\t\txmax = 2.0\n'
        '\t\tintervals: size = 2\n'
        '\t\tintervals [1]:\n'
        '\t\t\txmin = 0.0\n'
        '\t\t\txmax = 1.0\n'
        '\t\t\ttext = "hello"\n'
        '\t\tintervals [2]:\n'
        '\t\t\txmin = 1.0\n'
        '\t\t\txmax = 2.0\n'
        '\t\t\ttext = "world"\n',
        encoding="utf-8",
    )

    intervals = parse_textgrid_intervals(textgrid_path)

    assert [(interval["text"], interval["start"], interval["end"]) for interval in intervals] == [
        ("hello", 0.0, 1.0),
        ("world", 1.0, 2.0),
    ]


def test_prepare_mfa_inputs_creates_segment_wavs_and_text(tmp_path):
    audio_path = tmp_path / "sample.wav"
    textgrid_path = tmp_path / "sample.TextGrid"
    output_dir = tmp_path / "mfa"

    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)
    textgrid_path.write_text(
        'File type = "ooTextFile"\n'
        'Object class = "TextGrid"\n\n'
        'xmin = 0\n'
        'xmax = 1.0\n'
        'tiers? <exists>\n'
        'size = 1\n'
        'item []:\n'
        '\titem [1]:\n'
        '\t\tclass = "IntervalTier"\n'
        '\t\tname = "utterance"\n'
        '\t\txmin = 0\n'
        '\t\txmax = 1.0\n'
        '\t\tintervals: size = 1\n'
        '\t\tintervals [1]:\n'
        '\t\t\txmin = 0.0\n'
        '\t\t\txmax = 1.0\n'
        '\t\t\ttext = "hello"\n',
        encoding="utf-8",
    )

    prepared = prepare_mfa_inputs(audio_path, textgrid_path, output_dir)

    assert prepared["segments_dir"].exists()
    assert any(prepared["segments_dir"].iterdir())
    assert (prepared["segments_dir"] / "utt001_000.000.txt").exists()


def _write_utterance_textgrid(path, xmax, tier_name, intervals):
    lines = [
        'File type = "ooTextFile"\n',
        'Object class = "TextGrid"\n',
        "xmin = 0\n",
        f"xmax = {xmax}\n",
        "tiers? <exists>\n",
        "size = 1\n",
        "item []:\n",
        "\titem [1]:\n",
        '\t\tclass = "IntervalTier"\n',
        f'\t\tname = "{tier_name}"\n',
        "\t\txmin = 0\n",
        f"\t\txmax = {xmax}\n",
        f"\t\tintervals: size = {len(intervals)}\n",
    ]
    for index, (start, end, text) in enumerate(intervals, start=1):
        lines.append(f"\t\tintervals [{index}]:\n")
        lines.append(f"\t\t\txmin = {start}\n")
        lines.append(f"\t\t\txmax = {end}\n")
        lines.append(f'\t\t\ttext = "{text}"\n')
    path.write_text("".join(lines), encoding="utf-8")


def test_write_recombined_textgrid_shifts_and_fills_gaps(tmp_path):
    audio_path = tmp_path / "sample.wav"
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000 * 5)  # 5 seconds

    aligned_dir = tmp_path / "aligned"
    aligned_dir.mkdir()
    _write_utterance_textgrid(aligned_dir / "utt001_1000.000.TextGrid", 1.0, "phones", [(0.0, 1.0, "hello")])
    _write_utterance_textgrid(aligned_dir / "utt002_3000.000.TextGrid", 1.0, "phones", [(0.0, 1.0, "world")])

    output_path = write_recombined_textgrid(
        audio_path, aligned_dir, tmp_path / "combined" / "sample.TextGrid", tier_names=("phones",)
    )

    assert output_path.exists()
    result = parse_textgrid_intervals(output_path, tier_name="phones")
    assert [(i["start"], i["end"], i["text"]) for i in result] == [
        (0.0, 1.0, ""),
        (1.0, 2.0, "hello"),
        (2.0, 3.0, ""),
        (3.0, 4.0, "world"),
        (4.0, 5.0, ""),
    ]
