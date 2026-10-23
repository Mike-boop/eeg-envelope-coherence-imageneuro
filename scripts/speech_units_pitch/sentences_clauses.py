#!/usr/bin/env python3
"""
Extracts sentence and clause onsets from the transcripts.

Sentences come from the `utterance` tier of the normalised transcripts, which is
already a sentence-level segmentation: `whisperx.align()` splits each ASR chunk's
text with NLTK's Punkt tokenizer and emits one segment per sentence. These are
Punkt boundaries over Whisper's punctuated text, not a human segmentation — 87%
of them agree within 100ms with the old project's spaCy sentences, 93% within
250ms.

Clauses are found by parsing each sentence with spaCy's Dutch model and taking
the verb-like heads carrying a clause dependency label. Onsets for both come from
the forced-aligned `words` tier, not the WhisperX timings.

Outputs JSON files with keys equal to audio names and values equal to a list of
onsets (in seconds) for each audio file, plus the matching rate files.

Based on /home/gu08wuvu/neural_tracking_across_spectrum/stimulus_analysis/textgrid_analysis/sentences_clauses.ipynb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from _common import (
    is_speech_label,
    load_paths,
    read_interval_tier,
    textgrid_files,
    write_json,
)

SPACY_MODEL = "nl_core_news_lg"

# Dependency labels that head a clause (UD + spaCy variants).
CLAUSE_DEPS = {
    "ROOT",
    # clausal complements / clausal subjects
    "ccomp", "xcomp", "csubj", "csubjpass", "csubj:pass",
    # adverbial & relative/clausal modifiers
    "advcl", "acl", "acl:relcl", "relcl",
    # loose attachment & coordination
    "parataxis", "conj",
}
# A clause needs a verb: `conj` and friends also attach noun phrases.
VERB_POS = {"VERB", "AUX"}
# Markers and coordinators belong to the clause they introduce.
CLAUSE_INTRODUCERS = {"mark", "cc"}


def clause_start_indices(doc) -> list[int]:
    """Token indices at which each detected clause begins."""
    heads = [tok for tok in doc if tok.dep_ in CLAUSE_DEPS and tok.pos_ in VERB_POS]
    for sent in doc.sents:
        if sent.root.pos_ in VERB_POS and sent.root not in heads:
            heads.append(sent.root)

    starts = []
    for head in heads:
        start = min(tok.i for tok in head.subtree)
        for child in head.children:
            if child.dep_ in CLAUSE_INTRODUCERS:
                start = min(start, child.i)
        starts.append(start)
    return sorted(set(starts))


def words_within(words, start: float, end: float) -> list[tuple[float, float, str]]:
    """Aligned words whose midpoint falls inside an utterance."""
    return [w for w in words if start - 1e-6 <= (w[0] + w[1]) / 2 <= end + 1e-6]


def onsets_for_recording(nlp, transcript_path: Path, textgrid_path: Path):
    """Sentence and clause onsets for one recording, in seconds."""
    utterances = [u for u in read_interval_tier(transcript_path, "utterance") if u[2].strip()]
    words = [w for w in read_interval_tier(textgrid_path, "words") if is_speech_label(w[2])]

    sentence_onsets: list[float] = []
    clause_onsets: list[float] = []
    unaligned = 0

    texts = [utt[2] for utt in utterances]
    for utterance, doc in zip(utterances, nlp.pipe(texts, batch_size=64)):
        start, end, _ = utterance
        aligned = words_within(words, start, end)
        if not aligned:
            unaligned += 1
            continue

        sentence_onsets.append(aligned[0][0])

        # spaCy tokens map onto the aligned words in order. The transcripts carry
        # no punctuation, so this is 1:1 in practice; skipping punctuation and
        # clamping keeps it safe where a token splits.
        word_index_of = {}
        cursor = 0
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            word_index_of[token.i] = min(cursor, len(aligned) - 1)
            cursor += 1

        for token_index in clause_start_indices(doc):
            word_index = word_index_of.get(token_index)
            if word_index is not None:
                clause_onsets.append(aligned[word_index][0])

    return sorted(set(sentence_onsets)), sorted(set(clause_onsets)), unaligned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=SPACY_MODEL, help=f"spaCy model (default: {SPACY_MODEL})")
    args = parser.parse_args()

    try:
        import spacy
    except ImportError:  # pragma: no cover - environment plumbing
        raise SystemExit("sentences_clauses.py needs spaCy: uv pip install -e '.[speech-units]'")

    try:
        # ner and lemmatiser are not used; the parser and tagger are.
        nlp = spacy.load(args.model, exclude=["ner", "lemmatizer"])
    except OSError:  # pragma: no cover - environment plumbing
        raise SystemExit(f"spaCy model {args.model} not installed: python -m spacy download {args.model}")

    paths = load_paths(require_textgrids=True)
    transcript_dir = paths.transcript_dir
    if not transcript_dir.is_dir():
        raise FileNotFoundError(f"Normalised transcript directory not found: {transcript_dir}")

    sentence_onsets: dict[str, list[float]] = {}
    sentence_rates: dict[str, float] = {}
    clause_onsets: dict[str, list[float]] = {}
    clause_rates: dict[str, float] = {}
    missing: list[str] = []

    for textgrid_path in tqdm(textgrid_files(paths.textgrid_dir), desc="Sentences and clauses", unit="file"):
        audio_name = textgrid_path.stem
        transcript_path = transcript_dir / f"{audio_name}.TextGrid"
        if not transcript_path.exists():
            missing.append(audio_name)
            continue

        sentences, clauses, unaligned = onsets_for_recording(nlp, transcript_path, textgrid_path)
        duration = max(float(w[1]) for w in read_interval_tier(textgrid_path, "words"))

        sentence_onsets[audio_name] = sentences
        clause_onsets[audio_name] = clauses
        sentence_rates[audio_name] = len(sentences) / duration
        clause_rates[audio_name] = len(clauses) / duration

        note = f"  ({unaligned} utterance(s) with no aligned words)" if unaligned else ""
        print(f"{audio_name}: {len(sentences)} sentences ({sentence_rates[audio_name]:.2f} Hz), "
              f"{len(clauses)} clauses ({clause_rates[audio_name]:.2f} Hz){note}")

    if missing:
        print(f"WARNING: no normalised transcript for {len(missing)}: {', '.join(missing)}")

    write_json(sentence_onsets, paths.output_dir / "sentence_onsets.json")
    write_json(sentence_rates, paths.output_dir / "sentence_rates.json")
    write_json(clause_onsets, paths.output_dir / "clause_onsets.json")
    write_json(clause_rates, paths.output_dir / "clause_rates.json")

    for label, rates in [("sentences", sentence_rates), ("clauses", clause_rates)]:
        values = np.array(list(rates.values()))
        print(f"{label}: n={len(values)} median {np.median(values):.3f} Hz")

    print(f"Wrote sentence and clause onsets for {len(sentence_onsets)} recording(s) to {paths.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
