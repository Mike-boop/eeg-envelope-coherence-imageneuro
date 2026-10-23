# EEG Envelope Coherence Neuroimage

## Project structure

- `src/eeg_analysis/` — EEG analysis code
- `src/forced_alignment/` — forced alignment utilities
- `src/audio_features/` — continuous audio feature extraction (gammatone cochleogram, envelopes)
- `src/package_data/` — package data helpers
- `tests/` — project tests

## Development

Install the base project and test dependencies in editable mode:

```bash
uv pip install -e ".[dev]"
```

Install the optional transcription and MFA tooling for the full pipeline:

```bash
uv pip install -e ".[dev,transcription,mfa]"
```

Or create a conda environment from the repository config:

```bash
conda env create -f environment.yml
conda activate eeg-envelope-coherence-neuroimage
```

Run tests:

```bash
pytest
```

Run the example entrypoint:

```bash
python main.py
```

Forced alignment workflow:

```bash
conda env create -f mfa_environment.yml
conda activate mfa
python scripts/forced_alignment/split_utterances.py
./scripts/forced_alignment/run_mfa.sh
python scripts/forced_alignment/recombine_textgrids.py
```

All three scripts read `raw_waves_dir`, `normalised_transcripts_dir`, and `mfa_output_dir` from [src/package_data/config.json](src/package_data/config.json). `split_utterances.py` writes per-audio utterance folders to `<mfa_output_dir>/split_utterances/<audio_stem>/`, `run_mfa.sh` aligns them into `<mfa_output_dir>/raw_mfa_outputs/<audio_stem>/` (one TextGrid per utterance, in clip-local time), and `recombine_textgrids.py` stitches each recording's utterance TextGrids back into a single master TextGrid — with `words`/`phones` intervals shifted to their original offsets and gaps between utterances filled blank — written to `<mfa_output_dir>/combined_textgrids/<audio_stem>.TextGrid`. `run_mfa.sh` also accepts an optional config file argument (`{"mfa_env_name": "..."}`) if the conda environment isn't named `mfa`.

The forced alignment's input, `normalised_transcripts_dir`, comes from [scripts/transcriptions/transcribe.py](scripts/transcriptions/transcribe.py), which writes two TextGrids per recording:

- `raw_transcripts_dir` — the segments `model.transcribe()` returns, i.e. VAD-merged chunks of up to 30s (median ~24s in this corpus). Not usable as MFA utterances.
- `normalised_transcripts_dir` — the same audio after `whisperx.align()`, which re-splits each chunk into sentences with NLTK Punkt and attaches word timings (median ~4s). **This is what the forced alignment consumes.**

Feature extraction workflow:

```bash
python scripts/feature_extraction/extract_features.py
```

Reads every `.wav` in `raw_waves_dir` and writes one `.npz` per recording to `features_dir` (both from [src/package_data/config.json](src/package_data/config.json)). The default feature factory ([src/audio_features/factory.py](src/audio_features/factory.py)) runs two extractors:

- `GammatoneCochleogramExtractor` — low-pass filters audio to the frequency response of the insert-tube stimulus delivery, resamples it down (default 30kHz, to keep the per-channel FFT convolutions memory-bounded), then runs a 64-channel gammatone filterbank spaced on the ERB-rate scale (50Hz-8kHz by default). Writes `subband_envelopes` (channels x time), `gammatone_envelope` (their sum), and `gammatone_center_freqs`.
- `BroadbandEnvelopeExtractor` — the Hilbert-transform envelope of the raw audio. Writes `broadband_envelope`.

Both resample to a shared `target_fs` (1024Hz by default) so outputs align in time. Add a new continuous feature by writing a callable with a `name` attribute and a `(audio, fs) -> dict[str, np.ndarray]` signature, then including it in the `extractors` tuple passed to `extract_features` / `extract_features_from_wav`.

Speech unit and pitch extraction workflow:

```bash
uv pip install -e ".[speech-units]"
python scripts/speech_units_pitch/words.py
python scripts/speech_units_pitch/phonemes.py
python scripts/speech_units_pitch/syllabic_nuclei.py
python scripts/speech_units_pitch/intonation_units.py
python -m spacy download nl_core_news_lg
python scripts/speech_units_pitch/sentences_clauses.py
python scripts/speech_units_pitch/formants_and_pitch.py
```

These scripts describe the stimuli at the linguistic timescales the coherence analysis is compared against. They read `<mfa_output_dir>/combined_textgrids/` and `raw_waves_dir`, and write to `speech_units_dir` (all from [src/package_data/config.json](src/package_data/config.json)); shared path plumbing lives in [scripts/speech_units_pitch/_common.py](scripts/speech_units_pitch/_common.py). Onset files are JSON `{audio_name: [onset_seconds, ...]}` and rate files are JSON `{audio_name: units_per_second}`.

| script | outputs | method |
|---|---|---|
| [words.py](scripts/speech_units_pitch/words.py) | `word_onsets.json`, `word_rates.json` | interval starts on the `words` tier |
| [phonemes.py](scripts/speech_units_pitch/phonemes.py) | `phoneme_onsets.json`, `phoneme_rates.json`, `categorical_phoneme_rates.json`, `{vowels,consonants,sonorants,fricatives,plosives}_onsets.json` | interval starts on the `phones` tier, grouped by the `dutch_cv` IPA inventory |
| [syllabic_nuclei.py](scripts/speech_units_pitch/syllabic_nuclei.py) | `nuclei_onsets_{heuristic,dejong}.json`, `syllabic_rates_{heuristic,dejong}.json` | two independent detectors, see below |
| [sentences_clauses.py](scripts/speech_units_pitch/sentences_clauses.py) | `sentence_onsets.json`, `sentence_rates.json`, `clause_onsets.json`, `clause_rates.json` | transcript sentence split + spaCy dependency parse |
| [intonation_units.py](scripts/speech_units_pitch/intonation_units.py) | `iu_onsets.json`, `iu_rates.json`, `word_boundary_scores.json` | wavelet prosody word boundary strength + pooled K-means |
| [formants_and_pitch.py](scripts/speech_units_pitch/formants_and_pitch.py) | `pitch/<stem>_pitch.npz`, `formants/<stem>_formants.npz` | Praat (parselmouth) F0 and Burg formant histograms |

Every script skips MFA's non-speech labels (`<eps>`, `sil`, `spn`, and the blank gaps `recombine_textgrids.py` inserts) so they never contribute an onset or inflate a rate.

`syllabic_nuclei.py --method {heuristic,dejong,both}` runs two detectors that disagree by design, so both are kept:

- **heuristic** — Dutch vowel/diphthong rules over the aligned `phones` tier, with a sonorant fallback for words containing no vowel. Nucleus times are phone midpoints. Needs the forced alignment.
- **dejong** — de Jong & Wempe's `syllable_nuclei_v3.praat` intensity-peak detector run over the raw audio (via the `praat` binary on `PATH`, with Dutch filled-pause detection). Needs no alignment, so it covers every stimulus rather than only the aligned ones. Output `.auto.TextGrid`s are cached in `<speech_units_dir>/dejong_textgrids/` and reused unless `--force` is passed.

`intonation_units.py` needs the [wavelet prosody toolkit](https://github.com/asuni/wavelet_prosody_toolkit), which is not on PyPI. Clone it and set `WAVELET_PROSODY_TOOLKIT_DIR` (it defaults to `/home/gu08wuvu/wavelet_prosody_toolkit`); the `speech-units` extra installs its transitive dependencies. Word boundary strength is computed per stimulus from the combined f0 / energy / duration prosodic signal ([pwt_settings.yaml](scripts/speech_units_pitch/pwt_settings.yaml)) and cached to `word_boundary_scores.json`, so re-running is cheap; K-means (K=2) is then fit on the **pooled** non-zero strengths across all stimuli, and the higher-strength cluster is taken as the intonation unit boundaries. Pooling puts every recording on one common threshold rather than a per-recording one — which also means `iu_onsets.json` shifts as stimuli are added, so regenerate it once the forced alignment covers every recording (the cache makes the refit near-instant).

`sentences_clauses.py` takes its sentence boundaries from the `utterance` tier of `normalised_transcripts_dir`, which is already a sentence-level segmentation: `whisperx.align()` runs each ASR chunk's text through NLTK's Punkt sentence tokenizer and emits one segment per sentence (see `transcribe.py` below). These are therefore Punkt sentence boundaries over Whisper's punctuated text — two models' output stacked, not a human segmentation. Measured against the old project's spaCy-on-punctuated-text sentences, 87% of boundaries agree within 100ms and 93% within 250ms, and 97.4% of utterances contain exactly one spaCy sentence onset. Clauses do need the parse: each sentence goes through spaCy's `nl_core_news_lg`, and every verb-like token carrying a clause dependency label (`ROOT`, `ccomp`, `advcl`, `acl:relcl`, `conj`, ...) starts a clause, extended left over any `mark`/`cc` that introduces it. Onsets for both come from the forced-aligned `words` tier rather than the WhisperX word timings. Reproduces the old clause onsets at r = 0.995.

`formants_and_pitch.py` writes one npz per recording with fields `counts`, `bin_centres`, and `audio_duration`. Pitch bin centres are 0-512Hz at 0.2Hz — the 2561-point frequency axis the coherence spectra use, so the two can be overlaid directly. Formant counts have shape `(5, 800)` for F1-F5 over 0-8000Hz at 10Hz, restricted to frames above the 25th percentile of the intensity contour (formant tracking in near-silence returns noise).
