#!/usr/bin/env python3
"""
Extracts intonation unit onsets from the speech files (.wav files).

Computes word boundary strength using the pwt toolbox. Pools the WBS and applies K-means (K=2) clustering to identify the intonation unit boundaries.
Outputs a JSON file with keys equal to audio names and values equal to a list of IU onsets (in seconds) for each audio file.

Based on the notebooks:
- /home/gu08wuvu/neural_tracking_across_spectrum/stimulus_analysis/intonational_units/get_IUs.py
- /home/gu08wuvu/neural_tracking_across_spectrum/stimulus_analysis/intonational_units/get_IUs_pooled.ipynb

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from sklearn.cluster import KMeans
from tqdm import tqdm

from _common import SCRIPT_DIR, is_speech_label, load_paths, paired_files, write_json

# The wavelet prosody toolkit is not on PyPI; it lives as a git checkout.
# Point WAVELET_PROSODY_TOOLKIT_DIR at it, or pip install the checkout.
TOOLKIT_DIR = Path(
    os.environ.get("WAVELET_PROSODY_TOOLKIT_DIR", "/home/gu08wuvu/wavelet_prosody_toolkit")
).expanduser()

try:
    import wavelet_prosody_toolkit  # noqa: F401
except ImportError:  # pragma: no cover - environment plumbing
    if not TOOLKIT_DIR.is_dir():
        raise ImportError(
            "wavelet_prosody_toolkit not importable and no checkout found at "
            f"{TOOLKIT_DIR}. Clone https://github.com/asuni/wavelet_prosody_toolkit "
            "and set WAVELET_PROSODY_TOOLKIT_DIR, or pip install it."
        ) from None
    sys.path.insert(0, str(TOOLKIT_DIR))

from wavelet_prosody_toolkit.prosody_tools import (  # noqa: E402
    cwt_utils,
    duration_processing,
    energy_processing,
    f0_processing,
    lab,
    loma,
    misc,
    smooth_and_interp,
)


def _patch_numpy2_compat() -> None:
    """
    The toolkit predates NumPy 2. Its only incompatible call is `ndarray.ptp()`
    in pitch_tracker.inst_freq_pitch, which this repo's numpy==2.2.5 removed.
    Re-exec that one module from source with the call rewritten to `np.ptp(...)`,
    in place, so f0_processing's reference to it picks the fix up. A no-op once
    upstream fixes it.
    """
    from wavelet_prosody_toolkit.prosody_tools import pitch_tracker

    source_path = Path(pitch_tracker.__file__)
    source = source_path.read_text(encoding="utf-8")
    patched = source.replace("mags.ptp()", "np.ptp(mags)")
    if patched != source:
        exec(compile(patched, str(source_path), "exec"), pitch_tracker.__dict__)  # noqa: S102


_patch_numpy2_compat()


SETTINGS_PATH = SCRIPT_DIR / "pwt_settings.yaml"
# lab.read_textgrid returns annotation times in frames at this rate.
LABEL_SAMPLE_RATE = 200.0
# K-means is seeded so the pooled boundary threshold is reproducible across runs.
KMEANS_SEED = 0


def load_cfg(settings_path: Path) -> dict:
    """Load the pwt configuration. `pwt_settings.yaml` is a complete config, not a patch."""
    with settings_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.load(handle, Loader=yaml.FullLoader)
    return deepcopy(cfg)


def compute_wbs(wav_path: Path, textgrid_path: Path, cfg: dict) -> dict:
    """
    Word-level boundary strength (WBS) for one wav + TextGrid.

    Ported from ntracking.wbs.compute_wbs_from_yaml. Returns the annotation
    intervals used, the prominence and boundary matrices, and the WBS itself
    (the last column of the boundary matrix, one value per word).
    """
    # 1) audio
    orig_sr, sig = misc.read_wav(str(wav_path))

    # 2) energy
    energy = energy_processing.extract_energy(
        sig,
        orig_sr,
        cfg["energy"]["band_min"],
        cfg["energy"]["band_max"],
        cfg["energy"]["calculation_method"],
    )
    energy = np.cbrt(energy + 1)
    if cfg["energy"].get("smooth_energy", False):
        energy = smooth_and_interp.peak_smooth(energy, 30, 3)
        energy = smooth_and_interp.smooth(energy, 10)

    # 3) f0
    raw_pitch = f0_processing.extract_f0(
        sig,
        orig_sr,
        f0_min=cfg["f0"]["min_f0"],
        f0_max=cfg["f0"]["max_f0"],
        voicing=cfg["f0"]["voicing_threshold"],
        configuration=cfg["f0"]["pitch_tracker"],
    )
    pitch = f0_processing.process(raw_pitch)

    # 4) annotations
    tiers = lab.read_textgrid(str(textgrid_path))

    # 5) duration / speech rate
    if not cfg["duration"]["acoustic_estimation"]:
        dur_tiers = []
        for level in cfg["duration"]["duration_tiers"]:
            key = level.lower()
            if key not in tiers:
                raise KeyError(f'Duration tier "{level}" not found. Available tiers: {list(tiers.keys())}')
            dur_tiers.append(tiers[key])

        rate = duration_processing.get_duration_signal(
            dur_tiers,
            weights=cfg["duration"]["weights"],
            linear=cfg["duration"]["linear"],
            sil_symbols=cfg["duration"]["silence_symbols"],
            bump=cfg["duration"]["bump"],
        )
    else:
        rate = duration_processing.get_rate(energy)
        rate = smooth_and_interp.smooth(rate, 30)

    if cfg["duration"].get("delta_duration", False):
        rate = np.diff(rate)

    # 6) align lengths
    min_length = int(np.min([len(pitch), len(energy), len(rate)]))
    pitch = pitch[:min_length]
    energy = energy[:min_length]
    rate = rate[:min_length]

    # 7) combine into one prosodic parameter
    weights = cfg["feature_combination"]["weights"]
    if cfg["feature_combination"]["type"] == "product":
        params = (
            misc.normalize_minmax(pitch) ** weights["f0"]
            * misc.normalize_minmax(energy) ** weights["energy"]
            * misc.normalize_minmax(rate) ** weights["duration"]
        )
    else:
        params = (
            misc.normalize_std(pitch) * weights["f0"]
            + misc.normalize_std(energy) * weights["energy"]
            + misc.normalize_std(rate) * weights["duration"]
        )

    if cfg["feature_combination"].get("detrend", False):
        params = smooth_and_interp.remove_bias(params, 800)
    params = misc.normalize_std(params)

    # 8) continuous wavelet transform
    cwt, scales, freqs = cwt_utils.cwt_analysis(
        params,
        mother_name=cfg["wavelet"]["mother_wavelet"],
        period=cfg["wavelet"]["period"],
        num_scales=cfg["wavelet"]["num_scales"],
        scale_distance=cfg["wavelet"]["scale_distance"],
        apply_coi=False,
    )
    cwt = np.real(cwt)
    scales = (1.0 / freqs * LABEL_SAMPLE_RATE) * 0.5

    # 9) tier the boundaries are reported on
    ann_tier = cfg["labels"]["annotation_tier"].lower()
    if ann_tier not in tiers:
        raise KeyError(f'Annotation tier "{ann_tier}" not found. Available tiers: {list(tiers.keys())}')
    labels = tiers[ann_tier]

    # 10) lines of maximum amplitude -> prominences and boundaries
    scale_dist = cfg["wavelet"]["scale_distance"]
    unit_scale = misc.get_best_scale2(scales, labels)

    pos_loma = loma.get_loma(
        cwt,
        scales,
        unit_scale + int(cfg["loma"]["prom_start"] / scale_dist),
        unit_scale + int(cfg["loma"]["prom_end"] / scale_dist),
    )
    neg_loma = loma.get_loma(
        -cwt,
        scales,
        unit_scale + int(cfg["loma"]["boundary_start"] / scale_dist),
        unit_scale + int(cfg["loma"]["boundary_end"] / scale_dist),
    )

    max_loma = loma.get_prominences(pos_loma, labels)
    boundaries = np.array(loma.get_boundaries(max_loma, neg_loma, labels))

    if boundaries.ndim != 2 or boundaries.shape[0] != len(labels):
        raise ValueError(f"Unexpected boundaries shape: {boundaries.shape} (labels={len(labels)})")

    return {
        "labels": labels,
        "prominences": np.array(max_loma),
        "boundaries": boundaries,
        "wbs": boundaries[:, -1].astype(float),
    }


def textgrid_duration(textgrid_path: Path) -> float:
    """Recording duration from the TextGrid header, used to turn IU counts into rates."""
    for line in textgrid_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("xmax ="):
            return float(line.split("=", 1)[1])
    raise ValueError(f"No xmax in {textgrid_path}")


def compute_all_wbs(paths, cache_path: Path, *, force: bool) -> dict[str, dict]:
    """WBS per stimulus, cached to JSON — the expensive step, so it is resumable."""
    cached: dict[str, dict] = {}
    if cache_path.exists() and not force:
        with cache_path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)

    cfg = load_cfg(SETTINGS_PATH)
    pairs = paired_files(paths.textgrid_dir, paths.audio_dir)
    todo = [pair for pair in pairs if pair[0] not in cached]
    if not todo:
        print(f"  all {len(pairs)} stimuli already in {cache_path.name}")

    for audio_name, textgrid_path, wav_path in tqdm(todo, desc="Word boundary strength", unit="file"):
        output = compute_wbs(wav_path, textgrid_path, cfg)
        # labels are [start_frame, end_frame, text] at LABEL_SAMPLE_RATE; the
        # boundary sits at the *end* of each annotated word.
        cached[audio_name] = {
            "wbs": output["wbs"].tolist(),
            "onsets": [label[1] / LABEL_SAMPLE_RATE for label in output["labels"]],
            "labels": [label[2] for label in output["labels"]],
            "duration": textgrid_duration(textgrid_path),
        }
        # rewritten after every stimulus so an interrupted run resumes
        write_json(cached, cache_path, quiet=True)

    return cached


def _speech_wbs(entry: dict) -> np.ndarray:
    """WBS with the boundaries of MFA silence intervals zeroed out."""
    wbs = np.asarray(entry["wbs"], dtype=float)
    labels = entry.get("labels")
    if labels is None:  # cache written before labels were stored
        return wbs
    return np.where([is_speech_label(label) for label in labels], wbs, 0.0)


def pooled_iu_onsets(wbs_data: dict[str, dict]) -> tuple[dict[str, list[float]], KMeans, int]:
    """
    Split the pooled non-zero WBS into two clusters and keep the stronger one.

    Pooling across stimuli (rather than clustering each stimulus separately, as
    get_IUs.py did) puts every recording on one common boundary threshold.
    """
    scored = [_speech_wbs(entry) for entry in wbs_data.values()]
    all_wbs = np.concatenate(scored)
    nonzero = all_wbs[all_wbs != 0].reshape(-1, 1)

    kmeans = KMeans(n_clusters=2, n_init=10, random_state=KMEANS_SEED).fit(nonzero)
    # Cluster labels are arbitrary; the intonation-unit boundaries are the
    # higher-strength cluster.
    target_label = int(np.argmax(kmeans.cluster_centers_.ravel()))

    iu_onsets: dict[str, list[float]] = {}
    for audio_name, entry in wbs_data.items():
        wbs = _speech_wbs(entry)
        onsets = np.asarray(entry["onsets"], dtype=float)
        is_boundary = (kmeans.predict(wbs.reshape(-1, 1)) == target_label) & (wbs != 0)
        iu_onsets[audio_name] = onsets[is_boundary].tolist()

    return iu_onsets, kmeans, target_label


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute word boundary strengths instead of reusing the cached JSON.",
    )
    args = parser.parse_args()

    paths = load_paths(require_textgrids=True, require_audio=True)
    cache_path = paths.output_dir / "word_boundary_scores.json"

    wbs_data = compute_all_wbs(paths, cache_path, force=args.force)
    iu_onsets, kmeans, target_label = pooled_iu_onsets(wbs_data)

    centres = np.sort(kmeans.cluster_centers_.ravel())
    print(f"Pooled K-means cluster centres: {centres.round(3)} (boundary cluster = {target_label})")

    iu_rates: dict[str, float] = {}
    for audio_name, onsets in iu_onsets.items():
        duration = wbs_data[audio_name]["duration"]
        iu_rates[audio_name] = len(onsets) / duration
        print(f"{audio_name}: {len(onsets)} IUs, {iu_rates[audio_name]:.2f} Hz")

    write_json(iu_onsets, paths.output_dir / "iu_onsets.json")
    write_json(iu_rates, paths.output_dir / "iu_rates.json")

    print(f"Wrote intonation unit onsets for {len(iu_onsets)} recording(s) to {paths.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
