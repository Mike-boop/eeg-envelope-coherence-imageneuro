import importlib.util
import sys
from pathlib import Path

import mne
import numpy as np
import pytest

mne.set_log_level("WARNING")


def load_get_coherence_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "coherence" / "get_coherence.py"
    spec = importlib.util.spec_from_file_location("get_coherence_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Registered before exec: the module defines a @dataclass, and dataclasses resolves field
    # annotations via sys.modules[cls.__module__], which is None for an unregistered module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_stimulus_features(path: Path, n_subbands: int, n_samples: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    subband_envelopes = rng.standard_normal((n_subbands, n_samples))
    gammatone_envelope = subband_envelopes.sum(axis=0, keepdims=True)
    np.savez(
        path,
        subband_envelopes=subband_envelopes,
        gammatone_envelope=gammatone_envelope,
        gammatone_center_freqs=np.linspace(50.0, 8000.0, n_subbands),
    )
    return np.concatenate([subband_envelopes, gammatone_envelope], axis=0)


def _write_eeg_fif(path: Path, fs: float, n_samples: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ch_names = ["E1", "E2", "E3", "A1", "A2"]
    ch_types = ["eeg", "eeg", "eeg", "misc", "misc"]
    data = rng.standard_normal((len(ch_names), n_samples))
    info = mne.create_info(ch_names, sfreq=fs, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.save(str(path), overwrite=True, verbose=False)
    return data


def test_resolve_configured_inputs_uses_project_config_directories(tmp_path):
    module = load_get_coherence_module()

    config = {
        "audio_features_dir": str(tmp_path / "audio_features"),
        "eeg_dir": str(tmp_path / "eeg"),
        "coherence_dir": str(tmp_path / "coherence"),
        "stimulus_psd_dir": str(tmp_path / "stimulus_psd"),
    }

    audio_features_dir, eeg_dir, coherence_dir, stimulus_psd_dir = module.resolve_configured_inputs(
        config, root=tmp_path
    )

    assert audio_features_dir == (tmp_path / "audio_features").resolve()
    assert eeg_dir == (tmp_path / "eeg").resolve()
    assert coherence_dir == (tmp_path / "coherence").resolve()
    assert stimulus_psd_dir == (tmp_path / "stimulus_psd").resolve()


def test_find_matching_eeg_files_matches_by_audio_name(tmp_path):
    module = load_get_coherence_module()
    eeg_dir = tmp_path / "eeg"
    eeg_dir.mkdir()

    matching = [
        "sub-001_-_ses-001_-_trial-001_-_cond-natural_-_audio01-podcast_14_-_eeg.fif",
        "sub-002_-_ses-001_-_trial-003_-_cond-sin_-_audio01-podcast_14_-_eeg.fif",
    ]
    other = "sub-001_-_ses-001_-_trial-002_-_cond-natural_-_audio01-podcast_15_-_eeg.fif"
    for name in [*matching, other]:
        (eeg_dir / name).write_bytes(b"")

    found = module.find_matching_eeg_files(eeg_dir, "podcast_14")

    assert [path.name for path in found] == sorted(matching)


def test_load_stimulus_features_concatenates_subbands_and_gammatone(tmp_path):
    module = load_get_coherence_module()
    npz_path = tmp_path / "podcast_1.npz"
    expected = _write_stimulus_features(npz_path, n_subbands=4, n_samples=50)

    features = module.load_stimulus_features(npz_path)

    assert features.shape == (5, 50)
    assert np.allclose(features, expected)
    assert np.allclose(features[-1], expected[:-1].sum(axis=0))


def test_load_eeg_drops_non_eeg_channels_and_average_references(tmp_path):
    module = load_get_coherence_module()
    fif_path = tmp_path / "recording_eeg.fif"
    raw_data = _write_eeg_fif(fif_path, fs=200.0, n_samples=400)

    eeg_data, fs = module.load_eeg(fif_path)

    assert fs == 200.0
    assert eeg_data.shape == (3, 400)  # A1/A2 dropped
    expected = raw_data[:3] - raw_data[:3].mean(axis=0, keepdims=True)
    assert np.allclose(eeg_data, expected)


def test_compute_stimulus_psd_shape_and_split(tmp_path):
    module = load_get_coherence_module()
    n_subbands, n_samples, fs = 4, 2048, 100.0
    features = np.random.default_rng(0).standard_normal((n_subbands + 1, n_samples))

    freqs, psd = module.compute_stimulus_psd(features, fs=fs, nperseg=256, noverlap=128)

    assert freqs.shape[0] == psd.shape[0]
    assert psd.shape == (freqs.shape[0], n_subbands + 1)
    assert np.all(psd >= 0)


def test_null_shift_bounds_uses_duration_minus_margin_for_long_recordings():
    module = load_get_coherence_module()
    module.WINDOW_SECONDS = 5.0
    module.NULL_MIN_SHIFT_SECONDS = 60.0

    # podcast_1-sized recording (~953s): plenty of room, so min stays at the requested 1 minute
    # and max is duration minus that same margin.
    min_shift, max_shift = module.null_shift_bounds(953.0)

    assert min_shift == pytest.approx(60.0)
    assert max_shift == pytest.approx(953.0 - 5.0 - 60.0)


def test_null_shift_bounds_shrinks_for_short_recordings():
    """Regression test for the two real accou2023 stimuli (audiobook_7_5 at 119.4s and
    audiobook_7_6 at 121.0s) that are too short for a fixed 1 minute minimum shift — both bounds
    should shrink instead of leaving a range that null_coherence_shuffled_windows would reject."""
    module = load_get_coherence_module()
    module.WINDOW_SECONDS = 5.0
    module.NULL_MIN_SHIFT_SECONDS = 60.0

    for duration in (119.40234375, 121.0234375):
        min_shift, max_shift = module.null_shift_bounds(duration)
        available = duration - module.WINDOW_SECONDS
        assert 0 < min_shift < max_shift
        assert 2.0 * min_shift < available
        assert max_shift <= available - min_shift + 1e-9



def _use_small_windows(monkeypatch, window_seconds=0.5, overlap=0.5):
    """Shrink the Welch window so the fixtures can stay small.

    The windowing constants moved to ``coherence.pipeline``, which is a normal package module
    imported once into ``sys.modules`` -- so unlike the per-test-reloaded script module, assigning
    to them would leak into every later test. Monkeypatch undoes it.
    """
    import coherence.pipeline as pipeline

    monkeypatch.setattr(pipeline, "WINDOW_SECONDS", window_seconds)
    monkeypatch.setattr(pipeline, "OVERLAP_FRACTION", overlap)


def _run_pipeline(module, features_dir, eeg_dir, coherence_dir, stimulus_psd_dir, fs):
    """Drive the whole script over one temp dataset, sequentially.

    N_WORKERS=1 avoids spawning real subprocesses, which could not pickle this dynamically-loaded
    test module.
    """
    nperseg, noverlap = module.stimulus_window_sizes(fs)
    tasks = module.build_tasks(features_dir, eeg_dir, coherence_dir)
    context = module.build_context(features_dir, stimulus_psd_dir, coherence_dir, fs, nperseg, noverlap)
    failures = module.run_tasks(tasks, context, n_workers=1)
    assert failures == [], failures
    return tasks


def _make_dataset(tmp_path, n_subbands=4, n_samples=4000, fs=200.0, eeg_names=None, eeg_samples=None):
    features_dir = tmp_path / "features"
    eeg_dir = tmp_path / "eeg"
    coherence_dir = tmp_path / "coherence"
    stimulus_psd_dir = tmp_path / "stimulus_psd"
    for directory in (features_dir, eeg_dir, coherence_dir, stimulus_psd_dir):
        directory.mkdir()

    _write_stimulus_features(features_dir / "podcast_1.npz", n_subbands=n_subbands, n_samples=n_samples)
    names = eeg_names or ["sub-001_-_ses-001_-_trial-001_-_cond-natural_-_audio01-podcast_1_-_eeg.fif"]
    for index, eeg_name in enumerate(names):
        _write_eeg_fif(eeg_dir / eeg_name, fs=fs, n_samples=eeg_samples or n_samples, seed=index)
    return features_dir, eeg_dir, coherence_dir, stimulus_psd_dir, names


def test_pipeline_writes_expected_outputs(tmp_path, monkeypatch):
    module = load_get_coherence_module()
    _use_small_windows(monkeypatch)
    fs, n_subbands, n_samples = 200.0, 4, 4000

    features_dir, eeg_dir, coherence_dir, stimulus_psd_dir, names = _make_dataset(
        tmp_path, n_subbands=n_subbands, n_samples=n_samples, fs=fs
    )
    _run_pipeline(module, features_dir, eeg_dir, coherence_dir, stimulus_psd_dir, fs)

    stimulus_psd_path = stimulus_psd_dir / "podcast_1_stimulus_psd.npz"
    assert stimulus_psd_path.exists()
    stimulus_psd = np.load(stimulus_psd_path)
    assert str(stimulus_psd["audio_name"]) == "podcast_1"
    assert stimulus_psd["duration"] == pytest.approx(n_samples / fs)
    assert stimulus_psd["subband_psd"].shape[1] == n_subbands
    assert stimulus_psd["gammatone_psd"].shape[1] == 1

    coherence_path = coherence_dir / f"{Path(names[0]).stem}_coherence.npz"
    assert coherence_path.exists()
    coherence_data = np.load(coherence_path)
    assert str(coherence_data["audio_name"]) == "podcast_1"
    n_freqs = coherence_data["frequencies"].shape[0]
    assert coherence_data["subband_coherence"].shape == (n_freqs, n_subbands, 3)
    assert coherence_data["gammatone_coherence"].shape == (n_freqs, 1, 3)
    assert coherence_data["eeg_psd"].shape == (n_freqs, 3)
    assert coherence_data["subband_coherence"].dtype == np.complex64
    assert coherence_data["gammatone_coherence"].dtype == np.complex64

    assert coherence_data["null_subband_coherence"].shape == (module.N_NULL, n_freqs, n_subbands, 3)
    assert coherence_data["null_gammatone_coherence"].shape == (module.N_NULL, n_freqs, 1, 3)
    assert coherence_data["null_subband_coherence"].dtype == np.complex64
    assert coherence_data["null_gammatone_coherence"].dtype == np.complex64

    # Null draws are seeded, so re-running against the same inputs reproduces them exactly.
    coherence_path.unlink()
    _run_pipeline(module, features_dir, eeg_dir, coherence_dir, stimulus_psd_dir, fs)
    repeat_data = np.load(coherence_path)
    assert np.allclose(repeat_data["null_subband_coherence"], coherence_data["null_subband_coherence"])


def test_cached_stimulus_psd_matches_welch(tmp_path, monkeypatch):
    """The saved PSD comes from ``_auto_power`` over the cached spectrum rather than a second
    ``scipy.welch`` pass. Same window, detrend and density scaling, so the two must agree -- and if
    they ever stop agreeing, the stimulus PSDs the pooled scripts divide by would silently shift.
    """
    module = load_get_coherence_module()
    _use_small_windows(monkeypatch)
    fs, n_subbands, n_samples = 200.0, 4, 4000

    features_dir, eeg_dir, coherence_dir, stimulus_psd_dir, _ = _make_dataset(
        tmp_path, n_subbands=n_subbands, n_samples=n_samples, fs=fs
    )
    _run_pipeline(module, features_dir, eeg_dir, coherence_dir, stimulus_psd_dir, fs)

    features = module.load_stimulus_features(features_dir / "podcast_1.npz")
    nperseg, noverlap = module.stimulus_window_sizes(fs)
    welch_freqs, welch_psd = module.compute_stimulus_psd(features, fs, nperseg, noverlap)

    saved = np.load(stimulus_psd_dir / "podcast_1_stimulus_psd.npz")
    saved_psd = np.concatenate([saved["subband_psd"], saved["gammatone_psd"]], axis=1)
    assert np.allclose(saved["frequencies"], welch_freqs)
    assert np.allclose(saved_psd, welch_psd, rtol=1e-10, atol=0.0)


@pytest.mark.parametrize("n_eeg_samples", [3600, 4400])
def test_trial_duration_comes_from_eeg_not_stimulus(tmp_path, monkeypatch, n_eeg_samples):
    """A recording's saved ``duration`` must track the EEG, which in this dataset runs both
    shorter and longer than its stimulus (observed: -601 to +29949 samples).

    Every other test in this file gives the stimulus and the EEG the same number of samples, which
    makes "duration of the stimulus" and "duration of the EEG" indistinguishable -- exactly the
    blind spot that let this be wrong. Both directions are parametrised because only the
    EEG-shorter case changes ``analysed_duration``.
    """
    module = load_get_coherence_module()
    _use_small_windows(monkeypatch)
    fs, n_stim_samples = 200.0, 4000

    features_dir, eeg_dir, coherence_dir, stimulus_psd_dir, names = _make_dataset(
        tmp_path, n_subbands=4, n_samples=n_stim_samples, fs=fs, eeg_samples=n_eeg_samples
    )
    _run_pipeline(module, features_dir, eeg_dir, coherence_dir, stimulus_psd_dir, fs)

    data = np.load(coherence_dir / f"{Path(names[0]).stem}_coherence.npz")
    assert data["duration"] == pytest.approx(n_eeg_samples / fs)
    assert data["duration"] != pytest.approx(n_stim_samples / fs)
    assert data["analysed_duration"] == pytest.approx(min(n_eeg_samples, n_stim_samples) / fs)

    # n_frames is the count actually averaged, i.e. over the truncated overlap of the two.
    nperseg, noverlap = module.stimulus_window_sizes(fs)
    hop = nperseg - noverlap
    assert int(data["n_frames"]) == 1 + (min(n_eeg_samples, n_stim_samples) - nperseg) // hop

    # The stimulus PSD file describes the stimulus, so it keeps the stimulus's own duration.
    stimulus_psd = np.load(stimulus_psd_dir / "podcast_1_stimulus_psd.npz")
    assert stimulus_psd["duration"] == pytest.approx(n_stim_samples / fs)


def test_existing_outputs_are_skipped_when_overwrite_is_false(tmp_path, monkeypatch):
    module = load_get_coherence_module()
    module.OVERWRITE = False
    _use_small_windows(monkeypatch)
    fs = 200.0

    features_dir, eeg_dir, coherence_dir, stimulus_psd_dir, _ = _make_dataset(tmp_path, fs=fs)
    _run_pipeline(module, features_dir, eeg_dir, coherence_dir, stimulus_psd_dir, fs)

    def _fail_load_eeg(*_args, **_kwargs):
        raise AssertionError("load_eeg should not be called when output already exists")

    monkeypatch.setattr(module, "load_eeg", _fail_load_eeg)
    tasks = _run_pipeline(module, features_dir, eeg_dir, coherence_dir, stimulus_psd_dir, fs)
    assert tasks == []


def test_overwrite_recomputes_existing_outputs(tmp_path, monkeypatch):
    """OVERWRITE is what makes a deliberate re-run (e.g. after changing N_NULL) do anything at all;
    without it the skip-if-exists check turns the whole run into a no-op."""
    module = load_get_coherence_module()
    module.OVERWRITE = True
    _use_small_windows(monkeypatch)
    fs = 200.0

    features_dir, eeg_dir, coherence_dir, stimulus_psd_dir, names = _make_dataset(tmp_path, fs=fs)
    coherence_path = coherence_dir / f"{Path(names[0]).stem}_coherence.npz"
    coherence_path.write_bytes(b"")  # a stale, unreadable output

    tasks = _run_pipeline(module, features_dir, eeg_dir, coherence_dir, stimulus_psd_dir, fs)
    assert len(tasks) == 1
    assert np.load(coherence_path)["subband_coherence"].shape[1] == 4


def test_all_pending_recordings_are_processed(tmp_path, monkeypatch):
    module = load_get_coherence_module()
    _use_small_windows(monkeypatch)
    fs = 200.0
    eeg_names = [
        "sub-001_-_ses-001_-_trial-001_-_cond-natural_-_audio01-podcast_1_-_eeg.fif",
        "sub-002_-_ses-001_-_trial-001_-_cond-natural_-_audio01-podcast_1_-_eeg.fif",
    ]
    features_dir, eeg_dir, coherence_dir, stimulus_psd_dir, _ = _make_dataset(
        tmp_path, fs=fs, eeg_names=eeg_names
    )
    _run_pipeline(module, features_dir, eeg_dir, coherence_dir, stimulus_psd_dir, fs)

    for eeg_name in eeg_names:
        assert (coherence_dir / f"{Path(eeg_name).stem}_coherence.npz").exists()


def test_only_pending_recordings_are_processed(tmp_path, monkeypatch):
    """With OVERWRITE off, a recording with an existing output is skipped without touching others."""
    module = load_get_coherence_module()
    module.OVERWRITE = False
    _use_small_windows(monkeypatch)
    fs = 200.0

    done_name = "sub-001_-_ses-001_-_trial-001_-_cond-natural_-_audio01-podcast_1_-_eeg.fif"
    pending_name = "sub-002_-_ses-001_-_trial-001_-_cond-natural_-_audio01-podcast_1_-_eeg.fif"
    features_dir, eeg_dir, coherence_dir, stimulus_psd_dir, _ = _make_dataset(
        tmp_path, fs=fs, eeg_names=[done_name, pending_name]
    )
    (coherence_dir / f"{Path(done_name).stem}_coherence.npz").write_bytes(b"")

    tasks = _run_pipeline(module, features_dir, eeg_dir, coherence_dir, stimulus_psd_dir, fs)
    assert [task["eeg_path"].name for task in tasks] == [pending_name]


def test_stimulus_cache_reuses_one_fft_across_recordings(tmp_path, monkeypatch):
    """The cache is the largest optimisation in this script (617 stimulus FFTs -> ~72), and it is
    invisible in the outputs, so nothing else here would catch it silently regressing."""
    module = load_get_coherence_module()
    _use_small_windows(monkeypatch)
    fs = 200.0
    eeg_names = [
        f"sub-{i:03d}_-_ses-001_-_trial-001_-_cond-natural_-_audio01-podcast_1_-_eeg.fif" for i in range(1, 4)
    ]
    features_dir, eeg_dir, coherence_dir, stimulus_psd_dir, _ = _make_dataset(
        tmp_path, fs=fs, eeg_names=eeg_names
    )

    loads = []
    original = module.load_stimulus_features
    monkeypatch.setattr(
        module, "load_stimulus_features", lambda path: (loads.append(path), original(path))[1]
    )
    _run_pipeline(module, features_dir, eeg_dir, coherence_dir, stimulus_psd_dir, fs)

    assert len(eeg_names) == 3
    assert len(loads) == 1, f"stimulus loaded {len(loads)} times for 3 recordings of one stimulus"
