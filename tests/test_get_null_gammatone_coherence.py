import importlib.util
import sys
from pathlib import Path

import mne
import numpy as np
import pytest

mne.set_log_level("WARNING")

WINDOW_SECONDS = 0.5
OVERLAP_FRACTION = 0.5
STIMULUS_FS = 200.0


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "coherence" / "get_null_gammatone_coherence.py"
    spec = importlib.util.spec_from_file_location("get_null_gammatone_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Registered before exec so the worker pool can pickle _process_trial by qualified name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_stimulus_features(path: Path, n_subbands: int, n_samples: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    subband_envelopes = rng.standard_normal((n_subbands, n_samples))
    np.savez(
        path,
        subband_envelopes=subband_envelopes,
        gammatone_envelope=subband_envelopes.sum(axis=0, keepdims=True),
        gammatone_center_freqs=np.linspace(50.0, 8000.0, n_subbands),
    )


def _write_eeg_fif(path: Path, fs: float, n_samples: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    ch_names = ["E1", "E2", "E3", "A1", "A2"]
    ch_types = ["eeg", "eeg", "eeg", "misc", "misc"]
    info = mne.create_info(ch_names, sfreq=fs, ch_types=ch_types)
    raw = mne.io.RawArray(rng.standard_normal((len(ch_names), n_samples)), info, verbose=False)
    raw.save(str(path), overwrite=True, verbose=False)


@pytest.fixture(autouse=True)
def _small_windows(monkeypatch):
    """Shrink the Welch window for every test in this file.

    The windowing constants live in ``coherence.pipeline``, which is imported once into
    ``sys.modules`` -- so assigning to them would leak into other test files. Monkeypatch undoes it.
    """
    import coherence.pipeline as pipeline

    monkeypatch.setattr(pipeline, "WINDOW_SECONDS", WINDOW_SECONDS)
    monkeypatch.setattr(pipeline, "OVERLAP_FRACTION", OVERLAP_FRACTION)


def _patch_scales(module):
    """Import get_coherence and hold it to one process for fixture building.

    It is imported explicitly rather than fetched from ``sys.modules`` because the module under
    test no longer imports it: both now take their shared constants from ``coherence.pipeline``,
    which is the point of that module. ``load_module`` has already put the script directory on
    ``sys.path``.
    """
    import get_coherence as coherence_module

    coherence_module.N_WORKERS = 1
    module.STIMULUS_FS = STIMULUS_FS
    return coherence_module


def _build_dataset(tmp_path: Path, subjects, stimuli, n_samples=4000, n_subbands=4, eeg_lengths=None):
    """A miniature get_coherence.py-style dataset: every (subject, stimulus) pair is one trial.

    Runs the real get_coherence.py to produce the coherence/PSD fixtures rather than hand-crafting
    them, so the module under test consumes exactly the layout it will meet in production. The
    module's ``load_config`` is patched so ``main()`` resolves every directory into ``tmp_path``.
    """
    module = load_module()
    coherence_module = _patch_scales(module)

    dirs = {name: tmp_path / name for name in ("features", "eeg", "coherence", "psd", "null")}
    for d in dirs.values():
        d.mkdir(exist_ok=True)

    for stim_idx, stimulus in enumerate(stimuli):
        _write_stimulus_features(dirs["features"] / f"{stimulus}.npz", n_subbands, n_samples, seed=stim_idx)
        for sub_idx, subject in enumerate(subjects):
            name = (
                f"sub-{subject}_-_ses-001_-_trial-{stim_idx:03d}_-_cond-natural"
                f"_-_audio01-{stimulus}_-_eeg.fif"
            )
            n_eeg = n_samples if eeg_lengths is None else eeg_lengths[subject]
            _write_eeg_fif(dirs["eeg"] / name, fs=STIMULUS_FS, n_samples=n_eeg, seed=100 * stim_idx + sub_idx)

    nperseg, noverlap = coherence_module.stimulus_window_sizes(STIMULUS_FS)
    tasks = coherence_module.build_tasks(dirs["features"], dirs["eeg"], dirs["coherence"])
    context = coherence_module.build_context(
        dirs["features"], dirs["psd"], dirs["coherence"], STIMULUS_FS, nperseg, noverlap
    )
    assert coherence_module.run_tasks(tasks, context, n_workers=1) == []

    module.load_config = lambda _path: {
        "audio_features_dir": str(dirs["features"]),
        "eeg_dir": str(dirs["eeg"]),
        "coherence_dir": str(dirs["coherence"]),
        "stimulus_psd_dir": str(dirs["psd"]),
        "null_coherence_dir": str(dirs["null"]),
    }
    return module, dirs


def _run_main(module, dirs, n_null=3, workers=2, extra=()):
    assert module.main(["--n-null", str(n_null), "--workers", str(workers), *extra]) == 0
    return {
        grouping: {
            path.name.removesuffix("_gammatone_null.npz"): dict(np.load(path))
            for path in sorted((dirs["null"] / f"gammatone_null_by_{grouping}").glob("*.npz"))
        }
        for grouping in module.GROUPINGS
    }


# ---------------------------------------------------------------------------------------------
# Trial selection and the draw itself
# ---------------------------------------------------------------------------------------------


def test_excludes_sin_av_shifted_and_artefact(tmp_path):
    module = load_module()
    eeg_dir = tmp_path / "eeg"
    eeg_dir.mkdir()
    # Named, not written: build_trial_index parses filenames and reads no sample data.
    names = {
        "sub-001_-_ses-001_-_trial-001_-_cond-natural_-_audio01-podcast_1_-_eeg.fif": True,
        "sub-001_-_ses-001_-_trial-002_-_cond-control_-_audio01-podcast_1_-_eeg.fif": True,
        "sub-001_-_ses-001_-_trial-003_-_cond-sin_-_audio01-podcast_1_-_eeg.fif": False,
        "sub-001_-_ses-001_-_trial-004_-_cond-av_-_audio01-podcast_1_-_eeg.fif": False,
        "sub-001_-_ses-001_-_trial-005_-_cond-natural_-_audio01-podcast_1_shifted_-_eeg.fif": False,
        "sub-001_-_ses-001_-_trial-006_-_cond-natural_-_audio01-podcast_1_artefact_-_eeg.fif": False,
    }
    for name in names:
        (eeg_dir / name).touch()

    kept = set(module.build_trial_index(eeg_dir)["eeg_stem"])
    for name, should_keep in names.items():
        assert (Path(name).stem in kept) is should_keep, f"{name}: expected keep={should_keep}"


def test_shifted_cross_spectrum_matches_roll_reference():
    """The no-copy two-slice matmul must equal coherence.py's roll-then-matmul."""
    module = load_module()
    from coherence.coherence import _conj_transpose_for_matmul, _cross_spectrum, windowed_fft

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3000))
    y = rng.standard_normal((3, 3000))
    xs = windowed_fft(x, fs=200.0, nperseg=100, noverlap=50, axis=1)
    ys = windowed_fft(y, fs=200.0, nperseg=100, noverlap=50, axis=1)
    Xf_t = _conj_transpose_for_matmul(xs)
    Yf_t = ys.Xf.transpose(1, 0, 2)

    for shift in (0, 1, 13, ys.n_frames - 1):
        expected = _cross_spectrum(Xf_t, np.roll(ys.Xf, shift, axis=0), xs.scale, 100)
        actual = module.shifted_cross_spectrum(Xf_t, Yf_t, shift, xs.scale, 100)
        assert np.allclose(expected, actual), f"shift={shift}"


def test_contiguous_complex64_layout_matches_the_strided_float64_path():
    """The 6.6x layout fix and the complex64 working precision must not change the answer.

    ``contiguous_for_matmul`` exists purely for speed -- it removes a 36 MB frequency-axis stride
    that the batched matmul otherwise pays on every draw. This pins that it is the same
    computation to complex64 roundoff, which is the precision the result is stored at anyway.
    """
    module = load_module()
    from coherence.coherence import _conj_transpose_for_matmul, windowed_fft

    rng = np.random.default_rng(0)
    xs = windowed_fft(rng.standard_normal((1, 3000)), fs=200.0, nperseg=100, noverlap=50, axis=1)
    ys = windowed_fft(rng.standard_normal((3, 3000)), fs=200.0, nperseg=100, noverlap=50, axis=1)

    Xf_t_view = _conj_transpose_for_matmul(xs)
    Yf_t_view = ys.Xf.transpose(1, 0, 2)
    Xf_t_fast = Xf_t_view.astype(module.DRAW_DTYPE, order="C")
    Yf_t_fast = module.contiguous_for_matmul(ys.Xf, module.DRAW_DTYPE)

    assert Yf_t_fast.flags["C_CONTIGUOUS"]
    assert Yf_t_fast.dtype == module.DRAW_DTYPE
    assert np.array_equal(Yf_t_fast, Yf_t_view.astype(module.DRAW_DTYPE))  # same values, better layout

    for shift in (0, 7, 13):
        reference = module.shifted_cross_spectrum(Xf_t_view, Yf_t_view, shift, xs.scale, 100)
        fast = module.shifted_cross_spectrum(Xf_t_fast, Yf_t_fast, shift, xs.scale, 100)
        scale = np.abs(reference).max()
        assert np.abs(fast - reference).max() / scale < 1e-5, f"shift={shift}"


# ---------------------------------------------------------------------------------------------
# The single pass: both groupings from one set of draws
# ---------------------------------------------------------------------------------------------


def test_one_pass_produces_both_groupings_over_the_same_trials(tmp_path):
    subjects, stimuli = ["001", "002", "003"], ["podcast_1", "podcast_2"]
    module, dirs = _build_dataset(tmp_path, subjects, stimuli)
    results = _run_main(module, dirs)

    assert set(results["stimulus"]) == set(stimuli)
    assert set(results["subject"]) == set(subjects)
    for grouping in module.GROUPINGS:
        assert sum(int(r["n_trials"]) for r in results[grouping].values()) == len(subjects) * len(stimuli)
    stems_stim = sorted(s for r in results["stimulus"].values() for s in r["trial_stems"])
    stems_sub = sorted(s for r in results["subject"].values() for s in r["trial_stems"])
    assert stems_stim == stems_sub  # every trial counted exactly once in each grouping


def test_both_groupings_recombine_to_the_same_grand_average(tmp_path):
    """Re-pooling either grouping by its own weights must give the same overall average.

    This is the property the single-pass design rests on, and now tests it directly: both
    groupings are fed from one shared draw stack per trial, so the two sets of group averages are
    two partitions of one set of per-trial values and must recombine identically.
    """
    subjects, stimuli = ["001", "002", "003"], ["podcast_1", "podcast_2"]
    module, dirs = _build_dataset(tmp_path, subjects, stimuli)
    results = _run_main(module, dirs)

    def grand(groups):
        total = sum(float(r["total_duration"]) for r in groups.values())
        null = sum(r["null_gammatone_coherence"].astype(np.complex128) * float(r["total_duration"]) for r in groups.values())
        real = sum(r["gammatone_coherence"].astype(np.complex128) * float(r["total_duration"]) for r in groups.values())
        return null / total, real / total

    null_stim, real_stim = grand(results["stimulus"])
    null_sub, real_sub = grand(results["subject"])

    assert np.allclose(null_stim, null_sub, rtol=1e-4, atol=1e-8)
    assert np.allclose(real_stim, real_sub, rtol=1e-4, atol=1e-8)


def test_group_average_is_duration_weighted_not_plain_mean(tmp_path):
    """With unequal trial lengths the result must follow the duration weights."""
    lengths = {"001": 6000, "002": 3000}  # same stimulus, very different EEG lengths
    module, dirs = _build_dataset(
        tmp_path, list(lengths), ["podcast_1"], n_samples=6000, eeg_lengths=lengths
    )
    results = _run_main(module, dirs)

    group = results["stimulus"]["podcast_1"]
    assert int(group["n_trials"]) == 2
    assert float(group["total_duration"]) == pytest.approx(sum(lengths.values()) / STIMULUS_FS)
    # Durations come from each EEG recording, not from the (single, shared) stimulus.
    assert sorted(group["trial_durations"]) == pytest.approx(sorted(n / STIMULUS_FS for n in lengths.values()))


# ---------------------------------------------------------------------------------------------
# Resume: the journal must neither double-count nor drop a trial
# ---------------------------------------------------------------------------------------------


def test_resume_from_a_partial_journal_matches_an_uninterrupted_run(tmp_path):
    """Half a group folded in, then main() finishes it -- must equal a clean run.

    ``+=`` is not idempotent, so this is the property the journal exists to protect: a resumed run
    has to pick up exactly the trials that are missing, and no others.
    """
    subjects, stimuli = ["001", "002", "003"], ["podcast_1", "podcast_2"]
    module, dirs = _build_dataset(tmp_path, subjects, stimuli)
    clean = _run_main(module, dirs)

    # Wipe the outputs, then hand-fold one trial of one stimulus group before re-running.
    for grouping in module.GROUPINGS:
        for path in (dirs["null"] / f"gammatone_null_by_{grouping}").glob("*.npz"):
            path.unlink()
    work_dir = dirs["null"] / "gammatone_null_work"
    df = module.build_trial_index(dirs["eeg"])
    row = df[df["audio_name"] == "podcast_1"].iloc[0]

    nperseg, noverlap = module.stimulus_window_sizes(STIMULUS_FS)
    cache = module.StimulusCache(dirs["features"], dirs["psd"], STIMULUS_FS, nperseg, noverlap)
    stack, real, duration, analysed, _ = module.compute_trial_draws(
        row["audio_name"], row["eeg_stem"], "avg", dirs["eeg"], cache, STIMULUS_FS, nperseg, 3
    )
    # The real coherence is computed in this pass rather than read from get_coherence.py's
    # per-trial output, so it has to agree with that output to complex64 roundoff.
    trial = np.load(dirs["coherence"] / f"{row['eeg_stem']}_coherence.npz")
    assert np.allclose(real, trial["gammatone_coherence"], rtol=1e-5, atol=1e-10)
    store = module.GroupStore(work_dir, "stimulus", "podcast_1")
    store.create(stack.shape, real.shape)
    store.fold(stack, real, row["eeg_stem"], duration, analysed)
    assert store.folded_stems() == {row["eeg_stem"]}

    resumed = _run_main(module, dirs)
    for grouping in module.GROUPINGS:
        for group_id, expected in clean[grouping].items():
            got = resumed[grouping][group_id]
            assert int(got["n_trials"]) == int(expected["n_trials"])
            assert np.allclose(got["null_gammatone_coherence"], expected["null_gammatone_coherence"], rtol=1e-5, atol=1e-10)
            assert np.allclose(got["gammatone_coherence"], expected["gammatone_coherence"], rtol=1e-5, atol=1e-10)


def test_a_surviving_pending_marker_resets_that_group(tmp_path):
    """A group caught mid-fold is untrustworthy, so it must be rebuilt rather than resumed.

    The marker stands for "a += may have half-landed". The journal alone cannot tell, so the group
    is discarded -- here its accumulator is deliberately poisoned to prove the discard happens and
    the poisoned values do not reach the output.
    """
    subjects, stimuli = ["001", "002"], ["podcast_1"]
    module, dirs = _build_dataset(tmp_path, subjects, stimuli)
    clean = _run_main(module, dirs)

    for grouping in module.GROUPINGS:
        for path in (dirs["null"] / f"gammatone_null_by_{grouping}").glob("*.npz"):
            path.unlink()
    work_dir = dirs["null"] / "gammatone_null_work"
    store = module.GroupStore(work_dir, "stimulus", "podcast_1")
    shape = clean["stimulus"]["podcast_1"]["null_gammatone_coherence"].shape
    store.create((3, *shape[1:]) if len(shape) == 4 else shape, clean["stimulus"]["podcast_1"]["gammatone_coherence"].shape)
    poisoned = np.lib.format.open_memmap(store.null_path, mode="r+")
    poisoned[:] = 1e9
    poisoned.flush()
    del poisoned
    store.journal_path.write_text("sub-001_-_ses-001_-_trial-000_-_cond-natural_-_audio01-podcast_1_-_eeg\t1.0\t1.0\n")
    store.pending_path.write_text("interrupted")

    resumed = _run_main(module, dirs)
    got = resumed["stimulus"]["podcast_1"]
    assert int(got["n_trials"]) == len(subjects)
    assert np.allclose(got["null_gammatone_coherence"], clean["stimulus"]["podcast_1"]["null_gammatone_coherence"], rtol=1e-5, atol=1e-10)


def test_refuses_to_mix_draw_counts_in_one_output_dir(tmp_path):
    """A finished run at another n_null is worth keeping as a cross-check, but must not be resumed into."""
    subjects, stimuli = ["001"], ["podcast_1"]
    module, dirs = _build_dataset(tmp_path, subjects, stimuli)
    _run_main(module, dirs, n_null=3)

    with pytest.raises(SystemExit, match="n_null=3"):
        module.main(["--n-null", "5", "--workers", "1"])


# ---------------------------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------------------------


def _with_reference_dirs(module, dirs, reference, target):
    """Point one reference at its own null directory, leaving the rest as the fixture had them."""
    base = module.load_config(None)
    module.load_config = lambda _path: {
        **base,
        "references": {reference: {"coherence_dir": base["coherence_dir"],
                                   "null_coherence_dir": str(target)}},
    }


def test_linked_mastoid_reference_differs_from_average_and_is_recorded(tmp_path):
    """A reference changes the numbers, and every output says which reference produced it."""
    module, dirs = _build_dataset(tmp_path, subjects=["001", "002"], stimuli=["podcast_1"])
    average = _run_main(module, dirs)

    target = tmp_path / "null_linked"
    _with_reference_dirs(module, dirs, "linked_mastoids", target)
    assert module.main(["--n-null", "3", "--workers", "1", "--reference", "linked_mastoids"]) == 0

    saved = dict(np.load(target / "gammatone_null_by_stimulus" / "podcast_1_gammatone_null.npz"))
    assert str(saved["reference"]) == "linked_mastoids"
    # The fixture's EEG is E1/E2/E3 plus misc A1/A2, so the axis stays the three scalp channels.
    assert list(saved["channel_names"]) == ["E1", "E2", "E3"]
    assert saved["skipped_stems"].size == 0
    assert not np.allclose(saved["gammatone_coherence"],
                           average["stimulus"]["podcast_1"]["gammatone_coherence"])


def test_recordings_without_mastoids_are_skipped_only_for_linked_mastoids(tmp_path):
    """The 21-recordings-without-A1/A2 case: dropped up front, listed, and out of the denominator."""
    module, dirs = _build_dataset(tmp_path, subjects=["001", "002"], stimuli=["podcast_1"])
    dropped = sorted(dirs["eeg"].glob("sub-002_*.fif"))[0]
    raw = mne.io.read_raw_fif(dropped, preload=True, verbose=False)
    raw.drop_channels(["A1", "A2"]).save(str(dropped), overwrite=True, verbose=False)

    index = module.build_trial_index(dirs["eeg"])
    kept, skipped = module.filter_trials_for_reference(index, dirs["eeg"], "linked_mastoids")
    assert skipped == [dropped.stem] and len(kept) == len(index) - 1
    # Every other reference keeps it: only linked mastoids needs A1/A2.
    for reference in ("avg", "FCz"):
        assert module.filter_trials_for_reference(index, dirs["eeg"], reference)[1] == []

    target = tmp_path / "null_linked"
    _with_reference_dirs(module, dirs, "linked_mastoids", target)
    assert module.main(["--n-null", "3", "--workers", "1", "--reference", "linked_mastoids"]) == 0

    # sub-002's only recording was the dropped one, so that subject group has nothing to average
    # and must not be written -- an all-NaN file would look like a finished group.
    assert not (target / "gammatone_null_by_subject" / "002_gammatone_null.npz").exists()
    assert (target / "gammatone_null_by_subject" / "001_gammatone_null.npz").exists()

    saved = dict(np.load(target / "gammatone_null_by_stimulus" / "podcast_1_gammatone_null.npz"))
    assert list(saved["skipped_stems"]) == [dropped.stem]
    assert dropped.stem not in list(saved["trial_stems"])
    assert int(saved["n_trials"]) == len(kept)
    # Its duration never enters the denominator.
    assert float(saved["total_duration"]) == pytest.approx(float(saved["trial_durations"].sum()))


def test_refuses_to_mix_references_in_one_output_dir(tmp_path):
    """Without a per-reference directory the files would be indistinguishable, so this is fatal."""
    module, dirs = _build_dataset(tmp_path, subjects=["001"], stimuli=["podcast_1"])
    _run_main(module, dirs)  # average-referenced, into dirs["null"]

    # No references block, so linked mastoids resolves to the very same directory.
    with pytest.raises(SystemExit, match="already holds avg-referenced results"):
        module.main(["--n-null", "3", "--workers", "1", "--reference", "linked_mastoids"])
