import importlib.util
import sys
from pathlib import Path

import numpy as np

N_NULL, N_FREQS, N_EEG = 2, 3, 4
N_SUBBANDS, N_CHANNELS = 2, 3
TOTAL_DURATION = 2.0


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "coherence" / "get_null_cochleogram_coherence.py"
    spec = importlib.util.spec_from_file_location("get_null_cochleogram_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _accumulators(module, tmp_path: Path):
    return module.ChannelAccumulators(
        tmp_path / "work", tmp_path / "out", N_CHANNELS, N_SUBBANDS, N_NULL, N_FREQS, N_EEG
    )


def _fill(accumulators) -> None:
    """A distinct non-zero value per channel, so a zeroed rewrite cannot pass as the real thing."""
    accumulators.create()
    for channel, handle in enumerate(accumulators.open()):
        handle[...] = channel + 1
        handle.flush()


def _finalize(module, accumulators, tmp_path: Path) -> int:
    return module.finalize_channels(
        accumulators,
        real_sum=np.ones((N_FREQS, N_CHANNELS, N_EEG), dtype=np.complex128),
        output_dir=tmp_path / "out",
        freqs=np.arange(N_FREQS, dtype=float),
        centre_freqs=np.linspace(50.0, 8000.0, N_SUBBANDS),
        n_subbands=N_SUBBANDS,
        total_duration=TOTAL_DURATION,
        total_analysed=TOTAL_DURATION,
        stems=["sub-01_task-a"],
        durations=[TOTAL_DURATION],
        n_null=N_NULL,
        reference="avg",
        channel_names=[f"E{i}" for i in range(N_EEG)],
        skipped_stems=[],
    )


def test_finalize_writes_every_channel_and_removes_its_accumulator(tmp_path):
    module = load_module()
    accumulators = _accumulators(module, tmp_path)
    _fill(accumulators)

    assert _finalize(module, accumulators, tmp_path) == N_CHANNELS
    for channel, output_path in enumerate(accumulators.output_paths):
        assert np.allclose(np.load(output_path)["null_coherence"], (channel + 1) / TOTAL_DURATION)
    assert not any(path.exists() for path in accumulators.paths)


def test_a_rerun_after_a_finished_run_rewrites_nothing(tmp_path):
    """The regression: ``create`` used to put a zeroed .npy back for every finalised channel, and
    finalize then overwrote 61 finished .npz files with zeros -- right shape, right dtype, finite,
    so nothing downstream would have caught it."""
    module = load_module()
    accumulators = _accumulators(module, tmp_path)
    _fill(accumulators)
    _finalize(module, accumulators, tmp_path)
    before = {path.name: np.load(path)["null_coherence"] for path in accumulators.output_paths}

    resumed = _accumulators(module, tmp_path)
    resumed.create()
    assert not any(path.exists() for path in resumed.paths)
    assert all(handle is None for handle in resumed.open())
    assert resumed.finalized_channels() == list(range(N_CHANNELS))

    assert _finalize(module, resumed, tmp_path) == 0
    for path in resumed.output_paths:
        assert np.array_equal(np.load(path)["null_coherence"], before[path.name])


def test_a_crash_inside_finalize_resumes_at_the_remaining_channels(tmp_path):
    module = load_module()
    accumulators = _accumulators(module, tmp_path)
    _fill(accumulators)

    # Exactly what finalize_channels leaves behind after channel 0: the .npz renamed into place
    # and the .npy unlinked. Its contents are a marker, so a rewrite is visible.
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    marker = np.arange(5.0)
    np.savez(accumulators.output_paths[0], null_coherence=marker)
    accumulators.paths[0].unlink()

    resumed = _accumulators(module, tmp_path)
    resumed.create()
    assert [path.exists() for path in resumed.paths] == [False, True, True]
    assert resumed.finalized_channels() == [0]

    assert _finalize(module, resumed, tmp_path) == N_CHANNELS - 1
    assert np.array_equal(np.load(resumed.output_paths[0])["null_coherence"], marker)
    for channel in (1, 2):
        assert np.allclose(np.load(resumed.output_paths[channel])["null_coherence"], (channel + 1) / TOTAL_DURATION)
        assert not resumed.paths[channel].exists()


def test_a_finalised_channel_is_not_charged_for_disk_space_twice(tmp_path):
    module = load_module()
    accumulators = _accumulators(module, tmp_path)
    _fill(accumulators)
    allocated = accumulators.allocated_bytes()

    _finalize(module, accumulators, tmp_path)
    # The accumulators are gone but their bytes are still on disk as .npz, so a resumed run must
    # not be told it needs the whole 39.99 GB working set again.
    assert accumulators.allocated_bytes() > 0.5 * allocated
