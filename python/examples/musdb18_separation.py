"""RASE demo: DM-GMM and Def-MAP separation on MUSDB18 (7s preview).

Not a reproduction of the 2019 thesis numbers (A-Volute training data is
lost) -- this only checks that both separators do better than a naive
"mixture as estimate" baseline on real music.

Def-MAP is run as an *oracle* test: its library is the held-out test
track's own true vocals/accompaniment stems (n_sounds=1 per source), which
is exactly Def-MAP's designed use case (a known candidate sound per
source), not a blind-separation test. DM-GMM is trained statistically on a
handful of *other* tracks and applied to the held-out track, which is its
designed use case (generalization from training data).

N_FREQ_BINS is much smaller here than in the ESC-50 demo: DM-GMM inverts a
(2*freqbin, 2*freqbin) covariance matrix per component pair per frame, which
the thesis's classification-scale freqbin (205+) makes intractable.

Usage: MUSDB_ROOT=/path/to/musdb18-7s python musdb18_separation.py
"""

from __future__ import annotations

import os

import musdb
import numpy as np

from gasm.common.power_spectrum import complex_spectrum, frame_signal
from gasm.rase.defmap import DefMAPSeparator
from gasm.rase.dmgmm import DMGMMSeparator

FRAME_LENGTH = 2048
SHIFT_LENGTH = 2048
N_FREQ_BINS = 32
N_TRAIN_TRACKS_FOR_DMGMM = 5
DMGMM_COMPONENTS = [4, 4]


def _mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=-1) if audio.ndim > 1 else audio


def _track_spectrum(audio: np.ndarray) -> np.ndarray:
    frames = frame_signal(_mono(audio.astype(np.float64)), FRAME_LENGTH, SHIFT_LENGTH)
    return complex_spectrum(frames, N_FREQ_BINS)


def _spectral_sdr(estimate: np.ndarray, truth: np.ndarray) -> float:
    """10*log10(||truth||^2 / ||estimate-truth||^2) on complex spectra.
    NOT the standard time-domain BSS-eval SDR -- a quick, spectrum-domain
    stand-in, good enough to check "better than doing nothing"."""
    error_power = float(np.sum(np.abs(estimate - truth) ** 2))
    if error_power == 0.0:
        return float("inf")
    truth_power = float(np.sum(np.abs(truth) ** 2))
    return 10.0 * np.log10(truth_power / error_power)


def main() -> None:
    root = os.environ.get("MUSDB_ROOT")
    mus_train = musdb.DB(root=root, subsets="train", download=True)
    mus_test = musdb.DB(root=root, subsets="test", download=True)

    test_track = mus_test.tracks[0]
    mix_spec = _track_spectrum(test_track.audio)
    vocals_spec = _track_spectrum(test_track.targets["vocals"].audio)
    accompaniment_spec = _track_spectrum(test_track.audio - test_track.targets["vocals"].audio)

    baseline_vocals_sdr = _spectral_sdr(mix_spec, vocals_spec)
    baseline_accompaniment_sdr = _spectral_sdr(mix_spec, accompaniment_spec)
    print(f"naive baseline (mixture as estimate): vocals {baseline_vocals_sdr:.1f} dB, "
          f"accompaniment {baseline_accompaniment_sdr:.1f} dB")

    defmap = DefMAPSeparator().fit([vocals_spec[np.newaxis], accompaniment_spec[np.newaxis]])
    defmap_vocals, defmap_accompaniment = defmap.predict(mix_spec)
    defmap_vocals_sdr = _spectral_sdr(defmap_vocals, vocals_spec)
    defmap_accompaniment_sdr = _spectral_sdr(defmap_accompaniment, accompaniment_spec)
    print(f"Def-MAP (oracle library): vocals {defmap_vocals_sdr:.1f} dB, "
          f"accompaniment {defmap_accompaniment_sdr:.1f} dB")

    train_tracks = mus_train.tracks[:N_TRAIN_TRACKS_FOR_DMGMM]
    train_vocals = np.concatenate([_track_spectrum(t.targets["vocals"].audio) for t in train_tracks])
    train_accompaniment = np.concatenate(
        [_track_spectrum(t.audio - t.targets["vocals"].audio) for t in train_tracks]
    )
    dmgmm = DMGMMSeparator(DMGMM_COMPONENTS).fit([train_vocals, train_accompaniment], random_state=0)
    dmgmm_vocals, dmgmm_accompaniment = dmgmm.predict(mix_spec)
    dmgmm_vocals_sdr = _spectral_sdr(dmgmm_vocals, vocals_spec)
    dmgmm_accompaniment_sdr = _spectral_sdr(dmgmm_accompaniment, accompaniment_spec)
    print(f"DM-GMM (trained on {N_TRAIN_TRACKS_FOR_DMGMM} other tracks): "
          f"vocals {dmgmm_vocals_sdr:.1f} dB, accompaniment {dmgmm_accompaniment_sdr:.1f} dB")

    assert defmap_vocals_sdr > baseline_vocals_sdr
    assert defmap_accompaniment_sdr > baseline_accompaniment_sdr
    assert dmgmm_vocals_sdr > baseline_vocals_sdr
    assert dmgmm_accompaniment_sdr > baseline_accompaniment_sdr


if __name__ == "__main__":
    main()
