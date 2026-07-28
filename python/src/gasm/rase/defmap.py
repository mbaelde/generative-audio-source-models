"""Def-MAP (optimal deformation MAP), Proposition 2 of RASE. Thesis:
chapter3RTASS.tex (Proposition 2: uniform prior over sound indices,
frame-to-frame Markov chain via Pearson-correlation distance on dB
spectra, loss-to-probability conversion via exp(-alpha * loss)) and
annexes/calculus.tex sections B.4 (closed-form amplitude deformation) and
the complex-deformation section right after it (closed-form real/imaginary
transform).

ponytail: two sources only -- the thesis only develops Def-MAP's sequential
MAP decoding for a two-source mixture (unlike DM-GMM's explicit n-source
hierarchical cascade in dmgmm.py). Extend if a 3+ source Def-MAP need shows up.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_EPS = 1e-12


def _safe_div(numerator: FloatArray, denominator: FloatArray, identity: float) -> FloatArray:
    """numerator/denominator, falling back to `identity` where a source has
    zero energy at a bin (denominator == 0), a case the thesis never
    addresses since it treats spectra as generically non-zero."""
    safe_denominator = np.where(denominator > 0, denominator, 1.0)
    return np.where(denominator > 0, numerator / safe_denominator, identity)


def solve_amplitude_deformation(
    amplitude1: FloatArray, amplitude2: FloatArray, amplitude_mix: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Closed-form amplitude deformation (calculus.tex, appendix:calculus-amp):
    min ||T1-1||^2+||T2-1||^2 s.t. T1*amplitude1+T2*amplitude2=amplitude_mix.
    Exact when amplitude_mix == amplitude1+amplitude2 (thesis's own proof)."""
    denom = amplitude1**2 + amplitude2**2
    transform1 = _safe_div(amplitude2**2 + amplitude1 * (amplitude_mix - amplitude2), denom, 1.0)
    transform2 = _safe_div(amplitude1**2 + amplitude2 * (amplitude_mix - amplitude1), denom, 1.0)
    return transform1, transform2


def solve_complex_deformation(
    spec1: ComplexArray, spec2: ComplexArray, spec_mix: ComplexArray
) -> tuple[ComplexArray, ComplexArray]:
    """Closed-form complex deformation (calculus.tex, section right after
    appendix:calculus-amp). Real and imaginary parts are two INDEPENDENT
    real-valued deformations (not a complex multiplication): the target
    vector is [1_freqbin; 0_freqbin], so the real part is solved exactly like
    the amplitude case (target 1) while the imaginary part has no correction
    term (target 0, pure ratio mask). Apply the result with
    `apply_deformation`, not `transform * spectrum`."""
    re1, re2, re_mix = spec1.real, spec2.real, spec_mix.real
    im1, im2, im_mix = spec1.imag, spec2.imag, spec_mix.imag
    re_denom = re1**2 + re2**2
    im_denom = im1**2 + im2**2
    re_t1 = _safe_div(re2**2 + re1 * (re_mix - re2), re_denom, 1.0)
    re_t2 = _safe_div(re1**2 + re2 * (re_mix - re1), re_denom, 1.0)
    im_t1 = _safe_div(im_mix * im1, im_denom, 0.0)
    im_t2 = _safe_div(im_mix * im2, im_denom, 0.0)
    return re_t1 + 1j * im_t1, re_t2 + 1j * im_t2


def apply_deformation(transform: ComplexArray, spectrum: ComplexArray) -> ComplexArray:
    """Deformed-spectrum estimate: real/imaginary channels scaled
    independently by transform's matching channel (see solve_complex_deformation)."""
    return transform.real * spectrum.real + 1j * (transform.imag * spectrum.imag)


def _deformation_loss(transform: ComplexArray) -> float:
    """||Re(transform)-1||^2 + ||Im(transform)||^2, the optimization objective
    (calculus.tex), reused as Def-MAP's per-candidate loss(T) in eq:distdeform."""
    return float(((transform.real - 1.0) ** 2).sum() + (transform.imag**2).sum())


def _pearson_correlation(a: FloatArray, b: FloatArray) -> float:
    a_c, b_c = a - a.mean(), b - b.mean()
    denom = np.sqrt((a_c**2).sum() * (b_c**2).sum())
    return float((a_c @ b_c) / denom) if denom > 0 else 1.0


def _transition_distance(spec_a: ComplexArray, spec_b: ComplexArray) -> float:
    """Frame-to-frame transition kernel (chapter3RTASS.tex, Proposition 2
    "Formalisation"): d(x_dB,y_dB) = -2*log10(Pearson_correlation(x_dB,y_dB)).
    Correlation clipped to a small positive floor -- the thesis's formula
    implicitly assumes similar, positive-energy training sounds, which real
    silent bins or anti-correlated candidates would otherwise violate."""
    db_a = 20 * np.log10(np.abs(spec_a) + _EPS)
    db_b = 20 * np.log10(np.abs(spec_b) + _EPS)
    corr = _pearson_correlation(db_a, db_b)
    return float(-2 * np.log10(np.clip(corr, 1e-6, 1.0)))


class DefMAPSeparator:
    """Def-MAP separator (Proposition 2, chapter3RTASS.tex), two sources.

    fit() is given each source's training-sound library: for source s,
    library[s] has shape (n_sounds_s, n_frames, freqbin) complex -- n_sounds_s
    known recordings, each a full frame sequence (needed since the Markov
    transition kernel compares candidates' own frame-t spectra, and MAP
    decoding needs every sound's trajectory, not just its statistics).

    predict() runs the sequential MAP decoding of "Reconstruction des
    sources": frame 1 jointly picks the best (sound_1, sound_2) pair against
    the observed mixture (uniform prior, so only the deformation loss
    matters); subsequent frames add the pair's frame-to-frame transition cost
    relative to the previously selected pair (order-1 Markov chain, Z drops
    out of the argmin since it depends only on the previous state)."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, library: list[ComplexArray]) -> DefMAPSeparator:
        if len(library) != 2:
            raise ValueError("DefMAPSeparator supports exactly two sources")
        self.library_ = library
        return self

    def predict(self, mixed_spectra: ComplexArray) -> list[ComplexArray]:
        """mixed_spectra: (n_frames, freqbin) complex."""
        library1, library2 = self.library_
        n1, n2 = len(library1), len(library2)
        est1 = np.empty_like(mixed_spectra)
        est2 = np.empty_like(mixed_spectra)
        prev_y1 = prev_y2 = -1
        for t, spec_mix in enumerate(mixed_spectra):
            best_score = np.inf
            best = None
            for y1 in range(n1):
                for y2 in range(n2):
                    spec1, spec2 = library1[y1, t], library2[y2, t]
                    t1, t2 = solve_complex_deformation(spec1, spec2, spec_mix)
                    score = self.alpha * (_deformation_loss(t1) + _deformation_loss(t2))
                    if t > 0:
                        score += _transition_distance(spec1, library1[prev_y1, t])
                        score += _transition_distance(spec2, library2[prev_y2, t])
                    if score < best_score:
                        best_score = score
                        best = (y1, y2, t1, t2, spec1, spec2)
            assert best is not None
            y1, y2, t1, t2, spec1, spec2 = best
            est1[t] = apply_deformation(t1, spec1)
            est2[t] = apply_deformation(t2, spec2)
            prev_y1, prev_y2 = y1, y2
        return [est1, est2]
