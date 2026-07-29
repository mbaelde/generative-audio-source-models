# Third-party code

## Vendored

- `matlab/classification/demo/rare_program/jsonlab/` (Qianqian Fang, JSONLab):
  BSD License per each file's header (`% license: BSD License, see
  LICENSE_BSD.txt files for details`), compatible with this repo's
  BSD-3-Clause. Used by the standalone demo (`test_RARE.m`) to
  read/write its JSON config and results. Missed in the original
  2026-07-27 license triage below; added 2026-07-29.

## Dropped

Two files were staged here at various points and dropped after checking their
actual license terms:

- `compute_MFCC.m` (Malcolm Slaney, Auditory Toolbox, 1993/1998 Interval
  Research Corporation): the file carries a copyright notice but no license
  grant, so its redistribution terms are unverifiable. Dropped rather than
  guessed.
- `misc/v_findpeaks.m` (Mike Brookes, VOICEBOX, Imperial College London): the
  file header states GNU GPL v2-or-later — the same blocking license already
  excluded elsewhere in this repo (`gmmbayes`, GPL-2+), so it can't be kept
  regardless of attribution.

Both were called from `matlab/classification/**/compute_descriptors.m` and
`matlab/common/AudioDescriptors/misc/spec_getsins_f0.m`; those call sites are
left as-is (the archive already isn't meant to run, see README) rather than
rewritten, since the goal here is a clean license, not a working pipeline.
For the planned `python/` rewrite (phase 2), MFCC and peak-picking have
direct equivalents (`librosa.feature.mfcc` / `python_speech_features`,
`scipy.signal.find_peaks`).

Everything else originally staged under `matlab/third_party/` during import
was re-triaged on 2026-07-27:
- `Statistics/`, `PLCS mex/`, and most of `AudioDescriptors/` turned out to be
  Maxime Baelde's own code (no third-party author header), not vendored
  dependencies — merged into `matlab/common/`.
- `bss_eval/` (Emmanuel Vincent), `matlab2tikz/`, `Plot2LaTeX/`, and
  `progressbar.m` were genuinely vendored but dropped: each has a direct
  Python equivalent for the planned rewrite (`mir_eval`/`museval`, matplotlib's
  native export, `tqdm`), so there is no reason to carry the MATLAB copies
  forward.
