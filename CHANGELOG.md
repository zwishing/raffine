# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-14

First public release on crates.io. The library was rewritten and aligned
more closely with Python's [`affine`](https://github.com/rasterio/affine)
package; the items below describe changes against the pre-release
in-tree code.

### Added
- `Affine::from_array(&[[f64; 3]; 3])` — inverse of `to_array`.
- `Affine::iter()` and `IntoIterator for Affine` over the six coefficients.
- `Affine::transform_into(src, dst)` — zero-alloc out-of-place batch transform.
- `Affine::rowcol_into(xs, ys, &mut rows, &mut cols)` — buffer-reusing
  variant of `rowcol` for tight loops.
- `Default for Affine` (returns the identity).
- `reset_epsilon()` to restore `DEFAULT_EPSILON`.
- `AffineError::ParseError` now carries the offending token `index`.

### Changed
- **Performance**: `transform_vector`, `mul_affine`, `inverse` and the
  `rowcol` inner loop now use `f64::mul_add`, emitting fused FMA on
  x86_64 (FMA3) and AArch64. Results may differ from prior versions and
  from Python's `affine` by ≤1 ulp due to single-rounding semantics.
- `itransform` now takes `&mut [(f64, f64)]` (was `&mut Vec<(f64, f64)>`)
  and is marked `#[inline]`.
- `rowcol` signature changed from `(xs, ys, zs)` to `(xs, ys)`; the
  unused `zs` parameter was removed. `rowcol` now delegates to
  `rowcol_into`.
- Many factory and accessor methods became `const fn`.

### Removed
- The `pub determinant: Option<f64>` cached field on `Affine`. The
  determinant is now recomputed on demand from `a*e - b*d`. This
  eliminates a stale-cache footgun, but it does change the serde
  serialisation format.
- `impl Eq for Affine` — unsound over `f64`. `PartialEq` is retained.

### Fixed
- `loadsw` error reporting now includes the token index of the failure.
