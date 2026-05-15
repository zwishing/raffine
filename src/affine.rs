/*
 * Affine transformation matrices.
 *
 * The Affine crate is derived from Casey Duncan's Planar package. See the
 * copyright statement below.
 */

/*
 * Copyright (c) 2010 by Casey Duncan
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name(s) of the copyright holders nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AS IS AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

use std::fmt;
use std::fmt::Display;
use std::ops::{Mul, Not};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Default epsilon value for floating-point comparisons.
pub const DEFAULT_EPSILON: f64 = 1e-5;
/// Default square of epsilon.
pub const DEFAULT_EPSILON2: f64 = DEFAULT_EPSILON * DEFAULT_EPSILON;

static EPSILON_BITS: AtomicU64 = AtomicU64::new(DEFAULT_EPSILON.to_bits());
static EPSILON2_BITS: AtomicU64 = AtomicU64::new(DEFAULT_EPSILON2.to_bits());

/// Get the current epsilon value.
#[inline]
pub fn get_epsilon() -> f64 {
    f64::from_bits(EPSILON_BITS.load(Ordering::Relaxed))
}

/// Get the current epsilon-squared value.
#[inline]
pub fn get_epsilon2() -> f64 {
    f64::from_bits(EPSILON2_BITS.load(Ordering::Relaxed))
}

/// Error type for affine transformation operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AffineError {
    /// The transform could not be inverted.
    #[error("The transform could not be inverted")]
    TransformNotInvertible,
    /// The rotation angle could not be computed for this transform.
    #[error("The rotation angle could not be computed for this transform")]
    UndefinedRotation,
    /// The input array length mismatch.
    #[error("array length mismatch: xs has {xs_len}, ys has {ys_len}")]
    LengthMismatch { xs_len: usize, ys_len: usize },
    /// Invalid world file format.
    #[error("invalid world file format: {message}")]
    InvalidWorldFile { message: String },
    /// Parse error when reading world file.
    #[error("failed to parse world file at token {index}: {source}")]
    ParseError {
        index: usize,
        source: std::num::ParseFloatError,
    },
    /// Invalid ndarray shape (only emitted with the `ndarray` feature).
    #[error("invalid array shape: expected {expected}, got {actual:?}")]
    InvalidShape {
        expected: &'static str,
        actual: Vec<usize>,
    },
}

/// `x * a + b`.
///
/// On targets with hardware FMA (`target_feature = "fma"` on x86_64 or
/// `"vfp4"` / `"neon"` on ARM/AArch64) this lowers to a single fused
/// instruction with single rounding (most accurate). On targets without
/// hardware FMA we fall back to plain `mul + add` because `f64::mul_add`
/// becomes a libm function call on those targets, which is 20–50× slower
/// than two separate FP ops.
///
/// Build with `RUSTFLAGS="-C target-cpu=native"` (or any flag enabling
/// FMA) to opt into the more accurate hardware path.
#[inline(always)]
pub(crate) fn fma(x: f64, a: f64, b: f64) -> f64 {
    #[cfg(any(
        target_feature = "fma",
        target_feature = "vfp4",
        target_feature = "neon",
    ))]
    {
        x.mul_add(a, b)
    }
    #[cfg(not(any(
        target_feature = "fma",
        target_feature = "vfp4",
        target_feature = "neon",
    )))]
    {
        x * a + b
    }
}

/// Return the cosine and sine for the given angle in degrees, with
/// exact handling of multiples of 90.
#[inline]
fn cos_sin_deg(deg: f64) -> (f64, f64) {
    let deg_mod = deg % 360.0;
    let deg_norm = if deg_mod < 0.0 { deg_mod + 360.0 } else { deg_mod };
    if deg_norm == 0.0 {
        (1.0, 0.0)
    } else if deg_norm == 90.0 {
        (0.0, 1.0)
    } else if deg_norm == 180.0 {
        (-1.0, 0.0)
    } else if deg_norm == 270.0 {
        (0.0, -1.0)
    } else {
        let rad = deg_norm.to_radians();
        (rad.cos(), rad.sin())
    }
}

/// Two-dimensional affine transform for 2D linear mapping.
///
/// Internally the transform is stored as the first two rows of a 3x3
/// transformation matrix. The transform may be constructed directly by
/// specifying these 6 floats. The last row is always (0, 0, 1).
///
/// The transform can perform any combination of translations, scales/flips,
/// shears, and rotations. Parallel lines are preserved.
///
/// ```text
/// | x' |   | a  b  c | | x |
/// | y' | = | d  e  f | | y |
/// | 1  |   | 0  0  1 | | 1 |
/// ```
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Affine {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
    pub f: f64,
}

impl Affine {
    /// Create a new affine transformation from the six coefficients of the
    /// first two rows of the augmented 3x3 matrix.
    #[inline]
    pub const fn new(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> Self {
        Self { a, b, c, d, e, f }
    }

    /// Create a transformation from GDAL's GetGeoTransform() coefficient order.
    #[inline]
    pub const fn from_gdal(coeffs: &[f64; 6]) -> Self {
        Self::new(coeffs[1], coeffs[2], coeffs[0], coeffs[4], coeffs[5], coeffs[3])
    }

    /// Create a transformation from the upper two rows of a 3x3 array.
    /// The third row is assumed to be (0, 0, 1) and is ignored.
    #[inline]
    pub const fn from_array(arr: &[[f64; 3]; 3]) -> Self {
        Self::new(
            arr[0][0], arr[0][1], arr[0][2],
            arr[1][0], arr[1][1], arr[1][2],
        )
    }

    /// Return the identity transform.
    #[inline]
    pub const fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    }

    /// Create a translation transform from an offset vector.
    #[inline]
    pub const fn translation(xoff: f64, yoff: f64) -> Self {
        Self::new(1.0, 0.0, xoff, 0.0, 1.0, yoff)
    }

    /// Create a scaling transform.
    ///
    /// If only one scale factor is provided, it is used for both dimensions.
    #[inline]
    pub fn scale(sx: f64, sy: Option<f64>) -> Self {
        let sy = sy.unwrap_or(sx);
        Self::new(sx, 0.0, 0.0, 0.0, sy, 0.0)
    }

    /// Create a shear transform along one or both axes, in degrees.
    #[inline]
    pub fn shear(x_angle: f64, y_angle: f64) -> Self {
        let mx = x_angle.to_radians().tan();
        let my = y_angle.to_radians().tan();
        Self::new(1.0, mx, 0.0, my, 1.0, 0.0)
    }

    /// Create a rotation transform at the specified angle in degrees,
    /// counter-clockwise about the pivot point. If no pivot is provided,
    /// the origin (0, 0) is used.
    #[inline]
    pub fn rotation(angle: f64, pivot: Option<(f64, f64)>) -> Self {
        let (ca, sa) = cos_sin_deg(angle);
        match pivot {
            None => Self::new(ca, -sa, 0.0, sa, ca, 0.0),
            Some((px, py)) => Self::new(
                ca, -sa, px - px * ca + py * sa,
                sa, ca, py - px * sa - py * ca,
            ),
        }
    }

    /// Create the permutation transform.
    #[inline]
    pub const fn permutation() -> Self {
        Self::new(0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
    }

    /// Return coefficients in the order expected by GDAL's SetGeoTransform().
    #[inline]
    pub const fn to_gdal(&self) -> (f64, f64, f64, f64, f64, f64) {
        (self.c, self.a, self.b, self.f, self.d, self.e)
    }

    /// Return affine transformation parameters for shapely's affinity module.
    #[inline]
    pub const fn to_shapely(&self) -> (f64, f64, f64, f64, f64, f64) {
        (self.a, self.b, self.d, self.e, self.c, self.f)
    }

    /// X-axis translation. Alias for `c`.
    #[inline]
    pub const fn xoff(&self) -> f64 {
        self.c
    }

    /// Y-axis translation. Alias for `f`.
    #[inline]
    pub const fn yoff(&self) -> f64 {
        self.f
    }

    /// Evaluate the determinant of the transform matrix. This is the
    /// area scaling factor when the transform is applied to a shape.
    #[inline]
    pub fn determinant(&self) -> f64 {
        self.a * self.e - self.b * self.d
    }

    /// The absolute scaling factors of the transformation, sorted from
    /// larger to smaller. These are the singular values of the 2x2
    /// linear part of the transform.
    pub fn scaling(&self) -> (f64, f64) {
        let (a, b, d, e) = (self.a, self.b, self.d, self.e);
        let trace = a * a + b * b + d * d + e * e;
        let det2 = self.determinant().powi(2);

        let mut delta = trace.powi(2) / 4.0 - det2;
        if delta < get_epsilon2() {
            delta = 0.0;
        }
        let sqrt_delta = delta.sqrt();
        let l1 = (trace / 2.0 + sqrt_delta).sqrt();
        let l2 = (trace / 2.0 - sqrt_delta).sqrt();
        (l1, l2)
    }

    /// The eccentricity of an ellipse under this affine transformation.
    #[inline]
    pub fn eccentricity(&self) -> f64 {
        let (l1, l2) = self.scaling();
        ((l1 * l1 - l2 * l2) / (l1 * l1)).sqrt()
    }

    /// The rotation angle in degrees, assuming the transform is in the
    /// form `M = R * S` (rotation composed with scaling).
    ///
    /// # Errors
    ///
    /// Returns [`AffineError::UndefinedRotation`] for improper transforms
    /// (negative determinant) that are not degenerate.
    pub fn rotation_angle(&self) -> Result<f64, AffineError> {
        if self.is_proper() || self.is_degenerate() {
            let (l1, _) = self.scaling();
            let y = self.d / l1;
            let x = self.a / l1;
            Ok(y.atan2(x).to_degrees())
        } else {
            Err(AffineError::UndefinedRotation)
        }
    }

    /// True if this transform equals the identity matrix, within rounding limits.
    #[inline]
    pub fn is_identity(&self) -> bool {
        self == &IDENTITY || self.almost_equals(&IDENTITY, None)
    }

    /// True if the transform is rectilinear: a shape stays axis-aligned
    /// (within rounding limits) after applying it.
    #[inline]
    pub fn is_rectilinear(&self) -> bool {
        let epsilon = get_epsilon();
        (self.a.abs() < epsilon && self.e.abs() < epsilon)
            || (self.b.abs() < epsilon && self.d.abs() < epsilon)
    }

    /// True if the transform is conformal: angles between points are
    /// preserved (no effective shear), within rounding limits.
    #[inline]
    pub fn is_conformal(&self) -> bool {
        (self.a * self.b + self.d * self.e).abs() < get_epsilon()
    }

    /// True if the transform is orthonormal: a rigid motion with no
    /// effective scaling or shear.
    #[inline]
    pub fn is_orthonormal(&self) -> bool {
        let (a, b, d, e) = (self.a, self.b, self.d, self.e);
        let epsilon = get_epsilon();
        self.is_conformal()
            && (1.0 - (a * a + d * d)).abs() < epsilon
            && (1.0 - (b * b + e * e)).abs() < epsilon
    }

    /// True if this transform collapses shapes to zero area (det == 0).
    #[inline]
    pub fn is_degenerate(&self) -> bool {
        self.determinant() == 0.0
    }

    /// True if the transform has a positive determinant (no reflection).
    #[inline]
    pub fn is_proper(&self) -> bool {
        self.determinant() > 0.0
    }

    /// The transform as three 2D column vectors `(a, d)`, `(b, e)`, `(c, f)`.
    #[inline]
    pub const fn column_vectors(&self) -> ((f64, f64), (f64, f64), (f64, f64)) {
        ((self.a, self.d), (self.b, self.e), (self.c, self.f))
    }

    /// Compare transforms for approximate equality within `precision`.
    /// When `precision` is `None`, the global epsilon is used.
    #[inline]
    pub fn almost_equals(&self, other: &Self, precision: Option<f64>) -> bool {
        let precision = precision.unwrap_or_else(get_epsilon);
        (self.a - other.a).abs() < precision
            && (self.b - other.b).abs() < precision
            && (self.c - other.c).abs() < precision
            && (self.d - other.d).abs() < precision
            && (self.e - other.e).abs() < precision
            && (self.f - other.f).abs() < precision
    }

    /// Transform a single 2D point/vector.
    ///
    /// Uses fused multiply-add when the target supports it (FMA3 on
    /// x86_64, NEON/VFP4 on ARM); falls back to plain `mul + add`
    /// elsewhere — see [`fma`].
    #[inline]
    pub fn transform_vector(&self, vector: (f64, f64)) -> (f64, f64) {
        let (x, y) = vector;
        (
            fma(x, self.a, fma(y, self.b, self.c)),
            fma(x, self.d, fma(y, self.e, self.f)),
        )
    }

    /// Transform a sequence of points in-place. No-op for the identity.
    #[inline]
    pub fn itransform(&self, seq: &mut [(f64, f64)]) {
        if !self.is_identity() {
            for point in seq.iter_mut() {
                *point = self.transform_vector(*point);
            }
        }
    }

    /// Transform `src` into `dst`, which must have the same length.
    ///
    /// # Errors
    ///
    /// Returns [`AffineError::LengthMismatch`] if `src` and `dst` differ
    /// in length.
    pub fn transform_into(
        &self,
        src: &[(f64, f64)],
        dst: &mut [(f64, f64)],
    ) -> Result<(), AffineError> {
        if src.len() != dst.len() {
            return Err(AffineError::LengthMismatch {
                xs_len: src.len(),
                ys_len: dst.len(),
            });
        }
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            *d = self.transform_vector(*s);
        }
        Ok(())
    }

    /// Convert to a 3x3 array.
    #[inline]
    pub const fn to_array(&self) -> [[f64; 3]; 3] {
        [
            [self.a, self.b, self.c],
            [self.d, self.e, self.f],
            [0.0, 0.0, 1.0],
        ]
    }

    /// Convert to a 9-tuple of all matrix elements in row-major order.
    #[inline]
    pub const fn to_tuple(&self) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        (self.a, self.b, self.c, self.d, self.e, self.f, 0.0, 0.0, 1.0)
    }

    /// Iterate over the six coefficients `[a, b, c, d, e, f]`.
    #[inline]
    pub fn iter(&self) -> std::array::IntoIter<f64, 6> {
        [self.a, self.b, self.c, self.d, self.e, self.f].into_iter()
    }

    /// Inverse transform. Equivalent to the `!` operator.
    ///
    /// # Errors
    ///
    /// Returns [`AffineError::TransformNotInvertible`] when the transform
    /// is degenerate.
    #[inline]
    pub fn inverse(&self) -> Result<Self, AffineError> {
        !*self
    }

    /// Transform world coordinates `(xs, ys)` to pixel `(rows, cols)`.
    ///
    /// # Errors
    ///
    /// Returns [`AffineError::LengthMismatch`] when `xs` and `ys` differ
    /// in length, or [`AffineError::TransformNotInvertible`] when the
    /// transform cannot be inverted.
    pub fn rowcol(
        &self,
        xs: &[f64],
        ys: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), AffineError> {
        let mut rows = Vec::with_capacity(xs.len());
        let mut cols = Vec::with_capacity(xs.len());
        self.rowcol_into(xs, ys, &mut rows, &mut cols)?;
        Ok((rows, cols))
    }

    /// Like [`rowcol`](Self::rowcol), but appends into caller-provided
    /// buffers to avoid allocations on repeated calls. Pre-`clear()` the
    /// buffers if you want fresh results.
    ///
    /// # Errors
    ///
    /// Same as [`rowcol`](Self::rowcol).
    pub fn rowcol_into(
        &self,
        xs: &[f64],
        ys: &[f64],
        rows: &mut Vec<f64>,
        cols: &mut Vec<f64>,
    ) -> Result<(), AffineError> {
        if xs.len() != ys.len() {
            return Err(AffineError::LengthMismatch {
                xs_len: xs.len(),
                ys_len: ys.len(),
            });
        }
        if xs.is_empty() {
            return Ok(());
        }
        let inv = self.inverse()?;
        rows.reserve(xs.len());
        cols.reserve(xs.len());
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            rows.push(fma(x, inv.d, fma(y, inv.e, inv.f)));
            cols.push(fma(x, inv.a, fma(y, inv.b, inv.c)));
        }
        Ok(())
    }

    #[inline]
    fn mul_affine(&self, other: &Affine) -> Affine {
        Affine::new(
            fma(self.a, other.a, self.b * other.d),
            fma(self.a, other.b, self.b * other.e),
            fma(self.a, other.c, fma(self.b, other.f, self.c)),
            fma(self.d, other.a, self.e * other.d),
            fma(self.d, other.b, self.e * other.e),
            fma(self.d, other.c, fma(self.e, other.f, self.f)),
        )
    }
}

impl IntoIterator for Affine {
    type Item = f64;
    type IntoIter = std::array::IntoIter<f64, 6>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Mul for Affine {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        self.mul_affine(&other)
    }
}

impl Mul<Affine> for &Affine {
    type Output = Affine;
    #[inline]
    fn mul(self, other: Affine) -> Affine {
        self.mul_affine(&other)
    }
}

impl Mul<&Affine> for Affine {
    type Output = Affine;
    #[inline]
    fn mul(self, other: &Affine) -> Affine {
        self.mul_affine(other)
    }
}

impl Mul<&Affine> for &Affine {
    type Output = Affine;
    #[inline]
    fn mul(self, other: &Affine) -> Affine {
        self.mul_affine(other)
    }
}

impl Mul<(f64, f64)> for Affine {
    type Output = (f64, f64);
    #[inline]
    fn mul(self, point: (f64, f64)) -> (f64, f64) {
        self.transform_vector(point)
    }
}

impl Mul<(f64, f64)> for &Affine {
    type Output = (f64, f64);
    #[inline]
    fn mul(self, point: (f64, f64)) -> (f64, f64) {
        self.transform_vector(point)
    }
}

impl Mul<(isize, isize)> for Affine {
    type Output = (isize, isize);
    #[inline]
    fn mul(self, point: (isize, isize)) -> (isize, isize) {
        let (x, y) = self.transform_vector((point.0 as f64, point.1 as f64));
        (x.round() as isize, y.round() as isize)
    }
}

impl Mul<(isize, isize)> for &Affine {
    type Output = (isize, isize);
    #[inline]
    fn mul(self, point: (isize, isize)) -> (isize, isize) {
        let (x, y) = self.transform_vector((point.0 as f64, point.1 as f64));
        (x.round() as isize, y.round() as isize)
    }
}

impl Not for Affine {
    type Output = Result<Self, AffineError>;

    #[inline]
    fn not(self) -> Self::Output {
        if self.is_degenerate() {
            return Err(AffineError::TransformNotInvertible);
        }
        let idet = 1.0 / self.determinant();
        let ra = self.e * idet;
        let rb = -self.b * idet;
        let rd = -self.d * idet;
        let re = self.a * idet;
        Ok(Self::new(
            ra, rb, fma(-self.c, ra, -self.f * rb),
            rd, re, fma(-self.c, rd, -self.f * re),
        ))
    }
}

impl Not for &Affine {
    type Output = Result<Affine, AffineError>;
    #[inline]
    fn not(self) -> Self::Output {
        !*self
    }
}

impl PartialEq for Affine {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.a == other.a
            && self.b == other.b
            && self.c == other.c
            && self.d == other.d
            && self.e == other.e
            && self.f == other.f
    }
}

impl Default for Affine {
    #[inline]
    fn default() -> Self {
        Self::identity()
    }
}

impl Display for Affine {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "|{:6.2}, {:6.2}, {:6.2}|\n|{:6.2}, {:6.2}, {:6.2}|\n|{:6.2}, {:6.2}, {:6.2}|",
            self.a, self.b, self.c,
            self.d, self.e, self.f,
            0.0, 0.0, 1.0
        )
    }
}

/// The identity transform.
pub const IDENTITY: Affine = Affine::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0);

/// Parse an Affine from the contents of an Esri/ArcGIS world file string.
///
/// World files store coefficients for *pixel-center* coordinates; the
/// returned transform converts them to *pixel-corner* form, suitable for
/// composition with point geometries indexed from (0, 0) at the upper-left.
///
/// # Errors
///
/// Returns [`AffineError::ParseError`] for invalid numbers, or
/// [`AffineError::InvalidWorldFile`] when the file does not contain exactly
/// six coefficients.
pub fn loadsw(s: &str) -> Result<Affine, AffineError> {
    let coeffs: Vec<f64> = s
        .split_whitespace()
        .enumerate()
        .map(|(i, tok)| {
            tok.parse::<f64>()
                .map_err(|e| AffineError::ParseError { index: i, source: e })
        })
        .collect::<Result<Vec<f64>, _>>()?;

    if coeffs.len() != 6 {
        return Err(AffineError::InvalidWorldFile {
            message: format!("Expected 6 coefficients, found {}", coeffs.len()),
        });
    }

    let (a, d, b, e, c, f) = (coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5]);
    let center = Affine::new(a, b, c, d, e, f);
    Ok(center * Affine::translation(-0.5, -0.5))
}

/// Serialise an Affine to a world-file string, translating from
/// pixel-corner to pixel-center coordinates.
pub fn dumpsw(obj: &Affine) -> String {
    let center = obj * Affine::translation(0.5, 0.5);
    format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n",
        center.a, center.d, center.b, center.e, center.c, center.f
    )
}

/// Set the global absolute error value and rounding limit used by
/// approximate floating-point comparisons.
///
/// The default value of `1e-5` (see [`DEFAULT_EPSILON`]) suits values in
/// the "countable range". Use a larger epsilon for large magnitudes and a
/// smaller one for values near zero.
pub fn set_epsilon(epsilon: f64) {
    let epsilon2 = epsilon * epsilon;
    EPSILON_BITS.store(epsilon.to_bits(), Ordering::Relaxed);
    EPSILON2_BITS.store(epsilon2.to_bits(), Ordering::Relaxed);
}

/// Reset the global epsilon to [`DEFAULT_EPSILON`].
#[inline]
pub fn reset_epsilon() {
    set_epsilon(DEFAULT_EPSILON);
}

#[cfg(test)]
#[allow(clippy::op_ref, clippy::excessive_precision)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_identity() {
        let identity = Affine::identity();
        assert_eq!(identity.a, 1.0);
        assert_eq!(identity.b, 0.0);
        assert_eq!(identity.c, 0.0);
        assert_eq!(identity.d, 0.0);
        assert_eq!(identity.e, 1.0);
        assert_eq!(identity.f, 0.0);
        assert!(identity.is_identity());
        assert_eq!(identity, IDENTITY);
        assert_eq!(Affine::default(), IDENTITY);
    }

    #[test]
    fn test_translation() {
        let t = Affine::translation(10.0, 20.0);
        assert_eq!(t * (5.0, 5.0), (15.0, 25.0));
        assert_eq!(t.xoff(), 10.0);
        assert_eq!(t.yoff(), 20.0);
    }

    #[test]
    fn test_translation_isize() {
        let t = Affine::translation(10.0, 20.0);
        assert_eq!(t * (5isize, 5isize), (15isize, 25isize));
    }

    #[test]
    fn test_scale() {
        let s = Affine::scale(2.0, Some(3.0));
        assert_eq!(s * (5.0, 5.0), (10.0, 15.0));
        let uniform = Affine::scale(4.0, None);
        assert_eq!(uniform * (1.0, 1.0), (4.0, 4.0));
        assert_eq!(uniform.determinant(), 16.0);
    }

    #[test]
    fn test_shear() {
        let sh = Affine::shear(45.0, 0.0);
        let (x, y) = sh * (1.0, 1.0);
        assert_relative_eq!(x, 2.0, max_relative = 1e-12);
        assert_relative_eq!(y, 1.0, max_relative = 1e-12);
    }

    #[test]
    fn test_rotation() {
        let r = Affine::rotation(90.0, None);
        let (x, y) = r * (1.0, 0.0);
        assert!(x.abs() < TOL);
        assert!((y - 1.0).abs() < TOL);
    }

    #[test]
    fn test_rotation_with_pivot() {
        let r = Affine::rotation(180.0, Some((1.0, 1.0)));
        let (x, y) = r * (2.0, 2.0);
        assert!((x - 0.0).abs() < TOL);
        assert!((y - 0.0).abs() < TOL);
    }

    #[test]
    fn test_rotation_angle() {
        let r = Affine::rotation(30.0, None);
        assert_relative_eq!(r.rotation_angle().unwrap(), 30.0, max_relative = 1e-12);

        // Improper (reflection) transform should report undefined rotation
        let reflect = Affine::scale(-1.0, Some(1.0));
        assert!(matches!(
            reflect.rotation_angle(),
            Err(AffineError::UndefinedRotation)
        ));
    }

    #[test]
    fn test_permutation() {
        let p = Affine::permutation();
        assert_eq!(p * (3.0, 7.0), (7.0, 3.0));
        assert_eq!(p.determinant(), -1.0);
        assert!(!p.is_proper());
        assert!(!p.is_degenerate());
    }

    #[test]
    fn test_eccentricity() {
        // Uniform scale → circle → eccentricity 0
        let uniform = Affine::scale(3.0, None);
        assert_relative_eq!(uniform.eccentricity(), 0.0, epsilon = 1e-12);

        // 2x in x, 1x in y → eccentricity = sqrt(1 - 1/4) = sqrt(3)/2
        let stretched = Affine::scale(2.0, Some(1.0));
        assert_relative_eq!(
            stretched.eccentricity(),
            (3.0_f64).sqrt() / 2.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn test_predicates() {
        assert!(Affine::identity().is_rectilinear());
        assert!(Affine::identity().is_orthonormal());
        assert!(Affine::identity().is_conformal());
        assert!(Affine::identity().is_proper());
        assert!(!Affine::identity().is_degenerate());

        let r = Affine::rotation(45.0, None);
        assert!(r.is_orthonormal());
        assert!(r.is_conformal());
        assert!(!r.is_rectilinear());

        let degenerate = Affine::scale(0.0, Some(0.0));
        assert!(degenerate.is_degenerate());
        assert!(!degenerate.is_proper());

        let r90 = Affine::rotation(90.0, None);
        assert!(r90.is_rectilinear());

        let shear = Affine::shear(30.0, 0.0);
        assert!(!shear.is_conformal());
        assert!(!shear.is_orthonormal());
    }

    #[test]
    fn test_column_vectors() {
        let t = Affine::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let (c1, c2, c3) = t.column_vectors();
        // column_vectors returns ((a,d), (b,e), (c,f))
        assert_eq!(c1, (1.0, 4.0));
        assert_eq!(c2, (2.0, 5.0));
        assert_eq!(c3, (3.0, 6.0));
    }

    #[test]
    fn test_inversion() {
        let t = Affine::translation(10.0, 20.0);
        let t_inv = (!t).unwrap();
        assert_eq!(t_inv * (15.0, 25.0), (5.0, 5.0));

        let s = Affine::scale(2.0, Some(3.0));
        let s_inv = s.inverse().unwrap();
        assert_relative_eq!(s_inv.determinant(), 1.0 / s.determinant(), max_relative = 1e-12);
    }

    #[test]
    fn test_inversion_degenerate() {
        let degenerate = Affine::scale(0.0, Some(1.0));
        assert!(matches!(!degenerate, Err(AffineError::TransformNotInvertible)));
        assert!(matches!(
            degenerate.inverse(),
            Err(AffineError::TransformNotInvertible)
        ));
    }

    #[test]
    fn test_composition() {
        let t = Affine::translation(10.0, 20.0);
        let s = Affine::scale(2.0, Some(3.0));
        let c = t * s;
        assert_eq!(c * (5.0, 5.0), (20.0, 35.0));
    }

    #[test]
    fn test_reference_multiplication() {
        let t = Affine::translation(10.0, 20.0);
        let s = Affine::scale(2.0, Some(3.0));
        let c1 = &t * &s;
        let c2 = t * s;
        assert_eq!(c1 * (5.0, 5.0), c2 * (5.0, 5.0));
        assert_eq!(c1 * (5.0, 5.0), (20.0, 35.0));
    }

    #[test]
    fn test_from_to_gdal() {
        let params = [100.0, 1.0, 0.0, 200.0, 0.0, -1.0];
        let affine = Affine::from_gdal(&params);
        assert_eq!(affine.to_gdal(), (100.0, 1.0, 0.0, 200.0, 0.0, -1.0));
    }

    #[test]
    fn test_to_shapely() {
        let t = Affine::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        assert_eq!(t.to_shapely(), (1.0, 2.0, 4.0, 5.0, 3.0, 6.0));
    }

    #[test]
    fn test_to_array_roundtrip() {
        let t = Affine::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let arr = t.to_array();
        let back = Affine::from_array(&arr);
        assert_eq!(t, back);
    }

    #[test]
    fn test_iter() {
        let t = Affine::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let v: Vec<f64> = t.iter().collect();
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let v2: Vec<f64> = t.into_iter().collect();
        assert_eq!(v2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_almost_equals() {
        let a = Affine::identity();
        let b = Affine::new(1.0 + 1e-9, 0.0, 0.0, 0.0, 1.0, 0.0);
        assert!(a.almost_equals(&b, None));
        assert!(!a.almost_equals(&b, Some(1e-12)));
    }

    #[test]
    fn test_itransform() {
        let t = Affine::translation(1.0, 2.0);
        let mut pts = vec![(0.0, 0.0), (3.0, 4.0)];
        t.itransform(&mut pts);
        assert_eq!(pts, vec![(1.0, 2.0), (4.0, 6.0)]);

        // identity should be a no-op
        let mut pts2 = vec![(1.0, 1.0)];
        Affine::identity().itransform(&mut pts2);
        assert_eq!(pts2, vec![(1.0, 1.0)]);

        // also accepts plain slices
        let mut arr = [(1.0, 1.0), (2.0, 2.0)];
        t.itransform(&mut arr[..]);
        assert_eq!(arr, [(2.0, 3.0), (3.0, 4.0)]);
    }

    #[test]
    fn test_loadsw_dumpsw_roundtrip() {
        let original = Affine::new(2.0, 0.0, 100.0, 0.0, -2.0, 200.0);
        let s = dumpsw(&original);
        let parsed = loadsw(&s).unwrap();
        assert!(original.almost_equals(&parsed, Some(1e-9)));
    }

    #[test]
    fn test_loadsw_invalid() {
        assert!(matches!(
            loadsw("1 2 3"),
            Err(AffineError::InvalidWorldFile { .. })
        ));
        assert!(matches!(
            loadsw("1 2 3 4 5 nope"),
            Err(AffineError::ParseError { .. })
        ));
    }

    #[test]
    fn test_rowcol() {
        let aff = Affine::new(
            300.0379266750948,
            0.0,
            101985.0,
            0.0,
            -300.0417827298049929,
            2826915.0,
        );
        let left = 101985.0;
        let bottom = 2611485.0;
        let right = 339315.0;
        let top = 2826915.0;

        let pixel_width: f64 = 300.0379266750948;
        let pixel_height: f64 = -300.0417827298049929;
        let width = ((right - left) / pixel_width).round() as i32;
        let height = ((top - bottom) / pixel_height.abs()).round() as i32;

        let (row_result, col_result) = aff.rowcol(&[left], &[top]).unwrap();
        assert!(row_result[0].abs() < TOL);
        assert!(col_result[0].abs() < TOL);

        let (row_result, col_result) = aff.rowcol(&[right], &[top]).unwrap();
        assert!(row_result[0].abs() < TOL);
        assert!((col_result[0] - width as f64).abs() < TOL);

        let (row_result, col_result) = aff.rowcol(&[right], &[bottom]).unwrap();
        assert!((row_result[0] - height as f64).abs() < TOL);
        assert!((col_result[0] - width as f64).abs() < TOL);

        let (row_result, col_result) = aff.rowcol(&[left], &[bottom]).unwrap();
        assert!((row_result[0] - height as f64).abs() < TOL);
        assert!(col_result[0].abs() < TOL);

        let (rs, cs) = aff.rowcol(&[101985.0], &[2826915.0]).unwrap();
        assert!(rs[0].abs() < TOL);
        assert!(cs[0].abs() < TOL);
    }

    #[test]
    fn test_rowcol_length_mismatch() {
        let aff = Affine::identity();
        assert!(matches!(
            aff.rowcol(&[1.0, 2.0], &[1.0]),
            Err(AffineError::LengthMismatch { xs_len: 2, ys_len: 1 })
        ));
    }

    #[test]
    fn test_rowcol_empty() {
        let aff = Affine::identity();
        let (rows, cols) = aff.rowcol(&[], &[]).unwrap();
        assert!(rows.is_empty() && cols.is_empty());
    }

    #[test]
    fn test_transform_into() {
        let t = Affine::translation(1.0, 2.0);
        let src = vec![(0.0, 0.0), (3.0, 4.0)];
        let mut dst = vec![(0.0, 0.0); 2];
        t.transform_into(&src, &mut dst).unwrap();
        assert_eq!(dst, vec![(1.0, 2.0), (4.0, 6.0)]);

        let mut bad = vec![(0.0, 0.0); 1];
        assert!(matches!(
            t.transform_into(&src, &mut bad),
            Err(AffineError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn test_rowcol_into_buffer_reuse() {
        let aff = Affine::translation(10.0, 20.0);
        let mut rows = Vec::with_capacity(4);
        let mut cols = Vec::with_capacity(4);

        rows.clear();
        cols.clear();
        aff.rowcol_into(&[11.0, 12.0], &[21.0, 22.0], &mut rows, &mut cols).unwrap();
        assert_eq!(rows, vec![1.0, 2.0]);
        assert_eq!(cols, vec![1.0, 2.0]);

        // Reuse without reallocation (capacity preserved)
        let cap_rows = rows.capacity();
        let cap_cols = cols.capacity();
        rows.clear();
        cols.clear();
        aff.rowcol_into(&[13.0], &[23.0], &mut rows, &mut cols).unwrap();
        assert_eq!(rows, vec![3.0]);
        assert_eq!(cols, vec![3.0]);
        assert!(rows.capacity() >= cap_rows);
        assert!(cols.capacity() >= cap_cols);
    }

    #[test]
    fn test_rowcol_non_invertible() {
        let aff = Affine::scale(0.0, Some(0.0));
        assert!(matches!(
            aff.rowcol(&[1.0], &[1.0]),
            Err(AffineError::TransformNotInvertible)
        ));
    }

    #[test]
    fn test_epsilon_thread_safety() {
        set_epsilon(1e-3);
        assert!((get_epsilon() - 1e-3).abs() < 1e-15);
        assert!((get_epsilon2() - 1e-6).abs() < 1e-15);
        reset_epsilon();
        assert!((get_epsilon() - DEFAULT_EPSILON).abs() < 1e-15);
    }

    #[test]
    fn test_serde_roundtrip() {
        let t = Affine::new(1.5, 2.5, 3.5, 4.5, 5.5, 6.5);
        let json = serde_json::to_string(&t).unwrap();
        let back: Affine = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }
}
