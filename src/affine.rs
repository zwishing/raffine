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
use std::ops::{Mul, Not};
use std::fmt::Display;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

// Using constants instead of static mut for better safety and performance
/// Default epsilon value for floating-point comparisons
pub const DEFAULT_EPSILON: f64 = 1e-5;
/// Default square of epsilon
pub const DEFAULT_EPSILON2: f64 = DEFAULT_EPSILON * DEFAULT_EPSILON;

// Thread-safe global epsilon values
static EPSILON_BITS: AtomicU64 = AtomicU64::new(DEFAULT_EPSILON.to_bits());
static EPSILON2_BITS: AtomicU64 = AtomicU64::new(DEFAULT_EPSILON2.to_bits());

/// Get the current epsilon value
#[inline]
pub fn get_epsilon() -> f64 {
    f64::from_bits(EPSILON_BITS.load(Ordering::Relaxed))
}

/// Get the current epsilon squared value
#[inline]
pub fn get_epsilon2() -> f64 {
    f64::from_bits(EPSILON2_BITS.load(Ordering::Relaxed))
}

/// Error type for affine transformation operations
#[derive(Debug, Error)]
pub enum AffineError {
    /// The transform could not be inverted
    #[error("The transform could not be inverted")]
    TransformNotInvertible,
    /// The rotation angle could not be computed for this transform
    #[error("The rotation angle could not be computed for this transform")]
    UndefinedRotation,
}

/// Return the cosine and sine for the given angle in degrees.
///
/// With special-case handling of multiples of 90 for perfect right angles.
#[inline]
fn cos_sin_deg(deg: f64) -> (f64, f64) {
    let deg_mod = deg % 360.0;
    let deg_norm = if deg_mod < 0.0 { deg_mod + 360.0 } else { deg_mod };
    
    // Check for common angles using match expression
    match deg_norm {
        d if (d - 0.0).abs() < f64::EPSILON || (d - 360.0).abs() < f64::EPSILON => (1.0, 0.0),
        d if (d - 90.0).abs() < f64::EPSILON => (0.0, 1.0),
        d if (d - 180.0).abs() < f64::EPSILON => (-1.0, 0.0),
        d if (d - 270.0).abs() < f64::EPSILON => (0.0, -1.0),
        // For other angles, compute values
        _ => {
            let rad = deg_norm.to_radians();
            (rad.cos(), rad.sin())
        }
    }
}

/// Two dimensional affine transform for 2D linear mapping.
///
/// Internally the transform is stored as a 3x3 transformation matrix.
/// The transform may be constructed directly by specifying the first
/// two rows of matrix values as 6 floats. Since the matrix is an affine
/// transform, the last row is always (0, 0, 1).
///
/// The transform can perform any combination of translations, scales/flips,
/// shears, and rotations. Parallel lines are preserved by these transforms.
#[derive(Copy, Clone, Debug)]
pub struct Affine {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    // Cached values
    determinant: Option<f64>,
}

impl Affine {
    /// Create a new affine transformation.
    ///
    /// The parameters correspond to the coefficients of the 3x3 augmented
    /// affine transformation matrix:
    ///
    /// | x' |   | a  b  c | | x |
    /// | y' | = | d  e  f | | y |
    /// | 1  |   | 0  0  1 | | 1 |
    #[inline]
    pub fn new(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> Self {
        Self { a, b, c, d, e, f, determinant: None }
    }

    /// Create a transformation from GDAL's GetGeoTransform() coefficient order.
    #[inline]
    pub fn from_gdal(c: f64, a: f64, b: f64, f: f64, d: f64, e: f64) -> Self {
        Self::new(a, b, c, d, e, f)
    }

    /// Return the identity transform.
    #[inline]
    pub const fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 0.0,
            e: 1.0,
            f: 0.0,
            determinant: Some(1.0),
        }
    }

    /// Create a translation transform from an offset vector.
    #[inline]
    pub const fn translation(xoff: f64, yoff: f64) -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: xoff,
            d: 0.0,
            e: 1.0,
            f: yoff,
            determinant: Some(1.0),
        }
    }

    /// Create a scaling transform.
    ///
    /// If only one scale factor is provided, it will be used for both dimensions.
    #[inline]
    pub fn scale(sx: f64, sy: Option<f64>) -> Self {
        let sy = sy.unwrap_or(sx);
        Self {
            a: sx,
            b: 0.0,
            c: 0.0,
            d: 0.0,
            e: sy,
            f: 0.0,
            determinant: Some(sx * sy),
        }
    }

    /// Create a shear transform along one or both axes.
    #[inline]
    pub fn shear(x_angle: f64, y_angle: f64) -> Self {
        let mx = x_angle.to_radians().tan();
        let my = y_angle.to_radians().tan();
        Self {
            a: 1.0,
            b: mx,
            c: 0.0,
            d: my,
            e: 1.0,
            f: 0.0,
            determinant: Some(1.0 - mx * my),
        }
    }

    /// Create a rotation transform at the specified angle.
    ///
    /// The angle is in degrees, counter-clockwise about the pivot point.
    /// If no pivot point is provided, the coordinate system origin (0.0, 0.0) is used.
    #[inline]
    pub fn rotation(angle: f64, pivot: Option<(f64, f64)>) -> Self {
        let (ca, sa) = cos_sin_deg(angle);
        match pivot {
            None => Self {
                a: ca,
                b: -sa,
                c: 0.0,
                d: sa,
                e: ca,
                f: 0.0,
                determinant: Some(ca * ca + sa * sa),
            },
            Some((px, py)) => Self {
                a: ca,
                b: -sa,
                c: px - px * ca + py * sa,
                d: sa,
                e: ca,
                f: py - px * sa - py * ca,
                determinant: Some(ca * ca + sa * sa),
            },
        }
    }

    /// Create the permutation transform.
    ///
    /// For 2x2 matrices, there is only one permutation matrix that is not the identity.
    #[inline]
    pub const fn permutation() -> Self {
        Self {
            a: 0.0,
            b: 1.0,
            c: 0.0,
            d: 1.0,
            e: 0.0,
            f: 0.0,
            determinant: Some(-1.0),
        }
    }

    /// Return same coefficient order expected by GDAL's SetGeoTransform().
    #[inline]
    pub fn to_gdal(&self) -> (f64, f64, f64, f64, f64, f64) {
        (self.c, self.a, self.b, self.f, self.d, self.e)
    }

    /// Return affine transformation parameters for shapely's affinity module.
    #[inline]
    pub fn to_shapely(&self) -> (f64, f64, f64, f64, f64, f64) {
        (self.a, self.b, self.d, self.e, self.c, self.f)
    }

    /// Alias for 'c'.
    #[inline]
    pub fn xoff(&self) -> f64 {
        self.c
    }

    /// Alias for 'f'.
    #[inline]
    pub fn yoff(&self) -> f64 {
        self.f
    }

    /// Evaluate the determinant of the transform matrix.
    ///
    /// This value is equal to the area scaling factor when the
    /// transform is applied to a shape.
    #[inline]
    pub fn determinant(&self) -> f64 {
        match self.determinant {
            Some(det) => det,
            None => self.a * self.e - self.b * self.d
        }
    }

    /// The absolute scaling factors of the transformation.
    ///
    /// This tuple represents the absolute value of the scaling factors of the
    /// transformation, sorted from bigger to smaller.
    #[inline]
    fn scaling(&self) -> (f64, f64) {
        let a = self.a;
        let b = self.b;
        let d = self.d;
        let e = self.e;

        // The singular values are the square root of the eigenvalues
        // of the matrix times its transpose, M M*
        // Computing trace and determinant of M M*
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

    /// The eccentricity of the affine transformation.
    ///
    /// This value represents the eccentricity of an ellipse under
    /// this affine transformation.
    #[inline]
    pub fn eccentricity(&self) -> f64 {
        let (l1, l2) = self.scaling();
        ((l1 * l1 - l2 * l2) / (l1 * l1)).sqrt()
    }

    /// The rotation angle in degrees of the affine transformation.
    ///
    /// This is the rotation angle in degrees of the affine transformation,
    /// assuming it is in the form M = R S, where R is a rotation and S is a
    /// scaling.
    ///
    /// # Errors
    ///
    /// Returns an error for improper and degenerate transformations.
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

    /// True if the transform is rectilinear.
    ///
    /// i.e., whether a shape would remain axis-aligned, within rounding
    /// limits, after applying the transform.
    #[inline]
    pub fn is_rectilinear(&self) -> bool {
        let epsilon = get_epsilon();
        (self.a.abs() < epsilon && self.e.abs() < epsilon) ||
        (self.d.abs() < epsilon && self.b.abs() < epsilon)
    }

    /// True if the transform is conformal.
    ///
    /// i.e., if angles between points are preserved after applying the
    /// transform, within rounding limits. This implies that the
    /// transform has no effective shear.
    #[inline]
    pub fn is_conformal(&self) -> bool {
        (self.a * self.b + self.d * self.e).abs() < get_epsilon()
    }

    /// True if the transform is orthonormal.
    ///
    /// Which means that the transform represents a rigid motion, which
    /// has no effective scaling or shear. Mathematically, this means
    /// that the axis vectors of the transform matrix are perpendicular
    /// and unit-length. Applying an orthonormal transform to a shape
    /// always results in a congruent shape.
    #[inline]
    pub fn is_orthonormal(&self) -> bool {
        let a = self.a;
        let b = self.b;
        let d = self.d;
        let e = self.e;
        let epsilon = get_epsilon();
        
        self.is_conformal() &&
        (1.0 - (a * a + d * d)).abs() < epsilon &&
        (1.0 - (b * b + e * e)).abs() < epsilon
    }

    /// Return True if this transform is degenerate.
    ///
    /// A degenerate transform will collapse a shape to an effective area
    /// of zero, and cannot be inverted.
    #[inline]
    pub fn is_degenerate(&self) -> bool {
        self.determinant() == 0.0
    }

    /// Return True if this transform is proper.
    ///
    /// A proper transform (with a positive determinant) does not include
    /// reflection.
    #[inline]
    pub fn is_proper(&self) -> bool {
        self.determinant() > 0.0
    }

    /// The values of the transform as three 2D column vectors.
    ///
    /// Returns (a, d), (b, e), (c, f).
    #[inline]
    pub fn column_vectors(&self) -> ((f64, f64), (f64, f64), (f64, f64)) {
        ((self.a, self.d), (self.b, self.e), (self.c, self.f))
    }

    /// Compare transforms for approximate equality.
    #[inline]
    pub fn almost_equals(&self, other: &Self, precision: Option<f64>) -> bool {
        let precision = precision.unwrap_or_else(get_epsilon);
        (self.a - other.a).abs() < precision &&
        (self.b - other.b).abs() < precision &&
        (self.c - other.c).abs() < precision &&
        (self.d - other.d).abs() < precision &&
        (self.e - other.e).abs() < precision &&
        (self.f - other.f).abs() < precision
    }

    /// Transform a vector (x, y) using this transformation.
    #[inline]
    pub fn transform_vector(&self, vector: (f64, f64)) -> (f64, f64) {
        let (x, y) = vector;
        (
            x * self.a + y * self.b + self.c,
            x * self.d + y * self.e + self.f
        )
    }

    /// Transform a sequence of points or vectors in-place.
    pub fn itransform(&self, seq: &mut Vec<(f64, f64)>) {
        if !self.is_identity() {
            // Use chunks for better cache locality
            const CHUNK_SIZE: usize = 64;
            for chunk in seq.chunks_mut(CHUNK_SIZE) {
                for point in chunk {
                    *point = self.transform_vector(*point);
                }
            }
        }
    }
    
    /// Convert to a 3x3 array.
    #[inline]
    pub fn to_array(&self) -> [[f64; 3]; 3] {
        [
            [self.a, self.b, self.c],
            [self.d, self.e, self.f],
            [0.0, 0.0, 1.0]
        ]
    }
    
    /// Convert to a tuple of all 9 elements.
    #[inline]
    pub fn to_tuple(&self) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        (self.a, self.b, self.c, self.d, self.e, self.f, 0.0, 0.0, 1.0)
    }

    pub fn _mul(&self, other: &Affine) -> Affine {
        let sa = self.a;
        let sb = self.b;
        let sc = self.c;
        let sd = self.d;
        let se = self.e;
        let sf = self.f;
        
        let oa = other.a;
        let ob = other.b;
        let oc = other.c;
        let od = other.d;
        let oe = other.e;
        let of = other.f;
        
        Affine {
            a: sa * oa + sb * od,
            b: sa * ob + sb * oe,
            c: sa * oc + sb * of + sc,
            d: sd * oa + se * od,
            e: sd * ob + se * oe,
            f: sd * oc + se * of + sf,
            determinant: None,
        }
    }
}

/// Implement matrix multiplication for Affine transform.
impl Mul for Affine {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        self._mul(&other)
    }
}

/// Implement matrix multiplication for references
impl Mul<Affine> for &Affine {
    type Output = Affine;

    #[inline]
    fn mul(self, other: Affine) -> Affine {
        self._mul(&other)
    }
}

impl Mul<&Affine> for Affine {
    type Output = Affine;

    #[inline]
    fn mul(self, other: &Affine) -> Affine {
        self._mul(other)
    }
}

impl Mul<&Affine> for &Affine {
    type Output = Affine;

    #[inline]
    fn mul(self, other: &Affine) -> Affine {
        self._mul(other)
    }
}

/// Implement matrix multiplication for Affine transform and point.
impl Mul<(f64, f64)> for Affine {
    type Output = (f64, f64);

    #[inline]
    fn mul(self, point: (f64, f64)) -> (f64, f64) {
        self.transform_vector(point)
    }
}

/// Implement matrix multiplication for reference to Affine and point.
impl Mul<(f64, f64)> for &Affine {
    type Output = (f64, f64);

    #[inline]
    fn mul(self, point: (f64, f64)) -> (f64, f64) {
        self.transform_vector(point)
    }
}

/// Implement inversion (~) operator for Affine transform.
impl Not for Affine {
    type Output = Result<Self, AffineError>;

    #[inline]
    fn not(self) -> Self::Output {
        if self.is_degenerate() {
            return Err(AffineError::TransformNotInvertible);
        }
        
        let idet = 1.0 / self.determinant();
        let sa = self.a;
        let sb = self.b;
        let sc = self.c;
        let sd = self.d;
        let se = self.e;
        let sf = self.f;
        
        let ra = se * idet;
        let rb = -sb * idet;
        let rd = -sd * idet;
        let re = sa * idet;
        
        Ok(Self {
            a: ra,
            b: rb,
            c: -sc * ra - sf * rb,
            d: rd,
            e: re,
            f: -sc * rd - sf * re,
            determinant: Some(idet),
        })
    }
}

impl Not for &Affine {
    type Output = Result<Affine, AffineError>;

    #[inline]
    fn not(self) -> Self::Output {
        !(*self)
    }
}

impl PartialEq for Affine {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.a == other.a &&
        self.b == other.b &&
        self.c == other.c &&
        self.d == other.d &&
        self.e == other.e &&
        self.f == other.f
    }
}

impl Eq for Affine {}

impl Display for Affine {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "|{:6.2}, {:6.2}, {:6.2}|\n|{:6.2}, {:6.2}, {:6.2}|\n|{:6.2}, {:6.2}, {:6.2}|",
            self.a, self.b, self.c,
            self.d, self.e, self.f,
            0.0, 0.0, 1.0)
    }
}

// Identity transform
pub const IDENTITY: Affine = Affine {
    a: 1.0,
    b: 0.0,
    c: 0.0,
    d: 0.0,
    e: 1.0,
    f: 0.0,
    determinant: Some(1.0),
};

/// Return Affine from the contents of a world file string.
///
/// This method also translates the coefficients from center- to
/// corner-based coordinates.
pub fn loadsw(s: &str) -> Result<Affine, Box<dyn std::error::Error>> {
    let coeffs: Vec<f64> = s.split_whitespace()
        .map(|x| x.parse::<f64>())
        .collect::<Result<Vec<f64>, _>>()?;
    
    if coeffs.len() != 6 {
        return Err(format!("Expected 6 coefficients, found {}", coeffs.len()).into());
    }
    
    let a = coeffs[0];
    let d = coeffs[1];
    let b = coeffs[2];
    let e = coeffs[3];
    let c = coeffs[4];
    let f = coeffs[5];
    
    let center = Affine::new(a, b, c, d, e, f);
    Ok(center * Affine::translation(-0.5, -0.5))
}

/// Return string for a world file.
///
/// This method also translates the coefficients from corner- to
/// center-based coordinates.
pub fn dumpsw(obj: &Affine) -> String {
    let center = obj * Affine::translation(0.5, 0.5);
    format!("{}\n{}\n{}\n{}\n{}\n{}\n",
        center.a, center.d, center.b, center.e, center.c, center.f)
}

/// Set the global absolute error value and rounding limit.
///
/// # Parameters
///
/// * `epsilon` - The global absolute error value and rounding limit for
///   approximate floating point comparison operations.
///
/// # Notes
///
/// The default value of `0.00001` is suitable for values that are in
/// the "countable range". You may need a larger epsilon when using
/// large absolute values, and a smaller value for very small values
/// close to zero. Otherwise approximate comparison operations will not
/// behave as expected.
pub fn set_epsilon(epsilon: f64) {
    let epsilon2 = epsilon * epsilon;
    EPSILON_BITS.store(epsilon.to_bits(), Ordering::Relaxed);
    EPSILON2_BITS.store(epsilon2.to_bits(), Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }

    #[test]
    fn test_translation() {
        let t = Affine::translation(10.0, 20.0);
        let p = (5.0, 5.0);
        let result = t * p;
        assert_eq!(result, (15.0, 25.0));
    }

    #[test]
    fn test_scale() {
        let s = Affine::scale(2.0, Some(3.0));
        let p = (5.0, 5.0);
        let result = s * p;
        assert_eq!(result, (10.0, 15.0));
    }

    #[test]
    fn test_rotation() {
        let r = Affine::rotation(90.0, None);
        let p = (1.0, 0.0);
        let result = r * p;
        assert!((result.0).abs() < 1e-10);
        assert!((result.1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inversion() {
        let t = Affine::translation(10.0, 20.0);
        let t_inv = (!t).unwrap();
        let p = (15.0, 25.0);
        let result = t_inv * p;
        assert_eq!(result, (5.0, 5.0));
    }

    #[test]
    fn test_composition() {
        let t = Affine::translation(10.0, 20.0);
        let s = Affine::scale(2.0, Some(3.0));
        let c = t * s;
        let p = (5.0, 5.0);
        let result = c * p;
        assert_eq!(result, (20.0, 35.0));
    }
    
    #[test]
    fn test_reference_multiplication() {
        let t = Affine::translation(10.0, 20.0);
        let s = Affine::scale(2.0, Some(3.0));
        let c1 = &t * &s;
        let c2 = t * s;
        let p = (5.0, 5.0);
        assert_eq!(c1 * p, c2 * p);
        assert_eq!(c1 * p, (20.0, 35.0));
    }

    #[test]
    fn test_inversion_determinant() {
        let s = Affine::scale(2.0, Some(3.0));
        let inv = (!s).unwrap();
        let expected = 1.0 / s.determinant();
        assert!((inv.determinant() - expected).abs() < get_epsilon());
    }
}

