//! Optional `ndarray` integration.
//!
//! Enable with the `ndarray` feature. The methods below all operate
//! through [`ArrayView`](ndarray::ArrayView) / [`ArrayViewMut`](ndarray::ArrayViewMut)
//! so they are zero-copy. When inputs and outputs are contiguous
//! (standard layout) the implementation takes a slice fast path that
//! the compiler autovectorises with FMA; non-contiguous (strided)
//! inputs fall back to ndarray iteration.
//!
//! For ergonomics the parallel-array (SoA) APIs accept and return
//! one-dimensional arrays of `x` and `y`. The packed (AoS) variant
//! mutates an `[N, 2]` array in place; this avoids an allocation and
//! matches the natural memory layout of geometry libraries.

use ndarray::{Array1, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis};

use crate::{Affine, AffineError};

#[inline(always)]
fn tx(a: &Affine, x: f64, y: f64) -> (f64, f64) {
    (
        x.mul_add(a.a, y.mul_add(a.b, a.c)),
        x.mul_add(a.d, y.mul_add(a.e, a.f)),
    )
}

/// Forward transform on contiguous parallel-array buffers.
///
/// SAFETY: All four slices must already share the same length.
#[inline]
fn forward_slice(aff: &Affine, xs: &[f64], ys: &[f64], ox: &mut [f64], oy: &mut [f64]) {
    let n = xs.len();
    // Slice all four to the same length up front so the bound checks
    // hoist out of the loop and LLVM can vectorise the body.
    let xs = &xs[..n];
    let ys = &ys[..n];
    let ox = &mut ox[..n];
    let oy = &mut oy[..n];
    for i in 0..n {
        let x = xs[i];
        let y = ys[i];
        ox[i] = x.mul_add(aff.a, y.mul_add(aff.b, aff.c));
        oy[i] = x.mul_add(aff.d, y.mul_add(aff.e, aff.f));
    }
}

impl Affine {
    /// Transform parallel `xs` / `ys` coordinate arrays, returning newly
    /// allocated `(out_x, out_y)`.
    ///
    /// Prefer [`transform_xy_into`](Self::transform_xy_into) if you can
    /// reuse output buffers — it avoids two `Array1` allocations.
    ///
    /// # Errors
    ///
    /// Returns [`AffineError::LengthMismatch`] if `xs` and `ys` differ in length.
    pub fn transform_xy(
        &self,
        xs: ArrayView1<'_, f64>,
        ys: ArrayView1<'_, f64>,
    ) -> Result<(Array1<f64>, Array1<f64>), AffineError> {
        if xs.len() != ys.len() {
            return Err(AffineError::LengthMismatch {
                xs_len: xs.len(),
                ys_len: ys.len(),
            });
        }
        let mut out_x = Array1::<f64>::uninit(xs.len());
        let mut out_y = Array1::<f64>::uninit(xs.len());
        // Initialise via the slice fast path when possible.
        // Array1::uninit is contiguous so this always succeeds.
        let ox = out_x.as_slice_mut().expect("uninit Array1 is contiguous");
        let oy = out_y.as_slice_mut().expect("uninit Array1 is contiguous");
        // SAFETY: forward_slice and the strided fallback both fully
        // initialise the entire slice. After this call every element
        // is initialised so it is sound to call `assume_init`.
        unsafe {
            transform_dispatch(self, xs, ys, ox, oy);
            Ok((out_x.assume_init(), out_y.assume_init()))
        }
    }

    /// Transform parallel `xs` / `ys` into pre-allocated outputs without
    /// allocating. The four views must share the same length.
    ///
    /// # Errors
    ///
    /// Returns [`AffineError::LengthMismatch`] when lengths differ.
    pub fn transform_xy_into(
        &self,
        xs: ArrayView1<'_, f64>,
        ys: ArrayView1<'_, f64>,
        mut out_x: ArrayViewMut1<'_, f64>,
        mut out_y: ArrayViewMut1<'_, f64>,
    ) -> Result<(), AffineError> {
        let n = xs.len();
        if ys.len() != n || out_x.len() != n || out_y.len() != n {
            return Err(AffineError::LengthMismatch {
                xs_len: n,
                ys_len: ys.len().max(out_x.len()).max(out_y.len()),
            });
        }
        // Fast path: every view is contiguous standard layout.
        if let (Some(xs_s), Some(ys_s), Some(ox_s), Some(oy_s)) = (
            xs.as_slice(),
            ys.as_slice(),
            out_x.as_slice_mut(),
            out_y.as_slice_mut(),
        ) {
            forward_slice(self, xs_s, ys_s, ox_s, oy_s);
            return Ok(());
        }
        // Strided fallback.
        ndarray::Zip::from(&xs)
            .and(&ys)
            .and(&mut out_x)
            .and(&mut out_y)
            .for_each(|&x, &y, ox, oy| {
                let (rx, ry) = tx(self, x, y);
                *ox = rx;
                *oy = ry;
            });
        Ok(())
    }

    /// Transform an `[N, 2]` packed point array in place. Each row is
    /// one `(x, y)` point.
    ///
    /// # Errors
    ///
    /// Returns [`AffineError::InvalidShape`] if `pts.ncols() != 2`.
    pub fn itransform_pairs(
        &self,
        mut pts: ArrayViewMut2<'_, f64>,
    ) -> Result<(), AffineError> {
        if pts.ncols() != 2 {
            return Err(AffineError::InvalidShape {
                expected: "[N, 2]",
                actual: pts.shape().to_vec(),
            });
        }
        // Fast path: row-major contiguous → operate on the flat slice.
        if let Some(flat) = pts.as_slice_mut() {
            for chunk in flat.chunks_exact_mut(2) {
                let x = chunk[0];
                let y = chunk[1];
                chunk[0] = x.mul_add(self.a, y.mul_add(self.b, self.c));
                chunk[1] = x.mul_add(self.d, y.mul_add(self.e, self.f));
            }
            return Ok(());
        }
        // Strided fallback.
        for mut row in pts.axis_iter_mut(Axis(0)) {
            let x = row[0];
            let y = row[1];
            row[0] = x.mul_add(self.a, y.mul_add(self.b, self.c));
            row[1] = x.mul_add(self.d, y.mul_add(self.e, self.f));
        }
        Ok(())
    }

    /// Transform an `[N, 2]` packed point array into a freshly allocated
    /// output of the same shape.
    ///
    /// # Errors
    ///
    /// Returns [`AffineError::InvalidShape`] if `pts.ncols() != 2`.
    pub fn transform_pairs(
        &self,
        pts: ArrayView2<'_, f64>,
    ) -> Result<ndarray::Array2<f64>, AffineError> {
        if pts.ncols() != 2 {
            return Err(AffineError::InvalidShape {
                expected: "[N, 2]",
                actual: pts.shape().to_vec(),
            });
        }
        let mut out = ndarray::Array2::<f64>::uninit((pts.nrows(), 2));
        // Both contiguous fast path.
        if let (Some(src), Some(dst)) = (pts.as_slice(), out.as_slice_mut()) {
            // SAFETY: src.len() == dst.len() == 2 * nrows.
            let n = src.len();
            for i in (0..n).step_by(2) {
                let x = src[i];
                let y = src[i + 1];
                dst[i].write(x.mul_add(self.a, y.mul_add(self.b, self.c)));
                dst[i + 1].write(x.mul_add(self.d, y.mul_add(self.e, self.f)));
            }
            // SAFETY: fully initialised above.
            return Ok(unsafe { out.assume_init() });
        }
        // Strided fallback.
        for (i, row) in pts.axis_iter(Axis(0)).enumerate() {
            let x = row[0];
            let y = row[1];
            out[[i, 0]].write(x.mul_add(self.a, y.mul_add(self.b, self.c)));
            out[[i, 1]].write(x.mul_add(self.d, y.mul_add(self.e, self.f)));
        }
        // SAFETY: every element was written.
        Ok(unsafe { out.assume_init() })
    }

    /// Inverse transform world `(xs, ys)` to pixel `(rows, cols)`,
    /// returning newly allocated arrays.
    ///
    /// # Errors
    ///
    /// Returns [`AffineError::LengthMismatch`] when lengths differ, or
    /// [`AffineError::TransformNotInvertible`] when `self` is degenerate.
    pub fn rowcol_xy(
        &self,
        xs: ArrayView1<'_, f64>,
        ys: ArrayView1<'_, f64>,
    ) -> Result<(Array1<f64>, Array1<f64>), AffineError> {
        if xs.len() != ys.len() {
            return Err(AffineError::LengthMismatch {
                xs_len: xs.len(),
                ys_len: ys.len(),
            });
        }
        let inv = self.inverse()?;
        let mut rows = Array1::<f64>::uninit(xs.len());
        let mut cols = Array1::<f64>::uninit(xs.len());
        let r = rows.as_slice_mut().expect("uninit Array1 is contiguous");
        let c = cols.as_slice_mut().expect("uninit Array1 is contiguous");
        unsafe {
            inverse_dispatch(&inv, xs, ys, r, c);
            Ok((rows.assume_init(), cols.assume_init()))
        }
    }

    /// Inverse SoA into pre-allocated output buffers.
    ///
    /// # Errors
    ///
    /// Same as [`rowcol_xy`](Self::rowcol_xy).
    pub fn rowcol_xy_into(
        &self,
        xs: ArrayView1<'_, f64>,
        ys: ArrayView1<'_, f64>,
        mut rows: ArrayViewMut1<'_, f64>,
        mut cols: ArrayViewMut1<'_, f64>,
    ) -> Result<(), AffineError> {
        let n = xs.len();
        if ys.len() != n || rows.len() != n || cols.len() != n {
            return Err(AffineError::LengthMismatch {
                xs_len: n,
                ys_len: ys.len().max(rows.len()).max(cols.len()),
            });
        }
        let inv = self.inverse()?;
        if let (Some(xs_s), Some(ys_s), Some(rs), Some(cs)) = (
            xs.as_slice(),
            ys.as_slice(),
            rows.as_slice_mut(),
            cols.as_slice_mut(),
        ) {
            inverse_slice(&inv, xs_s, ys_s, rs, cs);
            return Ok(());
        }
        ndarray::Zip::from(&xs)
            .and(&ys)
            .and(&mut rows)
            .and(&mut cols)
            .for_each(|&x, &y, r, c| {
                *r = x.mul_add(inv.d, y.mul_add(inv.e, inv.f));
                *c = x.mul_add(inv.a, y.mul_add(inv.b, inv.c));
            });
        Ok(())
    }
}

#[inline]
fn inverse_slice(inv: &Affine, xs: &[f64], ys: &[f64], rs: &mut [f64], cs: &mut [f64]) {
    let n = xs.len();
    let xs = &xs[..n];
    let ys = &ys[..n];
    let rs = &mut rs[..n];
    let cs = &mut cs[..n];
    for i in 0..n {
        let x = xs[i];
        let y = ys[i];
        rs[i] = x.mul_add(inv.d, y.mul_add(inv.e, inv.f));
        cs[i] = x.mul_add(inv.a, y.mul_add(inv.b, inv.c));
    }
}

/// SAFETY: `ox.len() == oy.len() == xs.len()` already established; this
/// helper merely dispatches to the contiguous fast path or the strided
/// fallback. It fully initialises both output slices.
unsafe fn transform_dispatch(
    aff: &Affine,
    xs: ArrayView1<'_, f64>,
    ys: ArrayView1<'_, f64>,
    ox: &mut [std::mem::MaybeUninit<f64>],
    oy: &mut [std::mem::MaybeUninit<f64>],
) {
    // The two output slices come from freshly-allocated uninit Array1s,
    // so they are always contiguous; we still walk inputs with ndarray
    // to support strided views.
    if let (Some(xs_s), Some(ys_s)) = (xs.as_slice(), ys.as_slice()) {
        for i in 0..xs_s.len() {
            let x = xs_s[i];
            let y = ys_s[i];
            ox[i].write(x.mul_add(aff.a, y.mul_add(aff.b, aff.c)));
            oy[i].write(x.mul_add(aff.d, y.mul_add(aff.e, aff.f)));
        }
    } else {
        for (i, (x, y)) in xs.iter().zip(ys.iter()).enumerate() {
            let (rx, ry) = tx(aff, *x, *y);
            ox[i].write(rx);
            oy[i].write(ry);
        }
    }
}

/// SAFETY: same invariants as `transform_dispatch`.
unsafe fn inverse_dispatch(
    inv: &Affine,
    xs: ArrayView1<'_, f64>,
    ys: ArrayView1<'_, f64>,
    rs: &mut [std::mem::MaybeUninit<f64>],
    cs: &mut [std::mem::MaybeUninit<f64>],
) {
    if let (Some(xs_s), Some(ys_s)) = (xs.as_slice(), ys.as_slice()) {
        for i in 0..xs_s.len() {
            let x = xs_s[i];
            let y = ys_s[i];
            rs[i].write(x.mul_add(inv.d, y.mul_add(inv.e, inv.f)));
            cs[i].write(x.mul_add(inv.a, y.mul_add(inv.b, inv.c)));
        }
    } else {
        for (i, (x, y)) in xs.iter().zip(ys.iter()).enumerate() {
            rs[i].write(x.mul_add(inv.d, y.mul_add(inv.e, inv.f)));
            cs[i].write(x.mul_add(inv.a, y.mul_add(inv.b, inv.c)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_transform_xy_contiguous() {
        let t = crate::Affine::translation(10.0, 20.0);
        let xs = array![1.0, 2.0, 3.0];
        let ys = array![10.0, 20.0, 30.0];
        let (ox, oy) = t.transform_xy(xs.view(), ys.view()).unwrap();
        assert_eq!(ox.to_vec(), vec![11.0, 12.0, 13.0]);
        assert_eq!(oy.to_vec(), vec![30.0, 40.0, 50.0]);
    }

    #[test]
    fn test_transform_xy_into() {
        let t = crate::Affine::scale(2.0, Some(3.0));
        let xs = array![1.0, 2.0];
        let ys = array![10.0, 20.0];
        let mut ox = ndarray::Array1::<f64>::zeros(2);
        let mut oy = ndarray::Array1::<f64>::zeros(2);
        t.transform_xy_into(xs.view(), ys.view(), ox.view_mut(), oy.view_mut())
            .unwrap();
        assert_eq!(ox.to_vec(), vec![2.0, 4.0]);
        assert_eq!(oy.to_vec(), vec![30.0, 60.0]);
    }

    #[test]
    fn test_transform_xy_strided_fallback() {
        let t = crate::Affine::translation(1.0, 2.0);
        // 2D base, then take a strided view (step 2 along axis 0).
        let base = array![[1.0, 9.0], [2.0, 9.0], [3.0, 9.0], [4.0, 9.0]];
        let xs = base.slice(ndarray::s![..;2, 0]).to_owned(); // [1.0, 3.0]
        let ys = base.slice(ndarray::s![..;2, 0]).to_owned(); // [1.0, 3.0]
        let xs_v = xs.view();
        let ys_v = ys.view();
        let (ox, oy) = t.transform_xy(xs_v, ys_v).unwrap();
        assert_eq!(ox.to_vec(), vec![2.0, 4.0]);
        assert_eq!(oy.to_vec(), vec![3.0, 5.0]);
    }

    #[test]
    fn test_transform_xy_length_mismatch() {
        let t = crate::Affine::identity();
        let xs = array![1.0, 2.0];
        let ys = array![1.0];
        assert!(matches!(
            t.transform_xy(xs.view(), ys.view()),
            Err(AffineError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn test_itransform_pairs_contiguous() {
        let t = crate::Affine::translation(1.0, 2.0);
        let mut pts = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        t.itransform_pairs(pts.view_mut()).unwrap();
        assert_eq!(pts, array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]);
    }

    #[test]
    fn test_itransform_pairs_strided() {
        let t = crate::Affine::translation(1.0, 2.0);
        // Build a [6, 2] row-major array and pick every other row to get
        // a non-contiguous mutable [3, 2] view (stride along axis 0 is 2).
        let mut bigger = array![
            [0.0, 0.0],
            [99.0, 99.0],
            [1.0, 1.0],
            [99.0, 99.0],
            [2.0, 2.0],
            [99.0, 99.0],
        ];
        {
            let view = bigger.slice_mut(ndarray::s![..;2, ..]);
            t.itransform_pairs(view).unwrap();
        }
        assert_eq!(bigger.row(0).to_vec(), vec![1.0, 2.0]);
        assert_eq!(bigger.row(2).to_vec(), vec![2.0, 3.0]);
        assert_eq!(bigger.row(4).to_vec(), vec![3.0, 4.0]);
        // Spacer rows untouched.
        assert_eq!(bigger.row(1).to_vec(), vec![99.0, 99.0]);
    }

    #[test]
    fn test_itransform_pairs_bad_shape() {
        let t = crate::Affine::identity();
        let mut bad = Array2::<f64>::zeros((3, 3));
        assert!(matches!(
            t.itransform_pairs(bad.view_mut()),
            Err(AffineError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_transform_pairs() {
        let t = crate::Affine::scale(2.0, None);
        let pts = array![[1.0, 1.0], [2.0, 3.0]];
        let out = t.transform_pairs(pts.view()).unwrap();
        assert_eq!(out, array![[2.0, 2.0], [4.0, 6.0]]);
    }

    #[test]
    fn test_rowcol_xy_roundtrip() {
        let aff = crate::Affine::from_gdal(&[100.0, 1.0, 0.0, 200.0, 0.0, -1.0]);
        let xs = array![100.0, 101.0, 110.0];
        let ys = array![200.0, 199.0, 190.0];
        let (rows, cols) = aff.rowcol_xy(xs.view(), ys.view()).unwrap();
        assert_eq!(rows.to_vec(), vec![0.0, 1.0, 10.0]);
        assert_eq!(cols.to_vec(), vec![0.0, 1.0, 10.0]);
    }

    #[test]
    fn test_rowcol_xy_into() {
        let aff = crate::Affine::translation(10.0, 20.0);
        let xs = array![11.0, 12.0];
        let ys = array![21.0, 22.0];
        let mut rows = ndarray::Array1::<f64>::zeros(2);
        let mut cols = ndarray::Array1::<f64>::zeros(2);
        aff.rowcol_xy_into(xs.view(), ys.view(), rows.view_mut(), cols.view_mut())
            .unwrap();
        assert_eq!(rows.to_vec(), vec![1.0, 2.0]);
        assert_eq!(cols.to_vec(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_rowcol_xy_non_invertible() {
        let aff = crate::Affine::scale(0.0, Some(0.0));
        let xs = array![1.0];
        let ys = array![1.0];
        assert!(matches!(
            aff.rowcol_xy(xs.view(), ys.view()),
            Err(AffineError::TransformNotInvertible)
        ));
    }
}
