//! Optional `ndarray` integration.
//!
//! Enable with the `ndarray` feature. The methods below all operate
//! through [`ArrayView`](ndarray::ArrayView) / [`ArrayViewMut`](ndarray::ArrayViewMut)
//! so they are zero-copy. When inputs and outputs are contiguous
//! (standard layout) the implementation takes a slice fast path that
//! the compiler autovectorises; non-contiguous (strided) inputs fall
//! back to ndarray iteration.

use ndarray::{Array1, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis};

use crate::{fma, Affine, AffineError};

/// Forward transform on contiguous parallel-array buffers.
///
/// Pre-condition: all four slices have at least `xs.len()` elements.
#[inline]
fn forward_slice(aff: &Affine, xs: &[f64], ys: &[f64], ox: &mut [f64], oy: &mut [f64]) {
    let n = xs.len();
    // The asserts let the compiler elide bounds checks inside the loop.
    assert_eq!(ys.len(), n);
    assert_eq!(ox.len(), n);
    assert_eq!(oy.len(), n);
    // Hoist coefficients into locals so the optimiser doesn't need to
    // re-prove non-aliasing of `aff` on every iteration.
    let (a, b, c, d, e, f) = (aff.a, aff.b, aff.c, aff.d, aff.e, aff.f);
    for i in 0..n {
        let x = xs[i];
        let y = ys[i];
        ox[i] = fma(x, a, fma(y, b, c));
        oy[i] = fma(x, d, fma(y, e, f));
    }
}

#[inline]
fn inverse_slice(inv: &Affine, xs: &[f64], ys: &[f64], rs: &mut [f64], cs: &mut [f64]) {
    let n = xs.len();
    assert_eq!(ys.len(), n);
    assert_eq!(rs.len(), n);
    assert_eq!(cs.len(), n);
    let (a, b, c, d, e, f) = (inv.a, inv.b, inv.c, inv.d, inv.e, inv.f);
    for i in 0..n {
        let x = xs[i];
        let y = ys[i];
        rs[i] = fma(x, d, fma(y, e, f));
        cs[i] = fma(x, a, fma(y, b, c));
    }
}

impl Affine {
    /// Transform parallel `xs` / `ys` coordinate arrays, returning newly
    /// allocated `(out_x, out_y)`.
    ///
    /// Prefer [`transform_xy_into`](Self::transform_xy_into) if you can
    /// reuse output buffers — it avoids two `Array1` allocations.
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
        let n = xs.len();
        let mut out_x = Array1::<f64>::uninit(n);
        let mut out_y = Array1::<f64>::uninit(n);
        // SAFETY: write_dispatch initialises every element of both slices
        // before we call assume_init below.
        unsafe {
            let ox = out_x
                .as_slice_mut()
                .expect("uninit Array1 is contiguous");
            let oy = out_y
                .as_slice_mut()
                .expect("uninit Array1 is contiguous");
            write_dispatch_forward(self, xs, ys, ox, oy);
            Ok((out_x.assume_init(), out_y.assume_init()))
        }
    }

    /// Transform parallel `xs` / `ys` into pre-allocated outputs without
    /// allocating. The four views must share the same length.
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
        if let (Some(xs_s), Some(ys_s), Some(ox_s), Some(oy_s)) = (
            xs.as_slice(),
            ys.as_slice(),
            out_x.as_slice_mut(),
            out_y.as_slice_mut(),
        ) {
            forward_slice(self, xs_s, ys_s, ox_s, oy_s);
            return Ok(());
        }
        let (a, b, c, d, e, f) = (self.a, self.b, self.c, self.d, self.e, self.f);
        ndarray::Zip::from(&xs)
            .and(&ys)
            .and(&mut out_x)
            .and(&mut out_y)
            .for_each(|&x, &y, ox, oy| {
                *ox = fma(x, a, fma(y, b, c));
                *oy = fma(x, d, fma(y, e, f));
            });
        Ok(())
    }

    /// Transform an `[N, 2]` packed point array in place. Each row is
    /// one `(x, y)` point.
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
        let (a, b, c, d, e, f) = (self.a, self.b, self.c, self.d, self.e, self.f);
        // Fast path: row-major contiguous → operate on the flat slice.
        if let Some(flat) = pts.as_slice_mut() {
            for chunk in flat.chunks_exact_mut(2) {
                let x = chunk[0];
                let y = chunk[1];
                chunk[0] = fma(x, a, fma(y, b, c));
                chunk[1] = fma(x, d, fma(y, e, f));
            }
            return Ok(());
        }
        // Strided fallback.
        for mut row in pts.axis_iter_mut(Axis(0)) {
            let x = row[0];
            let y = row[1];
            row[0] = fma(x, a, fma(y, b, c));
            row[1] = fma(x, d, fma(y, e, f));
        }
        Ok(())
    }

    /// Transform an `[N, 2]` packed point array into a freshly allocated
    /// output of the same shape.
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
        let (a, b, c, d, e, f) = (self.a, self.b, self.c, self.d, self.e, self.f);
        let mut out = ndarray::Array2::<f64>::uninit((pts.nrows(), 2));
        if let (Some(src), Some(dst)) = (pts.as_slice(), out.as_slice_mut()) {
            // SAFETY: src.len() == dst.len() == 2 * nrows, and chunks_exact
            // is in lockstep so every dst element is written.
            for (s, d_chunk) in src.chunks_exact(2).zip(dst.chunks_exact_mut(2)) {
                let x = s[0];
                let y = s[1];
                d_chunk[0].write(fma(x, a, fma(y, b, c)));
                d_chunk[1].write(fma(x, d, fma(y, e, f)));
            }
            return Ok(unsafe { out.assume_init() });
        }
        // Strided fallback: zip row iterators directly to avoid 2D index lookups.
        for (src_row, mut dst_row) in pts.axis_iter(Axis(0)).zip(out.axis_iter_mut(Axis(0))) {
            let x = src_row[0];
            let y = src_row[1];
            dst_row[0].write(fma(x, a, fma(y, b, c)));
            dst_row[1].write(fma(x, d, fma(y, e, f)));
        }
        // SAFETY: every element was written above.
        Ok(unsafe { out.assume_init() })
    }

    /// Inverse transform world `(xs, ys)` to pixel `(rows, cols)`,
    /// returning newly allocated arrays.
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
        let n = xs.len();
        let mut rows = Array1::<f64>::uninit(n);
        let mut cols = Array1::<f64>::uninit(n);
        unsafe {
            let r = rows.as_slice_mut().expect("uninit Array1 is contiguous");
            let c = cols.as_slice_mut().expect("uninit Array1 is contiguous");
            write_dispatch_inverse(&inv, xs, ys, r, c);
            Ok((rows.assume_init(), cols.assume_init()))
        }
    }

    /// Inverse SoA into pre-allocated output buffers.
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
        let (a, b, c, d, e, f) = (inv.a, inv.b, inv.c, inv.d, inv.e, inv.f);
        ndarray::Zip::from(&xs)
            .and(&ys)
            .and(&mut rows)
            .and(&mut cols)
            .for_each(|&x, &y, r, col| {
                *r = fma(x, d, fma(y, e, f));
                *col = fma(x, a, fma(y, b, c));
            });
        Ok(())
    }
}

/// SAFETY: caller must ensure `ox.len() == oy.len() == xs.len() == ys.len()`.
/// All elements of `ox` and `oy` are initialised on return.
unsafe fn write_dispatch_forward(
    aff: &Affine,
    xs: ArrayView1<'_, f64>,
    ys: ArrayView1<'_, f64>,
    ox: &mut [std::mem::MaybeUninit<f64>],
    oy: &mut [std::mem::MaybeUninit<f64>],
) {
    let n = xs.len();
    let (a, b, c, d, e, f) = (aff.a, aff.b, aff.c, aff.d, aff.e, aff.f);
    if let (Some(xs_s), Some(ys_s)) = (xs.as_slice(), ys.as_slice()) {
        assert_eq!(ys_s.len(), n);
        assert_eq!(ox.len(), n);
        assert_eq!(oy.len(), n);
        for i in 0..n {
            let x = xs_s[i];
            let y = ys_s[i];
            ox[i].write(fma(x, a, fma(y, b, c)));
            oy[i].write(fma(x, d, fma(y, e, f)));
        }
    } else {
        assert_eq!(ox.len(), n);
        assert_eq!(oy.len(), n);
        for (i, (x, y)) in xs.iter().zip(ys.iter()).enumerate() {
            let x = *x;
            let y = *y;
            ox[i].write(fma(x, a, fma(y, b, c)));
            oy[i].write(fma(x, d, fma(y, e, f)));
        }
    }
}

/// SAFETY: same invariants as `write_dispatch_forward`.
unsafe fn write_dispatch_inverse(
    inv: &Affine,
    xs: ArrayView1<'_, f64>,
    ys: ArrayView1<'_, f64>,
    rs: &mut [std::mem::MaybeUninit<f64>],
    cs: &mut [std::mem::MaybeUninit<f64>],
) {
    let n = xs.len();
    let (a, b, c, d, e, f) = (inv.a, inv.b, inv.c, inv.d, inv.e, inv.f);
    if let (Some(xs_s), Some(ys_s)) = (xs.as_slice(), ys.as_slice()) {
        assert_eq!(ys_s.len(), n);
        assert_eq!(rs.len(), n);
        assert_eq!(cs.len(), n);
        for i in 0..n {
            let x = xs_s[i];
            let y = ys_s[i];
            rs[i].write(fma(x, d, fma(y, e, f)));
            cs[i].write(fma(x, a, fma(y, b, c)));
        }
    } else {
        assert_eq!(rs.len(), n);
        assert_eq!(cs.len(), n);
        for (i, (x, y)) in xs.iter().zip(ys.iter()).enumerate() {
            let x = *x;
            let y = *y;
            rs[i].write(fma(x, d, fma(y, e, f)));
            cs[i].write(fma(x, a, fma(y, b, c)));
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
