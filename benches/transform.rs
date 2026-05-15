use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array1, Array2};

use raffine::Affine;

fn make_transform() -> Affine {
    // A non-identity, non-rectilinear transform that exercises all six
    // coefficients (rotation 30deg + non-uniform scale + translation).
    Affine::translation(100.0, 200.0)
        * Affine::rotation(30.0, None)
        * Affine::scale(2.5, Some(1.7))
}

fn gen_points(n: usize) -> Vec<(f64, f64)> {
    (0..n)
        .map(|i| (i as f64 * 0.5, (i as f64 * 0.3).sin() * 100.0))
        .collect()
}

fn gen_soa(n: usize) -> (Array1<f64>, Array1<f64>) {
    let xs = Array1::from_iter((0..n).map(|i| i as f64 * 0.5));
    let ys = Array1::from_iter((0..n).map(|i| (i as f64 * 0.3).sin() * 100.0));
    (xs, ys)
}

fn gen_pairs(n: usize) -> Array2<f64> {
    let pts = gen_points(n);
    let mut a = Array2::<f64>::zeros((n, 2));
    for (i, (x, y)) in pts.into_iter().enumerate() {
        a[[i, 0]] = x;
        a[[i, 1]] = y;
    }
    a
}

fn bench_transform_vector(c: &mut Criterion) {
    let t = make_transform();
    c.bench_function("transform_vector_single", |b| {
        b.iter(|| {
            let p = black_box((3.14, 2.71));
            black_box(t.transform_vector(p))
        })
    });
}

fn bench_itransform(c: &mut Criterion) {
    let t = make_transform();
    let mut group = c.benchmark_group("itransform_slice");
    for &n in &[64usize, 1024, 1_000_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut data = gen_points(n);
            b.iter(|| {
                t.itransform(black_box(&mut data));
                black_box(&data);
            })
        });
    }
    group.finish();
}

fn bench_transform_into(c: &mut Criterion) {
    let t = make_transform();
    let mut group = c.benchmark_group("transform_into_slice");
    for &n in &[64usize, 1024, 1_000_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let src = gen_points(n);
            let mut dst = vec![(0.0_f64, 0.0_f64); n];
            b.iter(|| {
                t.transform_into(black_box(&src), black_box(&mut dst)).unwrap();
                black_box(&dst);
            })
        });
    }
    group.finish();
}

fn bench_transform_xy_into(c: &mut Criterion) {
    let t = make_transform();
    let mut group = c.benchmark_group("transform_xy_into");
    for &n in &[64usize, 1024, 1_000_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let (xs, ys) = gen_soa(n);
            let mut ox = Array1::<f64>::zeros(n);
            let mut oy = Array1::<f64>::zeros(n);
            b.iter(|| {
                t.transform_xy_into(
                    xs.view(),
                    ys.view(),
                    ox.view_mut(),
                    oy.view_mut(),
                )
                .unwrap();
                black_box(&ox);
                black_box(&oy);
            })
        });
    }
    group.finish();
}

fn bench_itransform_pairs(c: &mut Criterion) {
    let t = make_transform();
    let mut group = c.benchmark_group("itransform_pairs");
    for &n in &[64usize, 1024, 1_000_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut pts = gen_pairs(n);
            b.iter(|| {
                t.itransform_pairs(pts.view_mut()).unwrap();
                black_box(&pts);
            })
        });
    }
    group.finish();
}

fn bench_rowcol_into(c: &mut Criterion) {
    let t = make_transform();
    let mut group = c.benchmark_group("rowcol_into");
    for &n in &[1024usize, 1_000_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let xs: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
            let ys: Vec<f64> = (0..n).map(|i| 200.0 + i as f64 * 0.5).collect();
            let mut rows = Vec::with_capacity(n);
            let mut cols = Vec::with_capacity(n);
            b.iter(|| {
                rows.clear();
                cols.clear();
                t.rowcol_into(black_box(&xs), black_box(&ys), &mut rows, &mut cols)
                    .unwrap();
                black_box(&rows);
                black_box(&cols);
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_transform_vector,
    bench_itransform,
    bench_transform_into,
    bench_transform_xy_into,
    bench_itransform_pairs,
    bench_rowcol_into,
);
criterion_main!(benches);
