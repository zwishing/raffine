//! Performance comparison example for the raffine library
//!
//! This example demonstrates the performance improvements achieved through
//! various optimizations in the raffine library.

use raffine::Affine;
use std::time::Instant;

fn main() {
    println!("=== Raffine Performance Comparison ===\n");

    // Create a test transformation
    let transform = Affine::rotation(45.0, None) * Affine::scale(2.0, Some(1.5)) * Affine::translation(10.0, 20.0);
    
    // Test data sizes
    let sizes = vec![100, 1000, 10000, 100000];
    
    for size in sizes {
        println!("Testing with {} points:", size);
        
        // Generate test data
        let xs: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let ys: Vec<f64> = (0..size).map(|i| (i as f64) * 0.5).collect();
        let zs: Vec<f64> = vec![0.0; size];
        
        // Test individual point transformations
        let start = Instant::now();
        let mut individual_results = Vec::with_capacity(size);
        for i in 0..size {
            let point = (xs[i], ys[i]);
            let transformed = transform * point;
            individual_results.push(transformed);
        }
        let individual_time = start.elapsed();
        
        // Test batch processing with rowcol
        let start = Instant::now();
        let (batch_xs, batch_ys) = transform.rowcol(&xs, &ys, &zs);
        let batch_time = start.elapsed();
        
        // Test in-place transformation
        let mut inplace_points: Vec<(f64, f64)> = xs.iter().zip(ys.iter()).map(|(&x, &y)| (x, y)).collect();
        let start = Instant::now();
        transform.itransform(&mut inplace_points);
        let inplace_time = start.elapsed();
        
        // Verify results are the same
        let mut all_match = true;
        for i in 0..size {
            let individual = individual_results[i];
            let batch = (batch_xs[i], batch_ys[i]);
            let inplace = inplace_points[i];
            
            if (individual.0 - batch.0).abs() > 1e-10 || (individual.1 - batch.1).abs() > 1e-10 ||
               (individual.0 - inplace.0).abs() > 1e-10 || (individual.1 - inplace.1).abs() > 1e-10 {
                all_match = false;
                break;
            }
        }
        
        println!("  Individual transformations: {:?}", individual_time);
        println!("  Batch processing (rowcol): {:?}", batch_time);
        println!("  In-place transformation: {:?}", inplace_time);
        println!("  Results match: {}", all_match);
        
        if batch_time.as_nanos() > 0 {
            let speedup = individual_time.as_nanos() as f64 / batch_time.as_nanos() as f64;
            println!("  Batch speedup: {:.2}x", speedup);
        }
        
        if inplace_time.as_nanos() > 0 {
            let speedup = individual_time.as_nanos() as f64 / inplace_time.as_nanos() as f64;
            println!("  In-place speedup: {:.2}x", speedup);
        }
        
        println!();
    }
    
    // Test matrix multiplication performance
    println!("Matrix Multiplication Performance:");
    let transforms: Vec<Affine> = (0..1000).map(|i| {
        Affine::rotation(i as f64 * 0.1, None) * 
        Affine::scale(1.0 + i as f64 * 0.001, Some(1.0 + i as f64 * 0.001)) *
        Affine::translation(i as f64 * 0.1, i as f64 * 0.1)
    }).collect();
    
    let start = Instant::now();
    let mut result = Affine::identity();
    for transform in &transforms {
        result = result * transform;
    }
    let matrix_time = start.elapsed();
    println!("  1000 matrix multiplications: {:?}", matrix_time);
    println!("  Result determinant: {}", result.determinant());
    
    println!("\n=== Performance comparison completed! ===");
}
