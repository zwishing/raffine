//! Basic usage examples for the raffine library
//!
//! This example demonstrates the core functionality of the raffine library,
//! including basic transformations, matrix operations, and batch processing.

use raffine::Affine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Raffine Basic Usage Examples ===\n");

    // 1. Basic transformations
    println!("1. Basic Transformations:");
    let translation = Affine::translation(10.0, 20.0);
    let rotation = Affine::rotation(45.0, None);
    let scale = Affine::scale(2.0, Some(3.0));
    
    println!("Translation: {:?}", translation);
    println!("Rotation: {:?}", rotation);
    println!("Scale: {:?}", scale);

    // 2. Point transformation
    println!("\n2. Point Transformation:");
    let point = (1.0, 1.0);
    let translated = translation * point;
    let rotated = rotation * point;
    let scaled = scale * point;
    
    println!("Original point: {:?}", point);
    println!("Translated: {:?}", translated);
    println!("Rotated: {:?}", rotated);
    println!("Scaled: {:?}", scaled);

    // 3. Matrix composition
    println!("\n3. Matrix Composition:");
    let composite = translation * rotation * scale;
    let result = composite * point;
    println!("Composite transform: {:?}", composite);
    println!("Transformed point: {:?}", result);

    // 4. Matrix inversion
    println!("\n4. Matrix Inversion:");
    let inverse = (!translation)?;
    let original = inverse * translated;
    println!("Inverse of translation: {:?}", inverse);
    println!("Back to original: {:?}", original);

    // 5. Matrix properties
    println!("\n5. Matrix Properties:");
    println!("Determinant: {}", composite.determinant());
    println!("Is orthonormal: {}", composite.is_orthonormal());
    println!("Is conformal: {}", composite.is_conformal());
    println!("Is degenerate: {}", composite.is_degenerate());

    // 6. Batch processing with rowcol
    println!("\n6. Batch Processing:");
    let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let ys = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let zs = vec![0.0, 0.0, 0.0, 0.0, 0.0];
    
    let (new_xs, new_ys) = translation.rowcol(&xs, &ys, &zs);
    println!("Original xs: {:?}", xs);
    println!("Original ys: {:?}", ys);
    println!("Transformed xs: {:?}", new_xs);
    println!("Transformed ys: {:?}", new_ys);

    // 7. In-place transformation
    println!("\n7. In-place Transformation:");
    let mut points = vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)];
    println!("Before: {:?}", points);
    translation.itransform(&mut points);
    println!("After: {:?}", points);

    // 8. GDAL integration
    println!("\n8. GDAL Integration:");
    let gdal_params = [100.0, 1.0, 0.0, 200.0, 0.0, -1.0];
    let gdal_transform = Affine::from_gdal(
        gdal_params[0], gdal_params[1], gdal_params[2],
        gdal_params[3], gdal_params[4], gdal_params[5]
    );
    println!("GDAL transform: {:?}", gdal_transform);
    let gdal_export = gdal_transform.to_gdal();
    println!("GDAL export: {:?}", gdal_export);

    // 9. World file support
    println!("\n9. World File Support:");
    let world_content = "2.0\n0.0\n0.0\n-2.0\n100.0\n200.0\n";
    let world_transform = raffine::loadsw(world_content)?;
    println!("World file transform: {:?}", world_transform);
    let world_export = raffine::dumpsw(&world_transform);
    println!("World file export:\n{}", world_export);

    // 10. Precision control
    println!("\n10. Precision Control:");
    raffine::set_epsilon(1e-10);
    let t1 = Affine::identity();
    let t2 = Affine::translation(1e-11, 0.0);
    println!("Are transforms equal within precision? {}", t1.almost_equals(&t2, None));

    println!("\n=== All examples completed successfully! ===");
    Ok(())
}
