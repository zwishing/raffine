# Raffine

A high-performance, affine transformation library for Rust, inspired by Python's Affine package.

## 📦 Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
raffine = "0.1"
```

## 🎯 Quick Start

```rust
use raffine::Affine;

fn main() {
    // Create basic transformations
    let translation = Affine::translation(10.0, 20.0);
    let rotation = Affine::rotation(45.0, None);
    let scale = Affine::scale(2.0, Some(3.0));

    // Transform a point
    let point = (1.0, 1.0);
    let result = translation * point;
    println!("Translated point: {:?}", result); // (11.0, 21.0)

    // Compose transformations (applied right to left)
    let composite = translation * rotation * scale;
    let transformed = composite * point;
  
    // Invert transformations
    let inverse = (!translation).unwrap();
    let original = inverse * result;
    assert_eq!(original, point);
}
```

## 🌍 GIS Integration

### GDAL Compatibility

```rust
// Import from GDAL GeoTransform format
let gdal_params = [100.0, 1.0, 0.0, 200.0, 0.0, -1.0];
let transform = Affine::from_gdal(
    gdal_params[0], gdal_params[1], gdal_params[2],
    gdal_params[3], gdal_params[4], gdal_params[5]
);

// Export to GDAL format
let (c, a, b, f, d, e) = transform.to_gdal();
```

## ⚙️ Configuration

### Precision Control

```rust
use raffine::{set_epsilon, get_epsilon};

// Set global precision for floating-point comparisons
set_epsilon(1e-10);

// Custom precision for specific comparisons
let t1 = Affine::identity();
let t2 = Affine::translation(1e-11, 0.0);
assert!(t1.almost_equals(&t2, Some(1e-10)));
```

## 📄 License

MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

## 🙏 Credits

This library is derived from Casey Duncan's Planar package and inspired by Python's Affine package. Special thanks to the Python geospatial community for the excellent API design.
