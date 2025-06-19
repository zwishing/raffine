# Raffine

一个高性能的 Rust 仿射变换库，灵感来自 Python 的 [Affine](https://github.com/rasterio/affine) 包。

## 📦 安装

在 `Cargo.toml` 中添加：

```toml
[dependencies]
raffine = "0.1"
```

## 🎯 快速开始

```rust
use raffine::Affine;

fn main() {
    // 创建基本变换
    let translation = Affine::translation(10.0, 20.0);
    let rotation = Affine::rotation(45.0, None);
    let scale = Affine::scale(2.0, Some(3.0));

    // 变换一个点
    let point = (1.0, 1.0);
    let result = translation * point;
    println!("Translated point: {:?}", result); // (11.0, 21.0)

    // 变换整数点
    let ipoint = (1isize, 1isize);
    let iresult = translation * ipoint;
    println!("Translated point: {:?}", iresult); // (11, 21)

    // 组合变换（从右到左应用）
    let composite = translation * rotation * scale;
    let transformed = composite * point;

    // 反转变换
    let inverse = (!translation).unwrap();
    let original = inverse * result;
    assert_eq!(original, point);
}
```

## 🌍 GIS 集成

### 与 GDAL 兼容

```rust
// 从 GDAL GeoTransform 格式导入
let gdal_params = [100.0, 1.0, 0.0, 200.0, 0.0, -1.0];
let transform = Affine::from_gdal(
    gdal_params[0], gdal_params[1], gdal_params[2],
    gdal_params[3], gdal_params[4], gdal_params[5]
);

// 导出为 GDAL 格式
let (c, a, b, f, d, e) = transform.to_gdal();
```

## ⚙️ 配置

### 精度控制

```rust
use raffine::{set_epsilon, get_epsilon};

// 设置全局浮点比较精度
set_epsilon(1e-10);

// 针对特定比较自定义精度
let t1 = Affine::identity();
let t2 = Affine::translation(1e-11, 0.0);
assert!(t1.almost_equals(&t2, Some(1e-10)));
```

## 📄 许可证

MIT license ([LICENSE](LICENSE) 或 http://opensource.org/licenses/MIT)

## 🙏 鸣谢

本库源自 Casey Duncan 的 Planar 包，并受到 Python Affine 包的启发。
