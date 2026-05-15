#![cfg_attr(docsrs, feature(doc_cfg))]

mod affine;
pub use affine::*;

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
mod ndarray_support;
