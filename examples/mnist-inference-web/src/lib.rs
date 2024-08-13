#![cfg_attr(not(test), no_std)]

pub mod model;
pub mod model2;
pub mod state;
pub mod web;

pub use model2::mnist;

extern crate alloc;
