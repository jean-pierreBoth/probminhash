//! module implementing probminhash variations : probminhash2, probminhash3, probminhash3a


pub mod probminhash2;
pub mod probminhash3;

pub use probminhash2::ProbMinHash2;
pub use probminhash3::{ProbMinHash3, ProbMinHash3a};