//! a Trait for object needing to provide an id in the form of a [u8] slice possibly
//! larger than 8 (corresponding to u64) to feed in Sha hash functions with  
//!
//! The purpose is to get a hash as perfect as possible so the slice returned must identify
//! the object uniquely


/// return a u8 slice identifying the object.
/// For example a usize could return to_le_bytes, a 2-uple v = (usize, usize) could return
/// the concatenation of slices obtained for v.0 and v.1.
/// A String can return String.into::<Vec<u8>>
/// The purpose of this is to get a u8 slice that can be fed into Sha update methods
pub trait Sig {
    /// returns the object signature
    fn get_sig(&self) -> Vec<u8>;
}


impl Sig for Vec<u8>  {

    fn get_sig(&self) -> Vec<u8> {
        return self.clone();
    }
} // end of impl Sig for <Vec<u8>>