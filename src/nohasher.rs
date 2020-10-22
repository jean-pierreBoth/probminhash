//! This provides a struct implementing Hasher trait
//! for u64 hashed values and doing nothing
//! to use for example in counting structures when we
//! manipulate already hashed values!


use std::hash::{Hasher};


pub struct NoHashHasher(u64);

impl Default for NoHashHasher {
    #[inline]
    fn default() -> NoHashHasher {
        NoHashHasher(0x0000000000000000)
    }
}

impl Hasher for NoHashHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        match bytes.len() {
            4 => {
                *self = NoHashHasher(
                    ((bytes[0] as u64) << 24) +
                        ((bytes[1] as u64) << 16) +
                        ((bytes[2] as u64) << 8) +
                        (bytes[3] as u64));
            },
            
            8 => {
                *self = NoHashHasher(
                    ((bytes[0] as u64) << 56) +
                        ((bytes[1] as u64) << 48) +
                        ((bytes[2] as u64) << 40) +
                        ((bytes[3] as u64) << 32) +
                        ((bytes[4] as u64) << 24) +
                        ((bytes[5] as u64) << 16) +
                        ((bytes[6] as u64) << 8) +                       
                        (bytes[7] as u64));
            },
            
            _ => panic!("bad slice len in NoHashHasher write"),
        } // end match
    }
    //
    fn finish(&self) -> u64 { self.0 }
}
