//! This module provides inversible hash in 32 bit and 64 bits version.
//!
//! It uses Thomas Wang's invertible integer hash functions.
//! See <https://gist.github.com/lh3/59882d6b96166dfc3d8d> for a snapshot.

// Thomas Wang's 32 Bit Mix Function: http://www.cris.com/~Ttwang/tech/inthash.htm
// Ch also https://chromium.googlesource.com/chromium/blink/+/master/Source/wtf/HashFunctions.h
//  http://c42f.github.io/2015/09/21/inverting-32-bit-wang-hash.html  chris foster blog

// https://naml.us/post/inverse-of-a-hash-function/   pour la version sans mask
// cited in Squeakr an exact and approximate k-mer counting system. Pandey Bender 2017. Known as intHash
// cited in minimap and Miniasm H. Li 2016.
// https://gist.github.com/lh3/974ced188be2f90422cc site de Heng li version avec mask buggée. ne passe pas le test.
//
//  For any 1<k<=64, let mask=(1<<k)-1. hash_64() is a bijection on [0,1<<k), which means
//  hash_64(x, mask)==hash_64(y, mask) if and only if x==y. hash_64i() is the inversion of
//  hash_64(): hash_64i(hash_64(x, mask), mask) == hash_64(hash_64i(x, mask), mask) == x.
//

//===========   version 64 bits

/// computes a u32 hash value for a u32 key.
/// we can retrieve key applying int32_hash_inverse to hash value
pub fn int64_hash(key_arg: u64) -> u64 {
    let mut key = key_arg;
    //
    key = (!key).wrapping_add(key << 21); // key = (key << 21) - key - 1
    key = key ^ key >> 24;
    key = key.wrapping_add(key << 3).wrapping_add(key << 8); // key * 265
    key = key ^ key >> 14;
    key = key.wrapping_add(key << 2).wrapping_add(key << 4); // key * 21
    key = key ^ key >> 28;
    key = key.wrapping_add(key << 31);
    key
}

/// The inversion of int64_hash.
// Modified from <https://naml.us/blog/tag/invertible>
pub fn int64_hash_inverse(key_arg: u64) -> u64 {
    let mut key = key_arg;
    //
    // Invert key = key + (key << 31)
    let mut tmp: u64 = key.wrapping_sub(key << 31);
    //
    key = key.wrapping_sub(tmp << 31);
    // Invert key = key ^ (key >> 28)
    tmp = key ^ key >> 28;
    key ^= tmp >> 28;
    // Invert key *= 21
    key = key.wrapping_mul(14933078535860113213u64);
    // Invert key = key ^ (key >> 14)
    tmp = key ^ key >> 14;
    tmp = key ^ tmp >> 14;
    tmp = key ^ tmp >> 14;
    key ^= tmp >> 14;
    // Invert key *= 265
    key = key.wrapping_mul(15244667743933553977u64);
    // Invert key = key ^ (key >> 24)
    tmp = key ^ key >> 24;
    key ^= tmp >> 24;
    // Invert key = (~key) + (key << 21)
    tmp = !key;
    tmp = !(key.wrapping_sub(tmp << 21));
    tmp = !(key.wrapping_sub(tmp << 21));
    key = !(key.wrapping_sub(tmp << 21));
    key
}

//===============================  version 32 bits  ================================

/// computes a u32 hash value for a u32 key.
/// we can retrieve key applying int32_hash_inverse to hash value
pub fn int32_hash(tohash: u32) -> u32 {
    let mut key = tohash;
    key = key.wrapping_add(!(key << 15));
    key = key ^ (key >> 10);
    key = key.wrapping_add(key << 3);
    key = key ^ (key >> 6);
    key = key.wrapping_add(!(key << 11));
    key = key ^ (key >> 16);
    key
}

/// retrieves a u32 key from a u32 hash value
pub fn int32_hash_inverse(hash: u32) -> u32 {
    let mut val = hash;
    //
    val = val ^ (val >> 16);
    val = (!val).wrapping_mul(4290770943);
    val = val ^ (val >> 6) ^ (val >> 12) ^ (val >> 18) ^ (val >> 24) ^ (val >> 30);
    val = val.wrapping_mul(954437177);
    val = val ^ (val >> 10) ^ (val >> 20) ^ (val >> 30);
    val = (!val).wrapping_mul(3221192703);
    val
}

extern crate rand;

#[cfg(test)]
mod tests {

    use super::*;
    use log::*;
    use rand::thread_rng;
    use rand::RngCore;

    #[allow(dead_code)]
    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_inversible_hash_64() {
        log_init_test();
        // random generation of values and mask and check inversion
        let mut to_hash;
        let mut hashed;
        let mut i_hashed;
        for i in 0..1000000 {
            to_hash = rand::thread_rng().next_u64();
            hashed = int64_hash(to_hash);
            i_hashed = int64_hash_inverse(hashed);
            trace!("i hash unhash = {} {}  {} ", i, to_hash, i_hashed);
            assert!(to_hash == i_hashed);
        }
    } // end of test_inversible_hash_64

    #[test]
    fn test_inversible_hash_32() {
        log_init_test();
        // random generation of values and mask and check inversion
        let mut to_hash;
        let mut hashed;
        let mut i_hashed;
        for i in 0..1000000 {
            to_hash = thread_rng().next_u32();
            hashed = int32_hash(to_hash);
            i_hashed = int32_hash_inverse(hashed);
            trace!("i hash unhash = {} {}  {} ", i, to_hash, i_hashed);
            assert!(to_hash == i_hashed);
        }
    } // end of test_inversible_hash_32
}
