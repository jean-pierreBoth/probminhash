//! implementation of ordminhash based upon Probminhash2
//! 
//! See *Locality-sensitive hashing for the edit distance*
//! Marcais.G et al. BioInformatics 2019
//! and 
//! Ertl.O ProbMinHash , A Class Of Locality Sensitive Hash Probability Jaccard Similarity
//! IEEE transactions on knowledge and data engineering 2022 or [https://arxiv.org/abs/1911.00675]
//! 
//! This is a Rust reimplementation of Ertl.O
//! 

#![allow(unused)]


use std::hash::{Hash, Hasher};
use std::fmt::Debug;

/// A value for which there is a (natural) maximal value
pub trait MaxValue<V : PartialOrd> {
    //
    fn get_max() -> V;
} // end of trait Max


macro_rules! implement_maxvalue_for(
    ($ty:ty) => (
        impl MaxValue<$ty> for $ty {
            fn get_max() -> $ty {
                <$ty>::MAX
            }            
        }
    ) // end impl for ty
);


implement_maxvalue_for!(f64);
implement_maxvalue_for!(f32);
implement_maxvalue_for!(u32);
implement_maxvalue_for!(i32);
implement_maxvalue_for!(u64);
implement_maxvalue_for!(usize);



// This struct data to hash and permutation sorted l minimal values for each of the m hashed value
struct OrdMinHashStore<V> 
    where   V : MaxValue<V> + Copy + Eq+ PartialOrd+ Hash+ Debug  {
    // number of hash values per item
    m : usize,
    // number of minimal (value, occurence) we keep
    l : usize,
    //
    ml : usize,
    // indices[i] stores the index of data item
    indices : Vec<u64>,
    // values encountered in the data
    values : Vec<V>,
    // hashed 
    hashbuffer : Vec<u64>
} // end of struct OrdMinHashStore



impl<V> OrdMinHashStore<V> 
    where   V : MaxValue<V> + Copy + Eq + PartialOrd + Hash+ Debug {

    /// m is the number of hash values used, l size of minimum permutation associated values stored.
    pub fn new(m : usize, l : usize) -> Self {
        let ml = m*l;
        let indices = Vec::<u64>::with_capacity(ml);
        let values = Vec::<V>::with_capacity(m);
        let hashbuffer = Vec::<u64>::with_capacity(l);
        //
        assert!(l < 16);
        OrdMinHashStore{m, l, ml, indices, values, hashbuffer}
    } // end of new


    pub(crate) fn update(&mut self, hash_idx : usize, value: &V, data_idx : usize) {
        let first_idx : usize = hash_idx * self.l;
        let last_idx : usize = first_idx + self.l - 1;
        // if value is above the highest we have nothing to do, else we insert it at the right place
        if *value < self.values[last_idx] {
            let mut array_idx = last_idx;
            while(array_idx > first_idx && *value < self.values[array_idx - 1]) {
                self.values[array_idx] = self.values[array_idx - 1];
                self.indices[array_idx] = self.indices[array_idx - 1];
                array_idx -= 1;
            }
            self.values[array_idx] = *value;
            self.indices[array_idx] = data_idx as u64;
        }
    } // end of update


    pub(crate) fn reset_values(&mut self) {
        self.values.fill(V::get_max());
    }
} // end of impl OrdMinHashStore