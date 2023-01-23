//! implementation of ordminhash based upon Probminhash2
//! 
//! See *Locality-sensitive hashing for the edit distance*
//! Marcais.G et al. BioInformatics 2019
//! and 
//! Ertl.O ProbMinHash , A Class Of Locality Sensitive Hash Probability Jaccard Similarity
//! IEEE transactions on knowledge and data engineering 2022 or [https://arxiv.org/abs/1911.00675]
//! 
//! This is a Rust reimplementation  Ertl.O code
//! 

#![allow(unused)]


use std::hash::{Hash, Hasher, BuildHasher, BuildHasherDefault};
use std::fmt::Debug;
use std::collections::HashMap;

use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::fyshuffle::FYshuffle;
use crate::maxvaluetrack::MaxValueTracker;

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
implement_maxvalue_for!(u16);
implement_maxvalue_for!(i32);
implement_maxvalue_for!(u64);
implement_maxvalue_for!(usize);



// This struct data to hash and permutation sorted l minimal values for each of the m hashed value
struct OrdMinHashStore<V> 
    where   V : MaxValue<V> + Copy + PartialOrd + Debug  {
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
    where   V : MaxValue<V> + Copy + PartialOrd + Debug {

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

    /// return l (which is also the minimum data size to hash 
    pub(crate) fn get_l(&self) -> usize {
        self.l
    }
} // end of impl OrdMinHashStore



//=========================================================================


/// The equivalent of FastOrderMinHash2 in Ertl's ProbMinHash implementation
pub struct ProbOrdMinHasher2<H> {
    m : u32,
    ///
    b_hasher: BuildHasherDefault<H>,
    //
    max_tracker : MaxValueTracker,
    //
    min_store : OrdMinHashStore<f64>,
    ///
    g : Vec<f64>,
    /// random permutation generator
    permut_generator : FYshuffle,
    // object counter
    counter : HashMap<u64, u64>,
} // end of ProbOrdMinHasher2




impl <H> ProbOrdMinHasher2<H> 
        where H : Hasher+Default {
    //
    pub fn new(m : usize, l : usize) -> Self {

        let max_tracker = MaxValueTracker::new(m);
        let minstore = OrdMinHashStore::<f64>::new(m,l);
        let permut_generator = FYshuffle::new(m);
        //
        let mut g = Vec::<f64>::with_capacity(m-1);
        for i in 1..m {
            g[i - 1] = m as f64 / (m - i) as f64;
        }
        //
        let counter = HashMap::<u64, usize>::new();
        //
        std::panic!("not yet");
    } // end new


    /// hash a full batch of data
    pub fn hash_set<D:Eq+Hash>(&mut self, data : &[D]) {
        // check size
        let size = data.len();
        if size < self.min_store.get_l() {
            log::error!("data length must be greater than {:}", self.min_store.get_l());
            std::panic!("data length must be greater than {:}", self.min_store.get_l());
        }
        // reset to a clean state
        self.counter.clear();
        self.min_store.reset_values();
        self.max_tracker.reset();
        // now we can work  
        for i in 0..size {
            // hash data value to usize
            let mut hasher = self.b_hasher.build_hasher();
            data[i].hash(&mut hasher);
            let id_hash : u64 = hasher.finish();
            let count = match self.counter.get_mut(&id_hash) {
                Some(count) => {
                        *count = *count +1;
                        *count
                }
                _                    => {
                    self.counter.insert(id_hash, 1);
                    1
                }
            };
            // get a random generator initialized corresponding to couple(id_hash, count)
            // Xoshiro256PlusPlus use [u8; 32] as seed , we must fill seed_256 with (id_hash, count)
            let seed_256 = [0u8; 32];
            std::panic!("not yet");
            let mut rng = Xoshiro256PlusPlus::from_seed(seed_256);

        }
    } // end of hash_set

}  // end of impl ProbOrdMinHasher2