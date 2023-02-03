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

//#![allow(unused)]


use std::hash::{Hash, Hasher, BuildHasher, BuildHasherDefault};
use std::fmt::Debug;
use std::collections::HashMap;

use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand_distr::Exp1;

use wyhash::WyHash;

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
        // iniitalize indices with max so there can be no_confusion with a data index
        let indices = (0..ml).into_iter().map(|_| u64::MAX).collect::<Vec::<u64>>();
        let values = (0..ml).into_iter().map(|_| V::get_max()).collect::<Vec::<V>>();
        let hashbuffer = (0..l).into_iter().map(|_| 0).collect();
        //
        assert!(l < 16);
        OrdMinHashStore{m, l, indices, values, hashbuffer}
    } // end of new


    // returns true if value could be inserted, false otherwise
    pub(crate) fn update(&mut self, pos: usize, value: &V, data_idx : usize) -> bool {
        assert!(pos < self.m);
        let first_idx : usize = pos * self.l;
        let last_idx : usize = first_idx + self.l - 1;
        // if value is above the highest we have nothing to do, else we insert it at the right place
        if *value < self.values[last_idx] {
            let mut array_idx = last_idx;
            while array_idx > first_idx && *value < self.values[array_idx - 1] {
                self.values[array_idx] = self.values[array_idx - 1];
                self.indices[array_idx] = self.indices[array_idx - 1];
                array_idx -= 1;
            }
            self.values[array_idx] = *value;
            self.indices[array_idx] = data_idx as u64;
            return true;
        }
        else {
            return false;
        }
    } // end of update



    pub(crate) fn reset(&mut self) {
        self.values.fill(V::get_max());
        self.indices.fill(u64::MAX);
    }

    /// return l (which is also the minimum data size to hash 
    pub(crate) fn get_l(&self) -> usize {
        self.l
    }

    // The final job
    pub(crate) fn create_signature<D:Hash, H : Hasher+Default>(&mut self, data : &[D]) -> Vec<u64> {
        //
        let mut result = Vec::<u64>::with_capacity(self.m);
        //
        let b_hasher = BuildHasherDefault::<H>::default();
        let mut hasher = b_hasher.build_hasher();
        // We use wyrand as Ertl. 
        // TODO make generic over this hasher
        let mut combine_hasher = WyHash::with_seed(0xde10c59a9218c94a);
        //
        for i in 0..self.m {
            let start = i * self.l;
            let end = start + self.l;
            self.indices[start..end].sort_unstable();
            // fill self.hashbuffer
            for j in 0..self.l {
                data[start+j].hash(&mut hasher);
                self.hashbuffer[j] = hasher.finish();
                combine_hasher.write_u64(self.hashbuffer[j]);
            }
            // combine hash values
            result.push(combine_hasher.finish());
            // TODO to make generic over combiner?
        }
        return result;
    } // end of update_signature


} // end of impl OrdMinHashStore



//=========================================================================


/// The equivalent of FastOrderMinHash2 in Ertl's ProbMinHash implementation
pub struct ProbOrdMinHash2<H> {
    m : usize,
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
} // end of ProbOrdMinHash2




impl <H> ProbOrdMinHash2<H> 
        where H : Hasher+Default {
    //
    pub fn new(m_s : u32, l : usize) -> Self {

        let m = m_s as usize;
        let max_tracker = MaxValueTracker::new(m);
        let minstore = OrdMinHashStore::<f64>::new(m,l);
        //
        let mut g = (0..(m-1)).into_iter().map(|_| 0.).collect::<Vec<f64>>();
        for i in 1..m {
            g[i - 1] = m as f64 / (m - i) as f64;
        }
        //
        let counter = HashMap::<u64, u64>::new();
        //
        ProbOrdMinHash2{m, b_hasher : BuildHasherDefault::<H>::default(), max_tracker, min_store : minstore, g , 
                permut_generator : FYshuffle::new(m), counter}
    } // end new


    /// hash a full batch of data. All internal data are cleared at each new call
    pub fn hash_set<D:Eq+Hash>(&mut self, data : &[D]) -> Vec<u64> {
        // check size
        let size = data.len();
        if size < self.min_store.get_l() {
            log::error!("data length must be greater than {:}", self.min_store.get_l());
            std::panic!("data length must be greater than {:}", self.min_store.get_l());
        }
        // reset to a clean state
        self.counter.clear();
        self.min_store.reset();
        self.max_tracker.reset();
        // now we can work  
        let mut x : f64;
        for i in 0..size {
            self.permut_generator.reset();  // TODO ?

            // hash data value to usize
            let mut hasher = self.b_hasher.build_hasher();
            data[i].hash(&mut hasher);
            let id_hash : u64 = hasher.finish();
            let count = match self.counter.get_mut(&id_hash) {
                Some(count) => {
                    *count = *count + 1;
                    *count
                }
                _                    => {
                    self.counter.insert(id_hash, 1);
                    1
                }
            };
            // get a random generator initialized with seed corresponding to couple(id_hash, count)
            // Xoshiro256PlusPlus use [u8; 32] as seed , we must fill seed_256 with (id_hash, count)
            // TODO to optimize
            let mut seed_256 = [0u8; 32];
            seed_256[0..8].copy_from_slice(&id_hash.to_ne_bytes());
            seed_256[8..16].copy_from_slice(&count.to_ne_bytes());
            let mut rng = Xoshiro256PlusPlus::from_seed(seed_256);
            x = Exp1.sample(&mut rng);
            let mut nb_inserted = 0;
            let qmax = self.max_tracker.get_max_value();
            while x < qmax {
                let k = self.permut_generator.next(&mut rng);
                assert!(k < self.m);
                let inserted = self.min_store.update(k, &x, i);
                if !inserted {
                    break;
                }
                // x is growing, so even if last update was possible, if no update possible after preceding update, we can stop
                if !self.max_tracker.is_update_possible(x) {
                    break;
                }
                //
                if nb_inserted + 1 >= self.m {
                    break;
                }
                let y : f64 = Exp1.sample(&mut rng);
                x = x + y * self.g[nb_inserted];
                //
                nb_inserted += 1;
            } 
        }
        // we can update signature
        return self.min_store.create_signature::<D,H>(data);
    } // end of hash_set

}  // end of impl ProbOrdMinHash2


//============================================================================

#[cfg(test)]
mod tests {


use super::*;
use fnv::{FnvHasher};


fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}


// 
#[allow(unused)]
fn test_vectors(m : u32, l : usize, v1 : &[u64], v2 : &[u64], nb_iter : usize) {
    let mut pordminhash =  ProbOrdMinHash2::<FnvHasher>::new(m,l);
    //
    let hash1 = pordminhash.hash_set(&v1);
    let hash2 = pordminhash.hash_set(&v2);
} // end of test_vectors




#[test] 
fn ordminhash2_random_data() {
    //
    log_init_test();
    log::info!("in test_ordminhash2_random_data");
    //
    let mut rng = rand::thread_rng();
    let data_size = 50000;
    let data = (0..data_size).into_iter().map(|_| rng.next_u64()).collect::<Vec<u64>>();

    let m_s : u32 = 500;
    let l = 3;
    let mut pordminhash =  ProbOrdMinHash2::<FnvHasher>::new(m_s,l);
    //
    let _hash = pordminhash.hash_set(&data);

} // end of test_ordminhash2_random_data


#[test]
// test pattern comparisons
fn test_ordminhash2_dist_1() {
    //
    log_init_test();
} // end of test_ordminhash2_dist_1



} // end of mod tests

