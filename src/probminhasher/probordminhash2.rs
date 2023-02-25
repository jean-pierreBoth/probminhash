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
use rand_xoshiro::{Xoshiro256PlusPlus};
use rand_distr::Exp1;

use wyhash::WyHash;

use crate::fyshuffle::FYshuffle;

use crate::maxvaluetrack::{MaxValue, MaxValueTracker};



// This struct data to hash and permutation sorted l minimal values for each of the m hashed value
struct OrdMinHashStore<V> 
    where   V : MaxValue + Copy + PartialOrd + Debug  {
    // number of hash values per item
    m : usize,
    // number of minimal (value, occurence) we keep
    l : usize,
    // allocated to m*l. indices[i] stores the index of data item
    indices : Vec<u64>,
    // m * l minimal hashed values generted from the data
    values : Vec<V>,
    // hashed 
    hashbuffer : Vec<u64>,
    // to change seed
    seed_rng : ThreadRng,
    // seeds for WyHash
    wyhash_seed : u64
} // end of struct OrdMinHashStore



impl<V> OrdMinHashStore<V> 
    where   V : MaxValue + Copy + PartialOrd + Debug {

    /// m is the number of hash values used, l size of minimum permutation associated values stored.
    pub fn new(m : usize, l : usize) -> Self {
        let ml = m*l;
        // iniitalize indices with max so there can be no_confusion with a data index
        let indices = (0..ml).into_iter().map(|_| u64::MAX).collect::<Vec::<u64>>();
        let values = (0..ml).into_iter().map(|_| V::get_max()).collect::<Vec::<V>>();
        let hashbuffer = (0..l).into_iter().map(|_| 0).collect();
        let rng = ThreadRng::default();
        let wyhash_seed = 0xcf7355744a6e8145;
        //
        assert!(l < 16);
        OrdMinHashStore{m, l, indices, values, hashbuffer, seed_rng : rng, wyhash_seed}
    } // end of new


    #[allow(unused)]
    pub(crate) fn change_wyhash_seed(&mut self) {
        self.wyhash_seed = self.seed_rng.next_u64();
        log::trace!("OrdMinHashStore setting wyhash_seed to : {:?}", self.wyhash_seed);

    } 
    // returns true if value could be inserted, false otherwise
    // value is hashed value of data at index idx, coming from Probordminhash2
    // we insert value and data_idx in sorted arrays values and indices
    // after l values for a given permuted_idx all indices in self.indices[l*permuted_idx..permuted_idx*(l+1)[are finite!!
    pub(crate) fn update_with_maxtracker(&mut self, permuted_idx: usize, value: &V, data_idx : usize, maxtracker : &mut MaxValueTracker<V>) -> bool {
        assert!(permuted_idx < self.m);
        let first_idx : usize = permuted_idx * self.l;
        let last_idx : usize = first_idx + self.l - 1;
        log::trace!("OrdMinHashStore::update permut idx : {}", permuted_idx);
        log::trace!("indices : {:?}", self.indices);
        log::trace!("values : {:?}", self.values);
        // if value is above the highest we have nothing to do, else we insert it at the right place
        if *value < self.values[last_idx] {
            let mut array_idx = last_idx;
            while array_idx > first_idx && *value < self.values[array_idx - 1] {
                self.values[array_idx] = self.values[array_idx - 1];
                self.indices[array_idx] = self.indices[array_idx - 1];
                array_idx -= 1;
            }
            // so self.values[array_idx] is >= self.values[array_idx]-1 or array_idx = first_idx. self.values is increasing
            self.values[array_idx] = *value;
            self.indices[array_idx] = data_idx as u64;
            maxtracker.update(permuted_idx, self.values[last_idx]);
            log::trace!("exiting update indices : {:?}", self.indices);
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
        assert!(data.len() >= self.l , "data length must be greater than l");
        log::debug!("indices : {:?}", self.indices);
        //
        let mut result = Vec::<u64>::with_capacity(self.m);
        //
        let b_hasher = BuildHasherDefault::<H>::default();
        // We use wyrand as Ertl. 
        // TODO make generic over this hasher
        //
        for i in 0..self.m {
            let mut nb_bad_indices = 0;
            let mut combine_hasher = WyHash::with_seed(self.wyhash_seed);
            let start = i * self.l;
            let end = start + self.l;
            self.indices[start..end].sort_unstable();
            // fill self.hashbuffer
            for j in 0..self.l {
                let data_idx = usize::try_from(self.indices[start+j]).unwrap();
                if data_idx < data.len() {
                    let mut hasher = b_hasher.build_hasher();
                    data[data_idx].hash(&mut hasher);
                    self.hashbuffer[j] = hasher.finish();
                    combine_hasher.write_u64(self.hashbuffer[j]);
                }
                else {
                    nb_bad_indices += 1;
                }
            }
            // combine hash values
            result.push(combine_hasher.finish());
            if nb_bad_indices > 0 {
                log::error!("OrdMinHashStore::create_signature slot i : {} nb_bad_indices : {}", i, nb_bad_indices);
                assert_eq!(nb_bad_indices, 0);
            }
            // TODO to make generic over combiner?
        }
        log::debug!("create_signature : {:?}", result);
        return result;
    } // end of update_signature


} // end of impl OrdMinHashStore



//=========================================================================


/// The equivalent of FastOrderMinHash2 in Ertl's ProbMinHash implementation
/// data length must be greater than l
pub struct ProbOrdMinHash2<H> {
    m : usize,
    ///
    b_hasher: BuildHasherDefault<H>,
    //
    max_tracker : MaxValueTracker<f64>,
    //
    min_store : OrdMinHashStore<f64>,
    ///
    g : Vec<f64>,
    /// random permutation generator
    permut_generator : FYshuffle,
    // object counter
    counter : HashMap<u64, u64>,
    // to change seed
    seed_rng : ThreadRng,
    // seed used to modify sequences if necessary (as for testing, or see stability of results.)
    seed : u64
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
        let mut rng = ThreadRng::default();
        let seed = rng.next_u64();
        //
        ProbOrdMinHash2{m, b_hasher : BuildHasherDefault::<H>::default(), max_tracker, min_store : minstore, g , 
                permut_generator : FYshuffle::new(m), counter, seed_rng : rng.clone(), seed}
    } // end new


    /// hash a full batch of data and return a signature as Vec<u64> of size m.
    /// All internal data are cleared at each new call, so the structure ProbOrdMinHash2 can be reused.
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
            self.permut_generator.reset();
            // hash data value to usize
            let mut hasher = self.b_hasher.build_hasher();
            data[i].hash(&mut hasher);
            let id_hash : u64 = hasher.finish();
            let newcount = match self.counter.get_mut(&id_hash) {
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
            seed_256[8..16].copy_from_slice(&newcount.to_ne_bytes());
            seed_256[16..24].copy_from_slice(&self.seed.to_ne_bytes());
//            seed_256[24..32].copy_from_slice(&0xcf7355744a6e8145_u64.to_ne_bytes());
    
            let mut rng = Xoshiro256PlusPlus::from_seed(seed_256);
            x = Exp1.sample(&mut rng);
            let mut nb_inserted = 0;
            while x < self.max_tracker.get_max_value() {
                // we use sampling without replacement in [0..m-1] so we can have each k only once as we exit loop before m iterations!
                let k = self.permut_generator.next(&mut rng);
                assert!(k < self.m);
                let inserted = self.min_store.update_with_maxtracker(k, &x, i, &mut self.max_tracker);
                if !inserted {
                    break;
                }
                // x is growing, so even if last update was possible at slot k, it is possible another value of x
                // cannot be inserted (if k was last possible index), if no update possible after preceding update, we can exit
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

    /// This function changes the state of internal random generator.
    /// It is mainly useful to study variance of the estimator as in tests, and should be ignored for other purposes.
    /// **If a database of hashed value is used, it must not as to keep coherent hashing.**
    pub fn change_rng_seed(&mut self) {
        self.min_store.change_wyhash_seed();
        self.seed = self.seed_rng.next_u64();
    }
}  // end of impl ProbOrdMinHash2


//============================================================================

#[cfg(test)]
mod tests {



use super::*;
use fnv::{FnvHasher};


fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}


// generate 2 sequence of length n 
// The first one has k 0 then 1, the other has n-k 0 and then 1
// edit similarity is 2*k/n weighted jaccard is 1/(n-k)
fn gen_01seq(k : usize, n : usize) -> (Vec<u32>, Vec<u32>) {
    //
    assert!(k < n);
    //
    let vec1 = (0..n).into_iter().map(|i| if i < n-k {0} else {1}).collect::<Vec<u32>>();
    let vec2 = (0..n).into_iter().map(|i| if i < k {0} else {1}).collect::<Vec<u32>>();
    //
    (vec1, vec2)
}  // end of gen_01seq



// 
fn test_vectors(m : u32, l : usize, v1 : &[u32], v2 : &[u32], nb_iter : usize) {
    //
    let mut pordminhash =  ProbOrdMinHash2::<WyHash>::new(m,l);
    // get histo results
    let mut equals = (0..m+1).into_iter().map(|_| 0).collect::<Vec<usize>>();
    //
    for _ in 0..nb_iter {
        let hash1 = pordminhash.hash_set(&v1);
        let hash2 = pordminhash.hash_set(&v2);
        let mut  nb_equal : u32 = 0;
        for i in 0..m as usize {
            if hash1[i] == hash2[i] {
                nb_equal = nb_equal + 1;
            }
        }
        equals[nb_equal as usize] += 1;
        pordminhash.change_rng_seed();
    }
    //
    log::info!(" equalities at m slots : {:?}", equals);
} // end of test_vectors


fn get_pattern_1() -> (Vec<u32>, Vec<u32>) {
    let v1: Vec<u32> = vec![0, 0, 1, 2];
    let v2: Vec<u32> = vec![0, 1, 1, 2];
    //
    return (v1,v2)
} 

fn get_pattern_2() -> (Vec<u32>, Vec<u32>) {
    let v1: Vec<u32> = vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 2, 4, 5];
    let v2: Vec<u32> = vec![0, 1, 2, 6, 4, 0, 7, 1, 2, 3, 2, 4, 5];
    //
    return (v1,v2)
} 


fn get_pattern_3() -> (Vec<u32>, Vec<u32>) {
    let v1: Vec<u32> =  vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 2, 4, 5, 0, 1, 2, 3, 4, 0, 1, 2, 6, 2, 4, 5];
    let v2: Vec<u32> = vec![0, 1, 2, 6, 4, 0, 7, 1, 2, 3, 2, 4, 5];
    //
    return (v1,v2)
} 


#[test]
fn test_ordminhash_01seq() {
    log_init_test();
    //
    log::info!("in test_ordminhash_01seq");
    //
    let (v1, v2) =  gen_01seq(3, 100);

    test_vectors(1024, 1, &v1, &v2, 1000);
    test_vectors(1024, 3, &v1, &v2, 1000);
    test_vectors(1024, 5, &v1, &v2, 1000);
}


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
fn ordminhash2_equality() {
       //
    log_init_test();
    log::info!("in test_ordminhash2_equality");
    //
    let m_s : u32 = 5;
    let l = 2;
    let nb_iter = 10;
    //
    let v = [0, 0, 1, 2, 2 , 3, 4];
    //
    test_vectors(m_s, l , &v, &v, nb_iter);
} // end of test_ordminhash2_equality




#[test]
// test pattern comparisons
fn test_ordminhash2_p1() {
    //
    log_init_test();
    log::info!("in test_ordminhash2_1");
    //
    let m_s : u32 = 1024;
    let l = 3;
    let nb_iter = 100000;
    //
    let pattern = get_pattern_1();
    //
    test_vectors(m_s, l, &pattern.0, &pattern.1, nb_iter);
} // end of test_ordminhash2_p1



#[test]
fn test_ordminhash2_p2() {
    log_init_test();
    log::info!("in test_ordminhash2_p2");
    let pattern = get_pattern_2();
    //
    let nb_iter = 100000;

    //  pattern2 m = 32, l = 3
    //  0 0 7 23 103 347 1022 2415 4579 7443 10728 13314 14353 13844 11563 8556 5604 3207 1714 732 292 101 41 8 4 0 0 0 0 0 0 0 0
    //
    log::info!("\n m : 32, l : 3");
    test_vectors(32, 3, &pattern.0, &pattern.1, nb_iter);

    // pattern2 m = 32, l = 5
    // 713 3520 9579 16223 19522 18720 14512 8965 4837 2190 817 277 95 25 4 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    //
    log::info!("\n m : 32, l : 5");
    test_vectors(32, 5, &pattern.0, &pattern.1, nb_iter);
    //
    log::info!("\n m : 1024, l : 3");
    test_vectors(1024, 3, &pattern.0, &pattern.1, nb_iter);

    log::info!("\n m : 1024, l : 5");
    test_vectors(1024, 5, &pattern.0, &pattern.1, nb_iter);
} // end of test_ordminhash2_p2


#[test]
fn test_ordminhash2_p3() {
    //
    log_init_test();
    log::info!("in test_ordminhash2_p3");
    //
    let pattern = get_pattern_3();
    //
    let nb_iter = 100000;


    /* Ertl pattern 3 m = 32, l = 3
     6800 19111  25961 22759  14462  6950   2768   876    261 43 9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

     [651, 3392, 8604, 14253, 17917, 17572, 14376, 10012, 6463, 3513, 1739, 864, 384, 146, 67, 31, 11, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    */

    log::info!("\n m : 32, l : 3");
    test_vectors(32, 3, &pattern.0, &pattern.1, nb_iter);


    /* Ertl pattern3 m = 32, l = 5
    78348 19158 2308 180 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

    probordminhash2
    [70282, 24213, 4740, 656, 96, 12, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    */
    log::info!("\n m : 32, l : 5");
    test_vectors(32, 5, &pattern.0, &pattern.1, nb_iter);


    log::info!("\n m : 1024, l : 3");
    test_vectors(1024, 3, &pattern.0, &pattern.1, nb_iter);

    log::info!("\n m : 1024, l : 5");
    test_vectors(1024, 5, &pattern.0, &pattern.1, nb_iter);
} // end of test_ordminhash2_p3



#[test]
fn test_ordminhash2_p5() {
    //
    log_init_test();
    log::info!("in test_ordminhash2_dist_5");
    //
    let nbiter = 100000;
    //
    let size: usize = 25;
    let shift = 5;
    let v1 = (0..size).into_iter().map(|i| i as u32).collect::<Vec<u32>>();
    let v2 = (0..size).into_iter().map(|i| (i + shift) as u32).collect::<Vec<u32>>();

    log::info!("shift : {}", shift);
    test_vectors(4, 2, &v1, &v2, nbiter);

    let shift = 9;
    log::info!("shift : {}", shift);
    let v1 = (0..size).into_iter().map(|i| i as u32).collect::<Vec<u32>>();
    let v2 = (0..size).into_iter().map(|i| (i + shift) as u32).collect::<Vec<u32>>();
    test_vectors(4, 2, &v1, &v2, nbiter);


} // end of test_ordminhash2_dist_p5

}
