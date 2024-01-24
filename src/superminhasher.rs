//! An implementation of Superminhash from:    
//! *A new minwise Hashing Algorithm for Jaccard Similarity Estimation.*  
//!  Otmar Ertl (2017-2018) <https://arxiv.org/abs/1706.05698>
//!
//! The hash values can be computed before entering SuperMinHash methods
//! so that the structure just computes permutation according to the paper
//! or hashing can be delegated to the sketching method.  
//! In the first case, the build_hasher should be parametrized by NoHashHasher
//! (as in finch module).  
//! In the other case Fnv (fast when hashing small values as integer according to documentation) ,
//! or fxhash can be used.



use log::trace;

use std::cmp;
use std::hash::{BuildHasher, BuildHasherDefault, Hasher, Hash};
use std::marker::PhantomData;
use rand::prelude::*;
use rand::distributions::*;
use rand_distr::uniform::SampleUniform;
use rand_xoshiro::Xoshiro256PlusPlus;

use num::Float;


#[allow(unused_imports)]
use crate::invhash;




/// This type is used to store already hashed data and implements
/// a hash that does nothing. It just stores data (u64) inside itself
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

/// An implementation of Superminhash
/// A new minwise Hashing Algorithm for Jaccard Similarity Estimation
///  Otmar Ertl 2017-2018 arXiv <https://arxiv.org/abs/1706.05698>
/// 
/// The hash strategy can be chosen by specializing the H type 
/// to Fnv (fast when hashing small values as integer according to documentation),
/// of fxhash or any hasher chosen by the user.  
/// The hash values can also be computed before entering SuperMinHash methods
/// so that the structure  computes just the specific minhash part of the algorithm.
/// In this second case, the build_hasher should be parametrized by NoHashHasher
/// (as in finch module).   
/// 
/// The signature is a Vec\<F\> where F is f32 or f64
/// 
/// It runs in one pass on data so it can be used in streaming

pub struct SuperMinHash<F: Float, T: Hash, H: Hasher+Default> {
    /// size of sketch. sketch values lives in  [0, number of sketches], so a u16 is sufficient
    hsketch:Vec<F>,
    /// initialization marker
    q:Vec<i64>,
    /// permutation vector
    p:Vec<usize>,
    /// to store the current max of current signature value
    b:Vec<i64>,
    /// rank of item hashed . Necessary for the streaming case if sketch_slice is called several times iteratively
    item_rank:usize,
    ///
    a_upper : usize,
    /// the Hasher to use if data arrive unhashed. Anyway the data type we sketch must satisfy the trait Hash
    b_hasher: BuildHasherDefault<H>,
    /// just to mark the type we sketch
    t_marker: PhantomData<T>,
}  // end of struct SuperMinHash



impl <F: Float + SampleUniform + std::fmt::Debug, T:Hash ,  H : Hasher+Default> SuperMinHash<F, T, H> {
    /// allocate a struct to do superminhash.
    /// size is size of sketch. build_hasher is the build hasher for the type of Hasher we want.
    pub fn new(size:usize, build_hasher: BuildHasherDefault<H>) -> SuperMinHash<F, T, H> {
        log::info!("\n allocating-sketcher \n");
        let mut sketch_init = Vec::<F>::with_capacity(size);
        let mut q_init = Vec::<i64>::with_capacity(size);
        let mut p_init = Vec::<usize>::with_capacity(size);
        let mut b_init = Vec::<i64>::with_capacity(size);
        // we initialize sketches by something large.
        // If we initialize by f64::MAX we got a bug beccause f64::MAX as usize is 0! in cmp::min(skect, m-1) in j_2 line 216
        // computation in sketch_batch. 
        let large:F = F::from(u32::MAX).unwrap();  // is OK even for f32
        // ensure validity of inequality of line 215.
        assert!(size < large.to_usize().unwrap());
        for _i in 0..size {
            sketch_init.push(large);
            q_init.push(-1);
            p_init.push(0);
            b_init.push(0);
        }
        b_init[size-1] = size as i64;
        //
        SuperMinHash{hsketch: sketch_init, q: q_init, p: p_init, b: b_init, item_rank: 0, a_upper : size - 1,
                     b_hasher: build_hasher, t_marker : PhantomData,}
    }  // end of new


    /// Reinitialize minhasher, keeping size of sketches.  
    /// SuperMinHash can then be reinitialized and used again with sketch_slice.
    /// This methods puts an end to a streaming sketching of data and resets all counters.
    pub fn reinit(&mut self) {
        let size = self.hsketch.len();
        let large:F = F::from(u32::MAX).unwrap();
        for i in 0..size {
            self.hsketch[i] = large;
            self.q[i] = -1;
            self.p[i] = 0;
            self.b[i] = 0;
        }
        self.b[size-1] = size as i64;
        self.item_rank = 0;
        self.a_upper = size-1;
    }

    /// returns a reference to computed sketches
    pub fn get_hsketch(&self) -> &Vec<F> {
        return &self.hsketch;
    }

    /// returns an estimator of jaccard index between the sketch in this structure and the sketch passed as arg
    pub fn get_jaccard_index_estimate(&self, other_sketch : &Vec<F>)  -> Result<f64, ()> {
        if other_sketch.len() != self.hsketch.len() {
            return Err(());
        }
        let mut count:usize = 0;
        for i in 0..self.hsketch.len() {
            if self.hsketch[i] == other_sketch[i] {
                trace!(" intersection {:?} ", self.hsketch[i]);
                count += 1;
            }             
        }
        return Ok(count as f64/other_sketch.len() as f64);
    }  // end of get_jaccard_index_estimate



    /// Insert an item in the set to sketch.  
    /// It can be used in streaming to update current sketch
    pub fn sketch(&mut self, to_sketch : &T) -> Result <(),()> {
        let m = self.hsketch.len();
        let unit_range = Uniform::<F>::new(num::zero::<F>(), num::one::<F>());
        //
        // hash! even if with NoHashHasher. In this case T must be u64 or u32
        let mut hasher = self.b_hasher.build_hasher();
        to_sketch.hash(&mut hasher);
        let hval1 : u64 = hasher.finish();
        // Then initialize random numbers generators with seedxor,
        // we have one random generator for each element to sketch
        // In probminhash we imposed T to verifiy Into<usize>. We have to be coherent..
        let mut rand_generator = Xoshiro256PlusPlus::seed_from_u64(hval1);
//        trace!("sketch hval1 :  {:?} ",hval1); // needs T : Debug
        log::trace!("item-rank : {}", self.item_rank);
        //
        let mut j:usize = 0;
        let irank = (self.item_rank) as i64;
        while j <= self.a_upper {
            let r:F = unit_range.sample(&mut rand_generator);
            let k = Uniform::<usize>::new(j, m).sample(&mut rand_generator); // m beccause upper bound of range is excluded
            //
            if self.q[j] != irank {
                self.q[j] = irank;
                self.p[j] = j;
            }
            if self.q[k] != irank {
                self.q[k] = irank;
                self.p[k] = k;
            }
            self.p.swap(j,k);
            //
            // update hsketch and counters
            let rpj = r+ (F::from(j).unwrap());
            if rpj  < self.hsketch[self.p[j]]  {
                // update of signature of rank j
                let j_2 = cmp::min(self.hsketch[self.p[j]].to_usize().unwrap(), m-1);
                self.hsketch[self.p[j]] = rpj;
                if j < j_2 {
                    // we can decrease counter of upper parts of b and update upper
                    self.b[j_2] = self.b[j_2]-1;
                    self.b[j] += 1;
                    while self.b[self.a_upper] == 0 {
                        self.a_upper = self.a_upper - 1;
                    } // end if j < j_2
                    log::trace!("after : j {} k {} self.b[j] {}, upper : {}", j , k , self.b[j], self.a_upper);
                } // end update a_upper
            } // end if r+j < ...
            j+=1;
            log::trace!("after : j {} k {} self.p[k] {}, upper : {}", j , k , self.p[k], self.a_upper);
        } // end of while on j <= upper
        self.item_rank +=1;            
        //
        return Ok(());
    }  // end of sketch

    
    /// Arg to_sketch is an array ( a slice) of values to hash.
    /// It can be used in streaming to update current sketch
    pub fn sketch_slice(&mut self, to_sketch : &[T]) -> Result <(),()> {
        let nb_elem = to_sketch.len();
        //
        if nb_elem == 0 {
            println!(" empty arg");
            return Err(());
        }
        //
        for i in 1..nb_elem {
            self.sketch(&to_sketch[i]).unwrap();
        }
        //
        return Ok(());
    } // end of sketch_batch


} // end of impl SuperMinHash


//===============================================================================================



/// Returns an estimator of jaccard index between 2 sketches coming from the same
/// SuperMinHash struct (using reinit for example) or two SuperMinHash struct initialized with same Hasher parameters.
/// with the same number of hash signatures.  
///   
/// Note that if *jp* is the returned value of this function,  
/// the distance between siga and sigb, associated to the jaccard index is *1.- jp* 
pub fn compute_superminhash_jaccard<F:Float + std::fmt::Debug>(hsketch: &Vec<F>  , other_sketch: &Vec<F>)  -> Result<F, ()> {
    if hsketch.len() != other_sketch.len() {
        return Err(());
    }
    let mut count:usize = 0;
    for i in 0..hsketch.len() {
        if hsketch[i] == other_sketch[i] {
            trace!(" intersection {:?} ", hsketch[i]);
            count += 1;
        }             
    }
    return Ok(F::from(count).unwrap()/F::from(other_sketch.len()).unwrap());
}  // end of get_jaccard_index_estimate



/// just an alias for backward compatibility
#[inline]
pub fn get_jaccard_index_estimate<F: Float + std::fmt::Debug>(hsketch: &Vec<F>  , other_sketch: &Vec<F>)  -> Result<F, ()>  {
    return compute_superminhash_jaccard::<F>(hsketch, other_sketch);
}


//===========================================================================================


#[cfg(test)]
mod tests {
    use super::*;
    use fnv::FnvHasher;


    #[allow(dead_code)]
    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_build_hasher() {
        let bh = BuildHasherDefault::<FnvHasher>::default();
        let _new_hasher = bh.build_hasher();
        let _sminhash : SuperMinHash<f64, u64, FnvHasher>= SuperMinHash::new(10, bh);
    }  // end of test_build_hasher


    #[test]
    fn test_range_intersection_fnv_f64() {
        log_init_test();
        // we construct 2 ranges [a..b] [c..d], with a<b, b < d, c<d sketch them and compute jaccard.
        // we should get something like max(b,c) - min(b,c)/ (b-a+d-c)
        //
        let vamax = 1000;
        let va : Vec<usize> = (0..vamax).collect();
        let vbmin = 990;
        let vbmax = 2000;
        let vb : Vec<usize> = (vbmin..vbmax).collect();
        let inter = vamax - vbmin;
        let jexact = inter as f64 / vbmax as f64;
        let size = 10000;
        //
        let bh = BuildHasherDefault::<FnvHasher>::default();
        let mut sminhash : SuperMinHash<f64, usize, FnvHasher>= SuperMinHash::new(size, bh);
        // now compute sketches
        let resa = sminhash.sketch_slice(&va);
        if !resa.is_ok() {
            println!("error in sketcing va");
            return;
        }
        let ska = sminhash.get_hsketch().clone();
        sminhash.reinit();
        let resb = sminhash.sketch_slice(&vb);
        if !resb.is_ok() {
            println!("error in sketching vb");
            return;
        }
        let skb = sminhash.get_hsketch();
        //
        let jac = compute_superminhash_jaccard(&ska, &skb).unwrap();
        let sigma = (jexact * (1.- jexact) / size as f64).sqrt();
        log::info!(" jaccard estimate {:.3e}, j exact : {:.3e}, sigma : {:.3e} ", jac, jexact, sigma);
        // we have 10% common values and we sample a sketch of size 50 on 2000 values , we should see intersection
        assert!( jac > 0. && jac < jexact + 3. *sigma);
    } // end of test_range_intersection_fnv_f64


#[test]
    fn test_range_intersection_fnv_f32() {
        //
        log_init_test();
        // we construct 2 ranges [a..b] [c..d], with a<b, b < d, c<d sketch them and compute jaccard.
        // we should get something like max(b,c) - min(b,c)/ (b-a+d-c)
        //
        let vamax = 300000;
        let va : Vec<usize> = (0..vamax).collect();
        let vbmin = 290000;
        let vbmax = 2 * vamax;
        let vb : Vec<usize> = (vbmin..vbmax).collect();
        let inter = vamax - vbmin;
        let jexact = inter as f64 / vbmax as f64;
        let size = 100000;
        println!("sketching via superminhash1");
        let bh = BuildHasherDefault::<FnvHasher>::default();
        let mut sminhash : SuperMinHash<f32, usize, FnvHasher>= SuperMinHash::new(size, bh);
        // now compute sketches
        let resa = sminhash.sketch_slice(&va);
        if !resa.is_ok() {
            println!("error in sketcing va");
            return;
        }
        let ska = sminhash.get_hsketch().clone();
        sminhash.reinit();
        let resb = sminhash.sketch_slice(&vb);
        if !resb.is_ok() {
            println!("error in sketching vb");
            return;
        }
        let skb = sminhash.get_hsketch();
        //
        let jac = compute_superminhash_jaccard(&ska, &skb).unwrap();
        let sigma = (jexact * (1. - jexact) / size as f64).sqrt();
        log::info!(" jaccard estimate {:.3e}, j exact : {:.3e} sigma : {:.3e}", jac, jexact, sigma);
        // we have 10% common values and we sample a sketch of size 50 on 2000 values , we should see intersection
        assert!( jac > 0. && (jac as f64) < jexact + 3. * sigma);
    } // end of test_range_intersection_fnv_f32


    
    // the following tests when data are already hashed data and we use NoHashHasher inside minhash
    #[test]
    fn test_range_intersection_already_hashed_f64() {
        log_init_test();
        //  It seems that the log initialization in only one test is sufficient (and more cause a bug).
        // we construct 2 ranges [a..b] [c..d], with a<b, b < d, c<d sketch them and compute jaccard.
        // we should get something like max(b,c) - min(b,c)/ (b-a+d-c)
        //
        let va : Vec<u64> = (0..1000).map(|x| invhash::int64_hash(x)).collect();
        let vb : Vec<u64> = (900..2000).map(|x| invhash::int64_hash(x)).collect();
        // real minhash work now
        let inter = 100;  // intersection size
        let jexact = inter as f32 / 2000.;
        let size = 70;
        let bh = BuildHasherDefault::<NoHashHasher>::default();
        let mut sminhash : SuperMinHash<f64, u64, NoHashHasher>= SuperMinHash::new(size, bh);
        // now compute sketches
        trace!("sketching a ");
        let resa = sminhash.sketch_slice(&va);
        if !resa.is_ok() {
            println!("error in sketcing va");
            return;
        }
        let ska = sminhash.get_hsketch().clone();
        sminhash.reinit();
        trace!("\n \n sketching b ");
        let resb = sminhash.sketch_slice(&vb);
        if !resb.is_ok() {
            println!("error in sketching vb");
            return;
        }
        let skb = sminhash.get_hsketch();
        //
        let jac = compute_superminhash_jaccard(&ska, &skb).unwrap() as f32;
        let sigma = (jexact * (1. - jexact) / size as f32).sqrt();
        println!(" jaccard estimate : {:.3e}  exact value : {:.3e}  sigma : {:.3e}", jac, jexact, sigma);
        // we have 100 common values and we sample a sketch of size 50 on 2000 values , we should see intersection
        // J theo : 0.05
        assert!(jac > 0. && jac < jexact + 3. * sigma);
    } // end of test_range_intersection_already_hashed_f64
    



    #[test]
    fn test_range_intersection_already_hashed_f32() {
        log_init_test();
        //  It seems that the log initialization in only one test is sufficient (and more cause a bug).
        // we construct 2 ranges [a..b] [c..d], with a<b, b < d, c<d sketch them and compute jaccard.
        // we should get something like max(b,c) - min(b,c)/ (b-a+d-c)
        //
        let va : Vec<u64> = (0..1000).map(|x| invhash::int64_hash(x)).collect();
        let vb : Vec<u64> = (900..2000).map(|x| invhash::int64_hash(x)).collect();
        let inter = 100;  // intersection size
        let jexact = inter as f32 / 2000.;
        let size = 70;
        // real minhash work now
        let bh = BuildHasherDefault::<NoHashHasher>::default();
        let mut sminhash : SuperMinHash<f32, u64, NoHashHasher>= SuperMinHash::new(50, bh);
        // now compute sketches
        trace!("sketching a ");
        let resa = sminhash.sketch_slice(&va);
        if !resa.is_ok() {
            println!("error in sketcing va");
            return;
        }
        let ska = sminhash.get_hsketch().clone();
        sminhash.reinit();
        trace!("\n \n sketching b ");
        let resb = sminhash.sketch_slice(&vb);
        if !resb.is_ok() {
            println!("error in sketching vb");
            return;
        }
        let skb = sminhash.get_hsketch();
        //
        let jac = compute_superminhash_jaccard(&ska, &skb).unwrap();
        let sigma = (jexact * (1. - jexact) / size as f32).sqrt();
        println!(" jaccard estimate : {:.3e}  exact value : {:.3e} , sigma : {:.3e}", jac, jexact, sigma);
        // we have 100 common values and we sample a sketch of size 50 on 2000 values , we should see intersection
        // J theo : 0.05
        assert!(jac > 0. && jac < jexact + 3. * sigma);
    } // end of test_range_intersection_already_hashed_f32
} // end of module test

