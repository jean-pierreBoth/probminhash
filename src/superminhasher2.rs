//! An implementation of SuperMinHash2 from:    
//! A new minwise Hashing Algorithm for Jaccard Similarity Estimation.  
//!  Otmar Ertl (2017-2018) <https://arxiv.org/abs/1706.05698>.  
//! This version corresponds to the second implementation or Ertl as given in [probminhash](https://github.com/oertl/probminhash).
//! It is as fast as the first version in [super::superminhasher::SuperMinHash] and returns sketch as u32 u64 which is more adapted to Hamming distane
//!
//! The hash values can be computed before entering SuperMinHash2 methods
//! so that the structure just computes permutation according to the paper
//! or hashing can be delegated to the sketching method.  
//! In the first case, the build_hasher should be parametrized by NoHashHasher
//! (as in finch module).  
//! In the other case Fnv (fast when hashing small values as integer according to documentation) ,
//! or fxhash can be used.



use log::trace;

use std::hash::{BuildHasher, BuildHasherDefault, Hasher, Hash};
use std::marker::PhantomData;
use rand::prelude::*;
use rand::distributions::*;
use rand_xoshiro::Xoshiro256PlusPlus;

use num::{Integer, ToPrimitive, FromPrimitive, Bounded, Unsigned};

use crate::fyshuffle::*;

use crate::invhash::*;



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
                    ((bytes[0] as u64) << 24) |
                        ((bytes[1] as u64) << 16) |
                        ((bytes[2] as u64) << 8) |
                        (bytes[3] as u64));
            },
            
            8 => {
                *self = NoHashHasher(
                    ((bytes[0] as u64) << 56) |
                        ((bytes[1] as u64) << 48) |
                        ((bytes[2] as u64) << 40) |
                        ((bytes[3] as u64) << 32) |
                        ((bytes[4] as u64) << 24) |
                        ((bytes[5] as u64) << 16) |
                        ((bytes[6] as u64) << 8) |                       
                        (bytes[7] as u64));
            },
            
            _ => panic!("bad slice len in NoHashHasher write"),
        } // end match
    }
    //
    fn finish(&self) -> u64 { self.0 }
}



/// An implementation of SuperMinHash2
/// A new minwise Hashing Algorithm for Jaccard Similarity Estimation
///  Otmar Ertl 2017-2018 arXiv <https://arxiv.org/abs/1706.05698>
/// 
/// The hash strategy can be chosen by specializing the H type 
/// to Fnv (fast when hashing small values as integer according to documentation),
/// of fxhash or any hasher chosen by the user.  
/// The hash values can also be computed before entering SuperMinHash2 methods
/// so that the structure  computes just the specific minhash part of the algorithm.
/// In this second case, the build_hasher should be parametrized by NoHashHasher
/// 
/// The signature is a Vec\<I\> where I is u32 or u64
/// To get a u32 signature, a Hasher into 32 bit value must be used, see (fxhash32or)<https://crates.io/crates/twox-hash> or (xxhash32)<https://crates.io/crates/fxhash>
/// 
/// It runs in one pass on data so it can be used in streaming

pub struct SuperMinHash2<I: Integer, T: Hash, H: Hasher+Default> {
    /// 
    hsketch:Vec<I>,
    /// initialization marker
    values :Vec<usize>,
    /// 
    l:Vec<usize>,
    /// to store the current max of current signature value
    b:Vec<usize>,
    /// rank of item hashed . Necessary for the streaming case if sketch_slice is called several times iteratively
    item_rank:usize,
    ///
    a_upper : usize,
    /// random permutation generator
    permut_generator : FYshuffle,
    /// the Hasher to use if data arrive unhashed. Anyway the data type we sketch must satisfy the trait Hash
    b_hasher: BuildHasherDefault<H>,
    /// just to mark the type we sketch
    t_marker: PhantomData<T>,
    //
    f_marker: PhantomData<I>,
}  // end of struct SuperMinHash2



impl <I, T, H : Hasher+Default> SuperMinHash2<I, T, H> 
    where   I: Integer + Unsigned + ToPrimitive + FromPrimitive + Bounded + Copy + Clone + Send + Sync + std::fmt::Debug, 
            T:Hash {
    /// allocate a struct to do SuperMinHash2.
    /// size is size of sketch. build_hasher is the build hasher for the type of Hasher we want.
    pub fn new(size:usize, build_hasher: BuildHasherDefault<H>) -> SuperMinHash2<I, T, H> {
        //
        log::info!("\n allocating-sketcher \n");
        //
        let size_i = std::mem::size_of::<I>();
        assert!(size_i >= 4, "hash values should have at least 4 bytes,so u32 or u64"); // at least u32
        //
        let sketch_init: Vec<I> = (0..size).into_iter().map(|_| I::zero()).collect();
        // we initialize sketches by something large.
        // If we initialize by f64::MAX we got a bug beccause f64::MAX as usize is 0! in cmp::min(skect, m-1) in j_2 line 216
        // computation in sketch_batch. 
        let values: Vec<usize> = (0..size).into_iter().map(|_| usize::MAX).collect();
        let l : Vec<usize> = (0..size).into_iter().map(|_|  size - 1).collect();
        let mut b : Vec<usize> = (0..size).into_iter().map(|_| 0).collect();
        b[size-1] = size;
        let permut_generator = FYshuffle::new(size);
        //
        SuperMinHash2{hsketch: sketch_init, values, l , b, item_rank: 0, a_upper : size - 1,
                    permut_generator,  
                    b_hasher: build_hasher, t_marker : PhantomData, f_marker : PhantomData}
    }  // end of new


    /// Reinitialize minhasher, keeping size of sketches.  
    /// SuperMinHash2 can then be reinitialized and used again with sketch_slice.
    /// This methods puts an end to a streaming sketching of data and resets all counters.
    pub fn reinit(&mut self) {
        let size = self.hsketch.len();
        //
        self.values.fill(usize::MAX);
        self.l.fill(size - 1);
        self.b.fill(0);
        self.b[size-1] = size;
        //
        self.hsketch.fill(I::zero());
        self.item_rank = 0;
        self.a_upper = size-1;
        self.permut_generator.reset();
    }

    /// returns a reference to computed sketches
    pub fn get_hsketch(&self) -> &Vec<I> {
        return &self.hsketch;
    }

    /// returns an estimator of jaccard index between the sketch in this structure and the sketch passed as arg
    pub fn get_jaccard_index_estimate(&self, other_sketch : &Vec<I>)  -> Result<f64, ()> {
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
        //
        // hash! even if with NoHashHasher. In this case T must be u64 or u32
        let mut hasher = self.b_hasher.build_hasher();
        to_sketch.hash(&mut hasher);
        let hval_64 : u64 = hasher.finish();
        // this cause panic if I not adapted to hasher!  We do not use it...
        let hval_i =  I::from_u64(hval_64).unwrap();
        // Then initialize random numbers generators with seedxor,
        // we have one random generator for each element to sketch
        // In probminhash we imposed T to verifiy Into<usize>. We have to be coherent..
        let mut rand_generator = Xoshiro256PlusPlus::seed_from_u64(hval_64);
        self.permut_generator.reset();
        //
        log::trace!("item-rank : {}", self.item_rank);
//        trace!("sketch hval1 :  {:?} ",hval1); // needs T : Debug
        //
        let mut j:usize = 0;
        while j <= self.a_upper {
            let r : usize =  rand_generator.sample(Standard);
            let k = self.permut_generator.next(&mut rand_generator);
            //
            log::trace!("before: j {} k {} l[k] {}, upper : {}", j , k , self.l[k], self.a_upper);
            // update hsketch and counters
            if self.l[k] >= j {
                if self.l[k] == j {
                    if  r <= self.values[k] {
                        self.values[k] = r;
                        self.hsketch[k] = hval_i;
                    }
                }
                else {
                    self.b[self.l[k]] -=1;
                    self.b[j] += 1;
                    while self.b[self.a_upper] == 0 {
                        self.a_upper = self.a_upper - 1;
                    }
                    self.l[k] = j;
                    self.values[k] = r;
                    self.hsketch[k] = hval_i;
                    log::debug!("after : j {} k {} l[k] {}, upper : {}", j , k , self.l[k], self.a_upper);
                }
            }
            j+=1;
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


} // end of impl SuperMinHash2


//===============================================================================================



/// Returns an estimator of jaccard index between 2 sketches coming from the same
/// SuperMinHash2 struct (using reinit for example) or two SuperMinHash2 struct initialized with same Hasher parameters.
/// with the same number of hash signatures.  
///   
/// Note that if *jp* is the returned value of this function,  
/// the distance between siga and sigb, associated to the jaccard index is *1.- jp* 
pub fn compute_superminhash_jaccard<F: PartialEq + num::Zero + std::fmt::Debug>(hsketch: &Vec<F>  , other_sketch: &Vec<F>)  -> Result<f32, ()> {
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
    return Ok(count as f32/other_sketch.len() as f32);
}  // end of get_jaccard_index_estimate



/// just an alias for backward compatibility
#[inline]
pub fn get_jaccard_index_estimate<F: PartialEq + num::Zero + std::fmt::Debug>(hsketch: &Vec<F>  , other_sketch: &Vec<F>)  -> Result<f32, ()>  {
    return compute_superminhash_jaccard::<F>(hsketch, other_sketch);
}


//===========================================================================================


#[cfg(test)]
mod tests {
    use super::*;
    use fnv::FnvHasher;
    use twox_hash::XxHash32;
    use crate::invhash;
    #[allow(dead_code)]
    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_build_hasher() {
        let bh = BuildHasherDefault::<FnvHasher>::default();
        let _new_hasher = bh.build_hasher();
        let _sminhash : SuperMinHash2<u64, u64, FnvHasher>= SuperMinHash2::new(10, bh);
    }  // end of test_build_hasher


    #[test]
    fn test_range_intersection_fnv_u64() {
        log_init_test();
        // we construct 2 ranges [a..b] [c..d], with a<b, b < d, c<d sketch them and compute jaccard.
        // we should get something like max(b,c) - min(b,c)/ (b-a+d-c)
        //
        let vamax = 300000;
        let va : Vec<usize> = (0..vamax).collect();
        let vbmin = 5000;
        let vbmax = 1.6 * vamax as f64;
        let vb : Vec<usize> = (vbmin..vbmax as usize).collect();
        let inter = vamax as usize - vbmin;
        let jexact = inter as f32 / vbmax as f32;
        let size = 30000;
        //
        let bh = BuildHasherDefault::<FnvHasher>::default();
        let mut sminhash : SuperMinHash2<u64, usize, FnvHasher>= SuperMinHash2::new(size, bh);
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
        let sigma = (jac * (1. - jac) / size as f32).sqrt();
        println!(" jaccard estimate {:.3e}, j exact : {:.3e} , sigma : {:.3} ", jac, jexact, sigma);
        // we have 10% common values and we sample a sketch of size 50 on 2000 values , we should see intersection
        assert!( jac > 0. && jac < jexact + 3. * sigma as f32);
    } // end of test_range_intersection_fnv_f64


#[test]
    fn test_range_intersection_fnv_u32() {
        //
        log_init_test();
        // we construct 2 ranges [a..b] [c..d], with a<b, b < d, c<d sketch them and compute jaccard.
        // we should get something like max(b,c) - min(b,c)/ (b-a+d-c)
        //
        let vamax = 300000;
        let va : Vec<u64> = (0..vamax).collect();
        let vbmin = 290000;
        let vbmax = 2 * vamax;
        let vb : Vec<u64> = (vbmin..vbmax).collect();
        let inter = vamax - vbmin;
        let jexact = inter as f32 / vbmax as f32;
        let size = 100000;
        //
        println!("sketching via superminhash2");
        let bh = BuildHasherDefault::<XxHash32>::default();
        let mut sminhash : SuperMinHash2<u32, u64, XxHash32>= SuperMinHash2::new(size, bh);
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
        log::debug!("ska : {:?}", ska);
        log::debug!("skb : {:?}", skb);
        //
        let jac = compute_superminhash_jaccard(&ska, &skb).unwrap();
        let sigma = (jac * (1. - jac) / size as f32).sqrt();
        println!(" jaccard estimate {:.3e}, j exact : {:.3e}, sigma : {:.3e} ", jac, jexact, sigma);
        // we have 10% common values and we sample a sketch of size 50 on 2000 values , we should see intersection
        assert!( jac > 0. && jac < jexact + 3. * sigma);
    } // end of test_range_intersection_fnv_f32


    
    // the following tests when data are already hashed data and we use NoHashHasher inside minhash
    #[test]
    fn test_range_intersection_already_hashed_u64() {
        log_init_test();
        //  It seems that the log initialization in only one test is sufficient (and more cause a bug).
        // we construct 2 ranges [a..b] [c..d], with a<b, b < d, c<d sketch them and compute jaccard.
        // we should get something like max(b,c) - min(b,c)/ (b-a+d-c)
        //
        let va : Vec<u64> = (0..1000).map(|x| invhash::int64_hash(x)).collect();
        let vb : Vec<u64> = (900..2000).map(|x| invhash::int64_hash(x)).collect();
        // real minhash work now
        let inter = 100;
        let jexact = inter as f32 / 2000.;
        let size = 500;
        let bh = BuildHasherDefault::<NoHashHasher>::default();
        let mut sminhash : SuperMinHash2<u64, u64, NoHashHasher>= SuperMinHash2::new(size, bh);
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
        let sigma = (jac * (1. - jac) / size as f32).sqrt();
        println!(" jaccard estimate : {:.3e}  exact value : {:.3e} , sigma : {:.3e}", jac, jexact, sigma);
        // we have 100 common values and we sample a sketch of size 50 on 2000 values , we should see intersection
        // J theo : 0.05
        assert!(jac > 0. && jac < jexact + 3. * sigma);
    } // end of test_range_intersection_already_hashed_f64
    

}
