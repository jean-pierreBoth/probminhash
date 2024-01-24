//! Implementation of densification algorithms above One Permutation Hashing.  
//! They provides locally sensitive sketching of unweighted data in one pass.
//! 
//! - Optimal Densification for Fast and Accurate Minwise Hashing.   
//! Anshumali Shrivastava
//! Proceedings of the 34 th International Conference on Machine 2017.
//! [pmlr-2017](https://proceedings.mlr.press/v70/shrivastava17a.html)
//! 
//! - On densification for MinWise Hashing.  
//! Mai, Rao, Kapilevitch, Rossi, Abbasi-Yadkori, Sinha.  [pmlr-2020](http://proceedings.mlr.press/v115/mai20a/mai20a.pdf)
//! 
//! For both implementation the sketch is Vec on following possible types: u64, u32 or F:Float.
//! Both algorithms can be used in streaming, see methods [sketch](OptDensMinHash::sketch()) as opposed to method
//! [sketch_slice](OptDensMinHash::sketch_slice()) (Idem for RevOptDensMinHash)
//! 
//! **Note : When used in streaming with method [sketch](OptDensMinHash::sketch()) note the need to call [end_sketch](OptDensMinHash::end_sketch())**


use std::io::Cursor;

use std::hash::{BuildHasher, BuildHasherDefault, Hasher, Hash};
use std::marker::PhantomData;
use rand::prelude::*;
use rand::distributions::*;
use rand_distr::uniform::SampleUniform;
use rand_xoshiro::Xoshiro256PlusPlus;

use murmur3::murmur3_32;

use wyhash::WyRng;

use num::Float;


/// Optimal Densification for Fast and Accurate Minwise Hashing.  
/// 
/// For usual cases where the data size to sketch is larger or of size of same size as the sketch size this algorithm is optimal.
/// In case of sketch size really larger than data to sketch, consider using [RevOptDensMinHash]
pub struct OptDensMinHash<F: Float, D: Hash, H: Hasher+Default> {
    /// size of sketch. sketch values lives in  [0, number of sketches], so a u16 is sufficient
    hsketch:Vec<F>,
    /// stored data giving minima
    values:Vec<u64>,
    /// to keep track of initialized slot
    init : Vec<bool>,
    ///
    nb_empty : i64,
    /// the Hasher to use if data arrive unhashed. Anyway the data type we sketch must satisfy the trait Hash
    b_hasher: BuildHasherDefault<H>,
    /// just to mark the type we sketch
    t_marker: PhantomData<D>,
}  // end of struct OptDensMinHash



impl <F: Float + SampleUniform + std::fmt::Debug, D:Hash + Copy,  H : Hasher+Default> OptDensMinHash<F, D, H> {
    /// allocate a struct to do .
    /// size is size of sketch. build_hasher is the build hasher for the type of Hasher we want.
    pub fn new(size:usize, build_hasher: BuildHasherDefault<H>) -> OptDensMinHash<F, D, H> {
        log::info!("\n allocating-sketcher \n");
        let mut sketch_init = Vec::<F>::with_capacity(size);
        let mut values = Vec::<u64>::with_capacity(size);
        let mut init  = Vec::<bool>::with_capacity(size);
        let large:F = F::from(u32::MAX).unwrap();  // is OK even for f32
        let nb_empty = size as i64;
        for _i in 0..size {
            sketch_init.push(large);
            values.push(u64::MAX);
            init.push(false);
        }
        OptDensMinHash{hsketch: sketch_init, values, init, nb_empty, b_hasher: build_hasher, t_marker : PhantomData,}
    } // end of new


    /// Reinitialize minhasher, keeping size of sketches.  
    /// OptMinDens is reinitialized and can be used again to sketch a new slice
    /// This methods puts an end to sketching a slice of data and resets all counters, required before sketching a new slice of data
    pub fn reinit(&mut self) {
        let size = self.hsketch.len();
        let large:F = F::from(u32::MAX).unwrap();
        for i in 0..size {
            self.hsketch[i] = large;
            self.values[i] = u64::MAX;
            self.init[i] = false;
        }
        self.nb_empty = size as i64;
    } // end of reinit


    #[allow(unused)]
    fn get_nb_empty(&self) -> i64 {
        self.nb_empty
    }


    /// returns a reference to computed sketches of type F:Float.
    pub fn get_hsketch(&self) -> &Vec<F> {
        if self.nb_empty > 0 {
            log::error!("OptDensMinHash: end_sketch should have been called");
            std::panic!("OptDensMinHash: end_sketch should have been called")
        }
        return &self.hsketch;
    } // end of get_hsketch



    /// returns a u64 signature.
    pub fn get_hsketch_u64(&self) -> Vec<u64> {
        if self.nb_empty > 0 {
            log::error!("OptDensMinHash: end_sketch should have been called");
            std::panic!("OptDensMinHash: end_sketch should have been called")
        }
        return self.values.clone();
    }
    
    /// returns a u32 signature. Under memory constraints, it could be useful to accept a small overhead (rehashing and a rellocation)
    pub fn get_hsketch_u32(&self) -> Vec<u32> {
        if self.nb_empty > 0 {
            log::error!("OptDensMinHash: end_sketch should have been called");
            std::panic!("OptDensMinHash: end_sketch should have been called")
        }
        //
        let value_32 = self.values.iter().map(|v| murmur3_32(&mut Cursor::new(v.to_ne_bytes()), 127).unwrap()).collect();
        value_32
    } // end of get_hsketch_u32
    


    /// sketch the slice
    pub fn sketch_slice(&mut self, to_sketch : &[D]) -> Result <(),()> {

        let m: usize = self.hsketch.len();
        for d in to_sketch {
            self.sketch(d);
        }
        log::info!("optdensminhash::sketch_slice sketch size : {:?},  nb empy slots : {:?}", m, self.nb_empty);

        log::debug!("optdensminhash::sketch_slice sketch size : {:?},  nb empy slots : {:?}", m, self.nb_empty);
        //
        if self.nb_empty > 0 {
        // now we run densification if necessary
            let res = self.densify();
            assert!(res.is_ok());
        }
        //
        return Ok(());
    } // end of sketch_slice



    /// sketch an item. This provides for computing sketches incrementally.   
    /// **In this case the function [end_sketch](OptDensMinHash::end_sketch) must be called before calling methods get_hsketch!!**
    pub fn sketch(&mut self, to_sketch : &D)  {
        let m = self.hsketch.len();
        let unit_range = Uniform::<F>::new(num::zero::<F>(), num::one::<F>());
        // hash! even if with NoHashHasher. In this case T must be u64 or u32
        let mut hasher = self.b_hasher.build_hasher();
        to_sketch.hash(&mut hasher);
        let hval1 : u64 = hasher.finish();
        let mut rand_generator = Xoshiro256PlusPlus::seed_from_u64(hval1);
        let r:F = unit_range.sample(&mut rand_generator);
        let k: usize = Uniform::<usize>::new(0, m).sample(&mut rand_generator); // m beccause upper bound of range is excluded
        if r <= self.hsketch[k] {
            self.hsketch[k] = r;
            self.values[k] = hval1;
            if self.init[k] == false {
                self.init[k] = true;
                self.nb_empty -= 1;
            }
        }
    } // end of sketch


    /// This method (densification) must be called before get_hsketch, get_hsketch_u32 or get_hsketch_u64
    /// if the sketching of data was done step by step with iteration through [sketch()](OptDensMinHash::sketch) method.
    /// If sketching was done by method [sketch_slice](OptDensMinHash::sketch_slice) this is done internally
    pub fn end_sketch(&mut self) {
        if self.nb_empty == 0 {
            return;
        }
        let res = self.densify();
        assert!(res.is_ok());
    }

    // This method must be called before get_hsketch, get_hsketch_u32 or get_hsketch_u64
    // if the sketching of data was done step by step with iteration through sketch() method.
    // If sketching was done by (sketch_slice)[sketch_slice] this is done internally
    fn densify(&mut self) -> Result<(),()> {

        // now we run densification
        let m: usize = self.hsketch.len();
        let mut nbpass = 1u64;
        let inrange = Uniform::<usize>::new(0, m);
        for k in 0..m { 
            if self.init[k] == false {
                // change hash function for each, item. rng has no loop at expected horizon and provides independance so we get universal hash function
                let mut rng2 = WyRng::seed_from_u64(k as u64 + 123743);
                loop {
                    // we search a non empty bin to fill slot k
                    let j: usize = inrange.sample(&mut rng2);
                    if self.init[j] {
                        self.values[k] = self.values[j];
                        self.hsketch[k] = self.hsketch[j];
                        self.init[k] = true;
                        self.nb_empty -= 1;
                        break;                       
                    }
                    nbpass += 1;
                }
            }   
        }
        //
        log::info!("end of pass {}, nb empty : {}", nbpass, self.nb_empty);  
        log::debug!("end of pass {}, nb empty : {}", nbpass, self.nb_empty);      
        assert_eq!(self.nb_empty, 0);
        //
        Ok(())
    } // end of densify


} // end of impl OptDensMinHash




// ==============================================================================

/// On densification for MinWise Hashing.  
/// Mai, Rao, Kapilevitch, Rossi, Abbasi-Yadkori, Sinha
/// [pmlr-2020](http://proceedings.mlr.press/v115/mai20a/mai20a.pdf)
/// 
/// This algorithm may have a better variance if for some reason the sketch size can be much greater
/// than the data to sketch. (In which case apparition of empty bins in the sketch needs very robust densification).
///
pub struct RevOptDensMinHash<F: Float, D: Hash, H: Hasher+Default> {
    /// size of sketch. sketch values lives in  [0, number of sketches], so a u16 is sufficient
    hsketch:Vec<F>,
    /// stored data giving minima
    values:Vec<u64>,
    /// initialized status
    init : Vec<bool>,
    ///
    nb_empty : i64,
    /// the Hasher to use if data arrive unhashed. Anyway the data type we sketch must satisfy the trait Hash
    b_hasher: BuildHasherDefault<H>,
    /// just to mark the type we sketch
    t_marker: PhantomData<D>,
}  // end of struct FastDensMinHash


impl <F: Float + SampleUniform + std::fmt::Debug, D:Hash + Copy,  H : Hasher+Default> RevOptDensMinHash<F, D, H> {
    /// allocate a struct to do superminhash.
    /// size is size of sketch. build_hasher is the build hasher for the type of Hasher we want.
    pub fn new(size:usize, build_hasher: BuildHasherDefault<H>) -> RevOptDensMinHash<F, D, H> {
        log::info!("\n allocating-sketcher \n");
        let mut sketch_init = Vec::<F>::with_capacity(size);
        let mut values: Vec<u64> = Vec::<u64>::with_capacity(size);
        let mut init: Vec<bool> = Vec::<bool>::with_capacity(size);
        let large:F = F::from(u32::MAX).unwrap();  // is OK even for f32
        let nb_empty = size as i64;
        for _i in 0..size {
            sketch_init.push(large);
            values.push(u64::MAX);
            init.push(false);
        }
        RevOptDensMinHash{hsketch: sketch_init, values, init, nb_empty, b_hasher: build_hasher, t_marker : PhantomData,}
    } // end of new

    /// Reinitialize minhasher, keeping size of sketches.  
    /// This methods puts an end to sketching a slice of data and resets all counters.
    pub fn reinit(&mut self) {
        let size = self.hsketch.len();
        let large:F = F::from(u32::MAX).unwrap();
        for i in 0..size {
            self.hsketch[i] = large;
            self.values[i] = u64::MAX;
            self.init[i] = false;
        }
        self.nb_empty = size as i64;
    }

    /// returns a reference to computed sketches of type F:Float
    pub fn get_hsketch(&self) -> &Vec<F> {
        if self.nb_empty > 0 {
            log::error!("RevOptDensMinHash: end_sketch should have been called");
            std::panic!("RevOptDensMinHash: end_sketch should have been called")
        }
        return &self.hsketch;
    }

    /// returns a u64 signature. (requires a reallocation)
    pub fn get_hsketch_u64(&self) -> Vec<u64> {
        if self.nb_empty > 0 {
            log::error!("RevOptDensMinHash: end_sketch should have been called");
            std::panic!("RevOptDensMinHash: end_sketch should have been called")
        }
        return self.values.clone();
    }

    /// returns a u32 signature. Under memory constraints, it could be useful to accept a small overhead (rehashing and a rellocation)
    pub fn get_hsketch_u32(&self) -> Vec<u32> {
        if self.nb_empty > 0 {
            log::error!("OptDensMinHash: end_sketch should have been called");
            std::panic!("OptDensMinHash: end_sketch should have been called")
        }
        //
        let value_32 = self.values.iter().map(|v| murmur3_32(&mut Cursor::new(v.to_ne_bytes()), 127).unwrap()).collect();
        value_32
    } // end of get_hsketch_u32

 
    /// sketch an item. This provides for computing sketches incrementally.   
    /// **In this case the function [end_sketch()](RevOptDensMinHash::end_sketch) must be called before calling methods get_hsketch!!**
    pub fn sketch(&mut self, d : &D) {
        //
        let m: usize = self.hsketch.len();
        let mut hasher = self.b_hasher.build_hasher();
        d.hash(&mut hasher);
        let hval1 : u64 = hasher.finish();
        let mut rand_generator = Xoshiro256PlusPlus::seed_from_u64(hval1);
        let unit_range = Uniform::<F>::new(num::zero::<F>(), num::one::<F>());
        let r:F = unit_range.sample(&mut rand_generator);
        let k: usize = Uniform::<usize>::new(0, m).sample(&mut rand_generator); // m beccause upper bound of range is excluded
        if r <= self.hsketch[k] {
            self.hsketch[k] = r;
            self.values[k] = hval1;
            if self.init[k] == false {
                self.init[k] = true;
                self.nb_empty -= 1;
            }
        }      
    } // end of fn sketch



    /// Sketch a slice of data of type D.  
    /// implementation of sketching as in Algorithm1 in paragraph3 of [pmlr-2020](http://proceedings.mlr.press/v115/mai20a/mai20a.pdf)
    pub fn sketch_slice(&mut self, to_sketch : &[D]) -> Result <(),()> {

        let m: usize = self.hsketch.len();
        for d in to_sketch {
            self.sketch(d);
        }
        //
        if self.nb_empty > 0 {
            // now we run densification if necessary
            let res = self.densify();
            assert!(res.is_ok());
        }
        log::info!("fastdensminhash::sketch_slice sketch size : {:?},  nb empy slots : {:?}", m, self.nb_empty);

        log::debug!("fastdensminhash::sketch_slice sketch size : {:?},  nb empy slots : {:?}", m, self.nb_empty);
        //    
        return Ok(());
    } // end of sketch_slice



    /// This method must be called before get_hsketch, get_hsketch_u32 or get_hsketch_u64
    /// if the sketching of data was done step by step with iteration through (sketch)[sketch] method.
    /// If sketching was done by (sketch_slice)[sketch_slice] this is done internally
    fn densify(&mut self) -> Result<(),()> {

        // now we run densification
        let m: usize = self.hsketch.len();
        let mut pass : u64 = 1;
        while self.nb_empty > 0 {
            for k in 0..m { 
                if self.init[k] {
                    let mut rng2 = WyRng::seed_from_u64((k as u64 +1) * m as u64 + pass + 253713);
                    let j: usize = Uniform::<usize>::new(0, m).sample(&mut rng2);
                    if self.init[j] == false {
                        self.values[j] = self.values[k];
                        self.hsketch[j] = self.hsketch[k];
                        self.init[j] = true;
                        self.nb_empty = self.nb_empty - 1;                       
                    }
                }
            }   
            pass += 1;
            log::debug!("end of pass {}, nb empty : {}", pass, self.nb_empty);      
        }
        //
        log::info!("end of pass {}, nb empty : {}", pass, self.nb_empty);  
        log::debug!("end of pass {}, nb empty : {}", pass, self.nb_empty);      
        assert_eq!(self.nb_empty, 0);
        //
        Ok(())
    } // end of densify

    /// This method (densification) must be called before get_hsketch, get_hsketch_u32 or get_hsketch_u64
    /// if the sketching of data was done step by step with iteration through  [sketch()](RevOptDensMinHash::sketch) method.  
    /// If sketching was done by [sketch()](RevOptDensMinHash::sketch_slice) this is done internally
    pub fn end_sketch(&mut self) {
        if self.nb_empty == 0 {
            return;
        }
        let res = self.densify();
        assert!(res.is_ok());
    }
    
} // end of impl RevOptDensMinHash

//===================================================================================================


#[cfg(test)]
mod tests {

    use super::*;
    use fnv::FnvHasher;
    use crate::jaccard::*;


    #[allow(dead_code)]
    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

/// This test how jaccard estimator behave in case where data size is an order of magnitude
/// larger than sketch size. 
    #[test]
    fn test_optdens_manybins_fnv_f64() {
        log_init_test();
        // we construct 2 ranges [a..b] [c..d], with a<b, b < d, c<d sketch them and compute jaccard.
        // we should get something like max(b,c) - min(b,c)/ (b-a+d-c)
        //
        let vamax = 1000;
        let va : Vec<usize> = (0..vamax).collect();
        let vbmin = 900;
        let vbmax = 2000;
        let vb : Vec<usize> = (vbmin..vbmax).collect();
        let inter = vamax - vbmin;
        let jexact = inter as f64 / vbmax as f64;
        let size = 1000;
        let nbtest: usize = 50;       
        // growing nbtest up to 100 shows we get out of 3 sigma as test_revoptdens_manybins_fnv_f64 still OK.
        let mut deltavec = Vec::<f64>::with_capacity(nbtest);
        for j in 1..nbtest {
            let sketch_size = size * j;
            let opt_res = test_optdens(&va, &vb, jexact, sketch_size);
            let res = opt_res.unwrap();
            let delta = (jexact - res.0).abs()/ res.1;
            deltavec.push(delta);
        }
        let mean_delta = deltavec.iter().sum::<f64>()/deltavec.len() as f64;
        log::info!("test_optdens_manybins_fnv_f64 mean delta-j/sigma : {:.3e}", mean_delta);
        //
    } // end of test_optdens_manybins_fnv_f64


    #[test]
    fn test_optdens_fewbins_fnv_f64() {
        log_init_test();
        // we construct 2 ranges [a..b] [c..d], with a<b, b < d, c<d sketch them and compute jaccard.
        // we should get something like max(b,c) - min(b,c)/ (b-a+d-c)
        //
        let vamax = 300000;
        let va : Vec<usize> = (0..vamax).collect();
        let vbmin = 290000;
        let vbmax = 2.0 * vamax as f64;
        let vb : Vec<usize> = (vbmin..vbmax as usize).collect();
        let inter = vamax - vbmin;
        let jexact = inter as f64 / vbmax as f64;
        let size = 90000;
        //
        let _res = test_optdens(&va, &vb, jexact, size);
    } // end of test_optdens_fewbins_fnv_f64


    #[test]
    fn test_revoptdens_fewbins_fnv_f64() {
        log_init_test();
        // we construct 2 ranges [a..b] [c..d], with a<b, b < d, c<d sketch them and compute jaccard.
        // we should get something like max(b,c) - min(b,c)/ (b-a+d-c)
        //
        let vamax = 3000000;
        let va : Vec<usize> = (0..vamax).collect();
        let vbmin = 2900000;
        let vbmax = 2 * vamax;
        let vb : Vec<usize> = (vbmin..vbmax).collect();
        let inter = vamax - vbmin;
        let jexact = inter as f64 / vbmax as f64;
        let size = 100000;
        //
        let res = test_revoptdens(&va, &vb, jexact, size).unwrap();
        assert!( res.0 > 0. && (res.0 - jexact).abs() < 3. * res.1);
    } // end of test_fastdens_intersection_fnv_f64



    #[test]
    fn test_revoptdens_manybins_fnv_f64() {
        log_init_test();
        // we construct 2 ranges [a..b] [c..d], with a<b, b < d, c<d sketch them and compute jaccard.
        // we should get something like max(b,c) - min(b,c)/ (b-a+d-c)
        //
        let vamax = 1000;
        let va : Vec<usize> = (0..vamax).collect();
        let vbmin = 900;
        let vbmax = 2000;
        let vb : Vec<usize> = (vbmin..vbmax).collect();
        let inter = vamax - vbmin;
        let jexact = inter as f64 / vbmax as f64;
        let size = 1000;
        let nbtest: usize = 50;       
        //
        let mut deltavec = Vec::<f64>::with_capacity(nbtest);
        for j in 1..nbtest {
            let sketch_size = size * j;
            let opt_res = test_revoptdens(&va, &vb, jexact, sketch_size);
            let res = opt_res.unwrap();
            let delta = (jexact - res.0).abs()/ res.1;
            deltavec.push(delta);
        }
        //
        let mean_delta = deltavec.iter().sum::<f64>()/deltavec.len() as f64;
        log::info!("test_revoptdens_manybins_fnv_f64 mean delta-j/sigma : {:.3e}", mean_delta);
    } // end of test_revoptdens_manybins_fnv_f64 


    //========================================

    // return j estimate and sigma
    fn test_optdens(va : &Vec<usize>, vb : &Vec<usize>, jexact : f64, size : usize)  -> Result<(f64,f64), ()> {
        //
        let bh = BuildHasherDefault::<FnvHasher>::default();
        let mut sminhash : OptDensMinHash<f64, usize, FnvHasher>= OptDensMinHash::new(size, bh);
        // now compute sketches
        let resa = sminhash.sketch_slice(&va);
        if !resa.is_ok() {
            println!("error in sketcing va");
            return Err(());
        }
        let ska = sminhash.get_hsketch().clone();
        let ska_u64 = sminhash.get_hsketch_u64().clone();
        let ska_u32 = sminhash.get_hsketch_u32();
        sminhash.reinit();
        let resb = sminhash.sketch_slice(&vb);
        if !resb.is_ok() {
            println!("error in sketching vb");
            return Err(());
        }
        let skb = sminhash.get_hsketch();
        let skb_u64 = sminhash.get_hsketch_u64();
        let skb_u32 = sminhash.get_hsketch_u32();
        //
        let jac = get_jaccard_index_estimate(&ska, &skb).unwrap();
        let sigma = (jexact * (1.- jexact) / size as f64).sqrt();
        let delta = (jac - jexact).abs()/sigma;
        log::info!(" Float sketch jaccard estimate {:.3e}, j exact : {:.3e}, sigma : {:.3e}  j-error/sigma : {:.3e}", jac, jexact, sigma, delta);
        //
        // check result with u64 signature
        //
        let jac_u64 = get_jaccard_index_estimate(&ska_u64, &skb_u64).unwrap();        
        let sigma = (jexact * (1.- jexact) / size as f64).sqrt();
        let delta = (jac_u64 - jexact).abs()/sigma;
        log::info!(" u64 sketch jaccard estimate {:.3e}, j exact : {:.3e}, sigma : {:.3e}  j-error/sigma : {:.3e}", jac_u64, jexact, sigma, delta);
        //
        // check result with u32 signature
        //
        let jac_u32 = get_jaccard_index_estimate(&ska_u32, &skb_u32).unwrap();        
        let sigma = (jexact * (1.- jexact) / size as f64).sqrt();
        let delta = (jac_u32 - jexact).abs()/sigma;
        log::info!(" u32 sketch jaccard estimate {:.3e}, j exact : {:.3e}, sigma : {:.3e}  j-error/sigma : {:.3e}", jac_u32, jexact, sigma, delta);        
        //
        return Ok((jac, sigma));
    } // end of test_optdens



    // return j estimate and sigma
    fn test_revoptdens(va : &Vec<usize>, vb : &Vec<usize>, jexact : f64, size : usize)  -> Result<(f64,f64), ()>  {
        //
        let bh = BuildHasherDefault::<FnvHasher>::default();
        let mut sminhash : RevOptDensMinHash<f64, usize, FnvHasher>= RevOptDensMinHash::new(size, bh);
        // now compute sketches
        let resa = sminhash.sketch_slice(&va);
        if !resa.is_ok() {
            println!("error in sketcing va");
            return Err(());
        }
        let ska = sminhash.get_hsketch().clone();
        let ska_u64 = sminhash.get_hsketch_u64().clone();
        let ska_u32 = sminhash.get_hsketch_u32();
        sminhash.reinit();
        let resb = sminhash.sketch_slice(&vb);
        if !resb.is_ok() {
            println!("error in sketching vb");
            return Err(());
        }
        let skb = sminhash.get_hsketch();
        let skb_u64 = sminhash.get_hsketch_u64();
        let skb_u32 = sminhash.get_hsketch_u32();
        //
        let jac = get_jaccard_index_estimate(&ska, &skb).unwrap();
        let sigma = (jexact * (1.- jexact) / size as f64).sqrt();
        let delta = (jac - jexact).abs()/sigma;
        log::info!(" jaccard estimate {:.3e}, j exact : {:.3e}, sigma : {:.3e}  j-error/sigma : {:.3e}", jac, jexact, sigma, delta);
        // we have 10% common values and we sample a sketch of size 50 on 2000 values , we should see intersection
        assert!( jac > 0.);
        //
        // check result with u64 signature
        //
        let jac_u64 = get_jaccard_index_estimate(&ska_u64, &skb_u64).unwrap();        
        let sigma = (jexact * (1.- jexact) / size as f64).sqrt();
        let delta = (jac_u64 - jexact).abs()/sigma;
        log::info!(" u64 sketch jaccard estimate {:.3e}, j exact : {:.3e}, sigma : {:.3e}  j-error/sigma : {:.3e}", jac_u64, jexact, sigma, delta);
        //
        // check result with u32 signature
        //
        let jac_u32 = get_jaccard_index_estimate(&ska_u32, &skb_u32).unwrap();        
        let sigma = (jexact * (1.- jexact) / size as f64).sqrt();
        let delta = (jac_u32 - jexact).abs()/sigma;
        log::info!(" u32 sketch jaccard estimate {:.3e}, j exact : {:.3e}, sigma : {:.3e}  j-error/sigma : {:.3e}", jac_u32, jexact, sigma, delta); 
        //
        return Ok((jac, sigma));
    } // end of test_revoptdens

} // end of tests