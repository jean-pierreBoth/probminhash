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




use std::hash::{BuildHasher, BuildHasherDefault, Hasher, Hash};
use std::marker::PhantomData;
use rand::prelude::*;
use rand::distributions::*;
use rand_distr::uniform::SampleUniform;
use rand_xoshiro::Xoshiro256PlusPlus;
use wyhash::WyRng;

use num::Float;


/// Optimal Densification for Fast and Accurate Minwise Hashing.   
/// For usual cases where the data size to sketch is larger than the sketch size this algorithm is optimal.
pub struct OptDensMinHash<F: Float, T: Hash, H: Hasher+Default> {
    /// size of sketch. sketch values lives in  [0, number of sketches], so a u16 is sufficient
    hsketch:Vec<F>,
    /// stored data giving minima
    values:Vec<Option<T>>,
    /// the Hasher to use if data arrive unhashed. Anyway the data type we sketch must satisfy the trait Hash
    b_hasher: BuildHasherDefault<H>,
    /// just to mark the type we sketch
    t_marker: PhantomData<T>,
}  // end of struct OptDensMinHash



impl <F: Float + SampleUniform + std::fmt::Debug, T:Hash + Copy,  H : Hasher+Default> OptDensMinHash<F, T, H> {
    /// allocate a struct to do superminhash.
    /// size is size of sketch. build_hasher is the build hasher for the type of Hasher we want.
    pub fn new(size:usize, build_hasher: BuildHasherDefault<H>) -> OptDensMinHash<F, T, H> {
        let mut sketch_init = Vec::<F>::with_capacity(size);
        let mut values = Vec::<Option<T>>::with_capacity(size);
        let large:F = F::from(u32::MAX).unwrap();  // is OK even for f32
        for _i in 0..size {
            sketch_init.push(large);
            values.push(None);
        }
        OptDensMinHash{hsketch: sketch_init, values,b_hasher: build_hasher, t_marker : PhantomData,}
    } // end of new


    /// Reinitialize minhasher, keeping size of sketches.  
    /// SuperMinHash can then be reinitialized and used again with sketch_slice.
    /// This methods puts an end to a streaming sketching of data and resets all counters.
    pub fn reinit(&mut self) {
        let size = self.hsketch.len();
        let large:F = F::from(u32::MAX).unwrap();
        for i in 0..size {
            self.hsketch[i] = large;
            self.values[i] = None;
        }
    }

    /// returns a reference to computed sketches
    pub fn get_hsketch(&self) -> &Vec<F> {
        return &self.hsketch;
    }

    pub fn sketch_slice(&mut self, to_sketch : &[T]) -> Result <(),()> {

        let m = self.hsketch.len();
        let unit_range = Uniform::<F>::new(num::zero::<F>(), num::one::<F>());
        for d in to_sketch {
            // hash! even if with NoHashHasher. In this case T must be u64 or u32
            let mut hasher = self.b_hasher.build_hasher();
            d.hash(&mut hasher);
            let hval1 : u64 = hasher.finish();
            let mut rand_generator = Xoshiro256PlusPlus::seed_from_u64(hval1);
            let r:F = unit_range.sample(&mut rand_generator);
            let k: usize = Uniform::<usize>::new(0, m).sample(&mut rand_generator); // m beccause upper bound of range is excluded
            if r <= self.hsketch[k] {
                self.hsketch[k] = r;
                self.values[k] = Some(*d);
            }
        }
        let nb_empty : usize = self.values.iter().map(|x| if x.is_none() { 1} else {0}).sum();
        log::debug!("optdensminhash::sketch_slice sketch size : {:?},  nb empy slots : {:?}", m, nb_empty);
        if nb_empty == 0 {
            return Ok(());
        }
        // now we run densification
        let mut nbpass = 1u64;
        let inrange = Uniform::<usize>::new(0, m);
        for k in 0..m { 
            if self.values[k].is_none() {
                // change hash function for each, item. rng has no loop at expected horizon and provides independance so we get universal hash function
                let mut rng2 = WyRng::seed_from_u64(k as u64 + 123743);
                loop {
                    // we search a non empty bin to fill slot k
                    let j: usize = inrange.sample(&mut rng2);
                    if self.values[j].is_some() {
                        self.values[k] = self.values[j];
                        self.hsketch[k] = self.hsketch[j];
                        break;                       
                    }
                    nbpass += 1;
                }
            }   
        }
        //
        let nb_empty : usize = self.values.iter().map(|x| if x.is_none() { 1} else {0}).sum();
        log::debug!("end of pass {}, nb empty : {}", nbpass, nb_empty);      
        assert_eq!(nb_empty, 0);
        //
        return Ok(());
    } // end of sketch_slice


} // end of impl OptDensMinHash




// ==============================================================================

/// On densification for MinWise Hashing
/// Mai, Rao, Kapilevitch, Rossi, Abbasi-Yadkori, Sinha
/// [pmlr-2020](http://proceedings.mlr.press/v115/mai20a/mai20a.pdf)
/// 
/// This algorithm may have a better variance if there are many empty bins at first pass of OPH.
///
pub struct RevOptDensMinHash<F: Float, T: Hash, H: Hasher+Default> {
    /// size of sketch. sketch values lives in  [0, number of sketches], so a u16 is sufficient
    hsketch:Vec<F>,
    /// stored data giving minima
    values:Vec<Option<T>>,
    /// the Hasher to use if data arrive unhashed. Anyway the data type we sketch must satisfy the trait Hash
    b_hasher: BuildHasherDefault<H>,
    /// just to mark the type we sketch
    t_marker: PhantomData<T>,
}  // end of struct FastDensMinHash


impl <F: Float + SampleUniform + std::fmt::Debug, T:Hash + Copy,  H : Hasher+Default> RevOptDensMinHash<F, T, H> {
    /// allocate a struct to do superminhash.
    /// size is size of sketch. build_hasher is the build hasher for the type of Hasher we want.
    pub fn new(size:usize, build_hasher: BuildHasherDefault<H>) -> RevOptDensMinHash<F, T, H> {
        let mut sketch_init = Vec::<F>::with_capacity(size);
        let mut values = Vec::<Option<T>>::with_capacity(size);
        let large:F = F::from(u32::MAX).unwrap();  // is OK even for f32
        for _i in 0..size {
            sketch_init.push(large);
            values.push(None);
        }
        RevOptDensMinHash{hsketch: sketch_init, values,b_hasher: build_hasher, t_marker : PhantomData,}
    } // end of new

    /// Reinitialize minhasher, keeping size of sketches.  
    /// SuperMinHash can then be reinitialized and used again with sketch_slice.
    /// This methods puts an end to a streaming sketching of data and resets all counters.
    pub fn reinit(&mut self) {
        let size = self.hsketch.len();
        let large:F = F::from(u32::MAX).unwrap();
        for i in 0..size {
            self.hsketch[i] = large;
            self.values[i] = None;
        }
    }

    /// returns a reference to computed sketches
    pub fn get_hsketch(&self) -> &Vec<F> {
        return &self.hsketch;
    }

    /// implementation of sketching as in Algorithm1 in paragraph3 of [pmlr-2020](http://proceedings.mlr.press/v115/mai20a/mai20a.pdf)
    pub fn sketch_slice(&mut self, to_sketch : &[T]) -> Result <(),()> {

        let m = self.hsketch.len();
        let unit_range = Uniform::<F>::new(num::zero::<F>(), num::one::<F>());
        for d in to_sketch {
            // hash! even if with NoHashHasher. In this case T must be u64 or u32
            let mut hasher = self.b_hasher.build_hasher();
            d.hash(&mut hasher);
            let hval1 : u64 = hasher.finish();
            let mut rand_generator = Xoshiro256PlusPlus::seed_from_u64(hval1);
            let r:F = unit_range.sample(&mut rand_generator);
            let k: usize = Uniform::<usize>::new(0, m).sample(&mut rand_generator); // m beccause upper bound of range is excluded
            if r <= self.hsketch[k] {
                self.hsketch[k] = r;
                self.values[k] = Some(*d);
            }
        }
        let mut nb_empty : usize = self.values.iter().map(|x| if x.is_none() { 1} else {0}).sum();
        log::debug!("fastdensminhash::sketch_slice sketch size : {:?},  nb empy slots : {:?}", m, nb_empty);
        if nb_empty == 0 {
            return Ok(());
        }
        // now we run densification
        let mut pass : u64 = 1;
        while nb_empty > 0 {
            for k in 0..m { 
                if self.values[k].is_some() {
                    let mut rng2 = WyRng::seed_from_u64((k as u64 +1) * m as u64 + pass + 253713);
                    let j: usize = Uniform::<usize>::new(0, m).sample(&mut rng2);
                    if self.values[j].is_none() {
                        self.values[j] = self.values[k];
                        self.hsketch[j] = self.hsketch[k];
                        nb_empty = nb_empty - 1;                       
                    }
                }
            }   
            pass += 1;
            log::debug!("end of pass {}, nb empty : {}", pass, nb_empty);      
        }
        //
        let nb_empty : usize = self.values.iter().map(|x| if x.is_none() { 1} else {0}).sum();
        assert_eq!(nb_empty, 0);
        //
        return Ok(());
    } // end of sketch_slice




    /// returns an estimator of jaccard index between the sketch in this structure and the sketch passed as arg
    pub fn get_jaccard_index_estimate(&self, other_sketch : &Vec<F>)  -> Result<f64, ()> {
        if other_sketch.len() != self.hsketch.len() {
            return Err(());
        }
        let mut count:usize = 0;
        for i in 0..self.hsketch.len() {
            if self.hsketch[i] == other_sketch[i] {
                log::trace!(" intersection {:?} ", self.hsketch[i]);
                count += 1;
            }             
        }
        return Ok(count as f64/other_sketch.len() as f64);
    }  // end of get_jaccard_index_estimate

} // end of impl FastDensMinHash

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


    #[test]
    fn test_fastdens_manybins_fnv_f64() {
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
        let size = 50000;
        //
        test_fastdens(&va, &vb, jexact, size);
    } // end of test_fastdens_intersection_fnv_f64


    #[test]
    fn test_fastdens_fewbins_fnv_f64() {
        log_init_test();
        // we construct 2 ranges [a..b] [c..d], with a<b, b < d, c<d sketch them and compute jaccard.
        // we should get something like max(b,c) - min(b,c)/ (b-a+d-c)
        //
        let vamax = 300000;
        let va : Vec<usize> = (0..vamax).collect();
        let vbmin = 50000;
        let vbmax = 2 * vamax;
        let vb : Vec<usize> = (vbmin..vbmax).collect();
        let inter = vamax - vbmin;
        let jexact = inter as f64 / vbmax as f64;
        let size = 50000;
        //
        test_fastdens(&va, &vb, jexact, size);
    } // end of test_fastdens_intersection_fnv_f64



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
        let size = 50000;       
        //
        test_optdens(&va, &vb, jexact, size);
    } // end of test_optdens_intersection_fnv_f64



    #[test]
    fn test_optdens_fewbins_fnv_f64() {
        log_init_test();
        // we construct 2 ranges [a..b] [c..d], with a<b, b < d, c<d sketch them and compute jaccard.
        // we should get something like max(b,c) - min(b,c)/ (b-a+d-c)
        //
        let vamax = 300000;
        let va : Vec<usize> = (0..vamax).collect();
        let vbmin = 50000;
        let vbmax = 2 * vamax;
        let vb : Vec<usize> = (vbmin..vbmax).collect();
        let inter = vamax - vbmin;
        let jexact = inter as f64 / vbmax as f64;
        let size = 50000;
        //
        test_optdens(&va, &vb, jexact, size);
    } // end of test_optdens_intersection_fnv_f64



    fn test_optdens(va : &Vec<usize>, vb : &Vec<usize>, jexact : f64, size : usize)  {
        //
        let bh = BuildHasherDefault::<FnvHasher>::default();
        let mut sminhash : OptDensMinHash<f64, usize, FnvHasher>= OptDensMinHash::new(size, bh);
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
        let jac = get_jaccard_index_estimate(&ska, &skb).unwrap();
        let sigma = (jexact * (1.- jexact) / size as f64).sqrt();
        log::info!(" jaccard estimate {:.3e}, j exact : {:.3e}, sigma : {:.3e} ", jac, jexact, sigma);
        // we have 10% common values and we sample a sketch of size 50 on 2000 values , we should see intersection
        assert!( jac > 0. && (jac - jexact).abs() < 3. *sigma);
    } // end of test_optdens



    fn test_fastdens(va : &Vec<usize>, vb : &Vec<usize>, jexact : f64, size : usize)  {
        //
        let bh = BuildHasherDefault::<FnvHasher>::default();
        let mut sminhash : RevOptDensMinHash<f64, usize, FnvHasher>= RevOptDensMinHash::new(size, bh);
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
        let jac = get_jaccard_index_estimate(&ska, &skb).unwrap();
        let sigma = (jexact * (1.- jexact) / size as f64).sqrt();
        log::info!(" jaccard estimate {:.3e}, j exact : {:.3e}, sigma : {:.3e} ", jac, jexact, sigma);
        // we have 10% common values and we sample a sketch of size 50 on 2000 values , we should see intersection
        assert!( jac > 0. && (jac - jexact).abs() < 3. *sigma);
    } // end of test_fastdens

} // end of tests