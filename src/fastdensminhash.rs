//! implementation of the paper:
//! On densification for MinWise Hashing
//! Mai, Rao, Kapilevitch, Rossi, Abbasi-Yadkori, Sinha
//! [pmlr-2020](http://proceedings.mlr.press/v115/mai20a/mai20a.pdf)
//! 
//! It provides  locally sensitive sketching of unweighted data with densification



use std::cmp;
use std::hash::{BuildHasher, BuildHasherDefault, Hasher, Hash};
use std::marker::PhantomData;
use rand::prelude::*;
use rand::distributions::*;
use rand_distr::uniform::SampleUniform;
use rand_xoshiro::Xoshiro256PlusPlus;
use wyhash::WyRng;

use num::Float;


pub struct FastDensMinHash<F: Float, T: Hash, H: Hasher+Default> {
    /// size of sketch. sketch values lives in  [0, number of sketches], so a u16 is sufficient
    hsketch:Vec<F>,
    /// stored data giving minima
    values:Vec<Option<T>>,
    /// the Hasher to use if data arrive unhashed. Anyway the data type we sketch must satisfy the trait Hash
    b_hasher: BuildHasherDefault<H>,
    /// just to mark the type we sketch
    t_marker: PhantomData<T>,
}  // end of struct FastDensMinHash


impl <F: Float + SampleUniform + std::fmt::Debug, T:Hash + Copy,  H : Hasher+Default> FastDensMinHash<F, T, H> {
    /// allocate a struct to do superminhash.
    /// size is size of sketch. build_hasher is the build hasher for the type of Hasher we want.
    pub fn new(size:usize, build_hasher: BuildHasherDefault<H>) -> FastDensMinHash<F, T, H> {
        let mut sketch_init = Vec::<F>::with_capacity(size);
        let mut values = Vec::<Option<T>>::with_capacity(size);
        let large:F = F::from(u32::MAX).unwrap();  // is OK even for f32
        for _i in 0..size {
            sketch_init.push(large);
            values.push(None);
        }
        FastDensMinHash{hsketch: sketch_init, values,b_hasher: build_hasher, t_marker : PhantomData,}
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
                    // try to export filled bin
                    let mut hasher = self.b_hasher.build_hasher();
                    self.values[k].as_ref().unwrap().hash(&mut hasher);
                    let hval1 : u64 = hasher.finish();
                    let mut rng2 = WyRng::seed_from_u64(hval1 + k as u64 * pass);
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




    /// optimized version
    pub fn sketch_slice_2(&mut self, to_sketch : &[T]) -> Result <(),()> {

        let m = self.hsketch.len();
        //
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
        // now we run first densification pass as in paper pargaraph 5.
        let mut nb_empty : usize = self.values.iter().map(|x| if x.is_none() { 1} else {0}).sum();
        log::debug!("fastdensminhash::sketch_slice sketch size : {:?},  nb empy slots : {:?}", m, nb_empty);
        if nb_empty == 0 {
            return Ok(());
        }
        //
        for k in 0..m {
            // if self.values[k] is some, generate random value i in 0..m with seed self.values[k]
            if self.values[k].is_some() {
                // try to export filled bin
                let mut hasher = self.b_hasher.build_hasher();
                self.values[k].as_ref().unwrap().hash(&mut hasher);
                let hval1 : u64 = hasher.finish();
                let mut rng2 = WyRng::seed_from_u64(hval1);
                let j: usize = Uniform::<usize>::new(0, m).sample(&mut rng2);
                // is slot j empty?
                if self.values[j].is_none()  {
                    // if self.values[i] is none copy value from k to j
                    self.hsketch[j] = self.hsketch[k];
                    nb_empty = nb_empty - 1;
                    log::debug!("filling empty bin at {}", j);
                    self.values[j] = self.values[k];
                }
            }
        }
        log::debug!("after first densification pass , nb empty : {}", nb_empty);
        // now we run second densification pass as in paper pargaraph 5.
        for k in 0..m {
            // if self.values[k] is some, generate random value i in 0..m with seed self.values[k]
            if self.values[k].is_some() {
                // try to export filled bin
                let mut hasher = self.b_hasher.build_hasher();
                self.values[k].as_ref().unwrap().hash(&mut hasher);
                let hval1 : u64 = hasher.finish();
                let new_hval = hval1.overflowing_mul(2).0;
                let mut rng2 = WyRng::seed_from_u64(new_hval);
                let j: usize = Uniform::<usize>::new(k, k+1).sample(&mut rng2);
                // is slot j empty?
                if self.values[j].is_none() {
                    // if self.values[i] is none copy value from k to j
                    self.hsketch[j] = self.hsketch[k];
                    nb_empty = nb_empty - 1;
                    log::debug!("filling empty bin at {}", j);
                    self.values[j] = self.values[k];
                }
            }
        }
        //
        let nb_empty : usize = self.values.iter().map(|x| if x.is_none() { 1} else {0}).sum();
        assert_eq!(nb_empty, 0);
        //
        return Ok(());
    } // end of sketch


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
    fn test_range_intersection_fnv_f64() {
        log_init_test();
        // we construct 2 ranges [a..b] [c..d], with a<b, b < d, c<d sketch them and compute jaccard.
        // we should get something like max(b,c) - min(b,c)/ (b-a+d-c)
        //
        let va : Vec<usize> = (0..1000).collect();
        let vb : Vec<usize> = (900..2000).collect();
        let inter = 100;
        let jexact = inter as f32 / 2000.;
        let size = 2000;
        //
        let bh = BuildHasherDefault::<FnvHasher>::default();
        let mut sminhash : FastDensMinHash<f64, usize, FnvHasher>= FastDensMinHash::new(size, bh);
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
        let sigma = (jexact * (1.- jexact) / size as f32).sqrt();
        log::info!(" jaccard estimate {:.3e}, j exact : {:.3e}, sigma : {:.3e} ", jac, jexact, sigma);
        // we have 10% common values and we sample a sketch of size 50 on 2000 values , we should see intersection
        assert!( jac > 0. && (jac as f32) < jexact + 3. *sigma);
    } // end of test_range_intersection_fnv_f64
} // end of tests