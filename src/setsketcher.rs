
//! implementation of the paper :
//! *SetSkectch : filling the gap between MinHash and HyperLogLog*  
//! See  <https://arxiv.org/abs/2101.00314> or <https://vldb.org/pvldb/vol14/p2244-ertl.pdf>.
//! 
//! We implement Setsketch1 algorithm which supposes that the size of the data set
//! to sketch is large compared to the size of sketch.
//! The purpose of this implementation is to provide Local Sensitive sketching of a set
//! adapted to the Jaccard distance with some precaution, see function [get_jaccard_bounds](SetSketchParams::get_jaccard_bounds).  
//! 
//! The cardinal of the set can be estimated with the basic (unoptimized) function [get_cardinal_stats](SetSketcher::get_cardinal_stats)


use serde::{Deserialize, Serialize};
use serde_json::{to_writer};

use std::fs::OpenOptions;
use std::path::{Path};
use std::io::{BufReader, BufWriter };


use std::hash::{BuildHasher, BuildHasherDefault, Hasher, Hash};
use std::marker::PhantomData;
use rand::prelude::*;
use rand_distr::{Exp1};
use rand_xoshiro::Xoshiro256PlusPlus;

use num::{Integer, ToPrimitive, FromPrimitive, Bounded};

use crate::fyshuffle::*;

#[cfg_attr(doc, katexit::katexit)]
/// Parameters defining the Sketcher
/// - choice of a : given $\epsilon$ a is chosen verifying  $$ a \ge  \frac{1}{\epsilon} * log(\frac{m}{b})  $$ so that the probability 
///   of any sketch value being  negative is less than $\epsilon$
/// *(lemma 4 of setsketch paper)*.  
/// 
/// 
/// - choice of q:  if $$ q >=  log_{b} (\frac{m  n  a}{\epsilon})$$ then a sketch value is less than q+1 with proba less than $\epsilon$ up to n data to sketch.
///  *(see lemma 5 of paper)*.  
/// 
/// The default initialization corresponds to $m = 4096, b = 1.001,  a = 20 , q = 2^{16} -2 = 65534$ and guarantees the absence of negative value in sketch with proba $8.28 \space 10^{-6}$ and probability of
/// sketch value greater than q+1 with probability less than $2.93 \space 10^{-6}$.  
/// With low probability truncature the sketches can thus be represented by a u16 vector.
/// 
///    
/// 
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct SetSketchParams {
    // b must be <= 2
    b : f64,
    // size of sketch
    m : u64, 
    // default is 20.
    a : f64,
    //
    q : u64,
}

impl Default for SetSketchParams {
    fn default() -> Self {
        SetSketchParams{b : 1.001, m : 4096, a : 20., q : 2_u64.pow(16) - 2}
    }
} // end of SetSketchParams


impl SetSketchParams {
    #[cfg_attr(doc, katexit::katexit)]
    /// - m is the number of sketch. 
    /// - b is a parameter in the interval ]1., 2.[, in fact near 1. is better.
    /// - a is a parameter to be adjusted to reduce probability $\epsilon$ of having negative values in sketchs.
    /// - q minimal size in bits of sketch values. related to m,a, and  $\epsilon$ 
    pub fn new(b : f64, m : u64, a : f64, q : u64) -> Self {
        SetSketchParams{b, m, a, q}
    }

    ///
    pub fn get_a(&self) -> f64 { self.a}

    /// 
    pub fn get_b(&self) -> f64 { self.b} 

    ///
    pub fn get_q(&self) -> u64 { self.q}

    ///
    pub fn get_m(&self) -> u64 { self.m}

    ///
    pub fn set_m(&mut self, nb_sketch : usize) {
        self.m = nb_sketch as u64;
    }

    /// get bounds for J given parameters and first estimate for jaccard.    
    /// Returns a 2-uple (lower, upper).    
    /// For the parameters choice the difference between lower and upper bound should be smaller than accepted error over the range of jaccard
    /// needed by problem treated.  
    /// For the default parameters the difference between lower and upper is less 0.5% of jaccard value.
    pub fn get_jaccard_bounds(&self, jac : f64) -> (f64, f64) {
        assert!( jac <= 1.);
        // we want to compute b^(D0 / 2m) with article notation.
        // for us jac = D_0/m
        //  let b_aux = self.b.powf(D_0/ (2. * self.m as f64)); //   b^(jac/2m)
        let b_aux= self.b.powf(jac*0.5);        
        let jsup = (b_aux * b_aux - 1.) / ( self.b - 1.);
        //
        let b_inf = 2. * (b_aux * self.b.sqrt() - 1.) / (self.b - 1.) - 1.;
        let jinf = b_inf.max(0.);
        //
        log::debug!("b_inf : {:.5e}, b_aux : {:.3e}", b_inf, b_aux);
        //
        assert!(jinf <= jsup);
        //
        return (jinf, jsup);
    }


    pub fn dump_json(&self, dirpath: &Path) ->  Result<(), String> {
        //
        let filepath = dirpath.join("parameters.json");
        //
        log::info!("dumping SetSketchParams in json file : {:?}", filepath);
        //
        let fileres = OpenOptions::new().write(true).create(true).truncate(true).open(&filepath);
        if fileres.is_err() {
            log::error!("SetSketchParams dump : dump could not open file {:?}", filepath.as_os_str());
            println!("SetSketchParams dump: could not open file {:?}", filepath.as_os_str());
            return Err("SetSketchParams dump failed".to_string());
        }
        // 
        let mut writer = BufWriter::new(fileres.unwrap());
        let _ = to_writer(&mut writer, &self).unwrap();
        //
        Ok(())
    } // end of dump_json



    /// reload from a json dump. Used in request module to ensure coherence with database constitution
    pub fn reload_json(dirpath : &Path) -> Result<Self, String> {
        log::info!("in reload_json");
        //
        let filepath = dirpath.join("parameters.json");
        let fileres = OpenOptions::new().read(true).open(&filepath);
        if fileres.is_err() {
            log::error!("SetSketchParams reload_json : reload could not open file {:?}", filepath.as_os_str());
            println!("SetSketchParams reload_json: could not open file {:?}", filepath.as_os_str());
            return Err("SetSketchParams reload_json could not open file".to_string());            
        }
        //
        let loadfile = fileres.unwrap();
        let reader = BufReader::new(loadfile);
        let hll_parameters:Self = serde_json::from_reader(reader).unwrap();
        //
        Ok(hll_parameters)
    } // end of reload_json


} // end of impl SetSketchParams



/// This structure implements Setsketch1 algorithm which suppose that the size of
/// on which the algorithm runs is large compared to the size of sketch, see function [SetSketcher::get_nb_overflow].
///   
/// The default parameters ensure capacity to represent a set up to 10^28 elements.
/// I is an integer u16, u32. u16 should be sufficient for most cases (see [SetSketchParams])
pub struct SetSketcher<I : Integer, T, H:Hasher+Default> {
    // b must be <= 2. In fact we use lnb (precomputed log of b)
    _b : f64,
    // size of sketch
    m : u64, 
    // default is 20
    a : f64,
    //
    q : u64,
    // random values,
    k_vec : Vec<I>,
    // minimum of values stored in vec_k
    lower_k : f64,
    //
    nbmin : u64, 
    //
    permut_generator : FYshuffle,
    //
    nb_overflow : u64,
    // we store ln(b)
    lnb : f64,
    /// the Hasher to use if data arrive unhashed. Anyway the data type we sketch must satisfy the trait Hash
    b_hasher: BuildHasherDefault<H>,
    /// just to mark the type we sketch
    t_marker: PhantomData<T>,
}


impl <I, T, H> Default for SetSketcher<I, T, H > 
    where   I : Integer + Bounded + ToPrimitive + FromPrimitive + Copy + Clone,
            H: Hasher+Default {
    /// the default parameters give 4096 sketch with a capacity for counting up to 10^19 elements
    fn default() -> SetSketcher<I, T, H> {
        let params = SetSketchParams::default();
        let m : usize = 4096;
        let k_vec : Vec<I> = (0..m).into_iter().map(|_| I::zero()).collect();
        let lnb = (params.get_b() - 1.).ln_1p();  // this is ln(b) for b near 1.
        return SetSketcher::<I,T,H>{_b : params.get_b(), m : params.get_m(), a: params.get_a(), q: params.get_q(), 
                    k_vec, lower_k : 0., nbmin : 0, permut_generator :  FYshuffle::new(m), nb_overflow : 0, lnb,
                    b_hasher: BuildHasherDefault::<H>::default(), t_marker : PhantomData};
    }
}


impl <'a, I, T, H> SetSketcher<I, T, H>
    where   I : Integer + ToPrimitive + FromPrimitive + Bounded + Copy + Clone + std::fmt::Debug,
            T: Hash,
            H: Hasher+Default {


    /// allocate a new sketcher
    pub fn new(params : SetSketchParams, b_hasher: BuildHasherDefault::<H>) -> Self {
        //
        let k_vec : Vec<I> = (0..params.get_m()).into_iter().map(|_| I::zero()).collect();
        let lnb = (params.get_b() - 1.).ln_1p();  // this is ln(b) for b near 1.
        //
        return SetSketcher::<I,T,H>{_b : params.get_b(), m : params.get_m(), a: params.get_a(), q: params.get_q(), 
            k_vec, lower_k : 0., nbmin : 0, permut_generator :  FYshuffle::new(params.get_m() as usize), nb_overflow : 0, lnb,
            b_hasher, t_marker : PhantomData};
    }


    // We implement algo sketch1 as we will use it for large number of data and so correlation are expected to be very low.
    /// take into account one more data
    pub fn sketch(&mut self, to_sketch : &T) -> Result <(),()> {
        //
        let mut hasher = self.b_hasher.build_hasher();
        to_sketch.hash(&mut hasher);
        let hval1 : u64 = hasher.finish();
        //
        let imax : u64 = I::max_value().to_u64().unwrap();  // max value of I as a u64
        //
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(hval1);
        self.permut_generator.reset();
        //
        let iq1 : i64 = self.q as i64 + 1;
        let inva : f64 = 1. / self.a;
        //
        let mut x_pred : f64 = 0.;
        for j in 0..self.m {
            //
            let x_j = x_pred + (inva / (self.m - j) as f64) * rng.sample::<f64, Exp1>(Exp1);  // use Ziggurat
            x_pred = x_j;
            //
            let lb_xj =  x_j.ln()/self.lnb;   // log base b of x_j
            //
            if lb_xj > - self.lower_k  {
                break;
            } 
            //
            let z : i64 = iq1.min((1. - lb_xj).floor() as i64);
            log::trace!("j : {}, x_j : {:.5e} , lb_xj : {:.5e}, z : {:.5e}", j, x_j, lb_xj , z);
            let k= 0.max(z) as u64;
            // 
            if k as f64 <= self.lower_k {
                break;
            }
            // now work with permutation sampling
            let i = self.permut_generator.next(&mut rng);
            //
            if k > self.k_vec[i].to_u64().unwrap() {
                log::trace!("setting slot i: {}, f_k : {:.3e}", i, k);
                // we must enforce that f_k fits into I
                if k > imax {
                    self.nb_overflow += 1;
                    self.k_vec[i] = I::from_u64(imax).unwrap();
                    log::warn!("I overflow , got a k value {:.3e} over I::max : {:#}", k, imax);
                }
                else {
                    self.k_vec[i] = I::from_u64(k).unwrap();
                }
                self.nbmin = self.nbmin + 1;
                if self.nbmin % self.m == 0 {
                    let flow = self.k_vec.iter().fold(self.k_vec[0], |min : I, x| if x < &min { *x} else {min}).to_f64().unwrap();
                    if flow > self.lower_k {
                        // no register can decrease so self.lower_k must not decrease
                        log::debug!("j : {}, nbmin = {} , setting low to : {:?}", j, self.nbmin, flow);
                        self.lower_k = flow;
                    }
                }
            }
        }
        //
        return Ok(());
    }  // end of sketch

    /// returns the lowest value of sketch
    /// a null value is a diagnostic of bad skecthing. As the algorithm suppose
    /// that the size of the set sketched is large compared to the num ber of sketch this
    /// has a very low probability.
    pub fn get_low_sketch(&self) -> i64 {
        return self.lower_k.floor() as i64;
    }

    /// returns the number of time value sketcher overflowed the number of bits allocated
    /// should be less than number of values sketched / 100_000 if parameters are well chosen. 
    pub fn get_nb_overflow(&self) -> u64 {
        return self.nb_overflow;
    }

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
    } // end of sketch_slice


    /// The function returns a 2-uple with first field cardinal estimator and second field the **relative standard deviation**.  
    /// 
    /// It is a relatively cpu costly function (the computed logs are not cached in the SetSketcher structure) that involves log and exp calls on the whole sketch vector.
    pub fn get_cardinal_stats(&self) -> (f64, f64) {
        let sumbk = self.k_vec.iter().fold(0.0f64, | acc : f64, c| acc + (- c.to_f64().unwrap() * (self._b -1.).ln_1p()).exp());
        let cardinality : f64 = self.m as f64 * (1. - 1./ self._b) / ( self.a as f64 * self.lnb * sumbk);
        //
        let rel_std_dev = ((self._b + 1.) / (self._b - 1.) * self.lnb - 1.) / self.m as f64;
        let rel_std_dev = rel_std_dev.sqrt();
        return (cardinality, rel_std_dev);
    }

    // reset state
    pub fn reinit(&mut self) {
        //
        self.permut_generator.reset();
        self.k_vec = (0..self.m).into_iter().map(|_| I::zero()).collect();
        self.lower_k = 0.;
        self.nbmin = 0;
        self.nb_overflow = 0;
    }  // end of reinit


    /// get signature sketch. Same as get_hsketch
    pub fn get_signature(&self) -> &Vec<I> {
        return &self.k_vec;
    }

    // in fact I prefer get_signature
#[inline(always)]
    pub fn get_hsketch(&self) -> &Vec<I> {
        self.get_signature()
    }
} // end of impl SetSketch<F:



//======================================================================================================


#[cfg(test)]
mod tests {

    use super::*;
    use fnv::FnvHasher;
    use crate::jaccard::*;
    use rand::distributions::{Uniform};

    #[allow(dead_code)]
    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_params_bounds() {
        //
        log_init_test();
        //
        let params = SetSketchParams::default();
        log::info!("params default : {:?}", params);
        //
        let nb_frac = 20;

        for j in 1..=nb_frac {
            let jac = (j as f64)/ (nb_frac as f64);
            let (jinf, jsup) = params.get_jaccard_bounds(jac);
            let delta = 100. * (jsup - jinf) / jac;
            log::info!("j = {},  jinf : {:.5e}, jsup = {:.5e}, delta% : {:.3}", jac, jinf, jsup, delta);
        }
    } // end of test_params_bounds


    #[test]
    fn test_range_inter1_hll_fnv_f32() {
        //
        log_init_test();
        // we construct 2 ranges [a..b] [c..d], with a<b, b < d, c<d sketch them and compute jaccard.
        // we should get something like max(b,c) - min(b,c)/ (b-a+d-c)
        //
        let va : Vec<usize> = (0..1000).collect();
        let vb : Vec<usize> = (900..2000).collect();
        let inter = 100;  // intersection size
        let jexact = inter as f32 / 2000 as f32;
        let nb_sketch = 2000;
        //
        let mut params = SetSketchParams::default();
        params.set_m(nb_sketch);
        let mut sethasher : SetSketcher<u16, usize, FnvHasher>= SetSketcher::new(params, BuildHasherDefault::<FnvHasher>::default());
        // now compute sketches
        let resa = sethasher.sketch_slice(&va);
        if !resa.is_ok() {
            println!("error in sketcing va");
            return;
        }
        let ska = sethasher.get_signature().clone();
        //
        sethasher.reinit();
        //
        let resb = sethasher.sketch_slice(&vb);
        if !resb.is_ok() {
            println!("error in sketching vb");
            return;
        }
        let skb = sethasher.get_signature();
        //
        log::debug!("ska = {:?}",ska);
        log::debug!("skb = {:?}",skb);
        //
        let jac = get_jaccard_index_estimate(&ska, &skb).unwrap();
        let sigma = (jexact * (1. - jexact) / params.get_m() as f32).sqrt();
        log::info!(" jaccard estimate {:.3e}, j exact : {:.3e} , sigma : {:.3e}", jac, jexact, sigma);
        // we have 10% common values and we sample a sketch of size 50 on 2000 values , we should see intersection
        assert!( jac > 0. && (jac as f32) < jexact + 3.* sigma);
    }  // end of test_range_intersection_fnv_f32



    // a test with very different size of slices
    #[test]
    fn test_range_inter2_hll_fnv_f32() {
        //
        log_init_test();
        // we construct 2 ranges [a..b] [c..d], with a<b, b < d, c<d sketch them and compute jaccard.
        // we should get something like max(b,c) - min(b,c)/ (b-a+d-c)
        //
        let vbmax = 20000;
        let va : Vec<usize> = (0..1000).collect();
        let vb : Vec<usize> = (900..vbmax).collect();
        let inter = 100;  // intersection size
        let jexact = inter as f32 / vbmax as f32;
        let nb_sketch = 800;
        //
        let mut params = SetSketchParams::default();
        params.set_m(nb_sketch);
        let mut sethasher : SetSketcher<u16, usize, FnvHasher>= SetSketcher::new(params, BuildHasherDefault::<FnvHasher>::default());
        // now compute sketches
        let resa = sethasher.sketch_slice(&va);
        if !resa.is_ok() {
            println!("error in sketcing va");
            return;
        }
        let low_sketch = sethasher.get_low_sketch();
        log::info!("lowest sketch : {}", low_sketch);
        assert!(low_sketch > 0);
        let cardinal = sethasher.get_cardinal_stats();
        log::info!("cardinal of set a : {:.3e} relative stddev : {:.3e}", cardinal.0, cardinal.1);

        let ska = sethasher.get_signature().clone();
        //
        sethasher.reinit();
        //
        let resb = sethasher.sketch_slice(&vb);
        if !resb.is_ok() {
            println!("error in sketching vb");
            return;
        }
        let skb = sethasher.get_signature();
        let cardinal = sethasher.get_cardinal_stats();
        log::info!("cardinal of set b : {:.3e} relative stddev : {:.3e}", cardinal.0, cardinal.1);
        //
        log::debug!("ska = {:?}",ska);
        log::debug!("skb = {:?}",skb);
        //
        let jac = get_jaccard_index_estimate(&ska, &skb).unwrap();
        let sigma = (jexact * (1. - jexact) / params.get_m() as f32).sqrt();
        log::info!(" jaccard estimate {:.3e}, j exact : {:.3e} , sigma : {:.3e}", jac, jexact, sigma);
        // we have 10% common values and we sample a sketch of size 50 on 2000 values , we should see intersection
        assert!( jac > 0. && (jac as f32) < jexact + 3.* sigma);
    }  // end of test_range_intersection_fnv_f32


    #[test]
    fn test_hll_card_with_repetition() {
        //
        log_init_test();
        //
        let vamax = 200;
        let nb_sketch = 5000;
        //
        let mut params = SetSketchParams::default();
        params.set_m(nb_sketch);
        let mut sethasher : SetSketcher<u16, usize, FnvHasher>= SetSketcher::new(params, BuildHasherDefault::<FnvHasher>::default());
        let unif = Uniform::<usize>::new(0, vamax);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(45679 as u64);

        for _ in 0..nb_sketch {
            sethasher.sketch(&unif.sample(&mut rng)).unwrap();
        }
        let cardinal = sethasher.get_cardinal_stats();
        log::info!("cardinal of set b : {:.3e} relative stddev : {:.3e}", cardinal.0, cardinal.1);
    } // end of test_hll_card_with_repetition

}  // end of mod tests