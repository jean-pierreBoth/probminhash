//! setsketcher implementation
//! implementation of SetSkectch : filling the gap between MinHash and and HyperLogLog <https://arxiv.org/abs/2101.00314>
//! or <https://vldb.org/pvldb/vol14/p2244-ertl.pdf>
//! 


use serde::{Deserialize, Serialize};
use serde_json::{to_writer};

use std::hash::{BuildHasher, BuildHasherDefault, Hasher, Hash};
use std::marker::PhantomData;
use rand::prelude::*;
use rand_distr::{Exp1};
use rand_xoshiro::Xoshiro256PlusPlus;

use num::{Integer, ToPrimitive, FromPrimitive, Bounded, NumCast};

use crate::fyshuffle::*;

#[cfg_attr(doc, katexit::katexit)]
/// Parameters defining the Sketcher
/// - choice of a : given $\epsilon$ a is chosen verifying  $$ a \ge  \frac{1}{\epsilon} * log(\frac{m}{b})  $$ so that the probability of any sketch value being  negative is less than $\epsil$
/// ( lemma 4 of setsketch paper).  
/// 
/// 
/// - choice of q:  if $$ q >=  log_{b} (\frac{m  n  a}{\epsilon})$$ then a sketch value is less than q+1 with proba less than $\epsilon$ up to n data to sketch.
///  see lemma 5 of paper.  
/// 
/// m = 4096, b = 1.001,  a = 20 , q = $2^{16} -2 = 65534 $ guarantee to negative value in sketch with proba 8.28 10^-6 and probability of
/// sketch value greater than q+1 with probability less than $2.93 \space 10^{-6} $
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
    /// m is the number of sketch. b is a parameter in the interval ]1., 2.[, in fact near 1. is better.
    /// a is a parameter to be adjusted to avoid negative values in sketchs and q is related to size in bits 
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

    /// get bounds for J given parameters and first estimate for jaccard. Returns a 2-uple (lower, upper)
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
} // end of impl SetSketchParams




// default parameters ensure capacity to represented a set up to 10^28 elements.
// F is f32 or f64. To spare memory in Hnsw f32 is better.
pub struct SetSketcher<I : Integer, T, H:Hasher+Default> {
    // b must be <= 2
    b : f64,
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
        return SetSketcher::<I,T,H>{b : params.get_b(), m : params.get_m(), a: params.get_a(), q: params.get_q(), 
                    k_vec, lower_k : 0., nbmin : 0, permut_generator :  FYshuffle::new(m),
                    b_hasher: BuildHasherDefault::<H>::default(), t_marker : PhantomData};
    }
}


impl <'a, I, T, H> SetSketcher<I, T, H>
    where   I : Integer + ToPrimitive + FromPrimitive + Bounded + Copy + Clone + std::fmt::Debug,
            T: Hash,
            H: Hasher+Default {


    /// allocate a new sketcher
    pub fn new(params : SetSketchParams) -> Self {
        //
        let k_vec : Vec<I> = (0..params.get_m()).into_iter().map(|_| I::zero()).collect();
        //
        return SetSketcher::<I,T,H>{b : params.get_b(), m : params.get_m(), a: params.get_a(), q: params.get_q(), 
            k_vec, lower_k : 0., nbmin : 0, permut_generator :  FYshuffle::new(params.get_m() as usize),
            b_hasher: BuildHasherDefault::<H>::default(), t_marker : PhantomData};
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
        let lb = (self.b - 1.).ln_1p();  // this is ln(b) for b near 1.
        //
        let mut x_pred : f64 = 0.;
        for j in 0..self.m {
            //
            let x_j = x_pred + (self.a as f64 / (self.m - j) as f64) * rng.sample::<f64, Exp1>(Exp1);  // use Ziggurat
            x_pred = x_j;
            //
            let lb_xj =  x_j.ln()/lb;   // log base b of x_j
            //
            if lb_xj > - self.lower_k  {
                break;
            } 
            //
            let z : u64 = (self.q+1).min((1. - lb_xj).floor() as u64);
            log::debug!("j : {}, x_j : {:.5e} , lb_xj : {:.5e}, z : {:.5e}", j, x_j, lb_xj , z);
            let k= 0.max(z);
            // 
            if k as f64 <= self.lower_k {
                break;
            }
            // now work with permutation sampling
            let i = self.permut_generator.next(&mut rng);
            //
            if k > self.k_vec[i].to_u64().unwrap() {
                log::debug!("setting slot i: {}, f_k : {:.3e}", i, k);
                // we must enforce that f_k fits into I
                if k > imax {
                    log::info!("got a k value over I range : {:.3e}, I::max : {:#}", k, imax);
                }

                self.k_vec[i] = I::from_u64(k).unwrap();
                self.nbmin = self.nbmin + 1;
                if self.nbmin % self.m == 0 {
                    log::debug!("nbmin = {}", self.nbmin);
                    let low = self.k_vec.iter().fold(self.k_vec[0], |min : I, x| if x < &min { *x} else {min});
                    log::debug!("setting low to : {:?}", low);
                    self.lower_k = NumCast::from::<I>(low).unwrap();
                }
            }
        }
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
    } // end of sketch_slice



    // reset state
    pub fn reinit(&mut self) {
        //
        self.permut_generator.reset();
        self.k_vec = (0..self.m).into_iter().map(|_| I::zero()).collect();
        self.lower_k = 0.;
        self.nbmin = 0;
    }  // end of reinit


    /// get sketch
    pub fn get_signature(&self) -> &Vec<I> {
        return &self.k_vec;
    }
} // end of impl SetSketch<F:


//======================================================================================================


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
    fn test_params_bounds() {
        //
        log_init_test();
        //
        let params = SetSketchParams::default();
        log::info!("params default : {:?}", params);
        //
        let nb_frac = 20;

        for j in 0..nb_frac {
            let jac = (j as f64)/ (nb_frac as f64);
            let (jinf, jsup) = params.get_jaccard_bounds(jac);
            log::info!("j = {},  jinf : {:.5e}, jsup = {:.5e}", jac, jinf, jsup);
        }
    } // end of test_params_bounds


    #[test]
    fn test_range_intersection_fnv_f32() {
        //
        log_init_test();
        // we construct 2 ranges [a..b] [c..d], with a<b, b < d, c<d sketch them and compute jaccard.
        // we should get something like max(b,c) - min(b,c)/ (b-a+d-c)
        //
        let va : Vec<usize> = (0..1000).collect();
        let vb : Vec<usize> = (900..2000).collect();
        let inter = 100;  // intersection size
        let jexact = inter as f32 / 2000 as f32;
        //
        let mut params = SetSketchParams::default();
        params.set_m(700);
        let mut sethasher : SetSketcher<u16, usize, FnvHasher>= SetSketcher::new(params);
        // now compute sketches
        let resa = sethasher.sketch_slice(&va);
        if !resa.is_ok() {
            println!("error in sketcing va");
            return;
        }
        let ska = sethasher.get_signature().clone();
        log::debug!("ska = {:?}",ska);
        //
        sethasher.reinit();
        //
        let resb = sethasher.sketch_slice(&vb);
        if !resb.is_ok() {
            println!("error in sketching vb");
            return;
        }
        let skb = sethasher.get_signature();
        log::debug!("skb = {:?}",skb);
        //
        let jac = get_jaccard_index_estimate(&ska, &skb).unwrap();
        let sigma = (jexact * (1. - jexact) / params.get_m() as f32).sqrt();
        log::info!(" jaccard estimate {:.3e}, j exact : {:.3e} , sigma : {:.3e}", jac, jexact, sigma);
        // we have 10% common values and we sample a sketch of size 50 on 2000 values , we should see intersection
        assert!( jac > 0. && (jac as f32) < jexact + 3.* sigma);
    }  // end of test_range_intersection_fnv_f32

}  // end of mod tests