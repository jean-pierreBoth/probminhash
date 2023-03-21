//! setsketcher implementation
//! implementation of SetSkectch : filling the gap between MinHash and and HyperLogLog <https://arxiv.org/abs/2101.00314>
//! 

#![allow(unused)]

use std::hash::{BuildHasher, BuildHasherDefault, Hasher, Hash};
use std::marker::PhantomData;
use rand::prelude::*;
use rand::distributions::*;
use rand_distr::{Exp1, uniform::SampleUniform};
use rand_xoshiro::Xoshiro256PlusPlus;

use num::{Float, ToPrimitive, NumCast};

use crate::fyshuffle::*;


/// Parameters defining the Sketcher
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
    fn new(b : f64, m : u64, a : f64, q : u64) -> Self {
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
} // end of impl SetSketchParams




// default parameters ensure capacity to represented a set up to 10^28 elements.
pub struct SetSketch<F: Float, T, H:Hasher+Default> {
    // b must be <= 2
    b : f64,
    // size of sketch
    m : u64, 
    // default is 20
    a : f64,
    //
    q : u64,
    // random values,
    k_vec : Vec<F>,
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


impl <F, T, H> Default for SetSketch<F, T, H > 
    where   F: Float + ToPrimitive ,
            H: Hasher+Default {
    /// the default parameters give 4096 sketch with a capacity for counting up to 10^19 elements
    fn default() -> SetSketch<F, T, H> {
        let params = SetSketchParams::default();
        let m : usize = 4096;
        let k_vec : Vec<F> = (0..m).into_iter().map(|_| F::zero()).collect();
        return SetSketch::<F,T,H>{b : params.get_b(), m : params.get_m(), a: params.get_a(), q: params.get_q(), 
                    k_vec, lower_k : 0., nbmin : 0, permut_generator :  FYshuffle::new(m),
                    b_hasher: BuildHasherDefault::<H>::default(), t_marker : PhantomData};
    }
}


impl <'a, F, T, H> SetSketch<F, T, H>
    where   F: Float + SampleUniform + std::fmt::Debug,
            T: Hash,
            H: Hasher+Default {


    // We implement algo sketch1 as we will use it for large number of data and so correlation are expected to be very low.
    /// take into account one more data
    pub fn sketch(&mut self, to_sketch : &T) -> Result <(),()> {
        let mut hasher = self.b_hasher.build_hasher();
        to_sketch.hash(&mut hasher);
        let hval1 : u64 = hasher.finish();
        //
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(hval1);
        self.permut_generator.reset();

        //
        let mut x_pred : f64 = 0.;
        for j in 0..self.m {
            //
            let x_j = x_pred + (self.a as f64 / (self.m - j) as f64) * rng.sample::<f64, Exp1>(Exp1);  // use Ziggurat
            x_pred = x_j;
            //
            let lb_xj =  x_j.log(self.b);   // log base b of x_j
            if lb_xj > - self.lower_k  {
                break;
            } 
            //
            let z : u64 = (self.q+1).min((1. - lb_xj).floor() as u64);
            let k= 0.min(z);
            //
            if k as f64 <= self.lower_k {
                break;
            }
            // now work with permutation sampling
            let i = self.permut_generator.next(&mut rng);
            if F::from(k).unwrap() > self.k_vec[i] {
                self.k_vec[i] = F::from(k).unwrap();
                self.nbmin = self.nbmin + 1;
                if self.nbmin % self.m == 0 {
                    let low = self.k_vec.iter().fold(self.k_vec[0], |min : F, x| if x < &min { *x} else {min});
                    self.lower_k = NumCast::from::<F>(low).unwrap();
                }
            }
        }
        //
        return Ok(());
    }  // end of sketch


    /// get sketch
    pub fn get_hsketch(&self) -> &Vec<F> {
        return &self.k_vec;
    }
} // end of impl SetSketch<F: