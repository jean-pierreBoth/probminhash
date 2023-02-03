//! Fisher Yates permutation generator

use log::{trace};

use rand::distributions::{Distribution,Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;


// Fisher Yates random permutation generation (sampling without replacement), with lazy generation
/// of an array of size n
pub struct FYshuffle {
    m: usize,
    /// uniform distribution on [0,1)
    unif_01 : Uniform<f64>,
    ///
    v : Vec<usize>,
    ///
    lastidx : usize,
}

impl FYshuffle {
    /// initialize a random permutation generator on a set of size m
    pub fn new(m: usize) -> FYshuffle {
        let v : Vec<usize> = (0..m).map(|x| x).collect();
        FYshuffle{m, unif_01: Uniform::<f64>::new(0., 1.), v, lastidx:m}
    }

    // See https://www.geeksforgeeks.org/generate-a-random-permutation-of-1-to-n/
    // and The algorithm design manual S.Skiena P.458

    /// generates next randomly choosen item of the set (you get sampling without replacement)
    /// After being call m times, it is possible to get the full permutation with the function get_values
    /// as the permutation is fully sampled.
    /// If called more than m times, it calls reset implicitly to generate a new permutation.
    /// It is possible (and recommended) to reset explicitly after m successive call to next method
    pub fn next(&mut self, rng : &mut Xoshiro256PlusPlus) -> usize {
        if self.lastidx >= self.m {
            self.reset();
            log::debug!("FYshuffle next calling reset ");
        }
        let xsi = self.unif_01.sample(rng);
        // sample between self.lastidx (included) and self.m (excluded)
        let idx = self.lastidx + (xsi * (self.m - self.lastidx) as f64) as usize;
        let val = self.v[idx];
        self.v[idx] = self.v[self.lastidx];
        self.v[self.lastidx] = val;
        self.lastidx += 1;
        val
    }

    pub fn reset(&mut self) {
        trace!("resetting shuffle lastidx = {}", self.lastidx);
        self.lastidx = 0;
        for i in 0..self.m {
            self.v[i] = i;
        }
    }

    /// returns the set of permuted index
    pub fn get_values(&self) -> &Vec<usize> {
        &self.v
    }

}  // end of impl FYshuffle

