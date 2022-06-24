//! probminhash2 implementation


use log::{trace};
use std::fmt::{Debug};

use rand::distributions::{Distribution,Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rand::prelude::*;
use rand_distr::Exp1;

use std::hash::{BuildHasher, BuildHasherDefault, Hasher, Hash};

use crate::maxvaluetrack::*;
use crate::weightedset::*;

/// Fisher Yates random permutation generation (sampling without replacement), with lazy generation
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
        FYshuffle{m:m, unif_01: Uniform::<f64>::new(0., 1.), v : v, lastidx:m}
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





/// implementation of the algorithm ProbMinHash2 as described in Ertl paper.  
///
/// D must be convertible injectively into a usize for random generator initialization hence the requirement Hash.  
/// If all data are referred to by an unsigned integer, and weight association is given in a tuple for example if
/// data comes in a Vec<(D,f64)> then D can be replaced by the rank in the Vector, then no hash is need and you can use NoHasher
pub struct ProbMinHash2<D,H> 
            where D:Copy+Eq+Hash+Debug,H:Hasher+Default    {
    m : usize,
    ///
    b_hasher : BuildHasherDefault<H>,
    /// field to keep track of max hashed values
    maxvaluetracker : MaxValueTracker,
    /// random permutation generator
    permut_generator : FYshuffle,
    ///
    betas : Vec<f64>, 
    ///  final signature of distribution. allocated to size m
    signature : Vec<D>,
} // end of struct ProbMinHash2




impl <D,H> ProbMinHash2<D,H>
        where D:Copy+Eq+Hash+Debug,H:Hasher+Default {

    /// Allocates a ProbMinHash2 structure with nbhash hash functions and initialize signature with initobj (typically 0 for numeric objects)
    pub fn new(nbhash:usize, initobj:D) -> Self {
        let h_signature = (0..nbhash).map( |_| initobj).collect();
        let betas : Vec<f64> = (0..nbhash).map(| x | (nbhash as f64)/ (nbhash - x - 1) as f64).collect();
        ProbMinHash2{ m:nbhash, 
                    b_hasher :  BuildHasherDefault::<H>::default(),
                    maxvaluetracker: MaxValueTracker::new(nbhash as usize), 
                    permut_generator : FYshuffle::new(nbhash),
                    betas : betas,
                    signature:h_signature}
    } // end of new


    /// Incrementally adds an item in hash signature. It can be used in streaming.  
    /// It is the building block of the computation, but this method 
    /// does not check for unicity of id added in hash computation.  
    /// It is user responsability to enforce that. See method hash_wset
    pub fn hash_item(&mut self, id:D, weight:f64) {
        assert!(weight > 0.);
        trace!("hash_item : id {:?}  weight {} ", id, weight);
        let winv : f64 = 1./weight;
        let mut hasher = self.b_hasher.build_hasher();
        id.hash(&mut hasher);
        let id_hash : u64 = hasher.finish();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(id_hash);
        self.permut_generator.reset();
        let mut i = 0;
        let x : f64 = Exp1.sample(&mut rng);
        let mut h : f64 = winv * x;
        let mut qmax = self.maxvaluetracker.get_max_value();
        //
        while h < qmax {
            let k = self.permut_generator.next(&mut rng);
            if h <  self.maxvaluetracker.get_value(k) {
                self.signature[k] = id;
                // 
                self.maxvaluetracker.update(k, h);
                qmax = self.maxvaluetracker.get_max_value();
                if h >= qmax { break;}
            }
            let x : f64 = Exp1.sample(&mut rng);
            h = h + winv * self.betas[i] * x;
            i = i+1;
            assert!(i < self.m);
        }
    }  // end of hash_item 


    /// hash data when given by an iterable WeightedSet
    pub fn hash_wset<T>(&mut self, data: &mut T)
    where T: WeightedSet<Object=D> + Iterator<Item=D> {
        while let Some(obj) = &data.next() {
            let weight = data.get_weight(&obj);
            self.hash_item(*obj, weight);
        }
    } // end of hash method

    /// return final signature.
    pub fn get_signature(&self) -> &Vec<D> {
        return &self.signature;
    }
}  // end of implementation block for ProbMinHash2


#[cfg(test)]
mod tests {

use log::*;

use fnv::{FnvHasher};


use crate::jaccard::*;

#[allow(dead_code)]
fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}

use super::*;



#[test]
// We check we have a unifom distribution of values at each rank of v
// variance is 5/4
fn test_fyshuffle() {

    log_init_test();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(45678 as u64);
    let m = 4;
    let mut fypermut = FYshuffle::new(m);
    let nb_permut = 500000;
    let mut freq : Vec<usize> = (0..m).map(|_| 0).collect();

    for _ in 0..nb_permut {
        for _ in 0..m {
            fypermut.next(&mut rng);
        }
        let v = fypermut.get_values();
        for k in 0..v.len() {
            freq[k] += v[k];
        }
        fypermut.reset();
    }

    let th_freq = 1.5;
    let th_var = 5./4.;
    let sigma = (th_var/ (nb_permut as f64)).sqrt();
    for i in 0..freq.len() {
        let rel_error = ((freq[i] as f64)/ (nb_permut as f64) - th_freq)/ sigma;
        info!(" slot i {} , rel error = {}", i, rel_error);
        assert!( rel_error.abs() < 3.)
    }
    info!("  freq = {:?}", freq);
    for _ in 0..15 {
        for _ in 0..m {
            fypermut.next(&mut rng);
        }
        println!("permut state {:?} ", fypermut.get_values());
        fypermut.reset();
    }
} // end of test_fyshuffle


#[test] 
// This test checks JaccardProbability with unequal weights inside sets
fn test_probminhash2_count_intersection_unequal_weights() {
    //
    log_init_test();
    //
    println!("test_probminhash2_count_intersection_unequal_weights");
    debug!("test_probminhash2_count_intersection_unequal_weights");
    // we construct 2 ranges [a..b] [c..d], with a<b, b < d, c<d sketch them and compute jaccard.
    // we should get something like max(b,c) - min(b,c)/ (b-a+d-c)
    //
    let set_size = 100;
    let nbhash = 50;
    //
    // choose weights for va and vb elements
    let mut wa = Vec::<f64>::with_capacity(set_size);
    let mut wb = Vec::<f64>::with_capacity(set_size);
    // initialize wa, weight 20 up to 130
    for i in 0..set_size {
        if i < 70 {
            wa.push(2. * i as f64);
        }
        else {
            wa.push(0.);
        }
    }
    // initialize wb weight 10 above 70
    for i in 0..set_size {
        if i < 50 {
            wb.push(0.);
        }
        else {
            wb.push( (i as f64).powi(4));
        }
    }        
    // compute Jp as in 
    let mut jp_exact = 0.;
    for i in 0..set_size {
        if wa[i] > 0. && wb[i] > 0. {
            let mut den = 0.;
            for j in 0..set_size {
                den += (wa[j]/wa[i]).max(wb[j]/wb[i]);
            }
            jp_exact += 1./den;
        }
    }
    trace!("Jp = {} ",jp_exact);
    // probminhash 
    trace!("\n\n hashing wa");
    let mut waprobhash = ProbMinHash2::<usize, FnvHasher>::new(nbhash, 0);
    for i in 0..set_size {
        if wa[i] > 0. {
            waprobhash.hash_item(i, wa[i]);
        }
    }
    // waprobhash.maxvaluetracker.dump();
    //
    trace!("\n\n hashing wb");
    let mut wbprobhash = ProbMinHash2::<usize, FnvHasher>::new(nbhash, 0);
    for i in 0..set_size {
        if wb[i] > 0. {
            wbprobhash.hash_item(i, wb[i]);
        }
    }        
    let siga = waprobhash.get_signature();
    let sigb = wbprobhash.get_signature();
    let jp_estimate = compute_probminhash_jaccard(siga, sigb);
    //
//    waprobhash.maxvaluetracker.dump();
//    wbprobhash.maxvaluetracker.dump();
//    println!("siga :  {:?}", siga);
//    println!("sigb :  {:?}", sigb);
    //
    info!("jp exact = {jp_exact:.3} , jp estimate {jp_estimate:.3} ", jp_exact=jp_exact, jp_estimate=jp_estimate);
    assert!(jp_estimate > 0.);
} // end of test_probminhash2_count_intersection_unequal_weights


#[test] 
fn test_probminhash2_count_intersection_equal_weights() {
    //
    log_init_test();
    //
    println!("test_probminhash2_count_intersection_equal_weights");
    debug!("test_probminhash2_count_intersection_equal_weights");
    // we construct 2 ranges [a..b] [c..d], with a<b, b < d, c<d sketch them and compute jaccard.
    // we should get something like max(b,c) - min(b,c)/ (b-a+d-c)
    //
    let set_size = 100;
    let nbhash = 50;
    //
    // choose weights for va and vb elements
    let mut wa = Vec::<f64>::with_capacity(set_size);
    let mut wb = Vec::<f64>::with_capacity(set_size);
    // initialize wa, weight 20 up to 130
    for i in 0..set_size {
        if i < 70 {
            wa.push(1.);
        }
        else {
            wa.push(0.);
        }
    }
    // initialize wb weight 10 above 70
    for i in 0..set_size {
        if i < 50 {
            wb.push(0.);
        }
        else {
            wb.push(1.);
        }
    }        
    // compute Jp as in 
    let mut jp_exact = 0.;
    for i in 0..set_size {
        if wa[i] > 0. && wb[i] > 0. {
            let mut den = 0.;
            for j in 0..set_size {
                den += (wa[j]/wa[i]).max(wb[j]/wb[i]);
            }
            jp_exact += 1./den;
        }
    }
    trace!("Jp = {} ",jp_exact);
    // probminhash 
    trace!("\n\n hashing wa");
    let mut waprobhash = ProbMinHash2::<usize, FnvHasher>::new(nbhash, 0);
    for i in 0..set_size {
        if wa[i] > 0. {
            waprobhash.hash_item(i, wa[i]);
        }
    }
    // waprobhash.maxvaluetracker.dump();
    //
    trace!("\n\n hashing wb");
    let mut wbprobhash = ProbMinHash2::<usize, FnvHasher>::new(nbhash, 0);
    for i in 0..set_size {
        if wb[i] > 0. {
            wbprobhash.hash_item(i, wb[i]);
        }
    }        
    let siga = waprobhash.get_signature();
    let sigb = wbprobhash.get_signature();
    let jp_estimate = compute_probminhash_jaccard(siga, sigb);
    //
//    waprobhash.maxvaluetracker.dump();
//    wbprobhash.maxvaluetracker.dump();
    //
    info!("jp exact = {jp_exact:.3} , jp estimate {jp_estimate:.3} ", jp_exact=jp_exact, jp_estimate=jp_estimate);
    assert!(jp_estimate > 0.);
} // end of test_probminhash2_count_intersection_unequal_weights



}