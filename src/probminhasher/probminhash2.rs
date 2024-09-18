//! probminhash2 implementation
//! Probminhash2 is statistically equivalent to P-Minhash as described in :
//! Moulton Jiang "Maximally consistent sampling and the Jaccard index of probability distributions"
//!
//! <https://ieeexplore.ieee.org/document/8637426> or <https://arxiv.org/abs/1809.04052>.  
//! It is given as a fallback in case ProbminHash3* algorithms do not perform well, or for comparison.

use log::trace;
use std::fmt::Debug;

use rand::distributions::Distribution;
use rand::prelude::*;
use rand_distr::Exp1;
use rand_xoshiro::Xoshiro256PlusPlus;

use std::collections::HashMap;
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};

use crate::fyshuffle::*;
use crate::maxvaluetrack::*;
use crate::weightedset::*;

/// implementation of the algorithm ProbMinHash2 as described in Ertl paper.  
///
/// D must be convertible injectively into a usize for random generator initialization hence the requirement Hash.  
/// If all data are referred to by an unsigned integer, and weight association is given in a tuple for example if
/// data comes in a Vec<(D,f64)> then D can be replaced by the rank in the Vector, then no hash is need and you can use NoHasher
pub struct ProbMinHash2<D, H>
where
    D: Copy + Eq + Hash + Debug,
    H: Hasher + Default,
{
    m: usize,
    /// initialization object
    initobj: D,
    //
    b_hasher: BuildHasherDefault<H>,
    /// field to keep track of max hashed values
    maxvaluetracker: MaxValueTracker<f64>,
    /// random permutation generator
    permut_generator: FYshuffle,
    //
    betas: Vec<f64>,
    ///  final signature of distribution. allocated to size m
    signature: Vec<D>,
} // end of struct ProbMinHash2

impl<D, H> ProbMinHash2<D, H>
where
    D: Copy + Eq + Hash + Debug,
    H: Hasher + Default,
{
    /// Allocates a ProbMinHash2 structure with nbhash hash functions and initialize signature with initobj (typically 0 for numeric objects)
    pub fn new(nbhash: usize, initobj: D) -> Self {
        let h_signature = (0..nbhash).map(|_| initobj).collect();
        let betas: Vec<f64> = (0..nbhash)
            .map(|x| (nbhash as f64) / (nbhash - x - 1) as f64)
            .collect();
        ProbMinHash2 {
            m: nbhash,
            initobj,
            b_hasher: BuildHasherDefault::<H>::default(),
            maxvaluetracker: MaxValueTracker::new(nbhash),
            permut_generator: FYshuffle::new(nbhash),
            betas,
            signature: h_signature,
        }
    } // end of new

    /// Incrementally adds an item in hash signature. It can be used in streaming.  
    /// It is the building block of the computation, but this method
    /// does not check for unicity of id added in hash computation.  
    /// It is user responsability to enforce that. See method hash_weigthed_hashmap or hash_wset
    pub fn hash_item(&mut self, id: D, weight: f64) {
        assert!(weight > 0.);
        trace!("hash_item : id {:?}  weight {} ", id, weight);
        let winv: f64 = 1. / weight;
        let id_hash: u64 = self.b_hasher.hash_one(&id);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(id_hash);
        self.permut_generator.reset();
        let mut i = 0;
        let x: f64 = Exp1.sample(&mut rng);
        let mut h: f64 = winv * x;
        let mut qmax = self.maxvaluetracker.get_max_value();
        //
        while h < qmax {
            let k = self.permut_generator.next(&mut rng);
            if h < self.maxvaluetracker.get_value(k) {
                self.signature[k] = id;
                //
                self.maxvaluetracker.update(k, h);
                qmax = self.maxvaluetracker.get_max_value();
                if h >= qmax {
                    break;
                }
            }
            let x: f64 = Exp1.sample(&mut rng);
            h += winv * self.betas[i] * x;
            i += 1;
            assert!(i < self.m);
        }
    } // end of hash_item

    /// hash data when given by an iterable WeightedSet
    pub fn hash_wset<T>(&mut self, data: &mut T)
    where
        T: WeightedSet<Object = D> + Iterator<Item = D>,
    {
        while let Some(obj) = &data.next() {
            let weight = data.get_weight(obj);
            self.hash_item(*obj, weight);
        }
    } // end of hash method

    /// computes set signature when set is given as an HashMap with weights corresponding to values.(equivalent to the method with and IndexMap)
    /// This ensures that objects are assigned a weight only once, so that we really have a set of objects with weight associated.  
    /// The raw method hash_item can be used with the constraint that objects are sent ONCE in the hash method.
    pub fn hash_weigthed_hashmap<Hidx>(&mut self, data: &HashMap<D, f64>) {
        let iter = data.iter();
        for (key, weight) in iter {
            trace!(" retrieved key {:?} ", key);
            // we got weight as something convertible to f64
            self.hash_item(*key, *weight);
        }
    } // end of hash_weigthed_hashmap

    /// return final signature.
    pub fn get_signature(&self) -> &Vec<D> {
        &self.signature
    }

    /// reinitialize structure for another hash pass
    pub fn reset(&mut self) {
        self.signature.fill(self.initobj);
        self.maxvaluetracker.reset();
        self.permut_generator.reset();
    } // end of reset
} // end of implementation block for ProbMinHash2

#[cfg(test)]
mod tests {

    use log::*;

    use fnv::FnvHasher;

    use crate::jaccard::*;

    #[allow(dead_code)]
    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    use super::*;

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
            } else {
                wa.push(0.);
            }
        }
        // initialize wb weight 10 above 70
        for i in 0..set_size {
            if i < 50 {
                wb.push(0.);
            } else {
                wb.push((i as f64).powi(4));
            }
        }
        // compute Jp as in
        let mut jp_exact = 0.;
        for i in 0..set_size {
            if wa[i] > 0. && wb[i] > 0. {
                let mut den = 0.;
                for j in 0..set_size {
                    den += (wa[j] / wa[i]).max(wb[j] / wb[i]);
                }
                jp_exact += 1. / den;
            }
        }
        trace!("Jp = {} ", jp_exact);
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
        info!(
            "jp exact = {jp_exact:.3} , jp estimate {jp_estimate:.3} ",
            jp_exact = jp_exact,
            jp_estimate = jp_estimate
        );
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
            } else {
                wa.push(0.);
            }
        }
        // initialize wb weight 10 above 70
        for i in 0..set_size {
            if i < 50 {
                wb.push(0.);
            } else {
                wb.push(1.);
            }
        }
        // compute Jp as in
        let mut jp_exact = 0.;
        for i in 0..set_size {
            if wa[i] > 0. && wb[i] > 0. {
                let mut den = 0.;
                for j in 0..set_size {
                    den += (wa[j] / wa[i]).max(wb[j] / wb[i]);
                }
                jp_exact += 1. / den;
            }
        }
        trace!("Jp = {} ", jp_exact);
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
        info!(
            "jp exact = {jp_exact:.3} , jp estimate {jp_estimate:.3} ",
            jp_exact = jp_exact,
            jp_estimate = jp_estimate
        );
        assert!(jp_estimate > 0.);
    } // end of test_probminhash2_count_intersection_unequal_weights
}
