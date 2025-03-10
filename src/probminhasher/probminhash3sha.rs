//! Implementation of ProbMinHash3a as described in O. Ertl  
//! <https://arxiv.org/abs/1911.00675>
//!
//! This module is similar to probminhash3a but dedicated to hashing some types that do not implement Copy!
//!
//! Contrary to the module probminhash3a the hash is based on sha2 hash functions so
//! the generic type D must satisfy the trait probminhash::probminhasher::sig::Sig
//! which implies some cost but counted objects do not need the Copy trait
//!
//! The hash value is computed on 256 bits and the random generator is initilized with a full 256 bits value
//! reducing collisions.
//! This implementation uses Sha512_256 hashing for initialization the random generator (Xoshiro256PlusPlus) with 256 bits seed and
//! reduces the risk of collisions.  
//!
//! Counted objects must satisfy the trait Sig (instead of **Hash** for the preceding algorithms),
//! so that it can be fed into Sha update methods and thus do not need to satisfy Copy.
//!
//! It is more adapted to hashing Strings or Vec\<u8\>  
//!
//! If this is not a requirement the Probminhash3 module is a solution.
//!  

#[allow(unused_imports)]
use log::{debug, trace};

use std::fmt::Debug;

use num;

use rand::distr::{Distribution, Uniform};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;

use sha2::{Digest, Sha512_256};

use indexmap::IndexMap;
use std::collections::HashMap;

use super::sig::Sig;
use crate::exp01::*;
use crate::maxvaluetrack::*;

/// Another implementation of the algorithm ProbMinHash3a as described in Etrl.  
///
/// This version of ProbMinHash3a is faster than ProbminHash3 but needs some more memory as it stores some states
/// between 2 passes on data. Probminhash3aSha needs at least 2 hash values to run.  
///
/// Contrary to ProbMinHash3a the type D of counted objects must satisfay D:AsRef<\[u8\]>,
pub struct ProbMinHash3aSha<D>
where
    D: Clone + Eq + Debug + Sig,
{
    m: usize,
    /// field to keep track of max hashed values
    maxvaluetracker: MaxValueTracker<f64>,
    /// a exponential law restricted to interval [0., 1)
    exp01: ExpRestricted01,
    /// Buffer to store object to be processed in second pass. Stores (object, inverse weight, generator)
    to_be_processed: Vec<(D, f64, Xoshiro256PlusPlus)>,
    ///  final signature of distribution. allocated to size m
    signature: Vec<D>,
} // end of struct ProbMinHash3a

impl<D> ProbMinHash3aSha<D>
where
    D: Clone + Eq + Debug + Sig,
{
    /// Allocates a new ProbMinHash3a structure with nbhash >= 2 functions and initial object initobj to fill signature.  
    /// The precision on the final estimation depends on the number of hash functions.   
    /// The initial object can be any object , typically 0 for numerical objects.
    pub fn new(nbhash: usize, initobj: D) -> Self {
        assert!(nbhash >= 2);
        let lambda = ((nbhash as f64) / ((nbhash - 1) as f64)).ln();
        let h_signature = (0..nbhash).map(|_| initobj.clone()).collect();
        ProbMinHash3aSha {
            m: nbhash,
            maxvaluetracker: MaxValueTracker::new(nbhash),
            exp01: ExpRestricted01::new(lambda),
            to_be_processed: Vec::<(D, f64, Xoshiro256PlusPlus)>::new(),
            signature: h_signature,
        }
    } // end of new

    /// It is the entry point of this hash algorithm.
    /// The indexmap gives multiplicity (or weight of type F) to the objects hashed of type D.
    /// The weight be positive and be convertible to a f64 without overflow (so some unsigned int)
    pub fn hash_weigthed_idxmap<Hidx, F>(&mut self, data: &IndexMap<D, F, Hidx>)
    where
        Hidx: std::hash::BuildHasher,
        F: num::ToPrimitive + std::fmt::Display,
    {
        //
        let unif0m = Uniform::<usize>::new(0, self.m).unwrap();
        let mut qmax: f64 = self.maxvaluetracker.get_max_value();

        let iter = data.iter();
        for (key, weight_t) in iter {
            trace!("hash_item : id {:?}  weight {} ", key, weight_t);
            let weight = weight_t.to_f64().unwrap();
            assert!(
                weight.is_finite() && weight >= 0.,
                "conversion to f64 failed"
            );
            let winv = 1. / weight;
            // rebuild a new hasher at each id for reproductibility
            let mut hasher = Sha512_256::new();
            // write input message
            hasher.update(&key.get_sig());
            // read hash digest and consume hasher
            let new_hash = hasher.finalize();
            let hashed_slice = new_hash.as_slice();
            assert_eq!(hashed_slice.len(), 32);
            let mut seed: [u8; 32] = [0; 32];
            seed.copy_from_slice(&hashed_slice[..32]);
            let mut rng = Xoshiro256PlusPlus::from_seed(seed);
            let h = winv * self.exp01.sample(&mut rng);
            qmax = self.maxvaluetracker.get_max_value();

            if h < qmax {
                let k = unif0m.sample(&mut rng);
                assert!(k < self.m);
                if h < self.maxvaluetracker.get_value(k) {
                    self.signature[k] = key.clone();
                    //
                    self.maxvaluetracker.update(k, h);
                    qmax = self.maxvaluetracker.get_max_value();
                }
                if winv < qmax {
                    // we store point for further processing in second step, if inequality is not verified
                    // it cannot be added anymore anywhere.
                    self.to_be_processed.push((key.clone(), winv, rng));
                }
            } // end if h < qmax
        } // end initial loop
          //
          // now we have second step
          //
        let mut i = 2; // by comparison to ProbMinHash3 we are not at i = 2 !!
                       //
        while !self.to_be_processed.is_empty() {
            let mut insert_pos = 0;
            trace!(
                " i : {:?}  , nb to process : {}",
                i,
                self.to_be_processed.len()
            );
            for j in 0..self.to_be_processed.len() {
                let (key, winv, rng) = &mut self.to_be_processed[j];
                let mut h = (*winv) * (i - 1) as f64;
                if h < self.maxvaluetracker.get_max_value() {
                    h += (*winv) * self.exp01.sample(rng);
                    let k = unif0m.sample(rng);
                    if h < self.maxvaluetracker.get_value(k) {
                        self.signature[k] = key.clone();
                        //
                        self.maxvaluetracker.update(k, h);
                        qmax = self.maxvaluetracker.get_max_value();
                    }
                    if (*winv) * (i as f64) < qmax {
                        self.to_be_processed[insert_pos] = (key.clone(), *winv, rng.clone());
                        insert_pos += 1;
                    }
                }
            } // end of for j
            self.to_be_processed.truncate(insert_pos);
            i += 1;
        } // end of while
    } // end of hash_weigthed_idxmap

    /// It is the entry point of this hash algorithm with a HashMap (same as with IndexMap just in case)
    /// The HashMap gives multiplicity (or weight) to the objects hashed.
    /// The weight be positive and be convertible to a f64 without overflow (so )
    pub fn hash_weigthed_hashmap<Hidx, F>(&mut self, data: &HashMap<D, F, Hidx>)
    where
        Hidx: std::hash::BuildHasher,
        F: num::ToPrimitive + std::fmt::Display,
    {
        //
        let unif0m = Uniform::<usize>::new(0, self.m).unwrap();
        let mut qmax: f64 = self.maxvaluetracker.get_max_value();
        let iter = data.iter();

        for (key, weight_t) in iter {
            trace!("hash_item : id {:?}  weight {} ", key, weight_t);
            let weight = weight_t.to_f64().unwrap();
            assert!(
                weight.is_finite() && weight >= 0.,
                "conversion to f64 failed"
            );
            let winv = 1. / weight;
            // rebuild a new hasher at each id for reproductibility
            let mut hasher = Sha512_256::new();
            // write input message
            hasher.update(&key.get_sig());
            // read hash digest and consume hasher
            let new_hash = hasher.finalize();
            let hashed_slice = new_hash.as_slice();
            assert_eq!(hashed_slice.len(), 32);
            let mut seed: [u8; 32] = [0; 32];
            seed.copy_from_slice(&hashed_slice[..32]);
            let mut rng = Xoshiro256PlusPlus::from_seed(seed);
            let h = winv * self.exp01.sample(&mut rng);
            qmax = self.maxvaluetracker.get_max_value();

            if h < qmax {
                let k = unif0m.sample(&mut rng);
                assert!(k < self.m);
                if h < self.maxvaluetracker.get_value(k) {
                    self.signature[k] = key.clone();
                    //
                    self.maxvaluetracker.update(k, h);
                    qmax = self.maxvaluetracker.get_max_value();
                }
                if winv < qmax {
                    // we store point for further processing in second step, if inequality is not verified
                    // it cannot be added anymore anywhere.
                    self.to_be_processed.push((key.clone(), winv, rng));
                }
            } // end if h < qmax
        } // end initial loop
          //
          // now we have second step
          //
        let mut i = 2; // by comparison to ProbMinHash3 we are not at i = 2 !!
                       //
        while !self.to_be_processed.is_empty() {
            let mut insert_pos = 0;
            trace!(
                " i : {:?}  , nb to process : {}",
                i,
                self.to_be_processed.len()
            );
            for j in 0..self.to_be_processed.len() {
                let (key, winv, rng) = &mut self.to_be_processed[j];
                let mut h = (*winv) * (i - 1) as f64;
                if h < self.maxvaluetracker.get_max_value() {
                    h += (*winv) * self.exp01.sample(rng);
                    let k = unif0m.sample(rng);
                    if h < self.maxvaluetracker.get_value(k) {
                        self.signature[k] = key.clone();
                        //
                        self.maxvaluetracker.update(k, h);
                        qmax = self.maxvaluetracker.get_max_value();
                    }
                    if (*winv) * (i as f64) < qmax {
                        self.to_be_processed[insert_pos] = (key.clone(), *winv, rng.clone());
                        insert_pos += 1;
                    }
                }
            } // end of for j
            self.to_be_processed.truncate(insert_pos);
            i += 1;
        } // end of while
    } // end of hash_weigthed_hashmap

    /// return final signature.
    pub fn get_signature(&self) -> &Vec<D> {
        &self.signature
    }
} // end of ProbMinHash3aSha

//=================================================================

#[cfg(test)]
mod tests {

    use log::*;

    use fnv::FnvBuildHasher;
    use indexmap::IndexMap;

    type FnvIndexMap<K, V> = IndexMap<K, V, FnvBuildHasher>;

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    use crate::jaccard::*;

    use super::*;

    fn generate_slices(nb_slices: usize, length: usize) -> Vec<Vec<u8>> {
        //
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(237);
        let unif = Uniform::<u8>::new_inclusive(0, 255).unwrap();
        let mut slices = Vec::<Vec<u8>>::with_capacity(nb_slices);
        for _ in 0..nb_slices {
            let mut slice = Vec::<u8>::with_capacity(length);
            for _ in 0..length {
                slice.push(unif.sample(&mut rng));
            }
            slices.push(slice);
        }
        return slices;
    } // end of generate_slices

    #[test]
    // This test checks JaccardProbability with unequal weights inside sets
    fn test_probminhash3asha_count_intersection_unequal_weights() {
        //
        log_init_test();
        //
        println!("test_probminhash3a_count_intersection_unequal_weights");
        debug!("test_probminhash3a_count_intersection_unequal_weights");
        // we construct 2 ranges [a..b] [c..d], with a<b, b < d, c<d sketch them and compute jaccard.
        // we should get something like max(b,c) - min(b,c)/ (b-a+d-c)
        //
        let set_size = 100;
        let nbhash = 2000;
        let objects = generate_slices(set_size, 256);
        //
        // choose weights for va and vb elements
        let mut wa: FnvIndexMap<Vec<u8>, f64> =
            FnvIndexMap::with_capacity_and_hasher(70, FnvBuildHasher::default());
        // initialize wa, weight 20 up to 130
        for i in 0..set_size {
            if i < 70 {
                *wa.entry(objects[i].clone()).or_insert(0.) += 2. * i as f64;
            }
        }
        let mut wb: FnvIndexMap<Vec<u8>, f64> =
            FnvIndexMap::with_capacity_and_hasher(70, FnvBuildHasher::default());
        // initialize wb weight 10 above 70
        for i in 0..set_size {
            if i >= 50 {
                //            *wb.entry(i).or_insert(0.) += 2. * i as f64;  // gives jp = 0.24
                wb.entry(objects[i].clone()).or_insert((i as f64).powi(4)); // gives jp = 0.119
            }
        }
        // probminhash
        trace!("\n\n hashing wa");
        let mut waprobhash = ProbMinHash3aSha::<Vec<u8>>::new(nbhash, [0u8; 256].to_vec());
        waprobhash.hash_weigthed_idxmap(&wa);
        //
        trace!("\n\n hashing wb");
        let mut wbprobhash = ProbMinHash3aSha::<Vec<u8>>::new(nbhash, [0u8; 256].to_vec());
        wbprobhash.hash_weigthed_idxmap(&wb);
        //
        let siga = waprobhash.get_signature();
        let sigb = wbprobhash.get_signature();
        let jp_approx = compute_probminhash_jaccard(siga, sigb);
        //
        // compute Jp as in Ertl paper
        let mut jp = 0.;
        for i in 0..set_size {
            let wa_i = *wa.get(&objects[i]).unwrap_or(&0.);
            let wb_i = *wb.get(&objects[i]).unwrap_or(&0.);
            if wa_i > 0. && wb_i > 0. {
                let mut den = 0.;
                for j in 0..set_size {
                    let wa_j = *wa.get(&objects[j]).unwrap_or(&0.);
                    let wb_j = *wb.get(&objects[j]).unwrap_or(&0.);
                    den += (wa_j / wa_i).max(wb_j / wb_i);
                }
                jp += 1. / den;
            }
        }
        debug!("Jp = {} ", jp);
        //
        //    waprobhash.maxvaluetracker.dump();
        //    wbprobhash.maxvaluetracker.dump();
        //
        info!(
            "jp exact= {jptheo:.3} , jp estimate = {jp_est:.3} ",
            jptheo = jp,
            jp_est = jp_approx
        );
        assert!(jp_approx > 0.);
    } // end of test_probminhash3asha_count_intersection_unequal_weights
} // end of module tests
