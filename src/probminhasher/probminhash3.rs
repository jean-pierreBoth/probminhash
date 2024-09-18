//! Implementation of ProbMinHash3 and ProbMinHash3a as described in O. Ertl  
//! <https://arxiv.org/abs/1911.00675>
//! * ProbminHash3a is the fastest but at the cost of some internal storage.
//! * Probminhash3 is the same algorithm without the time optimization requiring more storage.  
//!     It can be used in streaming
//!
//! The generic type D must satisfy D:Copy+Eq+Hash+Debug  
//! The algorithms requires random generators to be initialized by the objects so we need to map (at least approximately
//! injectively) objects into a u64 so the of objcts must satisfy Hash.
//! If D is of type u64 it is possible to use a NoHasher (cd module nohasher)

#[allow(unused_imports)]
use log::{debug, trace};

use std::fmt::Debug;

use num;

use rand::distributions::{Distribution, Uniform};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;

use indexmap::IndexMap;
use std::collections::HashMap;
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};

use crate::exp01::*;
use crate::maxvaluetrack::*;
use crate::weightedset::*;

/// implementation of the algorithm ProbMinHash3a as described in Etrl.  
/// It needs less memory than Probminhash3 but can be a little slower.   
/// Probminhash3 needs at least 2 hash values to run.
///  
/// The algorithms requires random generators to be initialized by objects hashed.
/// So it must possible to associate D (at least partially) injectively to a u64 for random generator initialization hence the requirement D:H.  
/// If all data are referred to by an unsigned integer, and weight association is given in a tuple for example
/// data comes in a Vec<(D,f64)> then D is in fact can be replaced by the rank in the Vector, the no hash is need and you can use NoHasher
pub struct ProbMinHash3<D, H: Hasher + Default>
where
    D: Copy + Eq + Hash + Debug,
{
    m: usize,
    //
    b_hasher: BuildHasherDefault<H>,
    /// field to keep track of max hashed values
    maxvaluetracker: MaxValueTracker<f64>,
    /// a exponential law restricted to interval [0., 1)
    exp01: ExpRestricted01,
    ///  final signature of distribution. allocated to size m
    signature: Vec<D>,
} // end of struct ProbMinHash3

impl<D, H> ProbMinHash3<D, H>
where
    D: Copy + Eq + Debug + Hash,
    H: Hasher + Default,
{
    /// Allocates a new ProbMinHash3 structure with nbhash functions and initial object initobj to fill signature.  
    /// nbhash must be greater or equal to 2.
    /// The precision on the final estimation depends on the number of hash functions.   
    /// The initial object can be any object , typically 0 for numerical objects.
    pub fn new(nbhash: usize, initobj: D) -> Self {
        assert!(nbhash >= 2);
        let lambda = ((nbhash as f64) / ((nbhash - 1) as f64)).ln();
        let h_signature = (0..nbhash).map(|_| initobj).collect();
        ProbMinHash3 {
            m: nbhash,
            b_hasher: BuildHasherDefault::<H>::default(),
            maxvaluetracker: MaxValueTracker::new(nbhash),
            exp01: ExpRestricted01::new(lambda),
            signature: h_signature,
        }
    } // end of new

    /// Incrementally adds an item in hash signature. It can be used in streaming.  
    /// It is the building block of the computation, but this method
    /// does not check for unicity of id added in hash computation.  
    /// It is the user's responsability to enforce that. See function hash_weigthed_idxmap
    pub fn hash_item<F>(&mut self, id: D, weight_a: &F)
    where
        F: num::ToPrimitive + std::fmt::Display,
    {
        //
        let weight = weight_a.to_f64().unwrap();
        assert!(weight > 0.);
        trace!("hash_item : id {:?}  weight {} ", id, weight);
        let winv = 1. / weight;
        let unif0m = Uniform::<usize>::new(0, self.m);
        let id_hash: u64 = self.b_hasher.hash_one(&id);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(id_hash);
        let mut h = winv * self.exp01.sample(&mut rng);
        let mut i = 1;
        let mut qmax = self.maxvaluetracker.get_max_value();
        while h < qmax {
            let k = unif0m.sample(&mut rng);
            assert!(k < self.m);
            if h < self.maxvaluetracker.get_value(k) {
                self.signature[k] = id;
                //
                self.maxvaluetracker.update(k, h);
                qmax = self.maxvaluetracker.get_max_value();
            }
            h = winv * i as f64;
            i += 1;
            if h >= qmax {
                break;
            }
            h += winv * self.exp01.sample(&mut rng);
            trace!("hash_item :  i h qmax =  {}   {}   {} ", i, h, qmax);
        }
    } // end of hash_item

    /// return final signature.
    pub fn get_signature(&self) -> &Vec<D> {
        &self.signature
    }

    /// hash data when given by an iterable WeightedSet
    pub fn hash_wset<T>(&mut self, data: &mut T)
    where
        T: WeightedSet<Object = D> + Iterator<Item = D>,
    {
        while let Some(obj) = &data.next() {
            let weight = data.get_weight(obj);
            self.hash_item(*obj, &weight);
        }
    } // end of hash method

    /// computes set signature when set is given as an IndexMap with weights corresponding to values.  
    /// This ensures that objects are assigned a weight only once, so that we really have a set of objects with weight associated.  
    /// The raw method hash_item can be used with the constraint that objects are sent ONCE in the hash method.
    pub fn hash_weigthed_idxmap<Hidx, F>(&mut self, data: &IndexMap<D, F, Hidx>)
    where
        Hidx: std::hash::BuildHasher,
        F: num::ToPrimitive + std::fmt::Display,
    {
        let iter = data.iter();
        for (key, weigth) in iter {
            trace!(" retrieved key {:?} ", key);
            // we must convert Kmer64bit to u64 and be able to retrieve the original Kmer64bit
            // we got weight as something convertible to f64
            self.hash_item(*key, weigth);
        }
    } // end of hash_weigthed_idxmap

    /// computes set signature when set is given as an HashMap with weights corresponding to values.(equivalent to the method with and IndexMap)
    /// This ensures that objects are assigned a weight only once, so that we really have a set of objects with weight associated.  
    /// The raw method hash_item can be used with the constraint that objects are sent ONCE in the hash method.
    pub fn hash_weigthed_hashmap<Hidx, F>(&mut self, data: &HashMap<D, F, Hidx>)
    where
        Hidx: std::hash::BuildHasher,
        F: num::ToPrimitive + std::fmt::Display,
    {
        let iter = data.iter();
        for (key, weight) in iter {
            trace!(" retrieved key {:?} ", key);
            // we got weight as something convertible to f64
            self.hash_item(*key, weight);
        }
    } // end of hash_weigthed_hashmap
} // end of impl ProbMinHash3

/// implementation of the algorithm ProbMinHash3a as described in Etrl.  
/// This version of ProbMinHash3 is faster but needs some more memory as it stores some states
/// between 2 passes on data.
///
/// Probminhash3a needs at least 2 hash values to run.
/// D must be convertible (at least partially) injectively into a usize for random generator initialization hence the requirement D:Hash.  
/// If all data are referred to by an unsigned integer, and weight association is given in a tuple for example if
/// data comes in a Vec<(D,f64)> then D can be in fact replaced by the rank in the Vector, then no hash is need and you can use NoHasher
pub struct ProbMinHash3a<D, H>
where
    D: Copy + Eq + Hash + Debug,
    H: Hasher + Default,
{
    m: usize,
    //
    b_hasher: BuildHasherDefault<H>,
    /// field to keep track of max hashed values
    maxvaluetracker: MaxValueTracker<f64>,
    /// a exponential law restricted to interval [0., 1)
    exp01: ExpRestricted01,
    /// Buffer to store object to be processed in second pass. Stores (object, inverse weight, generator)
    to_be_processed: Vec<(D, f64, Xoshiro256PlusPlus)>,
    ///  final signature of distribution. allocated to size m
    signature: Vec<D>,
} // end of struct ProbMinHash3a

impl<D, H> ProbMinHash3a<D, H>
where
    D: Copy + Eq + Debug + Hash,
    H: Hasher + Default,
{
    /// Allocates a new ProbMinHash3a structure with nbhash >= 2 functions and initial object initobj to fill signature.  
    /// The precision on the final estimation depends on the number of hash functions.   
    /// The initial object can be any object , typically 0 for numerical objects.
    pub fn new(nbhash: usize, initobj: D) -> Self {
        assert!(nbhash >= 2);
        let lambda = ((nbhash as f64) / ((nbhash - 1) as f64)).ln();
        let h_signature = (0..nbhash).map(|_| initobj).collect();
        ProbMinHash3a {
            m: nbhash,
            maxvaluetracker: MaxValueTracker::new(nbhash),
            b_hasher: BuildHasherDefault::<H>::default(),
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
        let unif0m = Uniform::<usize>::new(0, self.m);
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
            let new_hash: u64 = self.b_hasher.hash_one(&key);
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(new_hash);
            let h = winv * self.exp01.sample(&mut rng);
            qmax = self.maxvaluetracker.get_max_value();

            if h < qmax {
                let k = unif0m.sample(&mut rng);
                assert!(k < self.m);
                if h < self.maxvaluetracker.get_value(k) {
                    self.signature[k] = *key;
                    //
                    self.maxvaluetracker.update(k, h);
                    qmax = self.maxvaluetracker.get_max_value();
                }
                if winv < qmax {
                    // we store point for further processing in second step, if inequality is not verified
                    // it cannot be added anymore anywhere.
                    self.to_be_processed.push((*key, winv, rng));
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
                        self.signature[k] = *key;
                        //
                        self.maxvaluetracker.update(k, h);
                        qmax = self.maxvaluetracker.get_max_value();
                    }
                    if (*winv) * (i as f64) < qmax {
                        self.to_be_processed[insert_pos] = (*key, *winv, rng.clone());
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
        let unif0m = Uniform::<usize>::new(0, self.m);
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
            let new_hash: u64 = self.b_hasher.hash_one(&key);
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(new_hash);
            let h = winv * self.exp01.sample(&mut rng);
            qmax = self.maxvaluetracker.get_max_value();

            if h < qmax {
                let k = unif0m.sample(&mut rng);
                assert!(k < self.m);
                if h < self.maxvaluetracker.get_value(k) {
                    self.signature[k] = *key;
                    //
                    self.maxvaluetracker.update(k, h);
                    qmax = self.maxvaluetracker.get_max_value();
                }
                if winv < qmax {
                    // we store point for further processing in second step, if inequality is not verified
                    // it cannot be added anymore anywhere.
                    self.to_be_processed.push((*key, winv, rng));
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
                        self.signature[k] = *key;
                        //
                        self.maxvaluetracker.update(k, h);
                        qmax = self.maxvaluetracker.get_max_value();
                    }
                    if (*winv) * (i as f64) < qmax {
                        self.to_be_processed[insert_pos] = (*key, *winv, rng.clone());
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
} // end of ProbMinHash3a

//=================================================================

#[cfg(test)]
mod tests {

    use log::*;

    use fnv::{FnvBuildHasher, FnvHasher};
    use indexmap::IndexMap;

    type FnvIndexMap<K, V> = IndexMap<K, V, FnvBuildHasher>;

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    use crate::jaccard::*;

    use super::*;

    #[test]
    // This test checks that with equal weights we fall back to Jaccard estimate
    fn test_probminhash3_count_intersection_equal_weights() {
        //
        log_init_test();
        //
        debug!("test_probminhash3_count_intersection_equal_weights");
        println!("test_probminhash3_count_intersection_equal_weights");
        // we construct 2 ranges [a..b] [c..d], with a<b, b < d, c<d sketch them and compute jaccard.
        // we should get something like max(b,c) - min(b,c)/ (b-a+d-c)
        //
        let set_size = 100;
        let nbhash = 100;
        //
        // choose weights for va and vb elements
        let mut wa = Vec::<f64>::with_capacity(set_size);
        let mut wb = Vec::<f64>::with_capacity(set_size);
        // initialize wa, weight 20 up to 130
        for i in 0..set_size {
            if i < 70 {
                wa.push(20.);
            } else {
                wa.push(0.);
            }
        }
        // initialize wb weight 10 above 70
        for i in 0..set_size {
            if i < 50 {
                wb.push(0.);
            } else {
                wb.push(10.);
            }
        }

        // compute Jp as in
        let mut jp = 0.;
        for i in 0..set_size {
            if wa[i] > 0. && wb[i] > 0. {
                let mut den = 0.;
                for j in 0..set_size {
                    den += (wa[j] / wa[i]).max(wb[j] / wb[i]);
                }
                jp += 1. / den;
            }
        }
        trace!("Jp = {} ", jp);
        // probminhash
        trace!("\n\n hashing wa");
        let mut waprobhash = ProbMinHash3::<usize, FnvHasher>::new(nbhash, 0);
        for i in 0..set_size {
            if wa[i] > 0. {
                waprobhash.hash_item::<f64>(i, &wa[i]);
            }
        }
        // waprobhash.maxvaluetracker.dump();
        //
        trace!("\n\n hashing wb");
        let mut wbprobhash = ProbMinHash3::<usize, FnvHasher>::new(nbhash, 0);
        for i in 0..set_size {
            if wb[i] > 0. {
                wbprobhash.hash_item(i, &wb[i]);
            }
        }
        let siga = waprobhash.get_signature();
        let sigb = wbprobhash.get_signature();
        //
        let jp_approx = compute_probminhash_jaccard(siga, sigb);
        //
        //       waprobhash.maxvaluetracker.dump();
        //       wbprobhash.maxvaluetracker.dump();
        //
        info!("exact jp = {:.3e} ,jp estimated = {:.3e} ", jp, jp_approx);
        assert!(jp_approx > 0.);
    } // end of test_prob_count_intersection

    #[test]
    // This test checks JaccardProbability with unequal weights inside sets
    fn test_probminhash3a_count_intersection_unequal_weights() {
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
        //
        // choose weights for va and vb elements
        let mut wa: FnvIndexMap<usize, f64> =
            FnvIndexMap::with_capacity_and_hasher(70, FnvBuildHasher::default());
        // initialize wa, weight 20 up to 130
        for i in 0..set_size {
            if i < 70 {
                *wa.entry(i).or_insert(0.) += 2. * i as f64;
            }
        }
        let mut wb: FnvIndexMap<usize, f64> =
            FnvIndexMap::with_capacity_and_hasher(70, FnvBuildHasher::default());
        // initialize wb weight 10 above 70
        for i in 0..set_size {
            if i >= 50 {
                //            *wb.entry(i).or_insert(0.) += 2. * i as f64;  // gives jp = 0.24
                wb.entry(i).or_insert((i as f64).powi(4)); // gives jp = 0.119
            }
        }
        // probminhash
        trace!("\n\n hashing wa");
        let mut waprobhash = ProbMinHash3a::<usize, FnvHasher>::new(nbhash, 0);
        waprobhash.hash_weigthed_idxmap(&wa);
        //
        trace!("\n\n hashing wb");
        let mut wbprobhash = ProbMinHash3a::<usize, FnvHasher>::new(nbhash, 0);
        wbprobhash.hash_weigthed_idxmap(&wb);
        //
        let siga = waprobhash.get_signature();
        let sigb = wbprobhash.get_signature();
        let jp_approx = compute_probminhash_jaccard(siga, sigb);
        //
        // compute Jp as in Ertl paper
        let mut jp = 0.;
        for i in 0..set_size {
            let wa_i = *wa.get(&i).unwrap_or(&0.);
            let wb_i = *wb.get(&i).unwrap_or(&0.);
            if wa_i > 0. && wb_i > 0. {
                let mut den = 0.;
                for j in 0..set_size {
                    let wa_j = *wa.get(&j).unwrap_or(&0.);
                    let wb_j = *wb.get(&j).unwrap_or(&0.);
                    den += (wa_j / wa_i).max(wb_j / wb_i);
                }
                jp += 1. / den;
            }
        }
        trace!("Jp = {} ", jp);
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
    } // end of test_probminhash3a_count_intersection_unequal_weights

    #[test]
    // This test checks JaccardProbability with unequal weights inside sets
    fn test_probminhash3_count_intersection_unequal_weights() {
        //
        log_init_test();
        //
        println!("test_probminhash3_count_intersection_unequal_weights");
        debug!("test_probminhash3_count_intersection_unequal_weights");
        // we construct 2 ranges [a..b] [c..d], with a<b, b < d, c<d sketch them and compute jaccard.
        // we should get something like max(b,c) - min(b,c)/ (b-a+d-c)
        //
        let set_size = 100;
        let nbhash = 2000;
        //
        // choose weights for va and vb elements
        let mut wa = Vec::<f64>::with_capacity(set_size);
        let mut wb = Vec::<f64>::with_capacity(set_size);
        // initialize wa, weight 2*i up to 70
        for i in 0..set_size {
            if i < 70 {
                wa.push(2. * i as f64);
            } else {
                wa.push(0.);
            }
        }
        // initialize wb weight i^4 above 50
        for i in 0..set_size {
            if i < 50 {
                wb.push(0.);
            } else {
                wb.push((i as f64).powi(4));
            }
        }
        // probminhash
        trace!("\n\n hashing wa");
        let mut waprobhash = ProbMinHash3::<usize, FnvHasher>::new(nbhash, 0);
        for i in 0..set_size {
            if wa[i] > 0. {
                waprobhash.hash_item(i, &wa[i]);
            }
        }
        //
        trace!("\n\n hashing wb");
        let mut wbprobhash = ProbMinHash3::<usize, FnvHasher>::new(nbhash, 0);
        for i in 0..set_size {
            if wb[i] > 0. {
                wbprobhash.hash_item(i, &wb[i]);
            }
        }
        let siga = waprobhash.get_signature();
        let sigb = wbprobhash.get_signature();
        let jp_approx = compute_probminhash_jaccard(siga, sigb);
        //
        // compute Jp as in
        let mut jp = 0.;
        for i in 0..set_size {
            if wa[i] > 0. && wb[i] > 0. {
                let mut den = 0.;
                for j in 0..set_size {
                    den += (wa[j] / wa[i]).max(wb[j] / wb[i]);
                }
                jp += 1. / den;
            }
        }
        trace!("Jp = {} ", jp);
        //
        //   waprobhash.maxvaluetracker.dump();
        //   wbprobhash.maxvaluetracker.dump();
        //    println!("siga :  {:?}", siga);
        //   println!("sigb :  {:?}", sigb);
        //
        info!(
            "jp exact = {jp_exact:.3} , jp estimate {jp_estimate:.3} ",
            jp_exact = jp,
            jp_estimate = jp_approx
        );
        assert!(jp_approx > 0.);
    } // end of test_probminhash3_count_intersection_unequal_weights
} // end of module tests
