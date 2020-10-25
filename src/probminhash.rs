//! Implementation of ProbMinHash2, ProbMinHash3 and ProbMinHash3a as described in O. Ertl  
//! <https://arxiv.org/abs/1911.00675>
//! * ProbminHash3a is the fastest but at the cost of some internal storage.
//! * Probminhash3 is the same algorithm without the time optimization requiring more storage.  
//!     It can be used in streaming
//! * Probminhash2 is statistically equivalent to P-Minhash as described in :
//! Moulton Jiang "Maximally consistent sampling and the Jaccard index of probability distributions"
//! <https://ieeexplore.ieee.org/document/8637426> or <https://arxiv.org/abs/1809.04052>.  
//! It is given as a fallback in case ProbminHash3* algorithms do not perform well, or for comparison.
//! 
//! The generic type D must satisfy D:Copy+Eq+Hash+Debug  
//! The algorithms requires random generators to be initialized by the objects so we need to map (at least approximately
//! injectively) objects into a u64 so the of objcts must satisfy Hash.
//! If D is of type u64 it is possible to use a NoHasher (cd module nohasher)

use log::{trace};

use std::fmt::{Debug};

use rand::distributions::{Distribution,Uniform};
use rand_distr::Exp1;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;

use wyhash::*;
use std::hash::{BuildHasher, BuildHasherDefault, Hasher, Hash};
use indexmap::{IndexMap};

/// Structure for defining exponential sampling of parameter lambda with support restricted
/// to unit interval [0,1).  
// All comments follow notations in Ertl article
#[derive(Clone, Copy, Debug)]
pub struct ExpRestricted01 {
    /// parameter of exponential
    lambda : f64,
    c1 : f64,
    // abciss of point for which A3 is under exponential
    c2 : f64,
    c3 : f64,
    /// we build upon a uniform [0,1) sampling
    unit_range : Uniform<f64>,
} // end of struct ExpRestricted01


impl ExpRestricted01  {
    /// allocates a struct ExpRestricted01 for sampling an exponential law of parameter lambda, but restricted to [0,1.)]
    pub fn new(lambda : f64) -> Self {
        let c1 = lambda.exp_m1() / lambda;    // exp_m1 for numerical precision 
        let c2 = (2./(1. + (-lambda).exp())).ln()/ lambda;
        let c3 = (1. - (-lambda).exp()) / lambda;
        ExpRestricted01{lambda, c1, c2, c3, unit_range:Uniform::<f64>::new(0.,1.)}
    }

    /// return lambda parameter of exponential
    pub fn get_lambda(&self) -> f64 {
        self.lambda
    }
}


impl Distribution<f64> for  ExpRestricted01  {
    /// sample from ExpRestricted01
    fn sample<R : Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let mut x = self.c1 * rng.sample(&self.unit_range);
        if x < 1.  { return x }
        loop {
            // check if we can sample in A3
            x = rng.sample(&self.unit_range);
            if x < self.c2 { return x}
            // 
            let mut y = 0.5 * rng.sample(&self.unit_range);
            if y > 1. - x {
                // transform a point in A5 to a point in A6
                x = 1. - x;
                y = 1. - y;
            }
            if x <= self.c3 * (1. - y) { return x }
            if self.c1 * y <= (1. - x) { return x }
            if y * self.c1 * self.lambda <= (self.lambda * (1.- x)).exp_m1() { return x }
        }        
    } // end sample
} 

// structure to keep track of max values in hash set
// adapted from class MaxValueTracker
struct MaxValueTracker {
    m : usize,
    // last_index = 2*m-2. max of array is at slot last_index
    last_index : usize,
    // dimensioned to m hash functions
    values : Vec<f64>
}


impl MaxValueTracker {
    pub fn new(m:usize) -> Self {
        let last_index  = ((m << 1) - 2) as usize;  // 0-indexation for the difference with he paper, lastIndex = 2*m-2
        let vlen = last_index+1;
        let values : Vec::<f64> = (0..vlen).map( |_| f64::INFINITY).collect();
        MaxValueTracker{m, last_index, values}
    }

 
    
    // update slot k with value value
    // 0 indexation imposes some changes with respect to the the algo 4 of the paper
    // parent of k is m + (k/2)
    // and accordingly
    // sibling ok k is k+1 if k even, k-1 else so it is given by bitxor(k,1)
    fn update(&mut self, k:usize, value:f64) {
        assert!(k < self.m);
        trace!("\n max value tracker update k, value , value at k {} {} {} ", k, value, self.values[k]);
        let mut current_value = value;
        let mut current_k = k;
        let mut more = false;
        if current_value < self.values[current_k] {
            more = true;
        }
        
        while more {
            trace!("mxvt update k value {} {}", current_k, current_value);
            self.values[current_k] = current_value;
            let pidx = self.m + (current_k/2) as usize;   // m + upper integer value of k/2 beccause of 0 based indexation
            if pidx > self.last_index {
                break;
            }
            let siblidx = current_k^1;      // get sibling index of k with numeration beginning at 0
            assert!(self.values[siblidx] <= self.values[pidx]);
            assert!(self.values[current_k] <= self.values[pidx]);
            //
            if self.values[siblidx] >= self.values[pidx] && self.values[current_k] >= self.values[pidx] {
                break;     // means parent current and sibling are equals no more propagation needed
            }
            // now either self.values[siblidx] <self.values[pidx] or current_value < self.values[pidx]
            trace!("propagating current_value {} sibling  {} ? ", current_value, self.values[siblidx]);
            //
            if current_value < self.values[siblidx] {
                trace!("     propagating sibling value {} to parent {}", self.values[siblidx], pidx);
                current_value = self.values[siblidx];
            }
            else {
                trace!("     propagating current_value {} to parent {}", current_value, pidx);   
            }
            current_k = pidx;
            if current_value >= self.values[current_k]  {
                more = false;
            }
        }
    } // end of update function 


    /// return the maximum value maintained in the data structure
    pub fn get_max_value(&self) -> f64 {
        return self.values[self.last_index]
    }

    #[allow(dead_code)]
    pub fn get_parent_slot(&self, slot : usize) -> usize {
        assert!(slot <= self.m);
        return self.m + (slot/2) as usize   // m + upper integer value of k/2 beccause of 0 based indexation
    }

    /// get value MaxValueTracker at slot
    #[allow(dead_code)]
    pub fn get_value(&self, slot: usize) -> f64 {
        self.values[slot]
    } // end of get_value

    #[allow(dead_code)]
    pub fn dump(&self) {
        println!("\n\nMaxValueTracker dump : ");
        for i in 0..self.values.len() {
            println!(" i  value   {}   {} ", i , self.values[i]);
        }
    } // end of dump
} // end of impl MaxValueTracker




/// A Trait to define association of a weight to an object.
/// Typically we could implement trait WeightedSet for any collection of Object if we have a function giving a weight to each object
/// Then hash_wset function can be used.
pub trait WeightedSet {
    type Object;
    /// returns the weight of an object
    fn get_weight(&self, obj:&Self::Object) -> f64;
}


/// implementation of the algorithm ProbMinHash3a as described in Etrl.  
/// It needs less memory than Probminhash3 but can be a little slower.   
/// Probminhash3 needs at least 2 hash values to run.
///  
/// The algorithms requires random generators to be initialized by objects hashed. 
/// So it must possible to associate D (at least partially) injectively to a u64 for random generator initialization hence the requirement D:H.  
/// If all data are referred to by an unsigned integer, and weight association is given in a tuple for example 
/// data comes in a Vec<(D,f64)> then D is in fact can be replaced by the rank in the Vector, the no hash is need and you can use NoHasher
pub struct ProbMinHash3<D, H: Hasher+Default> 
            where D:Copy+Eq+Hash+Debug   {
    m : usize,
    ///
    b_hasher: BuildHasherDefault<H>,
    /// field to keep track of max hashed values
    maxvaluetracker : MaxValueTracker,
    /// a exponential law restricted to interval [0., 1)
    exp01 : ExpRestricted01,
    ///  final signature of distribution. allocated to size m
    signature : Vec<D>,
} // end of struct ProbMinHash3


impl<D,H> ProbMinHash3<D, H> 
            where D:Copy+Eq+Debug+Hash , H: Hasher+Default {

    /// Allocates a new ProbMinHash3 structure with nbhash functions and initial object initobj to fill signature.  
    /// nbhash must be greater or equal to 2.
    /// The precision on the final estimation depends on the number of hash functions.   
    /// The initial object can be any object , typically 0 for numerical objects.
    pub fn new(nbhash:usize, initobj : D) -> Self {
        assert!(nbhash >= 2);
        let lambda = ((nbhash as f64)/((nbhash - 1) as f64)).ln();
        let h_signature = (0..nbhash).map( |_| initobj).collect();
        ProbMinHash3{m:nbhash, b_hasher : BuildHasherDefault::<H>::default(), 
                    maxvaluetracker: MaxValueTracker::new(nbhash as usize), 
                    exp01:ExpRestricted01::new(lambda), signature:h_signature}
    } // end of new
    

    /// Incrementally adds an item in hash signature. It can be used in streaming.  
    /// It is the building block of the computation, but this method 
    /// does not check for unicity of id added in hash computation.  
    /// It is the user's responsability to enforce that. See function hash_weigthed_idxmap
    pub fn hash_item(&mut self, id:D, weight:f64) {
        assert!(weight > 0.);
        trace!("hash_item : id {:?}  weight {} ", id, weight);
        let winv = 1./weight;
        let unif0m = Uniform::<usize>::new(0, self.m);
        let mut hasher = self.b_hasher.build_hasher();
        id.hash(&mut hasher);
        let id_hash : u64 = hasher.finish();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(id_hash);
        let mut h = winv * self.exp01.sample(&mut rng);
        let mut i = 1;
        let mut qmax = self.maxvaluetracker.get_max_value();
        while h < qmax {
            let k = unif0m.sample(&mut rng);
            assert!(k < self.m);
            if h < self.maxvaluetracker.values[k] {
                self.signature[k] = id;
                // 
                self.maxvaluetracker.update(k, h);
                qmax = self.maxvaluetracker.get_max_value();
            }
            h = winv * i as f64;
            i = i + 1;
            if h >= qmax {
                break;
            }
            h = h + winv * self.exp01.sample(&mut rng);
            trace!("hash_item :  i h qmax =  {}   {}   {} ", i, h, qmax);
        }
    } // end of hash_item

    /// return final signature.
    pub fn get_signature(&self) -> &Vec<D> {
        return &self.signature
    }

    /// hash data when given by an iterable WeightedSet
    pub fn hash_wset<T>(&mut self, data: &mut T)
        where T: WeightedSet<Object=D> + Iterator<Item=D> {
            while let Some(obj) = &data.next() {
                let weight = data.get_weight(&obj);
                self.hash_item(*obj, weight);
            }
    } // end of hash method


    /// computes set signature when set is given as an IndexMap with weights corresponding to values.  
    /// This ensures that objects are assigned a weight only once, so that we really have a set of objects with weight associated.  
    /// The raw method hash_item can be used with the constraint that objects are sent ONCE in the hash method.
    pub fn hash_weigthed_idxmap<Hidx>(&mut self, data: &mut IndexMap<D, f64, Hidx>) 
                where   Hidx : std::hash::BuildHasher, 
    {
        let mut objects = data.keys();
        loop {
            match objects.next() {
                Some(key) => {
                    trace!(" retrieved key {:?} ", key);  
                   // we must convert Kmer64bit to u64 and be able to retrieve the original Kmer64bit
                    if let Some(weight) = data.get(key) {
                        // we got weight as something convertible to f64
                        self.hash_item(*key, *weight);
                    };
                },
                None => break,
            }
        }
    }  // end of hash_weigthed_idxmap

}  // end of impl ProbMinHash3




/// implementation of the algorithm ProbMinHash3a as described in Etrl.  
/// This version of ProbMinHash3 is faster but needs some more memory as it stores some states
/// between 2 passes on data.
/// 
/// Probminhash3a needs at least 2 hash values to run.
/// D must be convertible (at least partially) injectively into a usize for random generator initialization hence the requirement D:Hash.  
/// If all data are referred to by an unsigned integer, and weight association is given in a tuple for example if
/// data comes in a Vec<(D,f64)> then D can be in fact replaced by the rank in the Vector, then no hash is need and you can use NoHasher
pub struct ProbMinHash3a<D,H> 
            where D:Copy+Eq+Hash+Debug,
                  H:Hasher+Default  {
    m : usize,
    ///
    b_hasher : BuildHasherDefault::<H>,
    /// field to keep track of max hashed values
    maxvaluetracker : MaxValueTracker,
    /// a exponential law restricted to interval [0., 1)
    exp01 : ExpRestricted01,
    /// Buffer to store object to be processed in second pass. Stores (object, inverse weight, generator)
    to_be_processed : Vec<(D, f64, Xoshiro256PlusPlus)>,
    ///  final signature of distribution. allocated to size m
    signature : Vec<D>,
} // end of struct ProbMinHash3a



impl <D,H> ProbMinHash3a<D,H> 
        where D:Copy+Eq+Debug+Hash, H : Hasher+Default {

    /// Allocates a new ProbMinHash3a structure with nbhash >= 2 functions and initial object initobj to fill signature.  
    /// The precision on the final estimation depends on the number of hash functions.   
    /// The initial object can be any object , typically 0 for numerical objects.
    pub fn new(nbhash:usize, initobj : D) -> Self {
        assert!(nbhash >= 2);
        let lambda = ((nbhash as f64)/((nbhash - 1) as f64)).ln();
        let h_signature = (0..nbhash).map( |_| initobj).collect();
        ProbMinHash3a{m:nbhash, 
                maxvaluetracker: MaxValueTracker::new(nbhash as usize), 
                b_hasher : BuildHasherDefault::<H>::default(),
                exp01:ExpRestricted01::new(lambda), 
                to_be_processed : Vec::<(D, f64, Xoshiro256PlusPlus)>::new(),
                signature:h_signature}
    } // end of new


    /// It is the building block of the computation, but this method 
    /// does not check for unicity of id added in hash computation.  
    /// It is user responsability to enforce that. See method hashWSet
    pub fn hash_weigthed_idxmap<Hidx>(&mut self, data: &IndexMap<D, f64, Hidx>) 
                where   Hidx : std::hash::BuildHasher  {
        //
        let mut objects = data.keys();
        let unif0m = Uniform::<usize>::new(0, self.m);
        let mut qmax:f64 = self.maxvaluetracker.get_max_value();

        loop {
            if let Some(key) = objects.next() {
                if let Some(weight) = data.get(key) {
                    trace!("hash_item : id {:?}  weight {} ", key, weight);
                    let winv = 1./weight;
                    // rebuild a new hasher at each id for reproductibility
                    let mut hasher = self.b_hasher.build_hasher();
                    key.hash(&mut hasher);
                    let new_hash : u64 = hasher.finish();
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(new_hash);
                    let h = winv * self.exp01.sample(&mut rng);
                    qmax = self.maxvaluetracker.get_max_value();
                    
                    if h < qmax {
                        let k = unif0m.sample(&mut rng);
                        assert!(k < self.m);
                        if h < self.maxvaluetracker.values[k] {
                            self.signature[k] = *key;
                            // 
                            self.maxvaluetracker.update(k, h);
                            qmax = self.maxvaluetracker.get_max_value();
                        }
                        if winv < qmax {
                            // we store point for further processing in second step, if inequality is not verified
                            // it cannot be added anymore anywhere.
                            self.to_be_processed.push((*key,winv, rng));
                        }                       
                    } // end if h < qmax
                } // end some weight
            } // end some key
            else { 
                break;
            }
        } // end initial loop
        //
        // now we have second step
        //
        let mut i = 2;    // by comparison to ProbMinHash3 we are not at i = 2 !!
        // 
        while self.to_be_processed.len() > 0 {
            let mut insert_pos = 0;
            trace!(" i : {:?}  , nb to process : {}", i , self.to_be_processed.len());
            for j in 0..self.to_be_processed.len() {
                let (key, winv, rng) = &mut self.to_be_processed[j];
                let mut h = (*winv) * (i - 1) as f64;
                if h < self.maxvaluetracker.get_max_value() {
                    h = h + (*winv) * self.exp01.sample(rng);
                    let k = unif0m.sample(rng);
                    if h < self.maxvaluetracker.values[k] {
                        self.signature[k] = *key;
                        // 
                        self.maxvaluetracker.update(k, h);
                        qmax = self.maxvaluetracker.get_max_value();
                    }
                    if (*winv) * (i as f64) < qmax {
                        self.to_be_processed[insert_pos] = (*key, *winv, rng.clone());
                        insert_pos = insert_pos + 1;
                    }
                }
            }  // end of for j
            self.to_be_processed.truncate(insert_pos);
            i = i+1;
        }  // end of while 
    } // end of hash_weigthed_idxmap

    /// return final signature.
    pub fn get_signature(&self) -> &Vec<D> {
        return &self.signature;
    }
} // end of ProbMinHash3a

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Fisher Yates random permutation generation (sampling without replacement), with lazy generation
/// of an array of size n
pub struct FYshuffle {
    m: usize,
    rng : Xoshiro256PlusPlus,
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
        // seed Xoshiro256PlusPlus with WyRng...
        let mut rng_init = WyRng::default();
        FYshuffle{m:m, rng:Xoshiro256PlusPlus::seed_from_u64(rng_init.next_u64()), unif_01: Uniform::<f64>::new(0., 1.), v : v, lastidx:m}
    }

    // See https://www.geeksforgeeks.org/generate-a-random-permutation-of-1-to-n/
    // and The algorithm design manual S.Skiena P.458

    /// generates next randomly choosen item of the set (you get sampling without replacement)
    /// After being call m times, it is possible to get the full permutation with the function get_values
    /// as the permutation is fully sampled.
    /// If called more than m times, it calls reset implicitly to generate a new permutation.
    /// It is possible (and recommended) to reset explicitly after m successive call to next method
    pub fn next(&mut self) -> usize {
        if self.lastidx >= self.m {
            self.lastidx = 0;
        }
        let xsi = self.unif_01.sample(&mut self.rng);
        // sample between self.lastidx (included) and self.m (excluded)
        let idx = self.lastidx + (xsi * (self.m - self.lastidx) as f64) as usize;
        let val = self.v[idx];
        self.v[idx] = self.v[self.lastidx];
        self.v[self.lastidx] = val;
        self.lastidx += 1;
        val
    }

    pub fn reset(&mut self) {
        self.lastidx = 0;
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
        let betas : Vec<f64> = (0..nbhash).map(| x | (nbhash as f64)/ (nbhash - x) as f64).collect();
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
        let x : f64 = rng.sample(Exp1);
        let mut h : f64 = winv * x;
        let mut qmax = self.maxvaluetracker.get_max_value();
        //
        while h < qmax {
            let k = self.permut_generator.next();
            if h < self.maxvaluetracker.values[k] {
                self.signature[k] = id;
                // 
                self.maxvaluetracker.update(k, h);
                qmax = self.maxvaluetracker.get_max_value();
                if h >= qmax { break;}
            }
            i = i+1;
            let x : f64 = rng.sample(Exp1);
            // note : we have incremented i before accessing to betas to be coherent i initialization to 0
            // and beta indexing.
            h = h + winv * self.betas[i] * x;
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



/// computes the weighted jaccard index of 2 signatures.
/// The 2 signatures must come from two equivalent instances of the same ProbMinHash algorithm
/// with the same number of hash signatures 
pub fn compute_probminhash_jaccard<D:Eq>(siga : &Vec<D>, sigb : &Vec<D>) -> f64 {
    let sig_size = siga.len();
    assert_eq!(sig_size, sigb.len());
    let mut inter = 0;
    for i in 0..siga.len() {
        if siga[i] == sigb[i] {
            inter += 1;
        }
    }
    let jp = inter as f64/siga.len() as f64;
    jp
}  // end of compute_probminhash_jaccard

//=================================================================


#[cfg(test)]
mod tests {

use log::*;

use indexmap::{IndexMap};
use fnv::{FnvHasher,FnvBuildHasher};

type FnvIndexMap<K, V> = IndexMap<K, V, FnvBuildHasher>;

#[allow(dead_code)]
fn log_init_test() {
    let _ = env_logger::builder().is_test(true).try_init();
}

use super::*;


    #[test]
    // This tests exponential random sampling in [0,1) 
    // by comparing theoretical mean and estimated mean and checking for deviation 
    // with nb_sampled = 1_000_000_000 we get 
    // mu_th 0.4585059174632017 mean 0.45850733816056904  sigma  0.000009072128699429336 
    // test = (mu_th - mean)/sigma = -0.15660022189165437
    // But as it needs some time we set nb_sampled to 10_000_000. 
    // test is often negative, so mu_th is approximated by above ?. to check
    fn test_exp01() {
        log_init_test();
        //
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(234567 as u64);
        let mut xsi;
        let lambda = 0.5f64;
        let mut mu_th = - lambda * (-lambda).exp() - (-lambda).exp_m1();
        mu_th =  mu_th / (- lambda * (-lambda).exp_m1());
        //
        let nb_sampled = 10_000_000;
        let mut sampled = Vec::<f64>::with_capacity(nb_sampled);
        let exp01 = ExpRestricted01::new(lambda);
        //  
        for _ in 0..nb_sampled {
            xsi = exp01.sample(&mut rng);
            sampled.push(xsi);
        }
        let sum = sampled.iter().fold(0., |acc, x| acc +x);
        let mean = sum / nb_sampled as f64;
        //
        let mut s2 = sampled.iter().fold(0., |acc, x| acc +(x-mean)*(x-mean));
        s2 = s2 / (nb_sampled - 1)  as f64;
        // 
        println!("mu_th {} mean {}  sigma  {} ", mu_th, mean, (s2/nb_sampled as f64).sqrt());
        let test = (mu_th - mean) / (s2/nb_sampled as f64).sqrt();
        println!("test {}", test);
        assert!(test.abs() < 3.);
    }
    #[test]    
    // This test stores random values in a MaxValueTracker and check for max at higher end of array
    fn test_max_value_tracker() {
        log_init_test();
        //
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(45678 as u64);

        let nbhash = 10;
        let unif_01 = Uniform::<f64>::new(0., 1.);
        let unif_m = Uniform::<usize>::new(0, nbhash);

        let mut tracker = MaxValueTracker::new(nbhash);
        //
        let mut vmax = 0f64;
        let loop_size = 500;
        //
        for _ in 0..loop_size {
            let k = unif_m.sample(&mut rng);
            assert!(k < nbhash);
            let xsi = unif_01.sample(&mut rng);
            vmax = vmax.max(xsi);
            tracker.update(k,xsi);
            // check equality of max
            assert!( !( vmax > tracker.get_max_value() && vmax < tracker.get_max_value()) );
            // check for sibling and their parent coherence
        }
        // check for sibling and their parent coherence
        for i in 0..nbhash {
            let sibling = i^1;
            let sibling_value = tracker.get_value(sibling);
            let i_value = tracker.get_value(i);
            let pidx = tracker.get_parent_slot(i);
            let pidx_value = tracker.get_value(pidx);
            assert!(sibling_value <=  pidx_value && i_value <= pidx_value);
            assert!( !( sibling_value > pidx_value  &&   i_value >  pidx_value) );
        }
        assert!(!( vmax > tracker.get_max_value()  && vmax < tracker.get_max_value() ));
//        tracker.dump();
    } // end of test_probminhash_count_range_intersection

    #[test] 
    // This test checks that with equal weights we fall back to Jaccard estimate
    fn test_probminhash3_count_intersection_equal_weights() {
        //
        log_init_test();
        //
        debug!("test_probminhash_count_intersection_equal_weights");
        println!("test_probminhash_count_intersection_equal_weights");
        // we construct 2 ranges [a..b] [c..d], with a<b, b < d, c<d sketch them and compute jaccard.
        // we should get something like max(b,c) - min(b,c)/ (b-a+d-c)
        //
        let set_size = 100;
        let nbhash = 10;
        //
        // choose weights for va and vb elements
        let mut wa = Vec::<f64>::with_capacity(set_size);
        let mut wb = Vec::<f64>::with_capacity(set_size);
        // initialize wa, weight 20 up to 130
        for i in 0..set_size {
            if i < 70 {
                wa.push(20.);
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
                wb.push(10.);
            }
        }

        // compute Jp as in 
        let mut jp = 0.;
        for i in 0..set_size {
            if wa[i] > 0. && wb[i] > 0. {
                let mut den = 0.;
                for j in 0..set_size {
                    den += (wa[j]/wa[i]).max(wb[j]/wb[i]);
                }
                jp += 1./den;
            }
        }
        trace!("Jp = {} ",jp);
        // probminhash 
        trace!("\n\n hashing wa");
        let mut waprobhash = ProbMinHash3::<usize, FnvHasher>::new(nbhash, 0);
        for i in 0..set_size {
            if wa[i] > 0. {
                waprobhash.hash_item(i, wa[i]);
            }
        }
        // waprobhash.maxvaluetracker.dump();
        //
        trace!("\n\n hashing wb");
        let mut wbprobhash = ProbMinHash3::<usize, FnvHasher>::new(nbhash,0 );
        for i in 0..set_size {
            if wb[i] > 0. {
                wbprobhash.hash_item(i, wb[i]);
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
        info!("exact jp = {} ,jp estimated = {} ", jp, jp_approx);
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
    let nbhash = 50;
    //
    // choose weights for va and vb elements
    let mut wa : FnvIndexMap::<usize,f64> = FnvIndexMap::with_capacity_and_hasher(70, FnvBuildHasher::default());
    // initialize wa, weight 20 up to 130
    for i in 0..set_size {
        if i < 70 {
            *wa.entry(i).or_insert(0.) += 2. * i as f64;
        }
    }
    let mut wb : FnvIndexMap::<usize,f64> = FnvIndexMap::with_capacity_and_hasher(70, FnvBuildHasher::default());
    // initialize wb weight 10 above 70
    for i in 0..set_size {
        if i >= 50 {
            wb.entry(i).or_insert((i as f64).powi(4));
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
        if wa_i > 0. &&  wb_i > 0. {
            let mut den = 0.;
            for j in 0..set_size {
                let wa_j = *wa.get(&j).unwrap_or(&0.);
                let wb_j = *wb.get(&j).unwrap_or(&0.);
                den += (wa_j/wa_i).max(wb_j/wb_i);
            }
            jp += 1./den;
        }
    }
    trace!("Jp = {} ",jp);
    //
//    waprobhash.maxvaluetracker.dump();
//    wbprobhash.maxvaluetracker.dump();
    //
    info!("jp exact= {jptheo:.3} , jp estimate = {jp_est:.3} ", jptheo=jp, jp_est=jp_approx);
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
    let nbhash = 50;
    //
    // choose weights for va and vb elements
    let mut wa = Vec::<f64>::with_capacity(set_size);
    let mut wb = Vec::<f64>::with_capacity(set_size);
    // initialize wa, weight 2*i up to 70
    for i in 0..set_size {
        if i < 70 {
            wa.push(2. * i as f64);
        }
        else {
            wa.push(0.);
        }
    }
    // initialize wb weight i^4 above 50
    for i in 0..set_size {
        if i < 50 {
            wb.push(0.);
        }
        else {
            wb.push( (i as f64).powi(4));
        }
    }
    // probminhash 
    trace!("\n\n hashing wa");
    let mut waprobhash = ProbMinHash3::<usize, FnvHasher>::new(nbhash, 0);
    for i in 0..set_size {
        if wa[i] > 0. {
            waprobhash.hash_item(i, wa[i]);
        }
    }
    //
    trace!("\n\n hashing wb");
    let mut wbprobhash = ProbMinHash3::<usize, FnvHasher>::new(nbhash, 0);
    for i in 0..set_size {
        if wb[i] > 0. {
            wbprobhash.hash_item(i, wb[i]);
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
                den += (wa[j]/wa[i]).max(wb[j]/wb[i]);
            }
            jp += 1./den;
        }
    }
    trace!("Jp = {} ",jp);
    //
//    waprobhash.maxvaluetracker.dump();
//    wbprobhash.maxvaluetracker.dump();
    //
    info!("jp exact = {jp_exact:.3} , jp estimate {jp_estimate:.3} ", jp_exact=jp, jp_estimate=jp_approx);
} // end of test_probminhash3_count_intersection_unequal_weights




#[test]
// We check we have a unifom distribution of values at each rank of v
// variance is 5/4
fn test_fyshuffle() {

    log_init_test();

    let m = 4;
    let mut fypermut = FYshuffle::new(m);
    let nb_permut = 500000;
    let mut freq : Vec<usize> = (0..m).map(|_| 0).collect();

    for _ in 0..nb_permut {
        fypermut.next();
        let v = fypermut.get_values();
        for k in 0..v.len() {
            freq[k] += v[k];
        }
    }

    let th_freq = 1.5;
    let th_var = 5./4.;
    let sigma = (th_var/ (nb_permut as f64)).sqrt();
    for i in 0..freq.len() {
        let rel_error = ((freq[i] as f64)/ (nb_permut as f64) - th_freq)/ sigma;
        trace!(" slot i {} , rel error = {}", i, rel_error);
        assert!( rel_error.abs() < 3.)
    }
    trace!("  freq = {:?}", freq);
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
    let mut jp = 0.;
    for i in 0..set_size {
        if wa[i] > 0. && wb[i] > 0. {
            let mut den = 0.;
            for j in 0..set_size {
                den += (wa[j]/wa[i]).max(wb[j]/wb[i]);
            }
            jp += 1./den;
        }
    }
    trace!("Jp = {} ",jp);
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
    let mut inter = 0;
    for i in 0..siga.len() {
        if siga[i] == sigb[i] {
            inter += 1;
        }
    }
    //
    // waprobhash.maxvaluetracker.dump();
    // wbprobhash.maxvaluetracker.dump();
    //
    info!("jp = {} , inter / card = {} ", jp, inter as f64/siga.len() as f64);
} // end of test_probminhash2_count_intersection_unequal_weights



}  // end of module tests