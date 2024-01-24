//! This module implements original minhash algorithm (bottom-k) and is highly inspired by the finch module.
//! The implementation is somewhat more generic as it was designed to hash various type of compressed Kmers or 
//! in fact any type T that satisfies Hash+Clone+Copy.  
//! It implements a variation in the case of Kmer hashed with inversible hash.  
//! Moreover it can just computes Jaccard estimate or keep track of objects hashed.
//! 



// with the inspiration of the  finch module for NoHashHasher.
// perhaps use a bloomfilter instead of Hashmap. Especially if counts are not used in jaccard estimate.






#[allow(unused_imports)]
use log::{debug, trace};

use std::collections::{BinaryHeap, HashMap};

use std::hash::{BuildHasher, BuildHasherDefault, Hasher, Hash};
use std::mem;
use std::marker::PhantomData;

use std::fmt::Debug;
use crate::invhash::*;
use crate::hashed::{HashedItem,HashCount,InvHashedItem,InvHashCount};
use crate::hashed::ItemHash;

/// result of minhash distance computations a tuple for containment, jaccard, common, total
pub struct MinHashDist(pub f64, pub f64, pub u64, pub u64);

pub struct MinHashCount<T: Hash+Clone+Copy+Debug, H: Hasher+Default> {
    // if set to true the hashed item is pushed into HashItem along the hasshed value
    keep_item:bool,
    hashes: BinaryHeap<HashedItem<T>>,
    b_hasher: BuildHasherDefault<H>,
    counts: HashMap<ItemHash, u16, BuildHasherDefault<H>>,
    total_count: u64,
    size: usize,
    // heap_lock: Mutex<()>,
    // instead of map_lock, look into using https://docs.rs/chashmap/2.2
    // map_lock: Mutex<()>,
}



impl <T:Hash + Clone + Copy + Debug ,  H : Hasher+Default> MinHashCount<T, H> {
    /// an allocator , size is capacity measured as  max number of hashed item
    /// keep_item is to ask( or not) to keep the objects (kmers) hashed.
    /// if using an invertible hasher for compressed kmers we do not need to keep track of kmers
    /// as they can be recovered from hashed values.
    pub fn new(size: usize, keep_item: bool) -> Self {
        MinHashCount {
            keep_item: keep_item,
            b_hasher: BuildHasherDefault::<H>::default(),
            hashes: BinaryHeap::with_capacity(size + 1),
            counts: HashMap::with_capacity_and_hasher(size, BuildHasherDefault::<H>::default()),
            total_count: 0,
            size: size,
            // heap_lock: Mutex::new(()),
            // map_lock: Mutex::new(()),
        }
    }  // end of new


    /// push an item in the sketching
    pub fn push(&mut self, item : &T) {
        //
        // hash
        let mut hasher = self.b_hasher.build_hasher();
        item.hash(&mut hasher);
        let new_hash : u64 = hasher.finish();
        //
        // trace!(" pushing item {:?}, hash {}", item, new_hash);
        // do we insert
        let add_hash = match self.hashes.peek() {
            None => true,
            Some(old_max_hash) => (new_hash <= (*old_max_hash).hash) || (self.hashes.len() < self.size),
        };
        // if add_hash is true we must insert in hashes, 
        if add_hash {
            self.total_count += 1;
            if self.counts.contains_key(&new_hash) {
                // the item was already seen once.
                // let _lock = self.map_lock.lock().unwrap();
                let count = self.counts.entry(new_hash).or_insert(0u16);
                (*count) += 1;
                // drop(_lock);
            } else {
                // newhash is encountered for the first time
                // let _ = self.heap_lock.lock().unwrap();
                self.hashes.push(HashedItem {
                    hash: new_hash,
                    item: if self.keep_item {
                        Some(*item)
                    }
                    else {
                        None
                    }
                });
                // 
                self.counts.insert(new_hash, 1u16);
                if self.hashes.len() > self.size {
                    let hashitem = self.hashes.pop().unwrap();
                    let _old_count = self.counts.remove(&hashitem.hash).unwrap();
                }
                // drop(_lock);
                // drop(_map_lock);
            }
        } // end if add_hash        
    } // end push

    /// push a slice in the sketching
    pub fn sketch_slice(&mut self, to_sketch : &[T]) {
        trace!("sketching slice");
        to_sketch.into_iter().for_each(|x| self.push(x));
    } // end of sketch_slice


    /// returns a sorted vecotr of the sketch
    pub fn get_sketchcount(&self) -> Vec<HashCount<T> > {
        trace!("get_sketchcount  got nb hashes : {} ",self.hashes.len());
        let mut results = Vec::with_capacity(self.hashes.len());
        for item in self.hashes.iter() {
            trace!(" got hash : {:?}", item.hash);
            let counts = *self.counts.get(&item.hash).unwrap();
            let counted_item = HashCount {
                hashed: *item,
                count: counts,
            };
            results.push(counted_item);
        }
        results
    }  // end of get_sketchcount

    /// returns if keep_item was set to false
    pub fn get_signature(&self) -> Option<&BinaryHeap<HashedItem<T>> > {
        if self.keep_item == true {
            return None;
        }
        else {
            return Some(&self.hashes);
        }
    } // end of get_signature



}  // end of impl MinHashCount



/// compute different distances from sketch.
pub fn minhash_distance<T:Hash+Clone+Copy>(sketch1: &Vec<HashCount<T> >, sketch2: &Vec<HashCount<T> >) ->  MinHashDist {
    let mut i: usize = 0;
    let mut j: usize = 0;
    let mut common: u64 = 0;
    let mut total: u64 = 0;
    let sketch_size = sketch1.len();
    //
    trace!("sketch1 len : {}, sketch2 len : {}", sketch1.len(), sketch2.len());
    //
    let mut items1 : Vec<HashedItem<T>> = sketch1.iter().map(|x| x.hashed).collect();
    items1.sort_unstable();
    let mut items2 : Vec<HashedItem<T>> = sketch2.iter().map(|x| x.hashed).collect();
    items2.sort_unstable();
    //    
    while i < items1.len() && j < items2.len() {
        if items1[i] < items2[j] {
            i += 1;
        } else if items2[j] < items1[i] {
            j += 1;
        } else {
            i += 1;
            j += 1;
            common += 1;
        }
        total += 1;
        if total >= sketch1.len() as u64 {
            break;
        }
    } // end while
    //
    // try to increase total up to asked sketch size
    //
    if total < items1.len() as u64 {
        // try to increase total.
        if i < items1.len() {
            total += (items1.len() - i) as u64;
        }
        if j < items1.len() {
            total += (items1.len() - j) as u64;
        }
        // now if ever total increase too much we truncate it
        if total > sketch_size as u64 {
            total = sketch_size as u64;
        }            
    }        
    //
    let containment: f64 = common as f64 / i as f64;
    let jaccard: f64 = common as f64 / total as f64;
    MinHashDist(containment, jaccard, common, total)
}  // end of minhash_distance

////////////////////////////////////////////////////////////////////////////////////////:


#[cfg(test)]
mod tests {
    use super::*;
    extern crate fnv;
    #[allow(unused_imports)]
    use self::fnv::FnvHasher; // extern fnv declared in test so we use self::fnv , if declared above we use super::fnv
    #[allow(unused_imports)]
    use crate::nohasher::NoHashHasher;
    use crate::invhash::*;

    fn init_log_test() {
        let _ = env_logger::builder().is_test(true).try_init();
    }



    #[test]
    fn test_minhash_count_range_intersection_fnv() {
        init_log_test();
        // we construct 2 ranges [a..b] [c..d], with a<b, b < d, c<d sketch them and compute jaccard.
        // we should get something like max(b,c) - min(b,c)/ (b-a+d-c)
        //
        let vamax = 300000;
        let va : Vec<usize> = (0..vamax).collect();
        let vbmin = 290000;
        let vbmax = 2.0 * vamax as f64;
        let vb : Vec<usize> = (vbmin..vbmax as usize).collect();
        let inter = vamax - vbmin;
        let jexact = inter as f64 / vbmax as f64;
        let size = 90000;

        let _bh = BuildHasherDefault::<FnvHasher>::default();
        let mut minhash_a : MinHashCount<usize, FnvHasher>= MinHashCount::new(size, true);
        let mut minhash_b : MinHashCount<usize, FnvHasher>= MinHashCount::new(size, true);
        // now compute sketches
        println!("sketching a ");
        minhash_a.sketch_slice(&va);
        println!("\n \nsketching b ");
        minhash_b.sketch_slice(&vb);
        let sketch_a = minhash_a.get_sketchcount();
        let sketch_b = minhash_b.get_sketchcount();
        // 
        let resdist = minhash_distance(&sketch_a, &sketch_b);
        println!(" Exact Jaccard estimate {:.5e}:", jexact);
        log::info!("MinHash estimated distance:{:.5e}", resdist.1);
        if let Some(opthashes) = minhash_a.get_signature() {
            trace!(" nb objects {} ", opthashes.len());
        }
        else {
            trace!("minhash_a.get_signature() returned None");
        }
        // 
        assert!(resdist.2 > 0);
        //
    } // end of test_range_intersection
}  // end of mod test
