//! Basic stuff about hashed items

use std::cmp::Ordering;
use std::marker::PhantomData;


/// We use maximum size to store hash value but with invertible 32 hash
/// the value stored is in fact a u32.
/// We would like to template over item hash but Hasher has u64 as arrival type
pub type ItemHash = u64;



// If we use an inversible hash we do not need to keep item (the kmer)
// for other hash  we need copying and storing of Kmer... whence the Option<T> field

/// A HashedItem is a hashed item and possibly the associated object (of type T) if
/// we want to keep track of objects contributiong to minhash signature.
/// Note that using invertible hash if objects hashes 
/// are stored in a u32 or a u64 (as in some Kmer representation) we can retrive objects
/// from hashed value. (See module invhash)
#[derive(Debug,Clone,Copy)]
pub struct HashedItem<T:Clone+Copy> {
    pub(crate) hash: ItemHash,
    ///
#[allow(unused)]
    pub(crate) item: Option<T>,
}

impl<T:Clone+Copy> PartialEq for HashedItem<T> {
    fn eq(&self, other: &HashedItem<T>) -> bool {
        other.hash.eq(&self.hash)
    }
}

impl<T:Clone+Copy> Eq for HashedItem<T> {}

impl<T:Clone+Copy> Ord for HashedItem<T> {
    fn cmp(&self, other: &HashedItem<T>) -> Ordering {
        self.hash.cmp(&other.hash)
    }
}

impl<T:Clone+Copy> PartialOrd for HashedItem<T> {
    fn partial_cmp(&self, other: &HashedItem<T>) -> Option<Ordering> {
        Some(self.hash.cmp(&other.hash))
    }
}



// size is 2*8+2 bytes !!
/// to store count of object
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HashCount<T:Clone+Copy> {
    pub hashed: HashedItem<T>,
    pub count: u16,
}


//===================   For items hashed by invertible hash


/// possibly something can be hashed with some inversible hash so we do not need to store the original item
#[derive(Debug,Clone)]
pub struct InvHashedItem<T:Clone+Copy> {
    pub(crate) hash: ItemHash,
    pub(crate) t_marker: PhantomData<T>,
}



impl <T:Clone+Copy> InvHashedItem<T> {

    pub fn new(hash:ItemHash) -> Self {
        InvHashedItem{hash:hash, t_marker:PhantomData,}
    }
    pub fn get_hash(&self) -> ItemHash {
        return self.hash;
    } // end of get impl
    
}  // end of impl InvHashedItem




impl<T:Copy+Clone> PartialEq for InvHashedItem<T> {
    fn eq(&self, other: &InvHashedItem<T>) -> bool {
        other.hash.eq(&self.hash)
    }
}



impl<T:Copy+Clone> Eq for InvHashedItem<T> {}


impl<T:Clone+Copy> Ord for InvHashedItem<T> {
    fn cmp(&self, other: &InvHashedItem<T>) -> Ordering {
        self.hash.cmp(&other.hash)
    }
}


impl<T:Clone+Copy> PartialOrd for InvHashedItem<T> {
    fn partial_cmp(&self, other: &InvHashedItem<T>) -> Option<Ordering> {
        Some(self.hash.cmp(&other.hash))
    }
}


//====================================================================================//


/// To count occurences of an inversible hashed objects
// size is 8 + 1 bytes !!
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InvHashCount<T:Clone+Copy> {
    pub hashed: InvHashedItem<T>,
    pub(crate) count: u8,
}

impl <T:Clone+Copy> InvHashCount<T> {
    pub fn new(hashed: InvHashedItem<T>, count:u8) -> Self {
        InvHashCount { hashed: hashed, count:count,}
    }
    pub fn get_count(&self) -> u8 {
        self.count
    }
} // end of impl block for InvHashCount
