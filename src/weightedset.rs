//! trait weighted set
//! 
//! 

/// A Trait to define association of a weight to an object.
/// Typically we could implement trait WeightedSet for any collection of Object if we have a function giving a weight to each object
/// Then hash_wset function can be used.
pub trait WeightedSet {
    type Object;
    /// returns the weight of an object
    fn get_weight(&self, obj:&Self::Object) -> f64;
}
