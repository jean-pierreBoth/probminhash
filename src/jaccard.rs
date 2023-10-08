//! jaccard distance



/// Computes the weighted jaccard index of 2 signatures.
///   
/// The 2 signatures must come from two equivalent instances of the same ProbMinHash algorithm
/// with the same number of hash signatures.  
/// Note that if *jp* is the returned value of this function,  
/// the distance between siga and sigb, associated to the jaccard index is *1.- jp* 
pub fn compute_probminhash_jaccard<D:PartialEq>(siga : &[D], sigb : &[D]) -> f64 {
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



/// Computes the weighted jaccard index of 2 signatures.
///   
/// The 2 signatures must come from two equivalent instances of the same ProbMinHash algorithm
/// with the same number of hash signatures.  
/// Note that if *jp* is the returned value of this function,  
/// the distance between siga and sigb, associated to the jaccard index is *1.- jp* 
pub fn get_jaccard_index_estimate<F: PartialEq + std::fmt::Debug>(siga: &[F]  , sigb: &[F])  -> Result<f64, ()>  {
    let sig_size = siga.len();
    assert_eq!(sig_size, sigb.len());
    let mut inter = 0;
    for i in 0..siga.len() {
        if siga[i] == sigb[i] {
            inter += 1;
        }
    }
    let jp = inter as f64/siga.len() as f64;
    Ok(jp)
}
