//! Implementation of ProbMinHash2, ProbMinHash3 and ProbMinHash3a as described in O. Ertl  
//! <https://arxiv.org/abs/1911.00675>.  
//!
//! * ProbminHash3a is the fastest but at the cost of some internal storage.
//! * Probminhash3 is the same algorithm without the time optimization requiring more storage.  
//!   It can be used in streaming
//! * Probminhash2 is statistically equivalent to P-Minhash as described in :
//!   Moulton Jiang "Maximally consistent sampling and the Jaccard index of probability distributions"
//!   <https://ieeexplore.ieee.org/document/8637426> or <https://arxiv.org/abs/1809.04052>.  
//!   It is given as a fallback in case ProbminHash3* algorithms do not perform well, or for comparison.
//!  
//! * ProbMinHash3aSha is a variation of probminhash3a dedicated to hashing of types not implementing Copy.  
//!   This implementation uses Sha512_256 hashing for initialization the random generator (Xoshiro256PlusPlus) with 256 bits seed and
//!   reduces the risk of collisions.  
//!   Counted objects must satisfy the trait **Sig** instead of **Hash** for the preceding algorithms, but they do not need to satisfy Copy.
//!   It is more adapted to hashing Strings or Vec\<u8\>  
//!
//! * ProbOrminhash2 is a locality-sensitive hashing for the edit distance implemented over ProbMinHash2 as in  Ertl's [probordminhash2](https://github.com/oertl/probminhash).
//!   It is inspired by *Marcais.G et al. BioInformatics 2019*.  Cf <https://academic.oup.com/bioinformatics/article/35/14/i127/5529166>
//!

pub mod probminhash2;
pub mod probminhash3;
pub mod probminhash3sha;
pub mod probordminhash2;
pub mod sig;

pub use probminhash2::ProbMinHash2;
pub use probminhash3::{ProbMinHash3, ProbMinHash3a};
pub use probminhash3sha::ProbMinHash3aSha;
