# Some Minhash related algorithms

This crate provides implementation of some recent algorithms deriving from the original Minhash. They have better performance and are more general.  

It implements:

* ProbMinHash2, ProbMinHash3 and ProbMinHash3a as described in O. Ertl paper:
**ProbMinHash. A Class of of Locality-Sensitive Hash Algorithms for the Probability Jaccard Similarity (2020)**
[probminhash Ertl](https://arxiv.org/abs/1911.00675) or [IEEE-2022](https://ieeexplore.ieee.org/document/9185081)

These algorithms compute an estimation of the Jaccard weighted index via sensitive hashing.
It is an extension of the Jaccard index to the case where objects have a weight, or a multiplicity associated.  
This Jaccard  weighted index provides a metric on discrete probability distributions as explained in :
**Moulton Jiang. Maximally consistent sampling and the Jaccard index of probability distributions (2018)**
[Moulton-Jiang-ieee](https://ieeexplore.ieee.org/document/8637426) or [Moulton-Jiang-arxiv](https://arxiv.org/abs/1809.04052)

Noting *Jp* the Jaccard weighted index, then  *1. - Jp* defines a metric on finite discrete probabilities.  
This module is the core of the crate which has two other modules.

* Superminhash

An implementation of Superminhash :  
**A new minwise Hashing Algorithm for Jaccard Similarity Estimation**
Otmar Ertl 2017-2018 Cf [superminhash Ertl](https://arxiv.org/abs/1706.05698)

This algorithm runs on unweighted objects and can sketch on a laptop billions of objects into f32/f64 vectors.
The hash values are computed by the *sketch* method or can be computed before entering SuperMinHash methods.
  
It runs in one pass on data so it can be used in streaming.  

A variant of this algorithm, **Superminhash2**, sketch data into u32/u64 vectors but is slower. It is accessed with the **sminhash2** feature.

* SetSketch  
  
An implementation of the SetSketch :
**SetSketch: Filling the gap between MinHash and HyperLogLog**
Otmar Ertl 2021 [arxiv](https://arxiv.org/abs/2101.00314) or [vldb](https://vldb.org/pvldb/vol14/p2244-ertl.pdf)

This algorithm runs on unweighted objects. It is slower than SuperMinHash but can sketch billions of objects into vectors of 16 bytes integers. Morever sketches are mergeable.  
We provide sketching (adapted to LSH with Jaccard distance) and a cardinality estimator of the sketched set.

* ProbOrdMinHash2 is a locality-sensitive hashing for the edit distance implemented over ProbMinHash2 as in  Ertl's [probordminhash2](https://github.com/oertl/probminhash).  
It is inspired by *Marcais.G et al. BioInformatics 2019*, see  [Marcais](https://academic.oup.com/bioinformatics/article/35/14/i127/5529166)

* Invhash
  
It is just a module providing invertible hash from u32 to u32 or u64 to u64 and can be used to run a prehash on indexes.
(See reference to Thomas Wang's invertible integer hash functions in invhash.rs)

## Some examples

Some example of usage (more in the tests in each module) consisting to estimate intersection of contents of 2 vectors:

* Probminhash
  
An example of Prominhash3a with an IndexMap  
(see test probminhash::tests::test_probminhash3a_count_intersection_unequal_weights)

```rust
    type FnvIndexMap<K, V> = IndexMap<K, V, FnvBuildHasher>;
    ...
    let mut wa : FnvIndexMap::<usize,f64> = FnvIndexMap::with_capacity_and_hasher(70, FnvBuildHasher::default());
    // initialize wa ...

    let mut wb : FnvIndexMap::<usize,f64> = FnvIndexMap::with_capacity_and_hasher(70, FnvBuildHasher::default());
    // initialize ...
    let mut waprobhash = ProbMinHash3a::<usize, FnvHasher>::new(nbhash, 0);
    waprobhash.hash_weigthed_idxmap(&wa);
    //
    let mut wbprobhash = ProbMinHash3a::<usize, FnvHasher>::new(nbhash, 0);
    wbprobhash.hash_weigthed_idxmap(&wb);
    //
    let siga = waprobhash.get_signature();
    let sigb = wbprobhash.get_signature();
    let jp_approx = compute_probminhash_jaccard(siga, sigb);
```

An example of Probminhash3 with items sent one by one:

```rust
    let set_size = 100;
    let mut wa = Vec::<f64>::with_capacity(set_size);
    let mut wb = Vec::<f64>::with_capacity(set_size);
    // initialize wa, wb
    ....
    // probminhash
    let mut waprobhash = ProbMinHash3::<usize, FnvHasher>::new(nbhash, 0);
    for i in 0..set_size {
        if wa[i] > 0. {
            waprobhash.hash_item(i, wa[i]);
        }
    }
    //
    let mut wbprobhash = ProbMinHash3::<usize, FnvHasher>::new(nbhash, 0);
    for i in 0..set_size {
        if wb[i] > 0. {
            wbprobhash.hash_item(i, wb[i]);
        }
    }
    let siga = waprobhash.get_signature();
    let sigb = wbprobhash.get_signature();
    let jp_approx = compute_probminhash_jaccard(siga, sigb);
```

* Superminhash

```rust
      let va : Vec<usize> = (0..1000).collect();
      let vb : Vec<usize> = (900..2000).collect();
      let bh = BuildHasherDefault::<FnvHasher>::default();
      let mut sminhash : SuperMinHash<usize, FnvHasher>= SuperMinHash::new(70, &bh);
      // now compute sketches
      let resa = sminhash.sketch_slice(&va);
      // we decide to reuse sminhash instead of allocating another SuperMinHash structure
      let ska = sminhash.get_hsketch().clone();
      sminhash.reinit();
      let resb = sminhash.sketch_slice(&vb);
      let skb = sminhash.get_hsketch();
      //
      let jac = get_jaccard_index_estimate(&ska, &skb).unwrap();
        ...
```

## License

Licensed under either of

* Apache License, Version 2.0, [LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>
* MIT license [LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>

at your option.

This software was written on my own while working at [CEA](http://www.cea.fr/)
