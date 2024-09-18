//! max value tracker

use log::trace;

// A value for which there is a (natural) maximal value
pub trait MaxValue {
    //
    fn get_max() -> Self;
} // end of trait Max

macro_rules! implement_maxvalue_for(
    ($ty:ty) => (
        impl MaxValue for $ty {
            fn get_max() -> $ty {
                <$ty>::MAX
            }
        }
    ) // end impl for ty
);

implement_maxvalue_for!(f64);
implement_maxvalue_for!(f32);
implement_maxvalue_for!(u32);
implement_maxvalue_for!(u16);
implement_maxvalue_for!(i32);
implement_maxvalue_for!(u64);
implement_maxvalue_for!(usize);

// structure to keep track of max values in hash set
// adapted from class MaxValueTracker
pub(crate) struct MaxValueTracker<V> {
    m: usize,
    // last_index = 2*m-2. max of array is at slot last_index
    last_index: usize,
    // dimensioned to m hash functions
    values: Vec<V>,
}

impl<V> MaxValueTracker<V>
where
    V: MaxValue + PartialOrd + Copy + std::fmt::Debug,
{
    pub fn new(m: usize) -> Self {
        let last_index = (m << 1) - 2; // 0-indexation for the difference with he paper, lastIndex = 2*m-2
        let vlen = last_index + 1;
        let values: Vec<V> = (0..vlen).map(|_| V::get_max()).collect();
        MaxValueTracker {
            m,
            last_index,
            values,
        }
    }

    // update slot k with value value
    // 0 indexation imposes some changes with respect to the the algo 4 of the paper
    // parent of k is m + (k/2)
    // and accordingly
    // sibling ok k is k+1 if k even, k-1 else so it is given by bitxor(k,1)
    pub(crate) fn update(&mut self, k: usize, value: V) {
        assert!(k < self.m);
        trace!(
            "\n max value tracker update k, value , value at k {} {:?} {:?} ",
            k,
            value,
            self.values[k]
        );
        let mut current_value = value;
        let mut current_k = k;
        let mut more = false;
        if current_value < self.values[current_k] {
            more = true;
        }

        while more {
            trace!("mxvt update k value {} {:?}", current_k, current_value);
            self.values[current_k] = current_value;
            let pidx = self.m + (current_k / 2); // m + upper integer value of k/2 beccause of 0 based indexation
            if pidx > self.last_index {
                break;
            }
            let siblidx = current_k ^ 1; // get sibling index of k with numeration beginning at 0
            assert!(self.values[siblidx] <= self.values[pidx]);
            assert!(self.values[current_k] <= self.values[pidx]);
            //
            if self.values[siblidx] >= self.values[pidx]
                && self.values[current_k] >= self.values[pidx]
            {
                break; // means parent current and sibling are equals no more propagation needed
            }
            // now either self.values[siblidx] <self.values[pidx] or current_value < self.values[pidx]
            trace!(
                "propagating current_value {:?} sibling  {:?} ? ",
                current_value,
                self.values[siblidx]
            );
            //
            if current_value < self.values[siblidx] {
                trace!(
                    "     propagating sibling value {:?} to parent {}",
                    self.values[siblidx],
                    pidx
                );
                current_value = self.values[siblidx];
            } else {
                trace!(
                    "     propagating current_value {:?} to parent {}",
                    current_value,
                    pidx
                );
            }
            current_k = pidx;
            if current_value >= self.values[current_k] {
                more = false;
            }
        }
    } // end of update function

    /// return the maximum value maintained in the data structure
    pub fn get_max_value(&self) -> V {
        self.values[self.last_index]
    }

    // returns true if a value can be inserted, false it is too high
    pub fn is_update_possible(&self, value: V) -> bool {
        value < self.values[self.last_index]
    } // end of is_update_possible

    #[allow(dead_code)]
    pub(crate) fn get_parent_slot(&self, slot: usize) -> usize {
        assert!(slot <= self.m);
        self.m + (slot / 2) // m + upper integer value of k/2 beccause of 0 based indexation
    }

    /// get value MaxValueTracker at slot
    pub(crate) fn get_value(&self, slot: usize) -> V {
        self.values[slot]
    } // end of get_value

    /// reset to max value f64::MAX
    pub(crate) fn reset(&mut self) {
        self.values.fill(V::get_max());
    }

    #[allow(unused)]
    pub fn dump(&self) {
        println!("\n\nMaxValueTracker dump : ");
        for i in 0..self.values.len() {
            println!(" i  value   {}   {:?} ", i, self.values[i]);
        }
    } // end of dump
} // end of impl MaxValueTracker

#[cfg(test)]
mod tests {

    use rand::distributions::{Distribution, Uniform};
    use rand::prelude::*;
    use rand_xoshiro::Xoshiro256PlusPlus;

    use super::*;

    fn log_init_test() {
        let _ = env_logger::builder().is_test(true).try_init();
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
            tracker.update(k, xsi);
            // check equality of max
            assert!(!(vmax > tracker.get_max_value() && vmax < tracker.get_max_value()));
            // check for sibling and their parent coherence
        }
        // check for sibling and their parent coherence
        for i in 0..nbhash {
            let sibling = i ^ 1;
            let sibling_value = tracker.get_value(sibling);
            let i_value = tracker.get_value(i);
            let pidx = tracker.get_parent_slot(i);
            let pidx_value = tracker.get_value(pidx);
            assert!(sibling_value <= pidx_value && i_value <= pidx_value);
            assert!(!(sibling_value > pidx_value && i_value > pidx_value));
        }
        assert!(!(vmax > tracker.get_max_value() && vmax < tracker.get_max_value()));
        //        tracker.dump();
    } // end of test_max_value_tracker
}
