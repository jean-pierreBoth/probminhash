//! sampling exponential law with restriction of domain in [0,1)

use rand::distributions::{Distribution, Uniform};
use rand::prelude::*;

/// Structure implemneting exponential sampling of parameter lambda with support restricted
/// to unit interval [0,1).  
// All comments follow notations in Ertl article
#[derive(Clone, Copy, Debug)]
pub struct ExpRestricted01 {
    /// parameter of exponential
    lambda: f64,
    c1: f64,
    // abciss of point for which A3 is under exponential
    c2: f64,
    c3: f64,
    /// we build upon a uniform [0,1) sampling
    unit_range: Uniform<f64>,
} // end of struct ExpRestricted01

impl ExpRestricted01 {
    /// allocates a struct ExpRestricted01 for sampling an exponential law of parameter lambda, but restricted to [0,1.)]
    pub fn new(lambda: f64) -> Self {
        let c1 = lambda.exp_m1() / lambda; // exp_m1 for numerical precision
        let c2 = (2. / (1. + (-lambda).exp())).ln() / lambda;
        let c3 = (1. - (-lambda).exp()) / lambda;
        ExpRestricted01 {
            lambda,
            c1,
            c2,
            c3,
            unit_range: Uniform::<f64>::new(0., 1.),
        }
    }

    /// return lambda parameter of exponential
    pub fn get_lambda(&self) -> f64 {
        self.lambda
    }
}

impl Distribution<f64> for ExpRestricted01 {
    /// sample from ExpRestricted01
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let mut x = self.c1 * rng.sample(self.unit_range);
        if x < 1. {
            return x;
        }
        loop {
            // check if we can sample in A3
            x = rng.sample(self.unit_range);
            if x < self.c2 {
                return x;
            }
            //
            let mut y = 0.5 * rng.sample(self.unit_range);
            if y > 1. - x {
                // transform a point in A5 to a point in A6
                x = 1. - x;
                y = 1. - y;
            }
            if x <= self.c3 * (1. - y) {
                return x;
            }
            if self.c1 * y <= (1. - x) {
                return x;
            }
            if y * self.c1 * self.lambda <= (self.lambda * (1. - x)).exp_m1() {
                return x;
            }
        }
    } // end sample
}

#[cfg(test)]
mod tests {

    use rand::distributions::Distribution;
    use rand::prelude::*;
    use rand_xoshiro::Xoshiro256PlusPlus;

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
        let mut mu_th = -lambda * (-lambda).exp() - (-lambda).exp_m1();
        mu_th /= -lambda * (-lambda).exp_m1();
        //
        let nb_sampled = 10_000_000;
        let mut sampled = Vec::<f64>::with_capacity(nb_sampled);
        let exp01 = ExpRestricted01::new(lambda);
        //
        for _ in 0..nb_sampled {
            xsi = exp01.sample(&mut rng);
            sampled.push(xsi);
        }
        let sum = sampled.iter().fold(0., |acc, x| acc + x);
        let mean = sum / nb_sampled as f64;
        //
        let mut s2 = sampled
            .iter()
            .fold(0., |acc, x| acc + (x - mean) * (x - mean));
        s2 = s2 / (nb_sampled - 1) as f64;
        //
        println!(
            "mu_th {} mean {}  sigma  {} ",
            mu_th,
            mean,
            (s2 / nb_sampled as f64).sqrt()
        );
        let test = (mu_th - mean) / (s2 / nb_sampled as f64).sqrt();
        println!("test {}", test);
        assert!(test.abs() < 3.);
    }
} // end of mod tests
