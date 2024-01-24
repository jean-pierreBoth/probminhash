extern crate rand;

// for logging (debug mostly, switched at compile time in cargo.toml)
#[macro_use]
extern crate lazy_static;


mod maxvaluetrack;

pub mod jaccard;

pub mod probminhasher;

pub mod superminhasher;
pub mod setsketcher;
pub mod densminhash;
pub mod minhash;

pub mod exp01;
pub mod fyshuffle;
pub mod weightedset;
// we keep it in case. give integer signature but slower!
#[cfg(feature="sminhash2")]
pub mod superminhasher2;

pub mod invhash;
pub mod nohasher;
pub mod hashed;

// hashing stuff


lazy_static! {
    #[allow(dead_code)]
    pub static ref LOG: u64 = {
        let res = init_log();
        res
    };
}
// install a logger facility
// set RUST_LOG to trace, warn debug off ....
fn init_log() -> u64 {
    env_logger::Builder::from_default_env().init();
    println!("\n ************** initializing logger from env *****************\n");    
    return 1;
}
