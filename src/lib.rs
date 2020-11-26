extern crate rand;

// for logging (debug mostly, switched at compile time in cargo.toml)
#[macro_use]
extern crate lazy_static;



pub mod probminhasher;
pub mod superminhasher;
pub mod invhash;
pub mod nohasher;

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
