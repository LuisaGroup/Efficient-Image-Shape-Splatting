use super::pt::{self, PathTracer};

use crate::*;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
#[serde(tag = "type")]
pub enum Method {
    #[serde(rename = "kelemen")]
    Kelemen {
        exponential_mutation: bool,
        small_sigma: f32,
        large_step_prob: f32,
        image_mutation_prob: f32,
        image_mutation_size: Option<f32>,
        adaptive: bool,
    },
}
impl Default for Method {
    fn default() -> Self {
        Method::Kelemen {
            exponential_mutation: true,
            small_sigma: 0.01,
            large_step_prob: 0.1,
            image_mutation_prob: 0.0,
            image_mutation_size: None,
            adaptive: false,
        }
    }
}
pub struct Mcmc {
    pub device: Device,
    pub pt: PathTracer,
    pub method: Method,
    pub n_chains: usize,
    pub n_bootstrap: usize,
    pub mcmc_depth: u32,
    config: Config,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(crate = "serde")]
#[serde(default)]
pub struct Config {
    pub spp: u32,
    pub max_depth: u32,
    pub rr_depth: u32,
    pub mcmc_depth: Option<u32>,
    pub spp_per_pass: u32,
    pub use_nee: bool,
    pub method: Method,
    pub n_chains: usize,
    pub n_bootstrap: usize,
    pub direct_spp: i32,
    pub wis:bool,
    pub seed:u64,
}
impl Default for Config {
    fn default() -> Self {
        let default_pt = pt::Config::default();
        Self {
            spp: default_pt.spp,
            spp_per_pass: default_pt.spp_per_pass,
            use_nee: default_pt.use_nee,
            max_depth: default_pt.max_depth,
            mcmc_depth: None,
            rr_depth: default_pt.rr_depth,
            method: Method::default(),
            n_chains: 512,
            n_bootstrap: 100000,
            direct_spp: 64,
            wis:false,
            seed:0,
        }
    }
}
