use std::collections::HashSet;
use std::f32::consts::PI;
use std::fs::File;
use std::io::BufWriter;
use std::process::abort;
use std::sync::Arc;
use std::time::Instant;

use akari_render::rand::Rng;
use akari_render::svm::surface::{BsdfEvalContext, Surface};
use akari_render::util::distribution::{AliasTableEntry, BindlessAliasTableVar};
use luisa::device_log;
use luisa::lang::debug::{comment, is_cpu_backend};
use luisa::lang::external::CpuFn;
use luisa::runtime::KernelBuildOptions;

use super::mcmc::Method;
use super::pt::{self, PathTracer};
use super::{Integrator, IntermediateStats, RenderSession, RenderStats};
use crate::pt::{PathTracerBase, ReconnectionShiftMapping, ReconnectionVertex};
use crate::sampler::mcmc::{mutate_image_space_single, KelemenMutationRecord, KELEMEN_MUTATE};
use crate::sampling::sample_gaussian;
use crate::shape_splat_pt::ShapeComps;
use crate::util::distribution::{resample_with_f64, AliasTable};
use crate::util::{is_power_of_four, morton2d};
use crate::{color::*, film::*, sampler::*, scene::*, *};
use serde::*;
pub type PssBuffer = Buffer<PssSample>;
use super::shape_splat_pt::{BaseShape, MixingMethod, Shape, SpatialCurve};
#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(default, crate = "serde")]
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
    pub max_sampled_length: Option<u32>,
    pub incremental: bool,
    pub stride: u32,
    pub uniform: bool,
    pub reconnect: bool,
    pub shape_width: u32,
    pub denoiser_kernel: bool,
    pub curve: SpatialCurve,
    pub shape: BaseShape,
    pub mix: MixingMethod,
    pub async_compile: bool,
    pub sample_all: bool,
    pub seed: u64,
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
            n_chains: 8192,
            n_bootstrap: 100000,
            direct_spp: 64,
            shape_width: 6,
            max_sampled_length: None,
            mix: MixingMethod::None,
            shape: BaseShape::Square,
            curve: SpatialCurve::Hilbert,
            denoiser_kernel: true,
            incremental: true,
            stride: 2,
            uniform: false,
            reconnect: true,
            async_compile: true,
            sample_all: false,
            seed: 0,
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Value, Debug, Soa)]
#[luisa(crate = "luisa")]
#[value_new(pub)]
pub struct PssSample {
    pub cur: f32,
    pub backup: f32,
    pub last_modified: u32,
    pub modified_backup: u32,
}
pub struct ShapeSplattingPssmlt {
    pub device: Device,
    pub pt: PathTracer,
    pub method: Method,
    pub n_chains: usize,
    pub n_bootstrap: usize,
    pub mcmc_depth: u32,
    config: Config,
}

#[derive(Clone, Copy, Debug, Value)]
#[luisa(crate = "luisa")]
#[repr(C)]
#[value_new(pub)]
pub struct MarkovState {
    cur_pixel: Uint2,
    chain_id: u32,
    cur_f: f32,
    b: f32,
    b_cnt: u32,
    n_accepted: u32,
    n_mutations: u32,
    cur_iter: u32,
    last_large_iter: u32,
    cur_dim: u32,
}
struct RenderState {
    t_lo: i32,
    t_hi: i32,
    base_shape_center: u32,
    n_shift_pixels: u32,
    avg_per_pixel: f32,
    rng_states: Buffer<Pcg32>,
    samples: PssBuffer,
    states: Buffer<MarkovState>,
    cur_colors: ColorBuffer,
    b_init: f64,
    b_init_cnt: usize,
    pmf_ats: BindlessArray,
    tmp_color: ColorBuffer,
    path_jacobians: Buffer<f32>,
    sampled_indices: Buffer<i32>,
    sort_tmp: Buffer<i32>,
    shapes: Buffer<Shape>,
    shape_indices: Buffer<i8>,
    vertices: Buffer<ReconnectionVertex>,
    proposal_vertices: Buffer<ReconnectionVertex>,
    curve: Buffer<Int2>,
    curve_data: Vec<Int2>,
    albedo_buf: Tex2d<Float4>,
    normal_buf: Tex2d<Float4>,
    avgs_per_pixel: Buffer<f32>,
}
pub struct ReplaySampler<'a> {
    pub base: &'a IndependentSampler,
    pub samples: &'a PssBuffer,
    pub offset: Expr<u32>,
    pub cur_dim: Var<u32>,
    pub replay_dim: Expr<u32>,
}
impl<'a> Sampler for ReplaySampler<'a> {
    #[tracked(crate = "luisa")]
    fn next_1d(&self) -> Expr<f32> {
        if self.cur_dim.load().lt(self.replay_dim) {
            let ret = self.samples.var().read(self.offset + self.cur_dim).cur;
            *self.cur_dim += 1;
            ret
        } else {
            *self.cur_dim += 1;
            self.base.next_1d()
        }
    }
    fn is_metropolis(&self) -> bool {
        false
    }
    fn uniform(&self) -> Expr<f32> {
        self.base.next_1d()
    }
    fn start(&self) {
        self.cur_dim.store(0);
    }
    fn clone_box(&self) -> Box<dyn Sampler> {
        todo!()
    }
    fn forget(&self) {
        todo!()
    }
}
fn randomize_shape(base_curve: &[Int2], spp: u32) -> Vec<Int2> {
    let mut rng = rand::thread_rng();
    let stride = 1.0;
    // let rotation = rng.gen_range(0.0..std::f32::consts::PI);
    let perturb = rng.gen_range(-0.1..0.1);
    let rotation = (perturb + 22.5f32).to_radians() * (3 * spp) as f32;
    let rotation_matrix = glam::Mat2::from_angle(rotation);
    base_curve
        .iter()
        .map(|p| {
            let p = glam::ivec2(p.x, p.y);
            let p = rotation_matrix * p.as_vec2() * stride;
            let p = p.round().as_ivec2();
            Int2::new(p.x, p.y)
        })
        .collect()
}
pub struct LazyMcmcSampler<'a> {
    pub base: &'a IndependentSampler,
    pub samples: &'a PssBuffer,
    pub offset: Expr<u32>,
    pub cur_dim: Var<u32>,
    pub mcmc_dim: Expr<u32>,
    pub mutator: Option<Mutator>,
}
impl<'a> LazyMcmcSampler<'a> {
    pub fn new(
        base: &'a IndependentSampler,
        samples: &'a PssBuffer,
        offset: Expr<u32>,
        mcmc_dim: Expr<u32>,
        mutator: Option<Mutator>,
    ) -> Self {
        Self {
            base,
            samples,
            offset,
            cur_dim: 0u32.var(),
            mcmc_dim,
            mutator,
        }
    }
}

impl<'a> Sampler for LazyMcmcSampler<'a> {
    #[tracked(crate = "luisa")]
    fn next_1d(&self) -> Expr<f32> {
        if self.cur_dim.load().lt(self.mcmc_dim) {
            let ret = if let Some(m) = &self.mutator {
                m.mutate_one(self.samples, self.offset, **self.cur_dim, self.base)
                    .cur
            } else {
                self.samples.var().read(self.offset + self.cur_dim).cur
            };
            *self.cur_dim += 1;
            ret
        } else {
            *self.cur_dim += 1;
            self.base.next_1d()
        }
    }
    fn is_metropolis(&self) -> bool {
        false
    }
    fn uniform(&self) -> Expr<f32> {
        self.base.next_1d()
    }
    fn start(&self) {
        self.cur_dim.store(0);
    }
    fn clone_box(&self) -> Box<dyn Sampler> {
        todo!()
    }
    fn forget(&self) {
        todo!()
    }
}
pub struct Mutator {
    pub method: Method,
    pub is_large_step: Expr<bool>,
    pub is_image_mutation: Expr<bool>,
    pub last_large_iter: Expr<u32>,
    pub cur_iter: Expr<u32>,
    pub res: Expr<Float2>,
}
impl Mutator {
    #[tracked(crate = "luisa")]
    pub fn mutate_one(
        &self,
        samples: &PssBuffer,
        offset: Expr<u32>,
        i: Expr<u32>, // dim
        rng: &IndependentSampler,
    ) -> Expr<PssSample> {
        match self.method {
            Method::Kelemen {
                exponential_mutation,
                small_sigma,
                image_mutation_size,
                image_mutation_prob,
                ..
            } => {
                let kelemen_mutate_size_low = 1.0 / 1024.0f32;
                let kelemen_mutate_size_high = 1.0 / 64.0f32;
                let kelemen_log_ratio = -(kelemen_mutate_size_high / kelemen_mutate_size_low).ln();
                let ret = Var::<PssSample>::zeroed();
                maybe_outline(|| {
                    let sample = samples.var().read(offset + i).var();
                    comment("mcmc mutate_one");
                    let u = rng.next_1d();
                    if sample.last_modified.lt(self.last_large_iter) {
                        *sample.cur = rng.next_1d();
                        *sample.last_modified = self.last_large_iter;
                    };

                    *sample.backup = sample.cur;
                    *sample.modified_backup = sample.last_modified;
                    if self.is_large_step {
                        *sample.cur = u;
                    } else {
                        let is_cur_dim_under_image_mutation =
                            image_mutation_size.is_some() & self.is_image_mutation;
                        let should_cur_dim_be_mutated = !is_cur_dim_under_image_mutation | i.lt(2);
                        let target_iter = if should_cur_dim_be_mutated {
                            self.cur_iter
                        } else {
                            self.cur_iter - 1
                        };
                        lc_assert!(target_iter.ge(sample.last_modified));
                        let n_small = target_iter - sample.last_modified;
                        if exponential_mutation {
                            let x = sample.cur.var();
                            for_range(0u32.expr()..n_small, |_| {
                                let u = rng.next_1d();
                                if u.lt(1.0 - image_mutation_prob) {
                                    let u = u / (1.0 - image_mutation_prob);
                                    let record = KelemenMutationRecord::new_expr(
                                        **x,
                                        u,
                                        kelemen_mutate_size_low,
                                        kelemen_mutate_size_high,
                                        kelemen_log_ratio,
                                        0.0,
                                    )
                                    .var();
                                    KELEMEN_MUTATE.call(record);
                                    x.store(**record.mutated);
                                };
                            });
                            *sample.cur = x;
                        } else {
                            if n_small.gt(0) {
                                // let tmp1 = (-2.0 * (1.0 - rng.next_1d()).ln()).sqrt();
                                // let dv = tmp1 * (2.0 * PI * rng.next_1d()).cos();
                                let dv = sample_gaussian(u);
                                let new = sample.cur
                                    + dv * small_sigma
                                        * ((1.0 - image_mutation_prob) * n_small.cast_f32()).sqrt();
                                let new = new - new.floor();
                                let new = select(new.is_finite(), new, 0.0f32.expr());
                                *sample.cur = new;
                            };
                        };
                        if image_mutation_size.is_some() {
                            if self.is_image_mutation & i.lt(2) {
                                let new = mutate_image_space_single(
                                    **sample.cur,
                                    rng,
                                    image_mutation_size.unwrap().expr(),
                                    self.res,
                                    i,
                                );
                                *sample.cur = new;
                            };
                        }
                    };
                    *sample.last_modified = self.cur_iter;
                    samples.var().write(offset + i, **sample);
                    *ret = sample;
                });
                **ret
            }
        }
    }
}
impl ShapeSplattingPssmlt {
    fn sample_dimension(&self) -> usize {
        4 + 1 + (1 + self.mcmc_depth as usize) * (3 + 3 + 1)
    }
    pub fn new(device: Device, config: Config) -> Self {
        let pt_config = pt::Config {
            spp: config.spp,
            max_depth: config.max_depth,
            spp_per_pass: config.spp_per_pass,
            use_nee: config.use_nee,
            rr_depth: config.rr_depth,
            indirect_only: config.direct_spp >= 0,
            ..Default::default()
        };
        Self {
            device: device.clone(),
            pt: PathTracer::new(device.clone(), pt_config),
            method: config.method,
            n_chains: config.n_chains,
            n_bootstrap: config.n_bootstrap,
            mcmc_depth: config.mcmc_depth.unwrap_or(pt_config.max_depth),
            config,
        }
    }
    #[tracked(crate = "luisa")]
    fn evaluate_at(
        &self,
        scene: &Arc<Scene>,
        filter: PixelFilter,
        color_pipeline: ColorPipeline,
        sampler: &dyn Sampler,
        p: Expr<Uint2>,
        shift_mapping: Option<&ReconnectionShiftMapping>,
    ) -> (Color, Expr<SampledWavelengths>, Expr<f32>) {
        sampler.start();
        let _p = sampler.next_2d();
        let swl = sample_wavelengths(color_pipeline.color_repr, sampler).var();
        let (ray, _) =
            scene
                .camera
                .generate_ray(&scene, filter, p, sampler, color_pipeline.color_repr, **swl);
        let mut pt = PathTracerBase::new(
            scene,
            color_pipeline,
            self.config.max_depth.expr(),
            self.config.rr_depth.expr(),
            self.config.use_nee,
            self.config.direct_spp >= 0,
            swl,
        );
        pt.need_shift_mapping = self.config.reconnect;
        pt.run_pt_hybrid_shift_mapping(ray, sampler, shift_mapping, None);
        let l = pt.radiance.load();
        (l, **swl, self.scalar_contribution(&l))
    }
    #[tracked(crate = "luisa")]
    fn evaluate_with_sampler(
        &self,
        scene: &Arc<Scene>,
        filter: PixelFilter,
        color_pipeline: ColorPipeline,
        sampler: &dyn Sampler,
        shift_mapping: Option<&ReconnectionShiftMapping>,
    ) -> (Expr<Uint2>, Color, Expr<SampledWavelengths>, Expr<f32>) {
        sampler.start();
        let res = scene.camera.resolution().expr();
        let p = sampler.next_2d() * res.cast_f32();
        let p = p.cast_i32().clamp(0, res.cast_i32() - 1);
        let swl = sample_wavelengths(color_pipeline.color_repr, sampler).var();
        let (ray, ray_w) = scene.camera.generate_ray(
            &scene,
            filter,
            p.cast_u32(),
            sampler,
            color_pipeline.color_repr,
            **swl,
        );
        let mut pt = PathTracerBase::new(
            scene,
            color_pipeline,
            self.config.max_depth.expr(),
            self.config.rr_depth.expr(),
            self.config.use_nee,
            self.config.direct_spp >= 0,
            swl,
        );
        pt.need_shift_mapping = self.config.reconnect;
        pt.run_pt_hybrid_shift_mapping(ray, sampler, shift_mapping, None);
        let l = pt.radiance.load();
        (p.cast_u32(), l, **swl, self.scalar_contribution(&l))
    }
    #[tracked(crate = "luisa")]
    fn evaluate(
        &self,
        scene: &Arc<Scene>,
        filter: PixelFilter,
        color_pipeline: ColorPipeline,
        samples: &PssBuffer,
        independent: &IndependentSampler,
        chain_id: Expr<u32>,
        mutator: Option<Mutator>,
        is_bootstrap: bool,
        shift_mapping: Option<&ReconnectionShiftMapping>,
    ) -> (
        Expr<Uint2>,
        Color,
        Expr<SampledWavelengths>,
        Expr<f32>,
        Expr<u32>,
    ) {
        let sampler = LazyMcmcSampler::new(
            independent,
            samples,
            chain_id * (self.sample_dimension() as u32).expr(),
            if is_bootstrap {
                0u32
            } else {
                self.sample_dimension() as u32
            }
            .expr(),
            mutator,
        );
        sampler.start();
        let (p, l, swl, f) =
            self.evaluate_with_sampler(scene, filter, color_pipeline, &sampler, shift_mapping);
        (p, l, swl, f, **sampler.cur_dim)
    }
    pub fn scalar_contribution(&self, color: &Color) -> Expr<f32> {
        if !self.config.uniform {
            color.max().clamp(0.0f32.expr(), 1e5f32.expr())
        } else {
            1.0f32.expr()
        }
    }
    fn bootstrap(
        &self,
        scene: &Arc<Scene>,
        filter: PixelFilter,
        color_pipeline: ColorPipeline,
    ) -> RenderState {
        let seeds =
            init_pcg32_buffer_with_seed(self.device.clone(), self.n_bootstrap, self.config.seed);
        let fs = self
            .device
            .create_buffer_from_fn(self.n_bootstrap, |_| 0.0f32);
        let sample_buffer = self
            .device
            .create_buffer(self.sample_dimension() * self.n_chains);
        {
            let pss_samples =
                self.sample_dimension() * self.n_chains * std::mem::size_of::<PssSample>();
            let states = self.n_chains * std::mem::size_of::<MarkovState>();
            log::info!(
                "Mcmc memory consumption {:.2}MiB: PSS samples: {:.2}MiB, Markov states: {:.2}MiB",
                (pss_samples + states) as f64 / 1024.0 / 1024.0,
                pss_samples as f64 / 1024.0 / 1024.0,
                states as f64 / 1024.0 / 1024.0
            );
        }
        self.device
            .create_kernel::<fn()>(&|| {
                let i = dispatch_id().x;
                let seed = seeds.var().read(i);
                let sampler = IndependentSampler::from_pcg32(seed.var());
                // DON'T WRITE INTO sample_buffer
                let (_p, _l, _swl, f, _) = self.evaluate(
                    scene,
                    filter,
                    color_pipeline,
                    &sample_buffer,
                    &sampler,
                    i,
                    None,
                    true,
                    None,
                );
                fs.var().write(i, f);
            })
            .dispatch([self.n_bootstrap as u32, 1, 1]);

        let weights = fs.copy_to_vec();
        let (b, resampled) = resample_with_f64(&weights, self.n_chains);
        assert!(b > 0.0, "Bootstrap failed, please retry with more samples");
        log::info!(
            "Normalization factor initial estimate: {}",
            b / self.n_bootstrap as f64
        );
        let resampled = self.device.create_buffer_from_slice(&resampled);
        let states = self.device.create_buffer(self.n_chains);
        let cur_colors = ColorBuffer::new(
            self.device.clone(),
            self.n_chains,
            color_pipeline.color_repr,
        );
        let vertices = self.device.create_buffer(self.n_chains);
        let proposal_vertices = self.device.create_buffer(self.n_chains);
        self.device
            .create_kernel::<fn()>(&track!(|| {
                let i = dispatch_id().x;
                let seed_idx = resampled.var().read(i);
                let seed = seeds.var().read(seed_idx);
                let sampler = IndependentSampler::from_pcg32(seed.var());
                let dim = (self.sample_dimension() as u32).expr();
                for_range(0u32.expr()..dim, |j| {
                    sample_buffer.var().write(
                        i * dim + j,
                        PssSample::new_expr(sampler.next_1d(), 0.0, 0, 0),
                    );
                });
                let vertex = ReconnectionVertex::var_zeroed();
                let shift_mapping = if self.config.reconnect {
                    Some(ReconnectionShiftMapping {
                        min_dist: 0.03f32.expr(),
                        min_roughness: 0.2f32.expr(),
                        is_base_path: true.var(),
                        read_vertex: Box::new(|| **vertex),
                        write_vertex: Box::new(|v| *vertex = v),
                        jacobian: 0.0f32.var(),
                        success: false.var(),
                    })
                } else {
                    None
                };
                let (p, l, swl, f, dim) = self.evaluate(
                    scene,
                    filter,
                    color_pipeline,
                    &sample_buffer,
                    &sampler,
                    i,
                    None,
                    false,
                    shift_mapping.as_ref(),
                );
                cur_colors.write(i, l, swl);
                let state = MarkovState::new_expr(p, i, f, 0.0, 0, 0, 0, 0, 0, dim);
                states.var().write(i, state);
                vertices.write(i, **vertex);
            }))
            .dispatch([self.n_chains as u32, 1, 1]);
        let rng_states =
            init_pcg32_buffer_with_seed(self.device.clone(), self.n_chains, self.config.seed + 1);
        let n_shift_pixels = match self.config.shape {
            BaseShape::Square => {
                self.config.shape_width.pow(2) + (self.config.shape_width - 1).pow(2) - 1
            }
            _ => todo!(),
        };
        let mut k_pmf = (0..self
            .config
            .max_sampled_length
            .unwrap_or(n_shift_pixels)
            .min(n_shift_pixels))
            // .map(|_| 1.0f32)
            .map(|i| (i as f32 + 1.0).powf(-2.0))
            // .map(|i| 1.0 / (i as f32 + 1.0))
            // .map(|i| 1.0f32 / 2.0f32.powi(i as i32))
            .collect::<Vec<_>>();
        {
            let norm = k_pmf.iter().sum::<f32>();
            k_pmf.iter_mut().for_each(|x| *x /= norm);
        }
        let avgs_per_pixel = {
            let compute_avg = |k_pmf: &[f32]| {
                let norm = k_pmf.iter().sum::<f32>();
                k_pmf
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (i + 1) as f32 * p / norm)
                    .sum::<f32>()
                    + 1.0f32
            };
            let mut avgs = vec![1.0];
            for i in 1..=k_pmf.len() {
                if self.config.sample_all {
                    avgs.push(i as f32 + 1.0);
                } else {
                    let avg = compute_avg(&k_pmf[0..i]);
                    avgs.push(avg);
                }
            }

            avgs
        };
        let avg_per_pixel = avgs_per_pixel[avgs_per_pixel.len() - 1];
        log::info!("Shape size: {}", n_shift_pixels + 1);
        log::info!("Average paths per pixel: {}", avg_per_pixel);
        let pmf_ats = {
            let arr = self.device.create_bindless_array((1 + k_pmf.len()) * 2);
            for i in 1..=k_pmf.len() {
                let mut pmf = k_pmf[0..i].to_vec();
                if self.config.sample_all {
                    let len = pmf.len();
                    pmf[0..len - 1].fill(0.0);
                    pmf[len - 1] = 1.0;
                }
                let at = AliasTable::new(self.device.clone(), &pmf);
                arr.emplace_buffer(2 * i, &at.0);
                arr.emplace_buffer(2 * i + 1, &at.1);
            }
            arr
        };
        let tmp_color = ColorBuffer::new(
            self.device.clone(),
            self.n_chains * (1 + n_shift_pixels as usize),
            color_pipeline.color_repr,
        );
        let sampled_indices = self
            .device
            .create_buffer::<i32>(self.n_chains * (1 + n_shift_pixels) as usize);
        let sort_tmp = self
            .device
            .create_buffer::<i32>(self.n_chains * (1 + n_shift_pixels) as usize);
        let mut t_lo = i32::MAX;
        let mut t_hi = i32::MAX;
        let mut base_shape_center = u32::MAX;
        let curve_data = {
            let curve = self.config.curve;
            let n = (1 + n_shift_pixels) as usize;

            let shape_width = ((((2 * n - 1) as f32).sqrt() + 1.0) * 0.5) as u32;
            assert_eq!(shape_width.pow(2) + (shape_width - 1).pow(2), n as u32);

            let curve_width = 2 * shape_width - 1;
            let p = (curve_width as f32).log2().ceil() as u32;
            let mut pts = HashSet::new();
            let curve_center = (1 << (p - 1)) as i32;
            for x in 0..(1 << p) {
                for y in 0..(1 << p) {
                    if (x - curve_center).abs() + (y - curve_center).abs() <= shape_width as i32 - 1
                    {
                        pts.insert((x as u32, y as u32));
                    }
                }
            }
            // let shape_center =
            match curve {
                SpatialCurve::Morton => {
                    let mut morton_curve = vec![];
                    for x in 0..(1 << p) {
                        for y in 0..(1 << p) {
                            morton_curve.push((x, y, morton2d(x as u64, y as u64) as usize));
                        }
                    }
                    morton_curve.sort_by_key(|x| x.2);
                    let morton_curve = morton_curve
                        .into_iter()
                        .map(|(x, y, _)| (x, y))
                        .collect::<Vec<_>>();
                    let mut buf = vec![];
                    for (x, y) in morton_curve {
                        if pts.contains(&(x, y)) {
                            if x == curve_center as u32 && y == curve_center as u32 {
                                assert!(base_shape_center == u32::MAX);
                                base_shape_center = buf.len() as u32;
                                t_lo = -(buf.len() as i32);
                                t_hi = n as i32 - buf.len() as i32;
                                assert!(t_hi > 0);
                                assert!(t_hi - t_lo == n as i32);
                            }
                            let stride = self.config.stride as i32;
                            buf.push(Int2::new(
                                (x as i32 - curve_center) * stride,
                                (y as i32 - curve_center) * stride,
                            ));
                        }
                    }
                    assert_eq!(buf.len(), 1 + n_shift_pixels as usize);
                    buf
                }
                SpatialCurve::Hilbert => {
                    let hilbert_curve = util::generate_hilbert_curve(p);

                    let mut buf = vec![];
                    for (x, y) in hilbert_curve {
                        if pts.contains(&(x, y)) {
                            if x == curve_center as u32 && y == curve_center as u32 {
                                assert!(base_shape_center == u32::MAX);
                                base_shape_center = buf.len() as u32;
                                t_lo = -(buf.len() as i32);
                                t_hi = n as i32 - buf.len() as i32;
                                assert!(t_hi > 0);
                                assert!(t_hi - t_lo == n as i32);
                            }
                            let stride = self.config.stride as i32;
                            buf.push(Int2::new(
                                (x as i32 - curve_center) * stride,
                                (y as i32 - curve_center) * stride,
                            ));
                        }
                    }
                    assert_eq!(buf.len(), 1 + n_shift_pixels as usize);
                    buf
                }
            }
        };
        let curve = self.device.create_buffer_from_slice(&curve_data);

        let path_jacobians = self
            .device
            .create_buffer(self.n_chains * (1 + n_shift_pixels) as usize);
        let resolution = scene.camera.resolution();
        let shapes = self
            .device
            .create_buffer::<Shape>((resolution.x * resolution.y) as usize);
        let shape_indices = self.device.create_buffer::<i8>(
            (resolution.x * resolution.y) as usize * (1 + n_shift_pixels) as usize,
        );
        let albedo_buf =
            self.device
                .create_tex2d::<Float4>(PixelStorage::Byte4, resolution.x, resolution.y, 1);
        let normal_buf =
            self.device
                .create_tex2d::<Float4>(PixelStorage::Byte4, resolution.x, resolution.y, 1);
        let avgs_per_pixel = self.device.create_buffer_from_slice(&avgs_per_pixel);
        RenderState {
            t_hi,
            t_lo,
            base_shape_center,
            n_shift_pixels,
            avg_per_pixel,
            avgs_per_pixel,
            rng_states,
            samples: sample_buffer,
            cur_colors,
            states,
            b_init: b,
            b_init_cnt: self.n_bootstrap,
            pmf_ats,
            sampled_indices,
            sort_tmp,
            tmp_color,
            curve,
            vertices,
            proposal_vertices,
            path_jacobians,
            shapes,
            shape_indices,
            normal_buf,
            albedo_buf,
            curve_data,
        }
    }
    #[tracked(crate = "luisa")]
    fn telescoping(
        &self,
        resolution: Expr<Uint2>,
        cur_p: Expr<Float2>,
        primary_l: Color,
        swl: Expr<SampledWavelengths>,
        render_state: &RenderState,
        color_buf: &ColorBuffer,
        path_jacobians: &BufferVar<f32>,
        t_buf: &BufferVar<i32>,
        idx_lo: Expr<u32>,
        idx_hi: Expr<u32>,
        center: Expr<u32>,
        p_from_t: impl Fn(Expr<i32>) -> Expr<Int2>,
        t_lo: Expr<i32>,
        t_hi: Expr<i32>,
        scale: Expr<f32>,
        diff_scale: Expr<f32>,
        k: Expr<f32>,
        splat_fn: impl Fn(Expr<Float2>, Color, Expr<SampledWavelengths>, Expr<f32>),
    ) {
        outline(|| {
            let t = t_lo.var();
            let idx = idx_lo.var();
            let next_idx = || idx + 1;
            let prev_idx = || idx.as_i32() - 1;
            let next_next_idx = || idx + 2;
            let is_last = false.var();

            let acc_cur_l = ColorVar::zero(primary_l.repr());
            let cur_f = self.scalar_contribution(&primary_l);
            let shape = {
                let cur_p = cur_p.cast_u32();
                let p_idx = cur_p.x + cur_p.y * resolution.x;
                if self.config.denoiser_kernel {
                    render_state.shapes.read(p_idx)
                } else {
                    self.default_shape(render_state)
                }
            };
            while t < t_hi {
                // TODO: check this!
                if !is_last {
                    loop {
                        if next_idx() >= idx_hi {
                            *is_last = true;
                            break;
                        }
                        if t_buf.read(next_idx()) > t {
                            break;
                        }
                        *idx = next_idx();
                    }
                };
                let next = next_idx();

                let acc_proposal_l = ColorVar::zero(primary_l.repr());
                let ip = p_from_t(**t);
                let splat_mcmc = |l: Color, jacobian: Expr<f32>, scale: Expr<f32>| {
                    let f = self.scalar_contribution(&l);
                    let p_idx = ip.x.as_u32() + ip.y.as_u32() * resolution.x;
                    let shape_p = if self.config.denoiser_kernel {
                        render_state.shapes.read(p_idx)
                    } else {
                        self.default_shape(render_state)
                    };
                    let ratio = shape.n_pixels.as_f32() / shape_p.n_pixels.as_f32();

                    // let shape
                    let accept = select(
                        f.is_finite(),
                        select(
                            cur_f.eq(0.0) | !cur_f.is_finite(),
                            1.0f32.expr(),
                            (f / cur_f * jacobian * ratio).clamp(0.0f32.expr(), 1.0f32.expr()),
                        ),
                        0.0f32.expr(),
                    );
                    if f > 0.0 {
                        acc_proposal_l.store(acc_proposal_l.load() + l / f * accept * scale);
                    }
                    if cur_f > 0.0 {
                        acc_cur_l
                            .store(acc_cur_l.load() + primary_l / cur_f * (1.0 - accept) * scale);
                    }
                };

                if (t < t_buf.read(idx)) | ((idx == idx_lo) & (t <= t_buf.read(idx))) {
                    // first
                    let l = color_buf.read(idx).0;
                    let jacobian = path_jacobians.read(idx);
                    let next = next_idx();
                    let (l_minus, jacobian_minus) = if idx == center {
                        (l, jacobian)
                    } else {
                        if next < idx_hi {
                            (color_buf.read(next).0, path_jacobians.read(next))
                        } else {
                            (l, jacobian)
                        }
                    };
                    splat_mcmc(l, jacobian, 1.0 / k * diff_scale);
                    splat_mcmc(l_minus, jacobian_minus, -1.0 / k * diff_scale);
                } else if **is_last {
                    // last
                    let l = color_buf.read(idx).0;
                    let jacobian = path_jacobians.read(idx);
                    let (l_minus, jacobian_minus) = if idx == center {
                        (l, jacobian)
                    } else if idx > idx_lo {
                        lc_assert!(prev_idx().ge(idx_lo.as_i32()));
                        (
                            color_buf.read(prev_idx().as_u32()).0,
                            path_jacobians.read(prev_idx().as_u32()),
                        )
                    } else {
                        (l, jacobian)
                    };
                    splat_mcmc(l, jacobian, 1.0 / k * diff_scale);
                    splat_mcmc(l_minus, jacobian_minus, -1.0 / k * diff_scale);
                } else {
                    let compute_lerp = |idx: Expr<u32>, next: Expr<u32>| {
                        let t0 = t_buf.read(idx);
                        let t1 = t_buf.read(next);
                        // device_log!(
                        //     "{} {} {} {} {} {} {} {}",
                        //     dispatch_id().x,
                        //     idx_lo,
                        //     idx_hi,
                        //     idx,
                        //     next,
                        //     t0,
                        //     t1,
                        //     t
                        // );
                        lc_assert!(t1.ge(t));
                        lc_assert!(t.ge(t0));
                        lc_assert!(t1.ne(t0));
                        let fac = (t - t0).as_f32() / (t1 - t0).as_f32();
                        (
                            color_buf.read(idx).0 * (1.0 - fac) + color_buf.read(next).0 * fac,
                            path_jacobians.read(idx) * (1.0 - fac)
                                + path_jacobians.read(next) * fac,
                        )
                    };

                    let (l, jacobian) = compute_lerp(**idx, next);
                    let prev = prev_idx();
                    let next = next_idx();
                    let next_next = next_next_idx();
                    if t == t_buf.read(idx) {
                        // if we are at one of the samples
                        // the only other way to get contribution is to remove the current one
                        let (l_minus, jacobian_minus) = if idx == center {
                            // cannot remove center
                            (l, jacobian)
                        } else {
                            lc_assert!(idx.gt(idx_lo));
                            compute_lerp(prev.as_u32(), next)
                        };
                        splat_mcmc(l, jacobian, 1.0 / k * diff_scale);
                        splat_mcmc(l_minus, jacobian_minus, -1.0 / k * diff_scale);
                    } else {
                        // we have two other ways to get contribution
                        // we can either remove the previous one or the next one

                        // remove idx
                        let (l_minus_left, jacobian_minus_left) = if idx == center {
                            (l, jacobian)
                        } else {
                            lc_assert!(next.lt(idx_hi));
                            if prev < idx_lo.as_i32() {
                                (color_buf.read(next).0, path_jacobians.read(next))
                            } else {
                                compute_lerp(prev.as_u32(), next)
                            }
                        };
                        //remove next
                        let (l_minus_right, jacobian_minus_right) = if next == center {
                            // cannot remove next
                            (l, jacobian)
                        } else {
                            if next_next >= idx_hi {
                                (color_buf.read(idx).0, path_jacobians.read(idx))
                            } else {
                                compute_lerp(**idx, next_next)
                            }
                        };
                        // l - l_minus_left + l - l_minus_right
                        splat_mcmc(l, jacobian, 2.0 / k * diff_scale);
                        splat_mcmc(l_minus_left, jacobian_minus_left, -1.0 / k * diff_scale);
                        splat_mcmc(l_minus_right, jacobian_minus_right, -1.0 / k * diff_scale);
                    }
                };

                splat_mcmc(primary_l, 1.0f32.expr(), 1.0f32.expr());
                // device_log!("{} {} {} {}", primary_l.as_rgb(), diff.as_rgb(), k, diff_scale);
                // splat_fn(ip.cast_f32(), primary_l + diff / k * diff_scale, swl, scale);
                splat_fn(ip.cast_f32(), acc_proposal_l.load(), swl, scale);

                *t += 1;
            }
            splat_fn(cur_p, acc_cur_l.load(), swl, scale);
        });
    }
    #[tracked(crate = "luisa")]
    fn default_shape(&self, state: &RenderState) -> Expr<Shape> {
        let n = state.n_shift_pixels.expr() + 1;
        Shape::from_comps_expr(ShapeComps {
            center: state.base_shape_center.expr().as_u8(),
            n_pixels: n.as_u8(),
        })
    }
    #[tracked(crate = "luisa")]
    fn kernel_compute_denoiser_kernel(&self, scene: Arc<Scene>, state: &RenderState) {
        let resolution = scene.camera.resolution();
        set_block_size([32, 1, 1]);
        let px_idx = dispatch_id().x;
        let p = {
            let p = Uint2::expr(px_idx % resolution.x, px_idx / resolution.x);
            p
        };

        let p_from_t = self.get_p_from_t(p, resolution.expr(), state, false);

        let cnt = 0u32.var();
        let t_lo = state.t_lo.expr();
        let t_hi = state.t_hi.expr();
        let center_normal = state.normal_buf.read(p).xyz() * 2.0 - 1.0;
        let center_albedo = state.albedo_buf.read(p).xyz();
        let offset = px_idx;
        let shape = state.shapes.read(offset).var();
        for t in t_lo..t_hi {
            let write = false.var();
            if t == 0 {
                *shape.center = cnt.as_u8();
                *write = true;
            } else {
                let p = p_from_t(t).cast_u32();
                let normal = state.normal_buf.read(p).xyz() * 2.0 - 1.0;
                let albedo = state.albedo_buf.read(p).xyz();

                let group_normal = normal.dot(center_normal) >= 0.707;
                let group_albedo = (center_albedo - albedo).abs().reduce_max() < 0.2;
                if (!self.config.denoiser_kernel) | (group_normal & group_albedo) {
                    *write = true;
                }
            }
            if write {
                state
                    .shape_indices
                    .write(offset * (1 + state.n_shift_pixels) + cnt, t.cast_i8());
                *cnt += 1;
            }
        }
        *shape.n_pixels = cnt.as_u8();
        state.shapes.write(offset, shape);
    }
    #[tracked(crate = "luisa")]
    fn is_shape_full(&self, state: &RenderState, s: Expr<Shape>) -> Expr<bool> {
        s.n_pixels.as_u32() == (state.n_shift_pixels + 1u32)
    }
    #[tracked(crate = "luisa")]
    fn get_p_from_t<'a>(
        &'a self,
        p: Expr<Uint2>,
        resolution: Expr<Uint2>,
        state: &'a RenderState,
        use_shape_indices: bool,
    ) -> Box<dyn Fn(Expr<i32>) -> Expr<Int2> + 'a> {
        let n = state.n_shift_pixels.expr();

        let p_idx = p.x + p.y * resolution.x;
        let shape = if use_shape_indices {
            Some(state.shapes.read(p_idx))
        } else {
            None
        };
        let offset = (p_idx) * (1 + n);
        let map_i = move |i: Expr<i32>| -> Expr<i32> {
            if use_shape_indices {
                let shape = shape.unwrap();
                if self.is_shape_full(state, shape) {
                    i
                } else {
                    let center = shape.center;
                    let t = i + center.as_i32();
                    let shape_indices = &state.shape_indices;
                    shape_indices.read(offset + t.as_u32()).cast_i32()
                }
            } else {
                i
            }
        };

        let curve: BufferVar<Vector<i32, 2>> = state.curve.var();
        let p_from_t = move |i: Expr<i32>| -> Expr<Int2> {
            let i = map_i(i);
            let i = i + state.base_shape_center as i32;
            let morton_p = curve.read(i.as_u32());

            let ip = p.cast_i32() + morton_p;
            (ip + resolution.cast_i32()) % resolution.cast_i32()
        };
        Box::new(p_from_t) as Box<dyn Fn(Expr<i32>) -> Expr<Int2>>
    }
    #[tracked(crate = "luisa")]
    fn kernel_compute_auxillary(
        &self,
        scene: Arc<Scene>,
        color_pipeline: ColorPipeline,
        rngs: &Buffer<Pcg32>,
        state: &RenderState,
        spp: Expr<u32>,
        film: &Film,
    ) {
        let resolution = scene.camera.resolution();
        set_block_size([256, 1, 1]);
        let px_idx = dispatch_id().x;
        let p = {
            let p = Uint2::expr(px_idx % resolution.x, px_idx / resolution.x);
            p
        };
        let rng = IndependentSampler::from_pcg32(rngs.read(px_idx).var());
        let acc_normal = Float3::var_zeroed();
        let acc_albedo = ColorVar::zero(color_pipeline.color_repr);
        for _ in 0u32.expr()..spp {
            let swl = sample_wavelengths(color_pipeline.color_repr, &rng);
            let (ray, _) = scene.camera.generate_ray(
                &scene,
                film.filter(),
                p,
                &rng,
                color_pipeline.color_repr,
                swl,
            );
            let si = scene.intersect(ray);
            if si.valid {
                let n = si.ng;
                let albedo =
                    scene
                        .svm
                        .dispatch_surface(si.surface, color_pipeline, si, swl, |closure| {
                            closure.albedo(
                                -ray.d,
                                swl,
                                &BsdfEvalContext {
                                    color_repr: color_pipeline.color_repr,
                                    ad_mode: ADMode::None,
                                },
                            )
                        });
                *acc_normal += n;
                acc_albedo.store(acc_albedo.load() + albedo);
            }
        }
        state.normal_buf.write(
            p,
            (acc_normal / spp.as_f32()).normalize().extend(0.0) * 0.5 + 0.5,
        );
        state
            .albedo_buf
            .write(p, (acc_albedo.load() / spp.as_f32()).as_rgb().extend(0.0));
        rngs.write(px_idx, rng.state);
    }
    #[tracked(crate = "luisa")]
    fn denoising_splat(
        &self,
        scene: &Arc<Scene>,
        color_pipeline: ColorPipeline,
        render_state: &RenderState,
        mcmc_film: &Film,
        tele_film: &Film,
        contribution: Expr<f32>,
        state: Var<MarkovState>,
        cur_color_v: ColorVar,
        cur_swl: Var<SampledWavelengths>,
        rng: &IndependentSampler,
    ) {
        let n_shift_pixels = render_state.n_shift_pixels;
        let tmp_buf_offset = state.chain_id * (1 + n_shift_pixels);
        let sampled_indices = &render_state.sampled_indices;
        let tmp_color = &render_state.tmp_color;
        let sort_tmp = &render_state.sort_tmp;

        let resolution = scene.camera.resolution();
        // let t_lo = render_state.t_lo.expr();
        // let t_hi = render_state.t_hi.expr();
        let px_idx = {
            let p = **state.cur_pixel;
            p.x + p.y * resolution.x
        };
        let shape = {
            if self.config.denoiser_kernel {
                render_state.shapes.read(px_idx)
            } else {
                self.default_shape(&render_state)
            }
        };
        let n = shape.n_pixels.as_u32() - 1;
        let t_lo = -(shape.center.cast_i32());
        let t_hi = (1 + n - shape.center.as_u32()).cast_i32();
        let (k, pmf_k) = if n > 0 {
            let entries = render_state.pmf_ats.buffer::<AliasTableEntry>(n * 2);
            let pmf = render_state.pmf_ats.buffer::<f32>(n * 2 + 1);
            let pmf_at = BindlessAliasTableVar(entries, pmf);
            let (k, pmf_k, _) = pmf_at.sample_and_remap(rng.next_1d());
            let k = k + 1;
            (k, pmf_k)
        } else {
            (0u32.expr(), 1.0f32.expr())
        };

        let p_from_t = {
            let p = **state.cur_pixel;
            self.get_p_from_t(
                p,
                resolution.expr(),
                render_state,
                self.config.denoiser_kernel,
            )
        };
        let p_from_t = &p_from_t;
        let rng_backup = rng.state.load();
        let trace = |t: Expr<i32>| {
            let l = ColorVar::zero(color_pipeline.color_repr);
            let swl = Var::<SampledWavelengths>::zeroed();
            let jacobian = 0.0f32.var();
            outline(|| {
                *rng.state = rng_backup;

                let sampler = ReplaySampler {
                    base: rng,
                    samples: &render_state.samples,
                    offset: state.chain_id * self.sample_dimension() as u32,
                    cur_dim: 0u32.var(),
                    replay_dim: **state.cur_dim,
                };
                let ip = p_from_t(t);
                let cid = **state.chain_id;
                let shift_mapping = if self.config.reconnect {
                    Some(ReconnectionShiftMapping {
                        min_dist: 0.03f32.expr(),
                        min_roughness: 0.2f32.expr(),
                        is_base_path: false.var(),
                        read_vertex: Box::new(|| render_state.vertices.read(cid)),
                        write_vertex: Box::new(|v| render_state.vertices.write(cid, v)),
                        jacobian: 0.0f32.var(),
                        success: false.var(),
                    })
                } else {
                    None
                };
                let (l_, swl_, _) = self.evaluate_at(
                    scene,
                    mcmc_film.filter(),
                    color_pipeline,
                    &sampler,
                    ip.cast_u32(),
                    shift_mapping.as_ref(),
                );

                l.store(l_);
                if let Some(sm) = shift_mapping {
                    *jacobian = sm.jacobian;
                } else {
                    *jacobian = 1.0f32;
                }
                *swl = swl_;
            });
            (l.load(), **jacobian, **swl)
        };
        let center = u32::MAX.var();

        // lc_assert!((t_hi.sub(t_lo)).eq(n.add(1).cast_i32()));
        if k > 0 {
            let use_reservoir = true;
            if use_reservoir {
                // sample k integrers from [0, n)
                // initialize the resevoir
                for i in 0u32.expr()..k {
                    sampled_indices.write(tmp_buf_offset + i, i.as_i32());
                }
                let w = (rng.next_1d().ln() / k.as_f32()).exp().var();
                // for i in k..n {
                //     let j = rng.state.gen_u32() % (i + 1);
                //     if j < k {
                //         sampled_indices.write(tmp_buf_offset + j, i.as_i32());
                //     }
                // }
                let i = k.var();
                while i < n {
                    *i += (rng.next_1d().ln() / (1.0 - w).ln()).floor().as_u32() + 1;
                    if i < n {
                        let j = rng.state.gen_u32() % k;
                        sampled_indices.write(tmp_buf_offset + j, i.as_i32());
                        *w *= (rng.next_1d().ln() / k.as_f32()).exp();
                    }
                }
            } else {
                // shuffle n times
                for i in 0u32.expr()..n {
                    sampled_indices.write(tmp_buf_offset + i, i.as_i32());
                }
                for _ in 0u32.expr()..n {
                    let p = rng.state.gen_u32() % n;
                    let q = 1 + rng.state.gen_u32() % (n - 1);
                    let q = (p + q) % n;
                    lc_assert!(p.ne(q));
                    let tmp = sampled_indices.read(tmp_buf_offset + p);
                    sampled_indices
                        .write(tmp_buf_offset + p, sampled_indices.read(tmp_buf_offset + q));
                    sampled_indices.write(tmp_buf_offset + q, tmp);
                }
            }

            if !is_cpu_backend() {
                // sort the indices using counting sort
                for i in 0u32.expr()..n {
                    sort_tmp.write(tmp_buf_offset + i, 0);
                }
                for i in 0u32.expr()..k {
                    let idx = sampled_indices.read(tmp_buf_offset + i);
                    sort_tmp.write(tmp_buf_offset + idx.as_u32(), 1);
                }
                let cnt = 0u32.var();
                for i in 0u32.expr()..n {
                    let contained = sort_tmp.read(tmp_buf_offset + i) != 0;
                    let j = i.as_i32().var();
                    if contained {
                        if j.as_u32() < shape.center.as_u32() {
                            *j = j - shape.center.as_i32();
                        } else {
                            *j = j - shape.center.as_i32() + 1;
                        }
                        if center == u32::MAX {
                            if j > 0 {
                                *center = cnt;
                            }
                        }
                        if center != u32::MAX {
                            if cnt < center {
                                lc_assert!(j.lt(0));
                            } else {
                                lc_assert!(j.gt(0));
                            }
                        }

                        lc_assert!(j.ne(0));
                        let idx = ((center == u32::MAX) | (cnt < center)).select(**cnt, cnt + 1);
                        sampled_indices.write(tmp_buf_offset + idx, j);
                        *cnt += 1;
                    }
                }
                if center == u32::MAX {
                    //set center to the last
                    *center = k;
                }
                sampled_indices.write(tmp_buf_offset + center, 0);
            } else {
                let sampled_indices_ptr = sampled_indices.device_address();
                #[derive(Clone, Copy, Value)]
                #[repr(C)]
                #[luisa(crate = "luisa")]
                struct Sort {
                    sampled_indices_ptr: u64,
                    length: u32,
                    half_n: u32,
                    center: u32,
                }
                *center = CpuFn::<Sort>::new(|sort| {
                    escape!({
                        let half_n = sort.half_n;
                        let slice = unsafe {
                            std::slice::from_raw_parts_mut(
                                sort.sampled_indices_ptr as *mut i32,
                                sort.length as usize + 1,
                            )
                        };
                        // dbg!(&slice[..sort.length as usize]);
                        slice[..sort.length as usize].sort_unstable();
                        sort.center = slice[..sort.length as usize]
                            .partition_point(|x| *x < half_n as i32)
                            as u32;
                        for i in (0..sort.length as usize).rev() {
                            let j = unsafe { *slice.get_unchecked_mut(i) };
                            let j = if (j as u32) < half_n {
                                j - half_n as i32
                            } else {
                                j - half_n as i32 + 1
                            };
                            let idx = if i < sort.center as usize { i } else { i + 1 };
                            unsafe {
                                *slice.get_unchecked_mut(idx) = j;
                            }
                        }
                        unsafe {
                            *slice.get_unchecked_mut(sort.center as usize) = 0;
                        }
                        // dbg!(slice);
                    })
                })
                .call(Sort::from_comps_expr(SortComps {
                    sampled_indices_ptr: sampled_indices_ptr
                        + tmp_buf_offset.as_u64() * std::mem::size_of::<i32>() as u64,
                    length: k,
                    half_n: shape.center.as_u32(),
                    center: **center,
                }))
                .center;
            }
        } else {
            *center = 0;
        }
        let cur_f = state.cur_f.expr();
        let cur_p = state.cur_pixel.expr();
        let cur_l = cur_color_v.load();
        let cur_swl = cur_swl.expr();
        let splat_mcmc_contribution = |film: &Film,
                                       p: Expr<Float2>,
                                       l: Color,
                                       jacobian: Expr<f32>,
                                       swl: Expr<SampledWavelengths>,
                                       scale: Expr<f32>| {
            let f = self.scalar_contribution(&l);
            let p_idx = p.x.as_u32() + p.y.as_u32() * resolution.x;
            let shape_p = if self.config.denoiser_kernel {
                render_state.shapes.read(p_idx)
            } else {
                self.default_shape(&render_state)
            };
            let ratio = shape.n_pixels.as_f32() / shape_p.n_pixels.as_f32();
            let accept = select(
                f.is_finite(),
                select(
                    state.cur_f.eq(0.0) | !state.cur_f.is_finite(),
                    1.0f32.expr(),
                    (f / state.cur_f * jacobian * ratio).clamp(0.0f32.expr(), 1.0f32.expr()),
                ),
                0.0f32.expr(),
            );
            if f > 0.0 {
                film.add_splat(p, &(l / f), swl, accept * contribution * scale);
            }
            if cur_f > 0.0 {
                film.add_splat(
                    cur_p.cast_f32(),
                    &(cur_l / cur_f),
                    cur_swl,
                    (1.0 - accept) * contribution * scale,
                );
            }
        };
        let avg_per_pixel = render_state
            .avgs_per_pixel
            .read(shape.n_pixels.as_u32() - 1);
        for i in 0u32.expr()..k {
            let j = if i < center { i } else { i + 1 };
            let t = sampled_indices.read(tmp_buf_offset + j);
            // device_log!("{} {}", dispatch_id(), t);
            lc_assert!(t.ne(0));
            lc_assert!(t.ge(t_lo));
            if !t.lt(t_hi) {
                device_log!("{} {} {} {} {}", t, t_lo, t_hi, k, shape);
            }
            lc_assert!(t.lt(t_hi));
            // cpu_dbg!(**t);
            let (l, jacobian, swl) = trace(t);
            tmp_color.write(tmp_buf_offset + j, l, swl);
            render_state
                .path_jacobians
                .write(tmp_buf_offset + j, jacobian);
            splat_mcmc_contribution(
                mcmc_film,
                p_from_t(t).cast_f32(),
                l,
                jacobian,
                swl,
                1.0f32.expr() / avg_per_pixel,
            );
        }

        {
            let primary_l = cur_color_v.load();
            let swl = cur_swl;
            splat_mcmc_contribution(
                mcmc_film,
                cur_p.cast_f32(),
                primary_l,
                1.0f32.expr(),
                swl,
                1.0f32.expr() / avg_per_pixel,
            );
            tmp_color.write(tmp_buf_offset + center, primary_l, swl);
            render_state
                .path_jacobians
                .write(tmp_buf_offset + center, 1.0);
        }

        let primary_l = cur_color_v.load();
        let swl = cur_swl;
        if shape.n_pixels == 1 {
            // tele_film.add_splat(cur_p.cast_f32(), &primary_l, swl, 1.0f32.expr());
            let f = self.scalar_contribution(&primary_l);
            if f > 0.0 {
                tele_film.add_splat(cur_p.cast_f32(), &(primary_l / f), swl, 1.0f32.expr());
            }
        } else {
            self.telescoping(
                scene.camera.resolution().expr(),
                cur_p.cast_f32(),
                primary_l,
                swl,
                &render_state,
                tmp_color,
                &render_state.path_jacobians,
                &sampled_indices,
                tmp_buf_offset,
                tmp_buf_offset + k + 1,
                tmp_buf_offset + center,
                p_from_t,
                t_lo,
                t_hi,
                contribution / shape.n_pixels.as_f32(),
                1.0f32 / pmf_k,
                k.as_f32(),
                |p, l, swl, scale| {
                    tele_film.add_splat(p, &l, swl, scale);
                },
            );
        }
    }
    #[tracked(crate = "luisa")]
    fn mutate_chain(
        &self,
        scene: &Arc<Scene>,
        color_pipeline: ColorPipeline,
        render_state: &RenderState,
        main_mcmc_film: &Film,
        mcmc_film: &Film,
        tele_film: &Film,
        contribution: Expr<f32>,
        state: Var<MarkovState>,
        cur_color_v: ColorVar,
        cur_swl: Var<SampledWavelengths>,
        rng: &IndependentSampler,
    ) {
        let offset = state.chain_id * self.sample_dimension() as u32;
        *state.cur_iter += 1;
        // select a mutation strategy
        match self.method {
            Method::Kelemen {
                large_step_prob,
                image_mutation_prob,
                ..
            } => {
                let u = rng.next_1d();
                let is_large_step = u.lt(large_step_prob);
                let is_image_mutation = rng.next_1d().lt(image_mutation_prob);
                let mutator = Mutator {
                    is_large_step,
                    is_image_mutation,
                    method: self.method,
                    last_large_iter: **state.last_large_iter,
                    cur_iter: **state.cur_iter,
                    res: scene.camera.resolution().expr().cast_f32(),
                };
                let cid = **state.chain_id;
                let shift_mapping = if self.config.reconnect {
                    render_state
                        .proposal_vertices
                        .write(cid, ReconnectionVertex::expr_zeroed());
                    Some(ReconnectionShiftMapping {
                        min_dist: 0.03f32.expr(),
                        min_roughness: 0.2f32.expr(),
                        is_base_path: true.var(),
                        read_vertex: Box::new(|| render_state.proposal_vertices.read(cid)),
                        write_vertex: Box::new(|v| render_state.proposal_vertices.write(cid, v)),
                        jacobian: 0.0f32.var(),
                        success: false.var(),
                    })
                } else {
                    None
                };

                let (proposal_p, proposal_color, proposal_swl, f, proposal_dim) = self.evaluate(
                    scene,
                    mcmc_film.filter(),
                    color_pipeline,
                    &render_state.samples,
                    &rng,
                    **state.chain_id,
                    Some(mutator),
                    false,
                    shift_mapping.as_ref(),
                );
                // if self.config.reconnect {
                //     let v = render_state.proposal_vertices.read(cid);
                //     device_log!("{}", v.valid());
                // }
                let proposal_f = f;
                if is_large_step & state.b_cnt.lt(1024u32 * 1024) {
                    *state.b += proposal_f;
                    *state.b_cnt += 1;
                };
                let cur_f = **state.cur_f;
                let cur_p = **state.cur_pixel;
                let cur_color = cur_color_v.load();
                let accept = select(
                    proposal_f.is_finite(),
                    select(
                        cur_f.eq(0.0) | !cur_f.is_finite(),
                        1.0f32.expr(),
                        (proposal_f / cur_f).clamp(0.0f32.expr(), 1.0f32.expr()),
                    ),
                    0.0f32.expr(),
                );
                main_mcmc_film.add_splat(
                    proposal_p.cast_f32(),
                    &(proposal_color.clone() / proposal_f),
                    proposal_swl,
                    accept * contribution,
                );
                main_mcmc_film.add_splat(
                    cur_p.cast_f32(),
                    &(cur_color / cur_f),
                    **cur_swl,
                    (1.0 - accept) * contribution,
                );
                if rng.next_1d().lt(accept) {
                    *state.cur_f = proposal_f;
                    cur_color_v.store(proposal_color);
                    cur_swl.store(proposal_swl);
                    *state.cur_pixel = proposal_p;
                    *state.cur_dim = proposal_dim;
                    if !is_large_step {
                        *state.n_accepted += 1;
                    } else {
                        *state.last_large_iter = state.cur_iter;
                    };
                    if self.config.reconnect {
                        render_state
                            .vertices
                            .write(cid, render_state.proposal_vertices.read(cid));
                    }
                } else
                // reject
                {
                    *state.cur_iter -= 1;
                    let dim = proposal_dim.min_(self.sample_dimension() as u32);
                    for_range(0u32.expr()..dim, |i| {
                        let sample = render_state.samples.var().read(offset + i).var();
                        *sample.cur = sample.backup;
                        *sample.last_modified = sample.modified_backup;
                        render_state.samples.var().write(offset + i, **sample);
                    });
                };
                if !is_large_step {
                    *state.n_mutations += 1;
                };
                self.denoising_splat(
                    scene,
                    color_pipeline,
                    render_state,
                    mcmc_film,
                    tele_film,
                    contribution,
                    state,
                    cur_color_v,
                    cur_swl,
                    rng,
                )
            }
            _ => todo!(),
        }
    }
    #[tracked(crate = "luisa")]
    fn advance_chain(
        &self,
        scene: &Arc<Scene>,
        color_pipeline: ColorPipeline,
        render_state: &RenderState,
        main_mcmc_film: &Film,
        mcmc_film: &Film,
        tele_film: &Film,
        mutations_per_chain: Expr<u32>,
        contribution: Expr<f32>,
    ) {
        let i = dispatch_id().x;
        let markov_states = render_state.states.var();
        let sampler = IndependentSampler::from_pcg32(render_state.rng_states.var().read(i).var());
        let state = markov_states.read(i).var();
        let (cur_color, cur_swl) = render_state.cur_colors.read(i);
        let cur_color_v = ColorVar::new(cur_color);
        let cur_swl_v = cur_swl.var();
        for_range(0u32.expr()..mutations_per_chain, |_| {
            // we are about to overflow
            if state.cur_iter.eq(u32::MAX - 1) {
                // cpu_dbg!(i);
                let dim = self.sample_dimension();
                for_range(0u32.expr()..(dim as u32).expr(), |j| {
                    let sample = render_state.samples.var().read(i * dim as u32 + j).var();
                    if sample.last_modified.lt(state.last_large_iter) {
                        *sample.cur = sampler.next_1d();
                    };
                    *sample.last_modified = 0u32;
                });
                *state.cur_iter -= state.last_large_iter;
                *state.last_large_iter = 0u32;
            };
            self.mutate_chain(
                scene,
                color_pipeline,
                render_state,
                main_mcmc_film,
                mcmc_film,
                tele_film,
                contribution,
                state,
                cur_color_v,
                cur_swl_v,
                &sampler,
            );
        });

        render_state
            .cur_colors
            .write(i, cur_color_v.load(), **cur_swl_v);
        render_state.rng_states.var().write(i, sampler.state.load());
        markov_states.write(i, state.load());
    }

    fn render_loop(
        &self,
        scene: &Arc<Scene>,
        color_pipeline: ColorPipeline,
        state: &RenderState,
        film: &mut Film,
        session: &RenderSession,
    ) {
        let resolution = scene.camera.resolution();
        let npixels = resolution.x * resolution.y;
        let mut mcmc_film = Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let mcmc_film_tmp = Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let mut mcmc_sqr_film =
            Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let mut main_mcmc_film =
            Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let main_mcmc_film_tmp =
            Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        // let main_mcmc_weight_film =
        //     Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        // let mut main_mcmc_sqr_film =
        //     Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let mc_weight_film = Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let mut tele_film = Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let tele_film_tmp = Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let mut tele_sqr_film =
            Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let tele_weight_film =
            Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let tele_var_film = Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let mc_var_film = Film::new(self.device.clone(), resolution, film.repr(), film.filter());

        let async_compile = self.config.async_compile;
        let rng_tmp = init_pcg32_buffer_with_seed(self.device.clone(), npixels as usize, 0);
        let compute_normal_albedo = self.device.create_kernel_with_options::<fn(u32)>(
            KernelBuildOptions {
                async_compile,
                ..Default::default()
            },
            &track!(|spp: Expr<u32>| {
                self.kernel_compute_auxillary(
                    scene.clone(),
                    color_pipeline,
                    &rng_tmp,
                    state,
                    spp,
                    film,
                );
            }),
        );
        let compute_denoiser_kernel = self.device.create_kernel_with_options::<fn()>(
            KernelBuildOptions {
                async_compile,
                ..Default::default()
            },
            &track!(|| {
                self.kernel_compute_denoiser_kernel(scene.clone(), &state);
            }),
        );
        let update_denoiser_kernel = move |s: &Scope, spp: u32| {
            if self.config.denoiser_kernel {
                let aux_spp = (self.config.spp / 4).clamp(1, 16);

                if spp == 0 {
                    s.submit([compute_normal_albedo
                        .dispatch_async([resolution.x * resolution.y, 1, 1], &aux_spp)]);
                }
                let new_data = randomize_shape(&state.curve_data, spp);
                state.curve.copy_from(&new_data);
                s.submit([compute_denoiser_kernel.dispatch_async([
                    resolution.x * resolution.y,
                    1,
                    1,
                ])])
                .synchronize();
            }
        };

        let kernel = self.device.create_kernel_with_options::<fn(u32, f32)>(
            KernelBuildOptions {
                // max_registers: 96,
                async_compile,
                ..Default::default()
            },
            &|mutations_per_chain: Expr<u32>, contribution: Expr<f32>| {
                if is_cpu_backend() {
                    let num_threads = std::thread::available_parallelism().unwrap().get();
                    if self.n_chains <= num_threads * 20 {
                        set_block_size([1, 1, 1]);
                    } else {
                        set_block_size([(num_threads / 20).clamp(1, 256) as u32, 1, 1]);
                    }
                } else {
                    set_block_size([256, 1, 1]);
                }
                self.advance_chain(
                    scene,
                    color_pipeline,
                    state,
                    &main_mcmc_film_tmp,
                    &mcmc_film_tmp,
                    &tele_film_tmp,
                    mutations_per_chain,
                    contribution,
                )
            },
        );
        log::info!(
            "Render kernel has {} arguments, {} captures!",
            kernel.num_arguments(),
            kernel.num_capture_arguments()
        );
        // let n_shift_pixels = state.n_shift_pixels;
        // let avg_per_pixel = state.avg_per_pixel;
        let update_var = self
            .device
            .create_kernel::<fn(u32)>(&track!(|_spp: Expr<u32>| {
                let p = dispatch_id().xy();
                for c in 0u32..film.repr().nvalues() as u32 {
                    let idx = film.linear_index(p.cast_f32());
                    let splat_offset = film.splat_offset() + idx * film.repr().nvalues() as u32 + c;

                    let main_mcmc = main_mcmc_film_tmp.data().read(splat_offset);
                    let mcmc = mcmc_film_tmp.data().read(splat_offset);
                    let tele = tele_film_tmp.data().read(splat_offset);
                    let avg_per_pixel = state.avg_per_pixel.expr();
                    main_mcmc_film
                        .data()
                        .var()
                        .atomic_fetch_add(splat_offset, main_mcmc);
                    // main_mcmc_sqr_film
                    // .data()
                    // .var()
                    // .atomic_fetch_add(splat_offset, main_mcmc * main_mcmc);
                    let mcmc = (mcmc * avg_per_pixel + main_mcmc) / (1.0 + avg_per_pixel);
                    mcmc_film.data().var().atomic_fetch_add(splat_offset, mcmc);
                    mcmc_sqr_film
                        .data()
                        .var()
                        .atomic_fetch_add(splat_offset, mcmc * mcmc);

                    tele_film.data().var().atomic_fetch_add(splat_offset, tele);
                    tele_sqr_film
                        .data()
                        .var()
                        .atomic_fetch_add(splat_offset, tele * tele);

                    main_mcmc_film_tmp.data().write(splat_offset, 0.0f32.expr());
                    mcmc_film_tmp.data().write(splat_offset, 0.0f32.expr());
                    tele_film_tmp.data().write(splat_offset, 0.0f32.expr());
                }
            }));
        let mix_kernel =
            self.device.create_kernel::<fn(f32, f32)>(&track!(
                |mc_scale: Expr<f32>, tele_scale: Expr<f32>| {
                    let p = dispatch_id().xy();
                    // mix by inverse variance
                    for c in 0u32..film.repr().nvalues() as u32 {
                        let idx = film.linear_index(p.cast_f32());
                        let splat_offset =
                            film.splat_offset() + idx * film.repr().nvalues() as u32 + c;
                        // let sample_offset = idx * film.repr().nvalues() as u32 + c;
                        // let main_mcmc = main_mcmc_film.data().read(splat_offset) * mc_scale;
                        // let main_mcmc_sqr = main_mcmc_sqr_film.data().read(splat_offset) * mc_scale;
                        let mcmc = mcmc_film.data().read(splat_offset) * mc_scale;
                        let mcmc_sqr = mcmc_sqr_film.data().read(splat_offset) * mc_scale;
                        let tele = tele_film.data().read(splat_offset) * tele_scale;
                        let tele_sqr = tele_sqr_film.data().read(splat_offset) * tele_scale;
                        if self.config.sample_all {
                            film.data().write(splat_offset, mcmc);
                        } else {
                            if self.config.mix != MixingMethod::None {
                                let normalize_weights = |a: Expr<f32>, b: Expr<f32>| {
                                    // let c = c * c;
                                    let sum = a + b;
                                    let sum = (sum != 0.0).select(sum, 1.0f32.expr());
                                    let a = a / sum;
                                    let b = b / sum;
                                    (a, b)

                                    // one hot
                                    // let w_a = a.gt(b).select(1.0f32.expr(), 0.0f32.expr());
                                    // let w_b = b.gt(a).select(1.0f32.expr(), 0.0f32.expr());
                                    // (w_a, w_b)
                                };

                                // let main_mcmc_var =
                                //     (main_mcmc_sqr - main_mcmc * main_mcmc).max_(0.0);
                                let mcmc_var = (mcmc_sqr - mcmc * mcmc).max_(0.0);
                                let tele_var = (tele_sqr - tele * tele).max_(0.0);

                                // let main_mcmc_weight = (main_mcmc_var + 1e-6f32.expr()).recip();
                                let mcmc_weight = (mcmc_var + 1e-6f32.expr()).recip();
                                let tele_weight = (tele_var + 1e-6f32.expr()).recip();
                                let (mcmc_weight, tele_weight) =
                                    normalize_weights(mcmc_weight, tele_weight);
                                let mix = mcmc * mcmc_weight + tele * tele_weight;

                                film.data().write(splat_offset, mix);
                                // main_mcmc_weight_film
                                //     .data()
                                //     .write(splat_offset, main_mcmc_weight);
                                mc_weight_film.data().write(splat_offset, mcmc_weight);
                                tele_weight_film.data().write(splat_offset, tele_weight);
                                mc_var_film.data().write(splat_offset, mcmc_var);
                                tele_var_film.data().write(splat_offset, tele_var);
                            } else {
                                film.data().write(splat_offset, tele);
                            }
                        }
                    }
                }
            ));
        let reconstruct = |film: &mut Film,
                           spp: u32,
                           main_mcmc_film: &mut Film,
                           mcmc_film: &mut Film,
                           mcmc_sqr_film: &mut Film,
                           tele_film: &mut Film,
                           tele_sqr_film: &mut Film,
                           _mix: MixingMethod| {
            let states = state.states.copy_to_vec();
            let mut b = state.b_init as f64;
            let mut b_cnt = state.b_init_cnt as u64;
            let mut accepted = 0u64;
            let mut mutations = 0u64;
            for s in &states {
                b += s.b as f64;
                b_cnt += s.b_cnt as u64;
                accepted += s.n_accepted as u64;
                mutations += s.n_mutations as u64;
            }
            let accept_rate = accepted as f64 / mutations as f64;
            let b = b / b_cnt as f64;
            log::info!("#indenpentent proposals: {}", b_cnt);
            log::info!("Normalization factor: {}", b);
            log::info!("Acceptance rate: {:.2}%", accept_rate * 100.0);

            let normalization = b as f32;
            let mc_scale = normalization / spp as f32;
            let tele_scale = normalization / spp as f32;

            main_mcmc_film.set_splat_scale(mc_scale);
            mcmc_film.set_splat_scale(mc_scale);
            mcmc_sqr_film.set_splat_scale(mc_scale);
            tele_film.set_splat_scale(tele_scale);
            tele_sqr_film.set_splat_scale(tele_scale);

            mix_kernel.dispatch([resolution.x, resolution.y, 1], &mc_scale, &tele_scale);

            if let Some(channel) = &session.display {
                film.copy_to_rgba_image(channel.screen_tex(), false);
                channel.notify_update();
            }
        };
        let mut acc_time = 0.0f64;
        let mut stats: RenderStats = Default::default();
        {
            let mut cnt = 0;
            let spp_per_pass = self.pt.spp_per_pass;
            let mutations_per_chain = (npixels as u64 / self.n_chains as u64).max(1);
            let contribution =
                { npixels as f64 / (mutations_per_chain as f64 * self.n_chains as f64) } as f32;
            if mutations_per_chain > u32::MAX as u64 {
                panic!("Number of mutations per chain exceeds u32::MAX, please reduce spp per pass or increase number of chains");
            }
            let progress = util::create_progess_bar(self.pt.spp as usize, "spp");
            while cnt < self.pt.spp {
                let tic = Instant::now();
                let cur_pass = (self.pt.spp - cnt).min(spp_per_pass);
                self.device.default_stream().with_scope(|s| {
                    for i in 0..cur_pass {
                        update_denoiser_kernel(s, cnt + i);
                        s.submit([
                            kernel.dispatch_async(
                                [self.n_chains as u32, 1, 1],
                                &(mutations_per_chain as u32),
                                &contribution,
                            ),
                            update_var.dispatch_async([resolution.x, resolution.y, 1], &(cnt + i)),
                        ])
                        .synchronize();
                    }
                });
                progress.inc(cur_pass as u64);
                cnt += cur_pass;
                let toc = Instant::now();
                acc_time += toc.duration_since(tic).as_secs_f64();
                if session.save_intermediate || session.display.is_some() {
                    reconstruct(
                        film,
                        cnt,
                        &mut main_mcmc_film,
                        &mut mcmc_film,
                        &mut mcmc_sqr_film,
                        &mut tele_film,
                        &mut tele_sqr_film,
                        self.config.mix,
                    );
                }
                if session.save_intermediate {
                    let output_image: Tex2d<Float4> = self.device.create_tex2d(
                        PixelStorage::Float4,
                        scene.camera.resolution().x,
                        scene.camera.resolution().y,
                        1,
                    );
                    film.copy_to_rgba_image(&output_image, true);
                    let path = format!("{}-{}.exr", session.name, cnt);
                    util::write_image(&output_image, &path);
                    stats.intermediate.push(IntermediateStats {
                        time: acc_time,
                        spp: cnt,
                        path,
                    });
                }
            }
            progress.finish();
            if session.save_stats {
                let file = File::create(format!("{}.json", session.name)).unwrap();
                let json = serde_json::to_value(&stats).unwrap();
                let writer = BufWriter::new(file);
                serde_json::to_writer(writer, &json).unwrap();
            }
        }

        log::info!("Rendering finished in {:.2}s", acc_time);
        reconstruct(
            film,
            self.pt.spp,
            &mut main_mcmc_film,
            &mut mcmc_film,
            &mut mcmc_sqr_film,
            &mut tele_film,
            &mut tele_sqr_film,
            self.config.mix,
        );
    }
}
impl Integrator for ShapeSplattingPssmlt {
    fn render(
        &self,
        scene: Arc<Scene>,
        sampler_config: SamplerConfig,
        color_pipeline: ColorPipeline,
        film: &mut Film,
        options: &RenderSession,
    ) {
        let resolution = scene.camera.resolution();
        log::info!(
            "Resolution {}x{}\nconfig: {:#?}",
            resolution.x,
            resolution.y,
            &self.config
        );

        assert_eq!(resolution.x, film.resolution().x);
        assert_eq!(resolution.y, film.resolution().y);
        if self.config.direct_spp > 0 {
            log::info!(
                "Rendering direct illumination: {} spp",
                self.config.direct_spp
            );
            let direct = PathTracer::new(
                self.device.clone(),
                pt::Config {
                    max_depth: 1,
                    rr_depth: 1,
                    spp: self.config.direct_spp as u32,
                    indirect_only: false,
                    spp_per_pass: self.pt.spp_per_pass,
                    use_nee: self.pt.use_nee,
                    ..Default::default()
                },
            );
            direct.render(
                scene.clone(),
                sampler_config,
                color_pipeline,
                film,
                &Default::default(),
            );
        }
        let render_state = self.bootstrap(&scene, film.filter(), color_pipeline);
        self.render_loop(&scene, color_pipeline, &render_state, film, options);
    }
}

pub fn render(
    device: Device,
    scene: Arc<Scene>,
    sampler: SamplerConfig,
    color_pipeline: ColorPipeline,
    film: &mut Film,
    config: &Config,
    options: &RenderSession,
) {
    let mcmc = ShapeSplattingPssmlt::new(device.clone(), config.clone());
    mcmc.render(scene, sampler, color_pipeline, film, options);
}
