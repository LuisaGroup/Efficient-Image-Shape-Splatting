use std::cell::RefCell;
use std::collections::HashSet;
use std::env;
use std::f32::consts::PI;
use std::fs::File;
use std::io::BufWriter;
use std::sync::Arc;
use std::time::Instant;

use akari_render::luisa::runtime::KernelBuildOptions;
use akari_render::rand::Rng;
use akari_render::serde::de;
use akari_render::svm::surface::{BsdfEvalContext, Surface};
use akari_render::util::distribution::{AliasTableEntry, BindlessAliasTableVar};
// use akari_render::luisa::lang::debug::is_cpu_backend;
// use akari_render::shared::Shared;
use serde::{Deserialize, Serialize};

use super::pt::PathTracerBase;
use super::{Integrator, RenderSession};
use crate::geometry::Ray;

use crate::pt::{
    DenoisingFeatures, ReconnectionShiftMapping, ReconnectionVertex, SurfaceHit, VertexType,
};
use crate::util::distribution::AliasTable;
use crate::util::{is_power_of_four, morton2d};
use crate::{color::*, sampler::*, *};

#[derive(Clone)]
pub struct ShapeSplattingPathTracer {
    pub device: Device,
    pub spp: u32,
    pub max_depth: u32,
    pub spp_per_pass: u32,
    pub use_nee: bool,
    pub rr_depth: u32,
    pub indirect_only: bool,
    pub pixel_offset: Int2,
    pub force_diffuse: bool,
    pub n_shift_pixels: u32,

    config: Config,
}
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
#[serde(crate = "serde")]
pub enum BaseShape {
    #[serde(rename = "square")]
    Square,
    #[serde(rename = "circle")]
    Circle,
}
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
#[serde(crate = "serde")]
pub enum SpatialCurve {
    #[serde(rename = "morton")]
    Morton,
    #[serde(rename = "hilbert")]
    Hilbert,
}
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq)]
#[serde(crate = "serde")]
pub enum SplatMethod {
    #[serde(rename = "lerp")]
    Lerp,
    #[serde(rename = "subspace")]
    Subspace,
    #[serde(rename = "voronoi")]
    Voronoi,
    #[serde(rename = "inv_dist")]
    InverseDistance,
}
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq)]
#[serde(crate = "serde")]
pub enum MixingMethod {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "min")]
    Min,
    #[serde(rename = "inv_var")]
    InverseVariance,
    #[serde(rename = "inv_var2")]
    InverseVarianceNoPt,
    #[serde(rename = "uniform")]
    Uniform,
}
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
#[serde(default, crate = "serde")]
pub struct Config {
    pub spp: u32,
    pub max_depth: u32,
    pub spp_per_pass: u32,
    pub use_nee: bool,
    pub rr_depth: u32,
    pub indirect_only: bool,
    pub force_diffuse: bool,
    pub pixel_offset: [i32; 2],
    pub shape_width: u32,
    pub curve: SpatialCurve,
    pub shape: BaseShape,
    pub mix: MixingMethod,
    pub max_sampled_length: Option<u32>,
    pub stride: f32,
    pub reconnect: bool,
    pub min_dist: f32,
    pub min_roughness: f32,
    pub pmf_power: f32,
    pub denoiser_kernel: bool,
    ///
    pub async_compile: bool,
    pub sample_all: bool,
    pub randomize: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            spp: 256,
            max_depth: 7,
            rr_depth: 5,
            spp_per_pass: 64,
            use_nee: true,
            indirect_only: false,
            force_diffuse: false,
            pixel_offset: [0, 0],
            shape_width: 6,
            max_sampled_length: None,
            curve: SpatialCurve::Hilbert,
            shape: BaseShape::Square,
            mix: MixingMethod::InverseVariance,
            stride: 1.0,
            reconnect: false,
            min_dist: 0.0,
            min_roughness: 0.0,
            pmf_power: -2.0,
            denoiser_kernel: true,

            sample_all: false,
            randomize: true,
            async_compile: false,
        }
    }
}
impl ShapeSplattingPathTracer {
    pub fn new(device: Device, config: Config) -> Self {
        assert!(config.pmf_power <= 0.0);
        // center square numbers
        let n_shift_pixels = match config.shape {
            BaseShape::Square => config.shape_width.pow(2) + (config.shape_width - 1).pow(2) - 1,
            _ => todo!(),
        };
        Self {
            device,
            spp: config.spp,
            max_depth: config.max_depth,
            spp_per_pass: config.spp_per_pass,
            use_nee: config.use_nee,
            rr_depth: config.rr_depth,
            indirect_only: config.indirect_only,
            force_diffuse: config.force_diffuse,
            pixel_offset: Int2::new(config.pixel_offset[0], config.pixel_offset[1]),
            n_shift_pixels,
            config,
        }
    }
}

#[derive(Copy, Clone)]
struct RenderState<'a> {
    t_lo: i32,
    t_hi: i32,
    base_shape_center: u32,
    pmf_ats: &'a BindlessArray,
    tele_film: &'a Film,
    pt_film: &'a Film,
    pt_sqr_film: &'a Film,
    mc_film: &'a Film,
    mc_replay_film: &'a Film,
    mc_sqr_film: &'a Film,
    mc_replay_sqr_film: &'a Film,
    debug_film: &'a Film,
    albedo_buf: &'a Tex2d<Float4>,
    normal_buf: &'a Tex2d<Float4>,
    first_hit: &'a Buffer<pt::SurfaceHit>,
    reconnect_vertex_buf: &'a Buffer<ReconnectionVertex>,
    path_states: &'a Buffer<PathState>,
    path_counter_buf: &'a Buffer<u32>,
    path_sm_replay: &'a ColorBuffer,
    path_sm_reconnect: &'a ColorBuffer,
    path_jacobians: &'a Buffer<f32>,
    sampled_indices: &'a Buffer<i8>,
    shifted_path_states: &'a Buffer<ShiftedPathState>,
    sort_tmp: &'a Buffer<i8>,
    rotation: &'a Buffer<f32>,
    shapes: &'a Buffer<Shape>,
    shape_indices: &'a Buffer<i8>,
    curve: &'a Buffer<Int2>,
    rng_buf: &'a Buffer<Pcg32>,
    avg_per_pixel: &'a Buffer<f32>,
}
#[derive(Copy, Clone, Value)]
#[luisa(crate = "luisa")]
#[repr(C)]
pub struct Shape {
    // offset: u32,
    pub center: u8,
    pub n_pixels: u8,
}
#[derive(Copy, Clone, Value)]
#[luisa(crate = "luisa")]
#[repr(C, align(16))]
struct PathState {
    k: u32,
    pmf_k: f32,
    offset: u32,
    center: u32,
}
#[derive(Copy, Clone, Value)]
#[luisa(crate = "luisa")]
#[repr(C)]
struct ShiftedPathState {
    ps_idx: u32,
    sampled_index: i32,
    color_index: u32,
}
impl ShapeSplattingPathTracer {
    #[tracked(crate = "luisa")]
    fn radiance(
        &self,
        scene: &Arc<Scene>,
        color_pipeline: ColorPipeline,
        ray: Expr<Ray>,
        swl: Var<SampledWavelengths>,
        sampler: &dyn Sampler,
        sm: Option<&ReconnectionShiftMapping>,
    ) -> (Color, Color, DenoisingFeatures, Expr<u32>) {
        let mut pt = PathTracerBase::new(
            scene,
            color_pipeline,
            self.max_depth.expr(),
            self.rr_depth.expr(),
            self.use_nee,
            self.indirect_only,
            swl,
        );
        pt.need_shift_mapping = self.config.reconnect;
        pt.force_diffuse = self.force_diffuse;
        pt.denoising = Some(DenoisingFeatures {
            first_hit_albedo: ColorVar::zero(color_pipeline.color_repr),
            first_hit_normal: Float3::var_zeroed(),
            first_hit_roughness: 0.0f32.var(),
            first_hit: {
                let v = SurfaceHit::var_zeroed();
                *v.inst_id = u32::MAX;
                v
            },
        });
        pt.run_pt_hybrid_shift_mapping(ray, sampler, sm, None);
        let replay = if self.config.reconnect {
            pt.base_replay_throughput.load()
        } else {
            pt.radiance.load()
        };
        (
            replay,
            pt.radiance.load() - replay,
            pt.denoising.unwrap(),
            **pt.depth,
        )
    }
    #[tracked(crate = "luisa")]
    fn kernel_compute_auxillary(
        &self,
        scene: Arc<Scene>,
        color_pipeline: ColorPipeline,
        _sampler_creator: &dyn SamplerCreator,
        state: RenderState,
        spp: Expr<u32>,
    ) {
        let resolution = scene.camera.resolution();
        set_block_size([256, 1, 1]);
        let px_idx = dispatch_id().x;
        let p = {
            let p = Uint2::expr(px_idx % resolution.x, px_idx / resolution.x);
            p
        };
        let rng = IndependentSampler::from_pcg32(state.rng_buf.read(px_idx).var());
        let acc_normal = Float3::var_zeroed();
        let acc_albedo = ColorVar::zero(color_pipeline.color_repr);
        for _ in 0u32.expr()..spp {
            let swl = sample_wavelengths(color_pipeline.color_repr, &rng);
            let (ray, _) = scene.camera.generate_ray(
                &scene,
                state.mc_film.filter(),
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
        state.rng_buf.write(px_idx, rng.state);
    }
    #[tracked(crate = "luisa")]
    fn default_shape(&self, state: RenderState) -> Expr<Shape> {
        let n = self.n_shift_pixels.expr() + 1;
        Shape::from_comps_expr(ShapeComps {
            center: state.base_shape_center.expr().as_u8(),
            n_pixels: n.as_u8(),
        })
    }
    #[tracked(crate = "luisa")]
    fn sample_indices(
        &self,
        state: RenderState,
        rng: &IndependentSampler,
        px_idx: Expr<u32>,
        shape: Expr<Shape>,
        tmp_buf_offset: Expr<u32>,
        sort_buf_offset: Expr<u32>,
        shifted_buf_offset: Expr<u32>,
        k: Expr<u32>,
        n: Expr<u32>,
    ) -> Expr<u32> {
        let path_states = state.path_states;
        /*
         *   Sample indices
         */
        let center = u32::MAX.var();

        let sampled_indices = state.sampled_indices;
        let sort_tmp = state.sort_tmp;
        if k > 0 {
            let use_reservoir = true;
            if use_reservoir {
                // sample k integrers from [0, n)
                // initialize the resevoir
                for i in 0u32.expr()..k {
                    sampled_indices.write(tmp_buf_offset + i, i.as_i8());
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
                        sampled_indices.write(tmp_buf_offset + j, i.as_i8());
                        *w *= (rng.next_1d().ln() / k.as_f32()).exp();
                    }
                }
            } else {
                // shuffle n times
                for i in 0u32.expr()..n {
                    sampled_indices.write(tmp_buf_offset + i, i.as_i8());
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
            // sort the indices using counting sort
            for i in 0u32.expr()..n {
                sort_tmp.write(sort_buf_offset + i, 0);
            }
            for i in 0u32.expr()..k {
                let idx = sampled_indices.read(tmp_buf_offset + i);
                sort_tmp.write(sort_buf_offset + idx.as_u32(), 1);
            }
            let cnt = 0u32.var();
            for i in 0u32.expr()..n {
                let contained = sort_tmp.read(sort_buf_offset + i) != 0;
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

                    // lc_assert!(j.ne(0));
                    let idx = ((center == u32::MAX) | (cnt < center)).select(**cnt, cnt + 1);
                    // lc_assert!(j.ge(t_lo));
                    // lc_assert!(j.lt(t_hi));
                    sampled_indices.write(tmp_buf_offset + idx, j.as_i8());
                    let shifted = ShiftedPathState::var_zeroed();
                    *shifted.ps_idx = px_idx;
                    *shifted.sampled_index = j;
                    *shifted.color_index = tmp_buf_offset + idx;
                    state
                        .shifted_path_states
                        .write(shifted_buf_offset + cnt, shifted);
                    *cnt += 1;
                }
            }
            // lc_assert!(cnt.eq(k));
            if center == u32::MAX {
                *center = k;
            }
        } else {
            *center = 0;
        }
        {
            let ps = path_states.read(px_idx).var();
            *ps.center = center;
            *ps.offset = tmp_buf_offset;
            path_states.write(px_idx, ps);
        }
        if k > 0 {
            sampled_indices.write(tmp_buf_offset + center, 0);
        }
        **center
    }
    #[tracked(crate = "luisa")]
    fn is_shape_full(&self, s: Expr<Shape>) -> Expr<bool> {
        s.n_pixels.as_u32() == (self.n_shift_pixels + 1u32)
    }

    #[tracked(crate = "luisa")]
    fn kernel_compute_denoiser_kernel(&self, scene: Arc<Scene>, state: RenderState) {
        let resolution = scene.camera.resolution();
        set_block_size([64, 1, 1]);

        let px_idx = dispatch_id().x;
        let p = {
            let p = Uint2::expr(px_idx % resolution.x, px_idx / resolution.x);
            p
        };

        let p_from_t = self.get_p_from_t_raw(p, resolution.expr(), state, false);

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
                let p = p_from_t(t);
                if (p < Int2::expr(0, 0)).any()
                    | (p >= scene.camera.resolution().expr().cast_i32()).any()
                {
                    continue;
                }
                let p = p.cast_u32();
                let normal = state.normal_buf.read(p).xyz() * 2.0 - 1.0;
                let albedo = state.albedo_buf.read(p).xyz();

                let group_normal = normal.dot(center_normal) >= 0.707;
                let group_albedo = (center_albedo - albedo).abs().reduce_max() < 0.1;
                if (!self.config.denoiser_kernel) | (group_normal & group_albedo) {
                    *write = true;
                }
            }
            if write {
                state
                    .shape_indices
                    .write(offset * (1 + self.n_shift_pixels) + cnt, t.cast_i8());
                *cnt += 1;
            }
        }
        *shape.n_pixels = cnt.as_u8();
        state.shapes.write(offset, shape);
    }
    #[tracked(crate = "luisa")]
    fn trace(
        &self,
        scene: &Scene,
        state: &RenderState,
        color_pipeline: ColorPipeline,
        sampler: &dyn Sampler,
        pixel: Expr<Uint2>,
        is_primary: bool,
        shift_mapping: &ReconnectionShiftMapping,
    ) -> (Expr<SampledWavelengths>, Color) {
        let swl_v = SampledWavelengths::var_zeroed();
        let indirect = ColorVar::zero(color_pipeline.color_repr);
        outline(|| {
            sampler.forget();
            let swl = sample_wavelengths(color_pipeline.color_repr, sampler).var();
            let (ray, _ray_w) = scene.camera.generate_ray(
                &scene,
                state.pt_film.filter(),
                pixel,
                sampler,
                color_pipeline.color_repr,
                **swl,
            );
            let swl = swl.var();
            let mut pt = PathTracerBase::new(
                scene,
                color_pipeline,
                self.config.max_depth.expr(),
                self.config.rr_depth.expr(),
                self.config.use_nee,
                true,
                swl,
            );
            pt.need_shift_mapping = true;
            *shift_mapping.is_base_path = is_primary;
            pt.run_pt_hybrid_shift_mapping(
                ray,
                sampler,
                if self.config.reconnect {
                    Some(shift_mapping)
                } else {
                    None
                },
                None,
            );
            let direct = pt.base_replay_throughput.load();
            indirect.store(pt.radiance.load() - direct);
            *swl_v = swl;
        });
        // let indirect = Color::one(state.color_pipeline.color_repr);//# - direct;
        (**swl_v, indirect.load())
    }
    /**
     * Sample a primary path, store the reconnection vertex
     * Store the albedo
     * Sample telescoping path indices
     */
    #[tracked(crate = "luisa")]
    fn kernel_sample_primary(
        &self,
        scene: Arc<Scene>,
        color_pipeline: ColorPipeline,
        sampler_creator: &dyn SamplerCreator,
        state: RenderState,
    ) {
        let resolution = scene.camera.resolution();
        set_block_size([256, 1, 1]);
        let px_idx = dispatch_id().x;
        let p = {
            let p = Uint2::expr(px_idx % resolution.x, px_idx / resolution.x);
            p
        };

        let rng = IndependentSampler::from_pcg32(state.rng_buf.read(px_idx).var());

        // let pmf_at = state.pmf_at;
        let path_states = state.path_states;

        /*
         *   Sample shape config
         */
        let shape = {
            if self.config.denoiser_kernel {
                state.shapes.read(px_idx)
            } else {
                self.default_shape(state)
            }
        };
        let n = shape.n_pixels.as_u32() - 1;
        let k = if n > 0 {
            let entries = state.pmf_ats.buffer::<AliasTableEntry>(n * 2);
            let pmf = state.pmf_ats.buffer::<f32>(n * 2 + 1);
            let pmf_at = BindlessAliasTableVar(entries, pmf);
            let (k, pmf_k, _) = pmf_at.sample_and_remap(rng.next_1d());
            lc_assert!(k.lt(n));
            let k = k + 1;
            let ps = path_states.read(px_idx).var();
            *ps.k = k;
            *ps.pmf_k = pmf_k;
            path_states.write(px_idx, ps);
            k
        } else {
            let ps = path_states.read(px_idx).var();
            *ps.pmf_k = 1.0f32.expr();
            *ps.k = 0;
            path_states.write(px_idx, ps);
            0u32.expr()
        };
        // Perform atomic increment on path counter
        let tmp_buf_offset = state.path_counter_buf.atomic_fetch_add(0, 1 + k);
        let shifted_buf_offset = state.path_counter_buf.atomic_fetch_add(1, k);
        let sort_buf_offset = px_idx * (1 + self.n_shift_pixels);
        let center = self.sample_indices(
            state,
            &rng,
            px_idx,
            shape,
            tmp_buf_offset,
            sort_buf_offset,
            shifted_buf_offset,
            k,
            n,
        );
        state
            .reconnect_vertex_buf
            .write(px_idx, ReconnectionVertex::var_zeroed());
        // trace primary path
        let shift_mapping = if self.config.reconnect {
            Some(ReconnectionShiftMapping {
                min_dist: self.config.min_dist.expr(),
                min_roughness: self.config.min_roughness.expr(),
                is_base_path: true.var(),
                read_vertex: Box::new(|| state.reconnect_vertex_buf.read(px_idx)),
                write_vertex: Box::new(|v| state.reconnect_vertex_buf.write(px_idx, v)),
                jacobian: 0.0f32.var(),
                success: false.var(),
            })
        } else {
            None
        };
        let sampler = sampler_creator.create(p);
        sampler.forget();
        sampler.start();
        let (primary_l, primary_reconnect, swl, _primay_ray_w, _, _, first_hit) = self.trace_path(
            scene.clone(),
            &sampler,
            color_pipeline,
            state,
            sampler_creator,
            p,
            true.expr(),
            0i32.expr(),
            shift_mapping.as_ref(),
        );

        {
            state.first_hit.write(px_idx, first_hit);
            let avg_per_pixel = state.avg_per_pixel.read(shape.n_pixels.as_u32() - 1);

            state
                .path_sm_replay
                .write(tmp_buf_offset + center, primary_l, swl);

            state
                .path_jacobians
                .write(tmp_buf_offset + center, 1.0f32.expr());

            state
                .path_sm_reconnect
                .write(tmp_buf_offset + center, primary_reconnect, swl);

            state.mc_film.add_splat(
                p.cast_f32(),
                &(primary_l + primary_reconnect),
                swl,
                1.0 / avg_per_pixel,
            );

            {
                let l = primary_l + primary_reconnect;
                state
                    .pt_film
                    .add_sample(p.cast_f32(), &l, swl, 1.0f32.expr());
                state
                    .pt_sqr_film
                    .add_sample(p.cast_f32(), &(l * l), swl, 1.0f32.expr());
            }

            state.debug_film.add_sample(
                p.cast_f32(),
                &(Color::one(color_pipeline.color_repr)
                    * (shape.n_pixels.as_f32() / (1.0f32 + self.n_shift_pixels as f32))),
                swl,
                1.0f32.expr(),
            );
        }

        state.rng_buf.write(px_idx, rng.state);
    }
    #[tracked(crate = "luisa")]
    fn kernel_sample_shifted(
        &self,
        scene: Arc<Scene>,
        color_pipeline: ColorPipeline,
        sampler_creator: &dyn SamplerCreator,
        state: RenderState,
    ) {
        let resolution = scene.camera.resolution();
        set_block_size([256, 1, 1]);
        let tid = dispatch_id().x;
        if tid >= state.path_counter_buf.read(1) {
            return;
        }
        let shifted = state.shifted_path_states.read(tid);
        let px_idx = shifted.ps_idx;
        // device_log!("{} {}", px_idx, tid);

        let p = {
            let p = Uint2::expr(px_idx % resolution.x, px_idx / resolution.x);
            p
        };
        let sm = if self.config.reconnect {
            Some(ReconnectionShiftMapping {
                min_dist: self.config.min_dist.expr(),
                min_roughness: self.config.min_roughness.expr(),
                is_base_path: false.var(),
                read_vertex: Box::new(|| state.reconnect_vertex_buf.read(px_idx)),
                write_vertex: Box::new(|_| {}),
                jacobian: 0.0f32.var(),
                success: false.var(),
            })
        } else {
            None
        };
        let sampler = sampler_creator.create(p);
        sampler.start();
        sampler.forget();

        let t = shifted.sampled_index;
        let p_from_t: Box<dyn Fn(Expr<i32>) -> Expr<Vector<i32, 2>>> =
            self.get_p_from_t(p, resolution.expr(), state, self.config.denoiser_kernel);
        let proposal_p = p_from_t(t).cast_f32();
        let proposal_p_idx = proposal_p.x.as_u32() + proposal_p.y.as_u32() * resolution.x;

        let (l, reconnect, swl, _ray_w, jacobian, _, _) = self.trace_path(
            scene.clone(),
            &sampler,
            color_pipeline,
            state,
            sampler_creator,
            p,
            false.expr(),
            t,
            sm.as_ref(),
        );
        // let jacobian = 1.0f32.expr();//(jacobian > 0.0).select(1.0f32.expr(), 0.0f32.expr());

        let ps = state.path_states.read(px_idx);
        let buf_offset = ps.offset;
        // let shape = state.shapes.read(px_idx);
        let (primary_l, primary_swl) = state.path_sm_replay.read(buf_offset + ps.center);
        let (primary_reconnect, _) = state.path_sm_reconnect.read(buf_offset + ps.center);
        let cur_p = p;
        let shape = if self.config.denoiser_kernel {
            state.shapes.read(px_idx)
        } else {
            self.default_shape(state)
        };

        let avg_per_pixel = state.avg_per_pixel.read(shape.n_pixels.as_u32() - 1);

        let proposal_shape = if self.config.denoiser_kernel {
            state.shapes.read(proposal_p_idx)
        } else {
            self.default_shape(state)
        };
        {
            let scale = 1.0 / avg_per_pixel;
            let acc_proposal_replay = ColorVar::zero(primary_l.repr());
            let acc_proposal_replay_w = 0.0f32.var();
            let acc_proposal_reconnect = ColorVar::zero(primary_l.repr());
            let acc_proposal_reconnect_w = 0.0f32.var();
            let acc_primary_replay = ColorVar::zero(primary_l.repr());
            let acc_primary_replay_w = 0.0f32.var();
            let acc_primary_reconnect = ColorVar::zero(primary_l.repr());
            let acc_primary_reconnect_w = 0.0f32.var();

            let acc_sqr_proposal_replay = ColorVar::zero(primary_l.repr());
            let acc_sqr_proposal_replay_w = 0.0f32.var();
            let acc_sqr_proposal_reconnect = ColorVar::zero(primary_l.repr());
            let acc_sqr_proposal_reconnect_w = 0.0f32.var();
            let acc_sqr_primary_replay = ColorVar::zero(primary_l.repr());
            let acc_sqr_primary_replay_w = 0.0f32.var();
            let acc_sqr_primary_reconnect = ColorVar::zero(primary_l.repr());
            let acc_sqr_primary_reconnect_w = 0.0f32.var();

            self.splat_mcmc_contribution(
                shape,
                proposal_shape,
                primary_l,
                primary_reconnect,
                l,
                reconnect,
                jacobian,
                |v, w| {
                    acc_primary_replay.store(acc_primary_replay.load() + v * w);
                    *acc_primary_replay_w += w;
                    acc_sqr_primary_replay.store(acc_sqr_primary_replay.load() + v * v * w);
                    *acc_sqr_primary_replay_w += w;
                },
                |v, w| {
                    acc_primary_reconnect.store(acc_primary_reconnect.load() + v * w);
                    *acc_primary_reconnect_w += w;
                    acc_sqr_primary_reconnect.store(acc_sqr_primary_reconnect.load() + v * v * w);
                    *acc_sqr_primary_reconnect_w += w;
                },
                |v, w| {
                    acc_proposal_replay.store(acc_proposal_replay.load() + v * w);
                    *acc_proposal_replay_w += w;
                    acc_sqr_proposal_replay.store(acc_sqr_proposal_replay.load() + v * v * w);
                    *acc_sqr_proposal_replay_w += w;
                },
                |v, w| {
                    acc_proposal_reconnect.store(acc_proposal_reconnect.load() + v * w);
                    *acc_proposal_reconnect_w += w;
                    acc_sqr_proposal_reconnect.store(acc_sqr_proposal_reconnect.load() + v * v * w);
                    *acc_sqr_proposal_reconnect_w += w;
                },
            );

            state.mc_film.add_splat(
                proposal_p,
                &(acc_proposal_replay.load() + acc_proposal_reconnect.load()),
                swl,
                scale,
            );
            state.mc_film.add_splat(
                cur_p.cast_f32(),
                &(acc_primary_replay.load() + acc_primary_reconnect.load()),
                primary_swl,
                scale,
            );
        }

        let shifted = state.shifted_path_states.read(tid);
        state.path_sm_replay.write(shifted.color_index, l, swl);
        state
            .path_sm_reconnect
            .write(shifted.color_index, reconnect, swl);
        state.path_jacobians.write(shifted.color_index, jacobian);
    }
    #[tracked(crate = "luisa")]
    fn kernel_telelscoping_sum(
        &self,
        scene: Arc<Scene>,
        color_pipeline: ColorPipeline,
        _sampler_creator: &dyn SamplerCreator,
        state: RenderState,
    ) {
        let resolution = scene.camera.resolution();
        set_block_size([256, 1, 1]);
        let px_idx = dispatch_id().x;
        let p = {
            let p = Uint2::expr(px_idx % resolution.x, px_idx / resolution.x);
            p
        };
        let ps = state.path_states.read(px_idx);
        let k = ps.k;
        let pmf_k = ps.pmf_k;
        let tmp_buf_offset = ps.offset;
        let center = ps.center;
        let shape = if self.config.denoiser_kernel {
            state.shapes.read(px_idx)
        } else {
            self.default_shape(state)
        };
        let p_from_t = self.get_p_from_t(p, resolution.expr(), state, self.config.denoiser_kernel);
        let (primary_l, swl) = state.path_sm_replay.read(tmp_buf_offset + ps.center);
        let p_from_t = &p_from_t;
        let (primary_reconnect, _) = state.path_sm_reconnect.read(tmp_buf_offset + ps.center);
        let n = shape.n_pixels - 1;
        let t_lo = -(shape.center.cast_i32());
        let t_hi = (1 + n - shape.center).cast_i32();
        // device_log!(
        //     "{} {} {}",
        //     shape,
        //     primary_l.as_rgb(),
        //     primary_reconnect.as_rgb()
        // );
        if !self.config.sample_all {
            if shape.n_pixels == 1 {
                state.tele_film.add_splat(
                    p.cast_f32(),
                    &(primary_l + primary_reconnect),
                    swl,
                    1.0f32.expr(),
                );
            } else {
                self.telescoping_lerp(
                    state,
                    scene.camera.resolution().expr(),
                    state.shapes.var(),
                    state.path_states.var(),
                    p.cast_f32(),
                    primary_l,
                    primary_reconnect,
                    swl,
                    &state.path_sm_replay,
                    &state.path_sm_reconnect,
                    &state.path_jacobians,
                    &state.sampled_indices,
                    tmp_buf_offset,
                    tmp_buf_offset + k + 1,
                    tmp_buf_offset + center,
                    p_from_t,
                    t_lo,
                    t_hi,
                    1.0 / shape.n_pixels.as_f32(),
                    1.0f32 / pmf_k,
                    k.as_f32(),
                    |p, l, swl, scale| {
                        state.tele_film.add_splat(p, &l, swl, scale);
                    },
                );
            }
        }
    }
    #[tracked(crate = "luisa")]
    fn get_p_from_t_raw<'a>(
        &'a self,
        p: Expr<Uint2>,
        resolution: Expr<Uint2>,
        state: RenderState<'a>,
        use_shape_indices: bool,
    ) -> Box<dyn Fn(Expr<i32>) -> Expr<Int2> + 'a> {
        let n = self.n_shift_pixels.expr();

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
                if self.is_shape_full(shape) {
                    i
                } else {
                    let center = shape.center;
                    let t = i + center.as_i32();
                    let shape_indices = state.shape_indices;
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
            // lc_assert!(i.ge(0));
            // if i.as_u32() >= curve.len_expr_u32() {
            //     device_log!("i = {}, curve.len = {}", i, curve.len_expr_u32());
            // }
            let morton_p = curve.read(i.as_u32());

            let ip = p.cast_i32() + morton_p;
            ip
        };
        Box::new(p_from_t) as Box<dyn Fn(Expr<i32>) -> Expr<Int2>>
    }
    #[tracked(crate = "luisa")]
    fn get_p_from_t<'a>(
        &'a self,
        p: Expr<Uint2>,
        resolution: Expr<Uint2>,
        state: RenderState<'a>,
        use_shape_indices: bool,
    ) -> Box<dyn Fn(Expr<i32>) -> Expr<Int2> + 'a> {
        let n = self.n_shift_pixels.expr();

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
                if self.is_shape_full(shape) {
                    i
                } else {
                    let center = shape.center;
                    let t = i + center.as_i32();
                    let shape_indices = state.shape_indices;
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
            // lc_assert!(i.ge(0));
            // if i.as_u32() >= curve.len_expr_u32() {
            //     device_log!("i = {}, curve.len = {}", i, curve.len_expr_u32());
            // }
            let morton_p = curve.read(i.as_u32());

            let ip = p.cast_i32() + morton_p;
            (ip + resolution.cast_i32()) % resolution.cast_i32()
        };
        Box::new(p_from_t) as Box<dyn Fn(Expr<i32>) -> Expr<Int2>>
    }
    #[tracked(crate = "luisa")]
    fn trace_path(
        &self,
        scene: Arc<Scene>,
        sampler: &Box<dyn Sampler>,
        color_pipeline: ColorPipeline,
        state: RenderState,
        sampler_creator: &dyn SamplerCreator,
        p: Expr<Uint2>,
        is_primary: Expr<bool>,
        t: Expr<i32>,
        shift_mapping: Option<&ReconnectionShiftMapping>,
    ) -> (
        Color,
        Color,
        Expr<SampledWavelengths>,
        Expr<f32>,
        Expr<f32>,
        Color,
        Expr<SurfaceHit>,
    ) {
        let l = ColorVar::zero(color_pipeline.color_repr);
        let sm_throughput = ColorVar::zero(color_pipeline.color_repr);
        let swl = Var::<SampledWavelengths>::zeroed();
        let ray_w = 0.0f32.var();
        let jacobian = 0.0f32.var();
        let albedo = ColorVar::zero(color_pipeline.color_repr);
        let first_hit_var = SurfaceHit::var_zeroed();
        outline(|| {
            let sampler = sampler.as_ref();
            let ip = if is_primary {
                p.cast_i32()
            } else {
                let p_from_t = self.get_p_from_t(
                    p,
                    scene.camera.resolution().expr(),
                    state,
                    self.config.denoiser_kernel,
                );
                p_from_t(t)
            };
            let ray_v = Ray::var_zeroed();
            let sample_primary_ray = |sampler: &dyn Sampler| {
                *swl = sample_wavelengths(color_pipeline.color_repr, sampler);

                let (ray, ray_w_) = scene.camera.generate_ray(
                    &scene,
                    state.mc_film.filter(),
                    ip.cast_u32(),
                    sampler,
                    color_pipeline.color_repr,
                    **swl,
                );
                *ray_v = ray;
                *ray_w = ray_w_;
            };
            sample_primary_ray(sampler);

            if let Some(shift_mapping) = &shift_mapping {
                *shift_mapping.is_base_path = is_primary;
            }
            let (radiance, reconnect, features, depth) =
                self.radiance(&scene, color_pipeline, **ray_v, swl, sampler, shift_mapping);

            if let Some(sm) = &shift_mapping {
                *jacobian = sm.jacobian;
                sm_throughput.store(reconnect);
            } else {
                *jacobian = 1.0f32.expr();
            }
            albedo.store(features.first_hit_albedo.load());
            l.store(radiance);
            *first_hit_var = features.first_hit;
        });

        (
            l.load(),
            sm_throughput.load(),
            **swl,
            ray_w.load(),
            **jacobian,
            albedo.load(),
            **first_hit_var,
        )
    }
    #[tracked(crate = "luisa")]
    fn splat_mcmc_contribution(
        &self,
        cur_shape: Expr<Shape>,
        proposal_shape: Expr<Shape>,
        primary_replay: Color,
        primary_reconnect: Color,
        shifted_replay: Color,
        shifted_reconnect: Color,
        jacobian: Expr<f32>,
        acc_primary_replay: impl Fn(Color, Expr<f32>),
        acc_primary_reconnect: impl Fn(Color, Expr<f32>),
        acc_proposal_replay: impl Fn(Color, Expr<f32>),
        acc_proposal_reconnect: impl Fn(Color, Expr<f32>),
    ) {
        let ratio = cur_shape.n_pixels.as_f32() / proposal_shape.n_pixels.as_f32();
        let accept = (jacobian * ratio).clamp(0.0f32.expr(), 1.0f32.expr());

        let replay_accept = ratio.clamp(0.0f32.expr(), 1.0f32.expr());

        acc_primary_reconnect(primary_reconnect, 1.0 - accept);
        acc_proposal_reconnect(shifted_reconnect, accept);

        acc_primary_replay(primary_replay, 1.0 - replay_accept);
        acc_proposal_replay(shifted_replay, replay_accept);
    }
    #[tracked(crate = "luisa")]
    fn telescoping_lerp(
        &self,
        state: RenderState,
        resolution: Expr<Uint2>,
        shapes: BufferVar<Shape>,
        path_states: BufferVar<PathState>,
        cur_p: Expr<Float2>,
        primary_l: Color,
        primary_reconnect: Color,
        swl: Expr<SampledWavelengths>,
        color_buf: &ColorBuffer,
        path_sm_throughput: &ColorBuffer,
        path_jacobians: &BufferVar<f32>,
        t_buf: &BufferVar<i8>,
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

            let acc_cur_replay = ColorVar::zero(primary_l.repr());
            let acc_cur_reconnect = ColorVar::zero(primary_l.repr());
            let p_idx = cur_p.x.as_u32() + cur_p.y.as_u32() * resolution.x;
            let shape = if self.config.denoiser_kernel {
                shapes.read(p_idx)
            } else {
                self.default_shape(state)
            };
            while t < t_hi {
                // TODO: check this!
                if !is_last {
                    loop {
                        if next_idx() >= idx_hi {
                            *is_last = true;
                            break;
                        }
                        if t_buf.read(next_idx()).as_i32() > t {
                            break;
                        }
                        *idx = next_idx();
                    }
                };
                let next = next_idx();
                let ip = p_from_t(**t);
                let acc_proposal_replay = ColorVar::zero(primary_l.repr());
                let acc_proposal_reconnect = ColorVar::zero(primary_l.repr());
                let proposal_shape = if self.config.denoiser_kernel {
                    let p_idx = ip.x.as_u32() + ip.y.as_u32() * resolution.x;
                    shapes.read(p_idx)
                } else {
                    self.default_shape(state)
                };
                let compute_contribution = |primary_direct: Color,
                                            primary_indirect: Color,
                                            shifted_direct: Color,
                                            shifted_indirect: Color,
                                            jacobian: Expr<f32>,
                                            scale: Expr<f32>|
                 -> [Color; 4] {
                    let cur_replay = ColorVar::zero(primary_l.repr());
                    let cur_reconnect = ColorVar::zero(primary_l.repr());
                    let proposal_replay = ColorVar::zero(primary_l.repr());
                    let proposal_reconnect = ColorVar::zero(primary_l.repr());
                    self.splat_mcmc_contribution(
                        shape,
                        proposal_shape,
                        primary_direct,
                        primary_indirect,
                        shifted_direct,
                        shifted_indirect,
                        jacobian,
                        |v, w| {
                            cur_replay.store(cur_replay.load() + v * w * scale);
                        },
                        |v, w| {
                            cur_reconnect.store(cur_reconnect.load() + v * w * scale);
                        },
                        |v, w| {
                            proposal_replay.store(proposal_replay.load() + v * w * scale);
                        },
                        |v, w| {
                            proposal_reconnect.store(proposal_reconnect.load() + v * w * scale);
                        },
                    );
                    [
                        cur_replay.load(),
                        cur_reconnect.load(),
                        proposal_replay.load(),
                        proposal_reconnect.load(),
                    ]
                };
                let acc_contribution = |colors: [Color; 4], scale: Expr<f32>| {
                    let [primary_direct, primary_indirect, shifted_direct, shifted_indirect] =
                        colors;
                    acc_proposal_replay.store(acc_proposal_replay.load() + shifted_direct * scale);
                    acc_proposal_reconnect
                        .store(acc_proposal_reconnect.load() + shifted_indirect * scale);
                    acc_cur_replay.store(acc_cur_replay.load() + primary_direct * scale);
                    acc_cur_reconnect.store(acc_cur_reconnect.load() + primary_indirect * scale);
                };
                let lerp_contributions =
                    |a: [Color; 4], b: [Color; 4], t: Expr<f32>| -> [Color; 4] {
                        [
                            a[0] * (1.0 - t) + b[0] * t,
                            a[1] * (1.0 - t) + b[1] * t,
                            a[2] * (1.0 - t) + b[2] * t,
                            a[3] * (1.0 - t) + b[3] * t,
                        ]
                    };
                if k > 0.0 {
                    if (t < t_buf.read(idx).as_i32())
                        | ((idx == idx_lo) & (t <= t_buf.read(idx).as_i32()))
                    {
                        // first
                        let l: Color = color_buf.read(idx).0;
                        let jacobian = path_jacobians.read(idx);
                        let reconnect = path_sm_throughput.read(idx).0;
                        let next = next_idx();
                        let (l_minus, reconnect_minus, jacobian_minus) = if idx == center {
                            (l, reconnect, jacobian)
                        } else {
                            if next < idx_hi {
                                (
                                    color_buf.read(next).0,
                                    path_sm_throughput.read(next).0,
                                    path_jacobians.read(next),
                                )
                            } else {
                                (l, reconnect, jacobian)
                            }
                        };
                        acc_contribution(
                            compute_contribution(
                                primary_l,
                                primary_reconnect,
                                l,
                                reconnect,
                                jacobian,
                                1.0 / k * diff_scale,
                            ),
                            1.0f32.expr(),
                        );
                        acc_contribution(
                            compute_contribution(
                                primary_l,
                                primary_reconnect,
                                l_minus,
                                reconnect_minus,
                                jacobian_minus,
                                -1.0 / k * diff_scale,
                            ),
                            1.0f32.expr(),
                        );
                    } else if **is_last {
                        // last
                        let l = color_buf.read(idx).0;
                        let jacobian = path_jacobians.read(idx);
                        let reconnect = path_sm_throughput.read(idx).0;
                        let (l_minus, reconnect_minus, jacobian_minus) = if idx == center {
                            (l, reconnect, jacobian)
                        } else if idx > idx_lo {
                            // lc_assert!(prev_idx().ge(idx_lo.as_i32()));
                            (
                                color_buf.read(prev_idx().as_u32()).0,
                                path_sm_throughput.read(prev_idx().as_u32()).0,
                                path_jacobians.read(prev_idx().as_u32()),
                            )
                        } else {
                            (l, reconnect, jacobian)
                        };
                        acc_contribution(
                            compute_contribution(
                                primary_l,
                                primary_reconnect,
                                l,
                                reconnect,
                                jacobian,
                                1.0 / k * diff_scale,
                            ),
                            1.0f32.expr(),
                        );
                        acc_contribution(
                            compute_contribution(
                                primary_l,
                                primary_reconnect,
                                l_minus,
                                reconnect_minus,
                                jacobian_minus,
                                -1.0 / k * diff_scale,
                            ),
                            1.0f32.expr(),
                        );
                    } else {
                        let compute_lerp = |idx: Expr<u32>, next: Expr<u32>| -> [Color; 4] {
                            let t0 = t_buf.read(idx).as_i32();
                            let t1 = t_buf.read(next).as_i32();
                            let fac = (t - t0).as_f32() / (t1 - t0).as_f32();
                            let l_left = color_buf.read(idx).0;
                            let l_right = color_buf.read(next).0;
                            let reconnect_left = path_sm_throughput.read(idx).0;
                            let reconnect_right = path_sm_throughput.read(next).0;
                            let colors_left = compute_contribution(
                                primary_l,
                                primary_reconnect,
                                l_left,
                                reconnect_left,
                                path_jacobians.read(idx),
                                1.0f32.expr(),
                            );
                            let colors_right = compute_contribution(
                                primary_l,
                                primary_reconnect,
                                l_right,
                                reconnect_right,
                                path_jacobians.read(next),
                                1.0f32.expr(),
                            );
                            lerp_contributions(colors_left, colors_right, fac)
                        };
                        let l = compute_lerp(**idx, next);
                        let prev = prev_idx();
                        let next = next_idx();
                        let next_next = next_next_idx();
                        if t == t_buf.read(idx).as_i32() {
                            // if we are at one of the samples
                            // the only other way to get contribution is to remove the current one
                            let l_minus = if idx == center {
                                // cannot remove center
                                l
                            } else {
                                // lc_assert!(idx.gt(idx_lo));
                                compute_lerp(prev.as_u32(), next)
                            };
                            // splat_mcmc(l, reconnect, jacobian, 1.0 / k * diff_scale);
                            // splat_mcmc(
                            //     l_minus,
                            //     reconnect_minus,
                            //     jacobian_minus,
                            //     -1.0 / k * diff_scale,
                            // );
                            acc_contribution(l, 1.0 / k * diff_scale);
                            acc_contribution(l_minus, -1.0 / k * diff_scale);
                        } else {
                            // we have two other ways to get contribution
                            // we can either remove the previous one or the next one

                            // remove idx
                            let l_minus_left = if idx == center {
                                l
                            } else {
                                // lc_assert!(next.lt(idx_hi));
                                if prev < idx_lo.as_i32() {
                                    // (
                                    //     color_buf.read(next).0,
                                    //     path_sm_throughput.read(next).0,
                                    //     path_jacobians.read(next),
                                    // )
                                    let shifted_l = color_buf.read(next).0;
                                    let shifted_reconnect = path_sm_throughput.read(next).0;
                                    let shifted_jacobian = path_jacobians.read(next);
                                    compute_contribution(
                                        primary_l,
                                        primary_reconnect,
                                        shifted_l,
                                        shifted_reconnect,
                                        shifted_jacobian,
                                        1.0f32.expr(),
                                    )
                                } else {
                                    compute_lerp(prev.as_u32(), next)
                                }
                            };
                            //remove next
                            let l_minus_right = if next == center {
                                // cannot remove next
                                l
                            } else {
                                if next_next >= idx_hi {
                                    let shifted_l = color_buf.read(idx).0;
                                    let shifted_reconnect = path_sm_throughput.read(idx).0;
                                    let shifted_jacobian = path_jacobians.read(idx);
                                    compute_contribution(
                                        primary_l,
                                        primary_reconnect,
                                        shifted_l,
                                        shifted_reconnect,
                                        shifted_jacobian,
                                        1.0f32.expr(),
                                    )
                                } else {
                                    compute_lerp(**idx, next_next)
                                }
                            };
                            // l - l_minus_left + l - l_minus_right
                            // splat_mcmc(l, reconnect, jacobian, 2.0 / k * diff_scale);
                            // splat_mcmc(
                            //     l_minus_left,
                            //     reconnect_minus_left,
                            //     jacobian_minus_left,
                            //     -1.0 / k * diff_scale,
                            // );
                            // splat_mcmc(
                            //     l_minus_right,
                            //     reconnect_minus_right,
                            //     jacobian_minus_right,
                            //     -1.0 / k * diff_scale,
                            // );
                            acc_contribution(l, 2.0 / k * diff_scale);
                            acc_contribution(l_minus_left, -1.0 / k * diff_scale);
                            acc_contribution(l_minus_right, -1.0 / k * diff_scale);
                        }
                    };
                }

                // splat_mcmc(primary_l, primary_reconnect, 1.0f32.expr(), 1.0f32.expr());
                acc_contribution(
                    compute_contribution(
                        primary_l,
                        primary_reconnect,
                        primary_l,
                        primary_reconnect,
                        1.0f32.expr(),
                        1.0f32.expr(),
                    ),
                    1.0f32.expr(),
                );
                // device_log!("{} {} {} {}", primary_l.as_rgb(), diff.as_rgb(), k, diff_scale);
                // splat_fn(ip.cast_f32(), primary_l + diff / k * diff_scale, swl, scale);
                splat_fn(
                    ip.cast_f32(),
                    acc_proposal_reconnect.load() + acc_proposal_replay.load(),
                    swl,
                    scale,
                );

                *t += 1;
            }
            splat_fn(
                cur_p,
                acc_cur_reconnect.load() + acc_cur_replay.load(),
                swl,
                scale,
            );
        });
    }
}
fn randomize_shape(base_curve: &[Int2], spp: u32) -> Vec<Int2> {
    let mut rng = rand::thread_rng();
    let stride = 1.0;
    let perturb = rng.gen_range(-0.1..0.1);
    let rotation = (perturb + 30.0f32).to_radians() * (5 * spp) as f32;
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
impl Integrator for ShapeSplattingPathTracer {
    fn render(
        &self,
        scene: Arc<Scene>,
        sampler_config: SamplerConfig,
        color_pipeline: ColorPipeline,
        film: &mut Film,
        session: &RenderSession,
    ) {
        let resolution = scene.camera.resolution();
        log::info!(
            "Resolution {}x{}\nconfig:{:#?}",
            resolution.x,
            resolution.y,
            &self.config
        );
        assert_eq!(resolution.x, film.resolution().x);
        assert_eq!(resolution.y, film.resolution().y);
        let sampler_creator = sampler_config.creator(self.device.clone(), &scene, self.spp);
        let mut k_pmf = (0..self.n_shift_pixels)
            // .map(|_| 1.0f32)
            .map(|i| (i as f32 + 1.0).powf(self.config.pmf_power))
            // .map(|i| 1.0 / (i as f32 + 1.0))
            // .map(|i| 1.0f32 / 2.0f32.powi(i as i32))
            .collect::<Vec<_>>();
        if let Some(l) = self.config.max_sampled_length {
            k_pmf[l as usize..].fill(0.0);
        }
        if self.config.sample_all {
            let len = k_pmf.len();
            k_pmf[0..len - 1].fill(0.0);
            k_pmf[len - 1] = 1.0;
        }
        {
            let norm = k_pmf.iter().sum::<f32>();
            k_pmf.iter_mut().for_each(|x| *x /= norm);
        }
        // dbg!(&k_pmf);

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
        log::info!("Shape size: {}", self.n_shift_pixels + 1);
        log::info!("Average paths per pixel: {}", avg_per_pixel);
        let avg_per_pixel_buf = self.device.create_buffer_from_slice(&avgs_per_pixel);

        // let pmf_at = if k_pmf.len() > 0 {
        //     AliasTable::new(self.device.clone(), &k_pmf)
        // } else {
        //     AliasTable::new(self.device.clone(), &[1.0f32])
        // };
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
        let tmp_buf_size =
            (resolution.x * resolution.y) as usize * (1.25 * avg_per_pixel.ceil()) as usize;
        let tmp_color =
            ColorBuffer::new(self.device.clone(), tmp_buf_size, color_pipeline.color_repr);
        let tmp_sm_color =
            ColorBuffer::new(self.device.clone(), tmp_buf_size, color_pipeline.color_repr);
        let albedo_buffer =
            self.device
                .create_tex2d::<Float4>(PixelStorage::Byte4, resolution.x, resolution.y, 1);
        let normal_buffer =
            self.device
                .create_tex2d::<Float4>(PixelStorage::Byte4, resolution.x, resolution.y, 1);
        let first_hit = self
            .device
            .create_buffer::<SurfaceHit>((resolution.x * resolution.y) as usize);
        let shapes = self
            .device
            .create_buffer::<Shape>((resolution.x * resolution.y) as usize);
        let rotation = self
            .device
            .create_buffer::<f32>((resolution.x * resolution.y) as usize);
        let shape_indices = self.device.create_buffer::<i8>(
            (resolution.x * resolution.y) as usize * (1 + self.n_shift_pixels) as usize,
        );
        let reconnect_vertex_buffer = self
            .device
            .create_buffer::<ReconnectionVertex>((resolution.x * resolution.y) as usize);
        let new_reconnect_vertex_buffer = self
            .device
            .create_buffer::<ReconnectionVertex>((resolution.x * resolution.y) as usize);
        let path_states = self
            .device
            .create_buffer::<PathState>((resolution.x * resolution.y) as usize);
        let path_counter = self.device.create_buffer::<u32>(3);
        let sampled_indices = self
            .device
            .create_buffer_from_fn::<i8>(tmp_buf_size, |_| i8::MAX);
        let shifted_path_states = self
            .device
            .create_buffer::<ShiftedPathState>(tmp_buf_size as usize);
        let path_jacobians = self.device.create_buffer::<f32>(tmp_buf_size as usize);
        path_jacobians.fill(0.0);
        let sort_tmp = self.device.create_buffer::<i8>(
            (resolution.x * resolution.y * (1 + self.n_shift_pixels)) as usize,
        );
        let mut t_lo = i32::MAX;
        let mut t_hi = i32::MAX;
        let mut base_shape_center = u32::MAX;
        let curve_data = {
            let curve = self.config.curve;
            let n = (1 + self.n_shift_pixels) as usize;

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
                            // let dist = (x as i32 - curve_center as i32).abs()
                            //     + (y as i32 - curve_center as i32).abs();
                            // let stride = 1 + dist / 4;
                            let stride = self.config.stride as i32;
                            buf.push(Int2::new(
                                (x as i32 - curve_center) * stride,
                                (y as i32 - curve_center) * stride,
                            ));
                        }
                    }
                    // let mut buf = vec![Uint2::new(u32::MAX, u32::MAX); n];

                    // assert_eq!(width * width, n as u32);
                    // for x in 0..width {
                    //     for y in 0..width {
                    //         let i = morton2d(x as u64, y as u64) as usize;
                    //         if i >= 1 + self.n_shift_pixels as usize {
                    //             continue;
                    //         }
                    //         assert!(buf[i].x == u32::MAX);
                    //         assert!(buf[i].y == u32::MAX);
                    //         buf[i] = Uint2::new(x, y);
                    //     }
                    // }
                    // let buf = buf
                    //     .into_iter()
                    //     .filter(|x| x.x != u32::MAX)
                    //     .map(|v| {
                    //         Int2::new(
                    //             (v.x as i32 - lo_width as i32 / 2) * self.config.stride as i32,
                    //             (v.y as i32 - lo_width as i32 / 2) * self.config.stride as i32,
                    //         )
                    //     })
                    //     .collect::<Vec<_>>();
                    assert_eq!(buf.len(), 1 + self.n_shift_pixels as usize);
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
                            // let dist = (x as i32 - curve_center as i32).abs()
                            //     + (y as i32 - curve_center as i32).abs();
                            // let stride = 1 + dist / 4;
                            let stride = self.config.stride as i32;
                            buf.push(Int2::new(
                                (x as i32 - curve_center) * stride,
                                (y as i32 - curve_center) * stride,
                            ));
                        }
                    }
                    // let buf = util::generate_hilbert_curve(power)
                    //     .into_iter()
                    //     .take(1 + self.n_shift_pixels as usize)
                    //     .map(|v| {
                    //         let x = v.0 as i32 - lo_width as i32 / 2;
                    //         let y = v.1 as i32 - lo_width as i32 / 2;
                    //         // let nx = x as f32 / (lo_width as i32 / 2) as f32;
                    //         // let ny = y as f32 / (lo_width as i32 / 2) as f32;
                    //         // let len = (1.0 + ((nx * nx + ny * ny) as f32).sqrt()).powf(2.0);
                    //         // let x = (x as f32 * len).round() as i32;
                    //         // let y = (y as f32 * len).round() as i32;
                    //         Int2::new(x * self.config.stride as i32, y * self.config.stride as i32)
                    //     })
                    //     .collect::<Vec<_>>();
                    assert_eq!(buf.len(), 1 + self.n_shift_pixels as usize);
                    buf
                }
            }
        };
        let curve = self.device.create_buffer_from_slice(&curve_data);
        let rng = init_pcg32_buffer_with_seed(
            self.device.clone(),
            (resolution.x * resolution.y) as usize,
            0,
        );
        let curve = &curve;
        let curve_data = &curve_data;
        let mut debug_film = Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let pt_film = Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let pt_sqr_film = Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let mc_film_tmp = Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let mc_film_replay_tmp =
            Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let mut mc_film = Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let mut mc_sqr_film =
            Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let mut mc_replay_film =
            Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let mut mc_replay_sqr_film =
            Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let mc_weight_film = Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let pt_weight_film = Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let mut tele_film = Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let tele_film_tmp = Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let mut tele_sqr_film =
            Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let tele_weight_film =
            Film::new(self.device.clone(), resolution, film.repr(), film.filter());

        let async_compile = self.config.async_compile;
        struct Profiling {
            sample_primary: f32,
            sample_shifted: f32,
            telescoping_sum: f32,
            auxillary: f32,
            denoiser_kernel: f32,
        }
        let profiling = RefCell::new(Profiling {
            sample_primary: 0.0,
            sample_shifted: 0.0,
            telescoping_sum: 0.0,
            auxillary: 0.0,
            denoiser_kernel: 0.0,
        });
        let is_profiling = match env::var("AKR_PROFILE") {
            Ok(s) => s == "1",
            Err(_) => false,
        };
        let update_var = self.device.create_kernel::<fn()>(&track!(|| {
            let mc_scale = 1.0; // / avg_per_pixel;
            let tele_scale = 1.0; // / (self.n_shift_pixels + 1) as f32;
            let p = dispatch_id().xy();
            let idx = film.linear_index(p.cast_f32());
            for c in 0u32..film.repr().nvalues() as u32 {
                // let sample_offset = idx * film.repr().nvalues() as u32 + c;
                let splat_offset = film.splat_offset() + idx * film.repr().nvalues() as u32 + c;

                let tele = tele_film_tmp.data().read(splat_offset) * tele_scale;

                let mc = mc_film_tmp.data().read(splat_offset) * mc_scale;
                let mc_replay = mc_film_replay_tmp.data().read(splat_offset) * mc_scale;

                mc_film.data().var().atomic_fetch_add(splat_offset, mc);
                // mc_film.data().write(splat_offset, mc);
                mc_sqr_film
                    .data()
                    .var()
                    .atomic_fetch_add(splat_offset, mc * mc);

                tele_film.data().var().atomic_fetch_add(splat_offset, tele);
                // tele_film.data().write(splat_offset, tele);
                tele_sqr_film
                    .data()
                    .var()
                    .atomic_fetch_add(splat_offset, tele * tele);
                mc_film_tmp.data().write(splat_offset, 0.0f32.expr());
                mc_film_replay_tmp.data().write(splat_offset, 0.0f32.expr());

                tele_film_tmp.data().write(splat_offset, 0.0f32.expr());
            }
        }));
        let update_var = &update_var;
        let dispatch_kernels = {
            let render_state = RenderState {
                t_hi,
                t_lo,
                rotation: &rotation,
                first_hit: &first_hit,
                base_shape_center,
                pmf_ats: &pmf_ats,
                tele_film: &tele_film_tmp,
                mc_film: &mc_film_tmp,
                mc_replay_film: &mc_film_replay_tmp,
                pt_film: &pt_film,
                pt_sqr_film: &pt_sqr_film,
                mc_sqr_film: &mc_sqr_film,
                mc_replay_sqr_film: &mc_replay_sqr_film,
                debug_film: &debug_film,
                albedo_buf: &albedo_buffer,
                normal_buf: &normal_buffer,
                reconnect_vertex_buf: &reconnect_vertex_buffer,
                path_states: &path_states,
                path_counter_buf: &path_counter,
                path_sm_replay: &tmp_color,
                path_sm_reconnect: &tmp_sm_color,
                path_jacobians: &path_jacobians,
                sampled_indices: &sampled_indices,
                shifted_path_states: &shifted_path_states,
                sort_tmp: &sort_tmp,
                curve: &curve,
                rng_buf: &rng,
                shapes: &shapes,
                shape_indices: &shape_indices,
                avg_per_pixel: &avg_per_pixel_buf,
            };
            let reset = self.device.create_kernel_with_options::<fn()>(
                KernelBuildOptions {
                    async_compile,
                    ..Default::default()
                },
                &track!(|| {
                    let i = dispatch_id().x;
                    if i == 0 {
                        render_state.path_counter_buf.write(0, 0);
                        render_state.path_counter_buf.write(1, 0);
                        render_state.path_counter_buf.write(2, 0);
                    }
                }),
            );
            let sample_primary = self.device.create_kernel_with_options::<fn(u32)>(
                KernelBuildOptions {
                    async_compile,
                    ..Default::default()
                },
                &track!(|_spp: Expr<u32>| {
                    self.kernel_sample_primary(
                        scene.clone(),
                        color_pipeline,
                        sampler_creator.as_ref(),
                        render_state,
                    );
                }),
            );
            let sample_shifted = self.device.create_kernel_with_options::<fn()>(
                KernelBuildOptions {
                    async_compile,
                    ..Default::default()
                },
                &track!(|| {
                    self.kernel_sample_shifted(
                        scene.clone(),
                        color_pipeline,
                        sampler_creator.as_ref(),
                        render_state,
                    );
                }),
            );
            let telescoping_sum = self.device.create_kernel_with_options::<fn()>(
                KernelBuildOptions {
                    async_compile,
                    ..Default::default()
                },
                &track!(|| {
                    self.kernel_telelscoping_sum(
                        scene.clone(),
                        color_pipeline,
                        sampler_creator.as_ref(),
                        render_state,
                    );
                }),
            );
            let compute_normal_albedo = self.device.create_kernel_with_options::<fn(u32)>(
                KernelBuildOptions {
                    async_compile,
                    ..Default::default()
                },
                &track!(|spp: Expr<u32>| {
                    self.kernel_compute_auxillary(
                        scene.clone(),
                        color_pipeline,
                        sampler_creator.as_ref(),
                        render_state,
                        spp,
                    );
                }),
            );
            let compute_denoiser_kernel = self.device.create_kernel_with_options::<fn()>(
                KernelBuildOptions {
                    async_compile,
                    ..Default::default()
                },
                &track!(|| {
                    self.kernel_compute_denoiser_kernel(scene.clone(), render_state);
                }),
            );
            fn time_fn(f: impl FnOnce()) -> f32 {
                let t = std::time::Instant::now();
                f();
                t.elapsed().as_secs_f32()
            }

            let profiling = &profiling;
            let compute_aux = move |s: &Scope| {
                if self.config.denoiser_kernel {
                    let aux_spp = (self.config.spp / 4).clamp(1, 16);
                    let t1 = time_fn(|| {
                        s.submit([compute_normal_albedo
                            .dispatch_async([resolution.x * resolution.y, 1, 1], &aux_spp)])
                            .synchronize();
                    });

                    if is_profiling {
                        let mut profiling = profiling.borrow_mut();
                        profiling.auxillary += t1;
                    }
                }
            };
            let update_denoiser_kernel = move |s: &Scope, spp: u32| {
                if self.config.denoiser_kernel || self.config.randomize {
                    let t2 = time_fn(|| {
                        if self.config.randomize {
                            let new_data = randomize_shape(&curve_data, spp);
                            curve.copy_from(&new_data);
                        }
                        if self.config.denoiser_kernel {
                            s.submit([compute_denoiser_kernel.dispatch_async([
                                resolution.x * resolution.y,
                                1,
                                1,
                            ])])
                            .synchronize();
                        }
                    });
                    if is_profiling {
                        let mut profiling = profiling.borrow_mut();
                        profiling.denoiser_kernel += t2;
                    }
                }
            };
            let update_sampler_kernel = self.device.create_kernel_with_options::<fn()>(
                KernelBuildOptions {
                    async_compile,
                    ..Default::default()
                },
                &track!(|| {
                    let p = dispatch_id().xy();
                    let sampler = sampler_creator.create(p);
                    sampler.start();
                }),
            );
            move |spp, s: &Scope| {
                if spp == 0 {
                    compute_aux(s);
                }

                update_denoiser_kernel(s, spp);

                let reset_dispatch = reset.dispatch_async([
                    1, //resolution.x * resolution.y * (1 + self.n_shift_pixels),
                    1, 1,
                ]);
                let sample_primary_dispatch =
                    sample_primary.dispatch_async([resolution.x * resolution.y, 1, 1], &spp);
                let sampled_shifted_dispatch = sample_shifted.dispatch_async([
                    resolution.x * resolution.y * self.n_shift_pixels,
                    1,
                    1,
                ]);
                let telescoping_dispatch =
                    telescoping_sum.dispatch_async([resolution.x * resolution.y, 1, 1]);

                let update_sampler_kernel_dispatch =
                    update_sampler_kernel.dispatch_async([resolution.x, resolution.y, 1]);

                if is_profiling {
                    let mut profiling = profiling.borrow_mut();
                    profiling.sample_primary += time_fn(|| {
                        s.submit([reset_dispatch, sample_primary_dispatch])
                            .synchronize();
                    });
                    profiling.sample_shifted += time_fn(|| {
                        s.submit([sampled_shifted_dispatch]).synchronize();
                    });

                    profiling.telescoping_sum += time_fn(|| {
                        s.submit([telescoping_dispatch, update_sampler_kernel_dispatch])
                            .synchronize();
                    });
                } else {
                    s.submit([
                        reset_dispatch,
                        sample_primary_dispatch,
                        sampled_shifted_dispatch,
                        telescoping_dispatch,
                        update_sampler_kernel_dispatch,
                    ]);
                }

                // s.submit([pt_edge_kernel_dispatch]);
            }
        };

        let mut cnt = 0;
        let progress = util::create_progess_bar(self.spp as usize, "spp");
        let mut acc_time = 0.0;

        let mix_kernel =
            self.device.create_kernel::<fn(f32, f32)>(&track!(
                |mc_scale: Expr<f32>, tele_scale: Expr<f32>| {
                    let p = dispatch_id().xy();
                    let idx = film.linear_index(p.cast_f32());
                    let weight_offset = film.weight_offset() + idx;
                    let pt_weight = pt_film.data().read(weight_offset);
                    let pt_sqr_weight = pt_sqr_film.data().read(weight_offset);
                    // mix by inverse variance
                    for c in 0u32..film.repr().nvalues() as u32 {
                        let splat_offset =
                            film.splat_offset() + idx * film.repr().nvalues() as u32 + c;
                        let sample_offset = idx * film.repr().nvalues() as u32 + c;
                        let safe_div = |a: Expr<f32>, b: Expr<f32>| select(b == 0.0, a, a / b);
                        // let sample_offset = idx * film.repr().nvalues() as u32 + c;
                        let (mc, mc_sqr) = {
                            let mc = mc_film.data().read(splat_offset) * mc_scale;
                            let mc_sqr = mc_sqr_film.data().read(splat_offset) * mc_scale;
                            (mc, mc_sqr)
                        };
                        let pt = pt_film.data().read(sample_offset);
                        let pt_sqr = pt_sqr_film.data().read(sample_offset);

                        let pt = safe_div(pt, pt_weight);
                        let pt_sqr = safe_div(pt_sqr, pt_sqr_weight);

                        let tele = tele_film.data().read(splat_offset) * tele_scale;
                        let tele_sqr = tele_sqr_film.data().read(splat_offset) * tele_scale;

                        let mc_var = (mc_sqr - mc * mc).max_(1e-6f32.expr());
                        let tele_var = (tele_sqr - tele * tele).max_(1e-6f32.expr());
                        let pt_var = (pt_sqr - pt * pt).max_(1e-6f32.expr());

                        // let mc_weight = mc_var.recip();
                        // let tele_weight = tele_var.recip();
                        // let pt_weight = pt_var.recip();
                        let (pt_weight, mc_weight, tele_weight) =
                            if self.config.mix == MixingMethod::InverseVariance {
                                (pt_var.recip(), mc_var.recip(), tele_var.recip())
                            } else if self.config.mix == MixingMethod::InverseVarianceNoPt {
                                (0.0f32.expr(), mc_var.recip(), tele_var.recip())
                            } else {
                                if self.config.mix == MixingMethod::Min {
                                    let min = pt_var.min_(mc_var).min_(tele_var);
                                    let pt_weight = if min == pt_var {
                                        1.0f32.expr()
                                    } else {
                                        0.0f32.expr()
                                    };
                                    let mc_weight = if min == mc_var {
                                        1.0f32.expr()
                                    } else {
                                        0.0f32.expr()
                                    };
                                    let tele_weight = if min == tele_var {
                                        1.0f32.expr()
                                    } else {
                                        0.0f32.expr()
                                    };
                                    (pt_weight, mc_weight, tele_weight)
                                } else {
                                    (
                                        1.0f32.expr(),
                                        avg_per_pixel.expr(),
                                        (curve_data.len() as f32).expr(),
                                    )
                                }
                            };
                        let sum_weight = mc_weight + tele_weight + pt_weight;
                        let mc_weight = mc_weight / sum_weight;
                        let tele_weight = tele_weight / sum_weight;
                        let pt_weight = pt_weight / sum_weight;

                        let mix = (mc * mc_weight + tele * tele_weight + pt * pt_weight)
                            .max_(0.0f32.expr());
                        film.data().write(splat_offset, mix);

                        mc_weight_film.data().write(splat_offset, mc_weight);
                        tele_weight_film.data().write(splat_offset, tele_weight);
                        pt_weight_film.data().write(splat_offset, pt_weight);
                    }
                }
            ));
        let mut stats: RenderStats = Default::default();
        let update = |film: &mut Film,
                      spp: u32,
                      mc_film: &mut Film,
                      mc_sqr_film: &mut Film,
                      tele_film: &mut Film,
                      tele_sqr_film: &mut Film,
                      mix: MixingMethod| {
            // let mc_scale = 1.0 / spp as f32;
            // let tele_scale = 1.0 / spp as f32;
            let (mc_scale, tele_scale) = (1.0 / spp as f32, 1.0 / spp as f32);

            if !self.config.sample_all {
                if mix != MixingMethod::None {
                    mc_film.set_splat_scale(mc_scale);
                    mc_sqr_film.set_splat_scale(mc_scale);
                    tele_film.set_splat_scale(tele_scale);
                    tele_sqr_film.set_splat_scale(tele_scale);
                    mix_kernel.dispatch([resolution.x, resolution.y, 1], &mc_scale, &tele_scale);
                } else {
                    // copy tele_film to film,
                    tele_film.data().copy_to_buffer(film.data());
                    film.set_splat_scale(tele_scale);
                }
            } else {
                mc_film.set_splat_scale(mc_scale);
                mc_film.data().copy_to_buffer(film.data());
                film.set_splat_scale(tele_scale);
            }

            if let Some(channel) = &session.display {
                film.copy_to_rgba_image(channel.screen_tex(), false);
                channel.notify_update();
            }
        };
        if session.dry_run {
            return;
        }
        let output_image: Tex2d<Float4> = self.device.create_tex2d(
            PixelStorage::Float4,
            scene.camera.resolution().x,
            scene.camera.resolution().y,
            1,
        );
        while cnt < self.spp {
            let cur_pass = (self.spp - cnt).min(self.spp_per_pass);
            let tic = Instant::now();
            self.device.default_stream().with_scope(|s| {
                for i in 0..cur_pass {
                    dispatch_kernels(i + cnt, s);
                    s.submit([update_var.dispatch_async([resolution.x, resolution.y, 1])]);
                }
            });

            let toc = Instant::now();
            acc_time += toc.duration_since(tic).as_secs_f64();
            update(
                film,
                cnt + cur_pass,
                &mut mc_film,
                &mut mc_sqr_film,
                &mut tele_film,
                &mut tele_sqr_film,
                self.config.mix,
            );
            cnt += cur_pass;
            if session.save_intermediate {
                film.copy_to_rgba_image(&output_image, true);
                let path = format!("{}-{}.exr", session.name, cnt);
                util::write_image(&output_image, &path);
                stats.intermediate.push(IntermediateStats {
                    time: acc_time,
                    spp: cnt,
                    path,
                });
            }
            progress.inc(cur_pass as u64);
        }
        if session.save_stats {
            let file = File::create(format!("{}.json", session.name)).unwrap();
            let json = serde_json::to_value(&stats).unwrap();
            let writer = BufWriter::new(file);
            serde_json::to_writer(writer, &json).unwrap();
        }
        progress.finish();
        if is_profiling {
            let profiling = profiling.borrow();
            let total = profiling.sample_primary
                + profiling.sample_shifted
                + profiling.telescoping_sum
                + profiling.auxillary
                + profiling.denoiser_kernel;
            log::info!(
                "Auxillary:       {:8.2}s {:5.2}%",
                profiling.auxillary,
                profiling.auxillary / total * 100.0
            );
            log::info!(
                "Denoiser kernel: {:8.2}s {:5.2}%",
                profiling.denoiser_kernel,
                profiling.denoiser_kernel / total * 100.0
            );
            log::info!(
                "Sample primary:  {:8.2}s {:5.2}%",
                profiling.sample_primary,
                profiling.sample_primary / total * 100.0
            );
            log::info!(
                "Sample shifted:  {:8.2}s {:5.2}%",
                profiling.sample_shifted,
                profiling.sample_shifted / total * 100.0
            );
            log::info!(
                "Telescoping sum: {:8.2}s {:5.2}%",
                profiling.telescoping_sum,
                profiling.telescoping_sum / total * 100.0
            );
        }
        log::info!("Rendering finished in {:.2}s", acc_time);
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
    let pt = ShapeSplattingPathTracer::new(device.clone(), config.clone());
    pt.render(scene, sampler, color_pipeline, film, options);
}
