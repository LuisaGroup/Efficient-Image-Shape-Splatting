use crate::geometry::{face_forward, spherical_to_xyz2, xyz_to_spherical, Frame};
use crate::sampling::uniform_sample_disk;
use crate::*;
use lazy_static::lazy_static;
use std::f32::consts::PI;

pub trait MicrofacetDistribution {
    #[tracked(crate = "luisa")]
    fn g1(&self, w: Expr<Float3>, ad_mode: ADMode) -> Expr<f32> {
        1.0 / (1.0 + self.lambda(w, ad_mode))
    }
    #[tracked(crate = "luisa")]
    fn g(&self, wo: Expr<Float3>, wi: Expr<Float3>, ad_mode: ADMode) -> Expr<f32> {
        1.0 / (1.0 + self.lambda(wo, ad_mode) + self.lambda(wi, ad_mode))
    }
    fn d(&self, wh: Expr<Float3>, ad_mode: ADMode) -> Expr<f32>;
    fn lambda(&self, w: Expr<Float3>, ad_mode: ADMode) -> Expr<f32>;
    fn sample_wh(&self, wo: Expr<Float3>, u: Expr<Float2>, ad_mode: ADMode) -> Expr<Float3>;
    fn invert_wh(&self, wo: Expr<Float3>, wh: Expr<Float3>, ad_mode: ADMode) -> Expr<Float2>;
    fn pdf(&self, wo: Expr<Float3>, wh: Expr<Float3>, ad_mode: ADMode) -> Expr<f32>;
    fn roughness(&self, ad_mode: ADMode) -> Expr<f32>;
}

pub struct TrowbridgeReitzDistribution {
    pub alpha: Expr<Float2>,
    pub sample_visible: bool,
    roughness: Expr<f32>,
}
impl TrowbridgeReitzDistribution {
    pub const MIN_ALPHA: f32 = 1e-4;
    #[tracked(crate = "luisa")]
    pub fn from_alpha(alpha: Expr<Float2>, sample_visible: bool) -> Self {
        Self {
            alpha: alpha.max_(Self::MIN_ALPHA),
            sample_visible,
            roughness: (alpha.max_(Self::MIN_ALPHA).reduce_sum() * 0.5).sqrt(),
        }
    }
    pub fn from_roughness(roughness: Expr<Float2>, sample_visible: bool) -> Self {
        let alpha = roughness.sqr();
        Self::from_alpha(alpha, sample_visible)
    }
}
#[tracked(crate = "luisa")]
fn tr_d_impl_(wh: Expr<Float3>, alpha: Expr<Float2>) -> Expr<f32> {
    let tan2_theta = Frame::tan2_theta(wh);
    let cos4_theta = Frame::cos2_theta(wh).sqr();
    let ax = alpha.x;
    let ay = alpha.y;
    let e = tan2_theta * ((Frame::cos_phi(wh) / ax).sqr() + (Frame::sin_phi(wh) / ay).sqr());
    let inv_d = PI * ax * ay * cos4_theta * (1.0 + e).sqr();
    select(
        !tan2_theta.is_finite() | !inv_d.is_finite() | inv_d.eq(0.0),
        0.0f32.expr(),
        1.0 / inv_d,
    )
}
#[tracked(crate = "luisa")]
fn tr_lambda_impl_(w: Expr<Float3>, alpha: Expr<Float2>) -> Expr<f32> {
    let abs_tan_theta = Frame::tan_theta(w).abs();
    let alpha2 = Frame::cos2_phi(w) * alpha.x.sqr() + Frame::sin2_phi(w) * alpha.y.sqr();
    let alpha2_tan2_theta = alpha2 * abs_tan_theta.sqr();
    let l = (-1.0 + (1.0 + alpha2_tan2_theta).sqrt()) * 0.5;
    select(!abs_tan_theta.is_finite(), 0.0f32.expr(), l)
}
#[tracked(crate = "luisa")]
fn tr_sample_impl_(alpha: Expr<Float2>, u: Expr<Float2>) -> Expr<Float3> {
    let (phi, cos_theta) = if alpha.x.eq(alpha.y) {
        let phi = 2.0 * PI * u.y;
        let tan_theta2 = alpha.x.sqr() * u.x / (1.0 - u.x);
        let cos_theta = 1.0 / (1.0 + tan_theta2).sqrt();
        (phi, cos_theta)
    } else {
        let phi = (alpha.y / alpha.x * (2.0 * PI * u.y + PI * 0.5).tan()).atan();
        let phi = select(u.y.gt(0.5), phi + PI, phi);
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        let ax2 = alpha.x.sqr();
        let ay2 = alpha.y.sqr();
        let a2 = 1.0 / (cos_phi.sqr() / ax2 + sin_phi.sqr() / ay2);
        let tan_theta2 = a2 * u.x / (1.0 - u.x);
        let cos_theta = 1.0 / (1.0 + tan_theta2).sqrt();
        (phi, cos_theta)
    };
    let sin_theta = (1.0 - cos_theta.sqr()).max_(0.0).sqrt();
    let wh = spherical_to_xyz2(cos_theta, sin_theta, phi);
    let wh = face_forward(wh, Float3::expr(0.0, 0.0, 1.0));
    wh
}
lazy_static! {
    static ref TR_D_IMPL: Callable<fn(Expr<Float3>, Expr<Float2>) -> Expr<f32>> =
        Callable::<fn(Expr<Float3>, Expr<Float2>) -> Expr<f32>>::new_static(|wh, alpha| {
            tr_d_impl_(wh, alpha)
        });
    static ref TR_LAMBDA_IMPL: Callable<fn(Expr<Float3>, Expr<Float2>) -> Expr<f32>> =
        Callable::<fn(Expr<Float3>, Expr<Float2>) -> Expr<f32>>::new_static(|w, alpha| {
            tr_lambda_impl_(w, alpha)
        });
}
impl MicrofacetDistribution for TrowbridgeReitzDistribution {
    fn d(&self, wh: Expr<Float3>, ad_mode: ADMode) -> Expr<f32> {
        if ad_mode != ADMode::Backward {
            TR_D_IMPL.call(wh, self.alpha)
        } else {
            tr_d_impl_(wh, self.alpha)
        }
    }

    fn lambda(&self, w: Expr<Float3>, ad_mode: ADMode) -> Expr<f32> {
        if ad_mode != ADMode::Backward {
            TR_LAMBDA_IMPL.call(w, self.alpha)
        } else {
            tr_lambda_impl_(w, self.alpha)
        }
    }
    #[tracked(crate = "luisa")]
    fn sample_wh(&self, w: Expr<Float3>, u: Expr<Float2>, ad_mode: ADMode) -> Expr<Float3> {
        if self.sample_visible {
            let wh = Var::<Float3>::zeroed();
            outline(|| {
                *wh = Float3::expr(self.alpha.x * w.x, self.alpha.y * w.y, w.z).normalize();
                if wh.z < 0.0 {
                    *wh = -wh;
                }
                let t1 = (wh.z < 0.99999).select(
                    Float3::expr(0.0, 0.0, 1.0).cross(wh).normalize(),
                    Float3::expr(1.0, 0.0, 0.0),
                );
                let t2 = wh.cross(t1).normalize();
                let p = uniform_sample_disk(u).var();
                let h = (1.0f32 - p.x.sqr()).sqrt();
                *p.y = h.lerp(p.y, (1.0 + wh.z) * 0.5);
                let pz = (1.0 - p.length_squared()).max_(0.0f32.expr()).sqrt();
                let nh = p.x * t1 + p.y * t2 + pz * wh;
                *wh = Float3::expr(self.alpha.x * nh.x, self.alpha.y * nh.y, nh.z.max_(1e-6f32))
                    .normalize();
            });
            **wh
        } else {
            lazy_static! {
                static ref SAMPLE: Callable<fn(Expr<Float2>, Expr<Float2>) -> Expr<Float3>> =
                    Callable::<fn(Expr<Float2>, Expr<Float2>) -> Expr<Float3>>::new_static(
                        |alpha: Expr<Float2>, u: Expr<Float2>| { tr_sample_impl_(alpha, u) }
                    );
            }
            if ad_mode != ADMode::Backward {
                SAMPLE.call(self.alpha, u)
            } else {
                tr_sample_impl_(self.alpha, u)
            }
        }
    }
    fn invert_wh(&self, _wo: Expr<Float3>, wh: Expr<Float3>, _ad_mode: ADMode) -> Expr<Float2> {
        if self.sample_visible {
            unimplemented!("invert_wh is not available for visible wh sampling");
        } else {
            lazy_static! {
                static ref INVERT_SAMPLE: Callable<fn(Expr<Float2>, Expr<Float3>)-> Expr<Float2>> =
                    Callable::<fn(Expr<Float2>, Expr<Float3>)-> Expr<Float2>>::new_static(
                        track!(|alpha: Expr<Float2>, wh: Expr<Float3>| {
                            let (theta, phi) = xyz_to_spherical(wh);
                            let cos_theta = theta.cos();
                            if alpha.x.eq(alpha.y) {
                                // see https://github.com/tunabrain/tungsten/blob/master/src/core/bsdfs/Microfacet.hpp
                                // let phi = 2.0 * PI * u.y;
                                // let tan_theta2 = alpha.x.sqr() * u.x / (1.0 - u.x);
                                // let cos_theta = 1.0 / (1.0 + tan_theta2).sqrt();
                                // (phi, cos_theta)
                                let uy = (phi * FRAC_1_2PI).fract();
                                let tan_theta2 = cos_theta.sqr().recip() - 1.0;
                                let gamma = tan_theta2 / alpha.x.sqr();
                                let ux = gamma / (1.0 + gamma);
                                Float2::expr(ux, uy)
                            } else {
                                // let phi = (alpha.y / alpha.x * (2.0 * PI * u.y + PI * 0.5).tan()).atan();
                                let uy = (((phi.atan() * alpha.x / alpha.y).atan() - PI * 0.5) * FRAC_1_2PI).fract();
                                // let phi = select(u.y.gt(0.5), phi + PI, phi);
                                let sin_phi = phi.sin();
                                let cos_phi = phi.cos();
                                let ax2 = alpha.x.sqr();
                                let ay2 = alpha.y.sqr();
                                let a2 = 1.0 / (cos_phi.sqr() / ax2 + sin_phi.sqr() / ay2);
                                // let tan_theta2 = a2 * u.x / (1.0 - u.x);
                                // let cos_theta = 1.0 / (1.0 + tan_theta2).sqrt();
                                let tan_theta2 = cos_theta.sqr().recip() - 1.0;
                                let gamma = tan_theta2 / a2;
                                let ux = gamma / (1.0 + gamma);
                                Float2::expr(ux, uy)
                            }
                        }
                    ));
            }
            INVERT_SAMPLE.call(self.alpha, wh)
        }
    }
    fn pdf(&self, wo: Expr<Float3>, wh: Expr<Float3>, ad_mode: ADMode) -> Expr<f32> {
        let pdf = if self.sample_visible {
            track!(
                self.d(wh, ad_mode) * self.g1(wo, ad_mode) * wo.dot(wh).abs()
                    / Frame::abs_cos_theta(wo)
            )
        } else {
            track!(self.d(wh, ad_mode) * Frame::abs_cos_theta(wh))
        };
        pdf
    }
    #[tracked(crate = "luisa")]
    fn roughness(&self, _ad_mode: ADMode) -> Expr<f32> {
        self.roughness
    }
}
#[cfg(test)]
mod test {
    use std::env::current_exe;

    use crate::sampler::{init_pcg32_buffer, IndependentSampler, Sampler};

    use super::*;

    #[test]
    fn tr_sample_wh() {
        let ctx = luisa::Context::new(current_exe().unwrap());
        let device = ctx.create_cpu_device();
        let seeds = init_pcg32_buffer(device.clone(), 8192);
        let out = device.create_buffer::<f32>(seeds.len());
        let n_iters = 4098u32;
        let kernel = Kernel::<fn(Float3, Float2)>::new(
            &device,
            &track!(|wo: Expr<Float3>, alpha: Expr<Float2>| {
                let i = dispatch_id().x;
                let sampler = IndependentSampler::from_pcg32(seeds.read(i).var());
                let dist = TrowbridgeReitzDistribution::from_alpha(alpha, true);
                for_range(0u32.expr()..n_iters.expr(), |_| {
                    let wh = dist.sample_wh(wo, sampler.next_2d(), ADMode::None);
                    let pdf = dist.pdf(wo, wh, ADMode::None);
                    if pdf.gt(0.0) {
                        out.write(i, out.read(i) + 1.0 / pdf);
                    }
                });
            }),
        );
        let test_alpha = |theta: f32, alpha_x: f32, alpha_y: f32| {
            kernel.dispatch(
                [seeds.len() as u32, 1, 1],
                &Float3::new(theta.sin(), theta.cos(), 0.0),
                &Float2::new(alpha_x, alpha_y),
            );
            let out = out.copy_to_vec();
            let mean =
                out.iter().map(|x| *x as f64).sum::<f64>() / out.len() as f64 / n_iters as f64;
            println!("theta: {}, alpha: {}, mean: {}", theta, alpha_x, mean);
        };
        test_alpha(0.8, 0.1, 0.1);
    }
}
