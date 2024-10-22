use crate::{
    geometry::*,
    interaction::*,
    light::*,
    mesh::MeshHeader,
    sampling::uniform_sample_triangle,
    svm::{surface::Surface, ShaderRef},
};
#[derive(Clone, Copy, Value)]
#[luisa(crate = "luisa")]
#[repr(C)]
pub struct AreaLight {
    pub light_id: u32,
    pub instance_id: u32,
    pub geom_id: u32,
}

impl AreaLightExpr {
    fn emission(
        &self,
        wo: Expr<Float3>,
        si: SurfaceInteraction,
        swl: Expr<SampledWavelengths>,
        ctx: &LightEvalContext<'_>,
    ) -> Color {
        ctx.svm
            .dispatch_surface(si.surface, ctx.color_pipeline, si, swl, |closure| {
                closure.emission(wo, swl, &ctx.surface_eval_ctx)
            })
    }
}
impl Light for AreaLightExpr {
    fn id(&self) -> Expr<u32> {
        self.light_id
    }
    fn le(
        &self,
        ray: Expr<Ray>,
        si: SurfaceInteraction,
        swl: Expr<SampledWavelengths>,
        ctx: &LightEvalContext<'_>,
    ) -> Color {
        let emission = self.emission(-ray.d, si, swl, ctx);
        select(
            si.ng.dot(ray.d).lt(0.0),
            emission,
            Color::zero(ctx.color_repr()),
        )
    }
    #[tracked(crate = "luisa")]
    fn sample_direct(
        &self,
        pn: Expr<PointNormal>,
        u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        swl: Expr<SampledWavelengths>,
        ctx: &LightEvalContext<'_>,
    ) -> LightSample {
        let meshes = ctx.meshes;
        let at = meshes.mesh_area_samplers(self.instance_id);
        let (prim_id, pdf, _) = at.sample_and_remap(u_select);
        // let (prim_id, pdf) = (0u32.expr(), 1.0f32.expr());

        let bary = uniform_sample_triangle(u_sample);
        let si = meshes.surface_interaction(self.instance_id, prim_id, bary);

        let area = si.prim_area;
        let p = si.p;
        let n = si.ng;
        let wi = p - pn.p;
        if wi.length_squared() == 0.0 {
            LightSample {
                li: Color::zero(ctx.color_repr()),
                pdf,
                shadow_ray: Expr::<Ray>::zeroed(),
                wi,
                n,
                valid: false.expr(),
            }
        } else {
            let dist2 = wi.length_squared();
            let wi = wi / dist2.sqrt();
            let emission = self.emission(-wi, si, swl, ctx);
            let li = select(wi.dot(n).lt(0.0), emission, Color::zero(ctx.color_repr()));
            let cos_theta_i = n.dot(wi).abs();
            let pdf = pdf / area * dist2 / cos_theta_i;
            let ro = rtx::offset_ray_origin(pn.p, face_forward(pn.n, wi));
            let dist = dist2.sqrt();
            let shadow_ray = Ray::new_expr(
                ro,
                wi,
                0.0,
                dist * (1.0f32 - 1e-3),
                Uint2::expr(u32::MAX, u32::MAX),
                Uint2::expr(self.instance_id, prim_id),
            );
            // cpu_dbg!( u);
            LightSample {
                li,
                pdf,
                shadow_ray,
                wi,
                n,
                valid: pdf.is_finite(),
            }
        }
    }
    #[tracked(crate = "luisa")]
    fn pdf_direct(
        &self,
        si: SurfaceInteraction,
        pn: Expr<PointNormal>,
        ctx: &LightEvalContext<'_>,
    ) -> Expr<f32> {
        let prim_id = si.prim_id;
        let meshes = ctx.meshes;
        let at = meshes.mesh_area_samplers(self.instance_id);
        if debug_mode() {
            lc_assert!(si.inst_id.eq(self.instance_id));
        }
        let area = si.prim_area;
        let prim_pdf = at.pdf(prim_id);
        let ng = si.ng;
        let p = si.p;
        let wi = p - pn.p;
        let dist2 = wi.length_squared();
        let wi = wi / dist2.sqrt();
        let pdf = prim_pdf / area * dist2 / ng.dot(wi).abs().max_(1e-6);
        pdf
    }
}
impl_polymorphic!(Light, AreaLight);
