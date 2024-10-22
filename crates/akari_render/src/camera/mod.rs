use sampling::uniform_sample_disk;

use crate::{
    color::*, film::PixelFilter, geometry::*, heap::MegaHeap, sampler::*, scene::Scene, *,
};

pub trait Camera: Send + Sync {
    fn set_resolution(&mut self, resolution: Uint2);
    fn resolution(&self) -> Uint2;
    fn generate_ray(
        &self,
        scene: &Scene,
        filter: PixelFilter,
        pixel: Expr<Uint2>,
        sampler: &dyn Sampler,
        color_repr: ColorRepr,
        swl: Expr<SampledWavelengths>,
    ) -> (Expr<Ray>, Expr<f32>);
}
pub struct PerspectiveCamera {
    data_buf_index: u32,
    data: Buffer<PerspectiveCameraData>,
    resolution: Uint2,
    device: Device,
    transform: AffineTransform,
    fov: f32,
    lens_radius: f32,
    focal_length: f32,
}
impl PerspectiveCamera {
    pub fn new(
        device: Device,
        heap: &MegaHeap,
        resolution: Uint2,
        transform: AffineTransform,
        fov: f32,
        lens_radius: f32,
        focal_length: f32,
        depth_of_field: bool,
    ) -> Self {
        let (data, data_buf_index) = heap.alloc_buffer::<PerspectiveCameraData>(1);
        let mut camera = Self {
            data_buf_index,
            data,
            resolution,
            device,
            transform,
            fov,
            lens_radius: if depth_of_field { lens_radius } else { 0.0 },
            focal_length,
        };
        camera.set_resolution(resolution);
        camera
    }
}
impl Camera for PerspectiveCamera {
    fn set_resolution(&mut self, resolution: Uint2) {
        self.resolution = resolution;
        PerspectiveCameraData::new(
            self.device.clone(),
            resolution,
            self.transform,
            self.fov,
            self.lens_radius,
            self.focal_length,
            &self.data,
        )
    }
    fn resolution(&self) -> Uint2 {
        self.resolution
    }
    #[tracked(crate = "luisa")]
    fn generate_ray(
        &self,
        scene: &Scene,
        filter: PixelFilter,
        pixel: Expr<Uint2>,
        sampler: &dyn Sampler,
        _color_repr: ColorRepr,
        _swl: Expr<SampledWavelengths>,
    ) -> (Expr<Ray>, Expr<f32>) {
        let camera = scene
            .heap
            .var()
            .buffer::<PerspectiveCameraData>(self.data_buf_index)
            .read(0);
        let fpixel = pixel.cast_f32() + Float2::expr(0.5, 0.5);
        let (offset, w) = filter.sample(sampler.next_2d());
        let p_film = fpixel + offset;
        let ray = Ray::new_expr(
            Float3::expr(0.0, 0.0, 0.0),
            camera
                .r2c
                .transform_point(Float3::expr(p_film.x, p_film.y, 0.0))
                .normalize(),
            0.0,
            1e20,
            Uint2::expr(u32::MAX, u32::MAX),
            Uint2::expr(u32::MAX, u32::MAX),
        )
        .var();
        if camera.lens_radius > 0.0 {
            let p_lens = camera.lens_radius * uniform_sample_disk(sampler.next_2d());
            let ft = camera.focal_length / -ray.d.z;
            let p_focus = ray.at(ft);
            *ray.o = p_lens.extend(0.0);
            *ray.d = (p_focus - ray.o).normalize();
        }
        *ray.o = camera.c2w.transform_point(ray.o);
        *ray.d = camera.c2w.transform_vector(ray.d);
        // cpu_dbg!(ray);
        (**ray, w)
    }
}
#[derive(Clone, Copy, Debug, Value)]
#[luisa(crate = "luisa")]
#[repr(C)]
struct PerspectiveCameraData {
    resolution: Uint2,
    c2w: AffineTransform,
    w2c: AffineTransform,
    fov: f32,
    r2c: AffineTransform,
    c2r: AffineTransform,
    lens_area: f32,
    lens_radius: f32,
    focal_length: f32,
}
impl PerspectiveCameraData {
    fn new(
        device: Device,
        resolution: Uint2,
        transform: AffineTransform,
        fov: f32,
        lens_radius: f32,
        focal_length: f32,
        buffer: &Buffer<PerspectiveCameraData>,
    ) {
        let mut m = glam::Mat4::IDENTITY;
        let fres = glam::vec2(resolution.x as f32, resolution.y as f32);
        m = glam::Mat4::from_scale(glam::vec3(1.0 / fres.x, 1.0 / fres.y, 1.0)) * m;
        m = glam::Mat4::from_scale(glam::vec3(2.0, 2.0, 1.0)) * m;
        m = glam::Mat4::from_translation(glam::vec3(-1.0, -1.0, 0.0)) * m;
        m = glam::Mat4::from_scale(glam::vec3(1.0, -1.0, 1.0)) * m;
        let s = (fov / 2.0).tan();
        if resolution.x > resolution.y {
            m = glam::Mat4::from_scale(glam::vec3(s, s * fres.y / fres.x, 1.0)) * m;
        } else {
            m = glam::Mat4::from_scale(glam::vec3(s * fres.x / fres.y, s, 1.0)) * m;
        }
        m = glam::Mat4::from_translation(glam::vec3(0.0, 0.0, -1.0)) * m;
        let r2c = AffineTransform::from_matrix(&m);
        buffer.view(..).copy_from(&[Self {
            resolution,
            c2w: transform,
            w2c: transform.inverse(),
            fov,
            r2c,
            c2r: r2c.inverse(),
            lens_area: std::f32::consts::PI * lens_radius * lens_radius,
            lens_radius,
            focal_length,
        }]);
        if !is_dry_run() {
            Kernel::<fn()>::new(
                &device,
                track!(&|| {
                    let camera = buffer;
                    let c = camera.read(0);
                    let r2c = c.r2c;
                    let resolution = c.resolution;
                    let a = {
                        let p_min = r2c.transform_point(Float3::expr(0.0, 0.0, 0.0));
                        let p_max = r2c.transform_point(Float3::expr(
                            resolution.x.cast_f32(),
                            resolution.y.cast_f32(),
                            0.0,
                        ));
                        let p_min = p_min / p_min.z;
                        let p_max = p_max / p_max.z;
                        ((p_max.x - p_min.x) * (p_max.y - p_min.y)).abs()
                    };
                    let c = c.var();
                    *c.lens_area = a;
                    *c.w2c = c.c2w.inverse();
                    *c.c2r = c.r2c.inverse();
                    camera.write(0, c);
                }),
            )
            .dispatch([1, 1, 1]);
        }
    }
}
