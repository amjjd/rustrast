use std::{array, cell::*, fs::*, iter, path::*, slice::*, sync::*};
use core::arch::x86_64::*;
use windows::Win32::Graphics::Gdi::*;
use once_cell::sync::Lazy;
use scoped_threadpool::Pool;

pub mod time;
pub mod simd_vec;
mod obj;
#[macro_use]
mod transformation;
#[macro_use]
mod rasterisation;
#[macro_use]
mod binning;
mod shaders;

use time::*;
use simd_vec::*;
use obj::*;
use transformation::*;
use binning::*;
use shaders::*;
use rasterisation::*;

pub const LOG_TILE_WIDTH: usize = 7;
pub const LOG_TILE_HEIGHT: usize = 7;
pub const TILE_WIDTH: usize = 1 << LOG_TILE_WIDTH; // must be a multiple of BUFFER_X_ALIGNMENT
pub const TILE_HEIGHT: usize = 1 << LOG_TILE_HEIGHT; // must be a multiple of BUFFER_Y_ALIGNMENT
const SHADOW_MAP_SIZE: usize = 1024;
const AA_MODE: AntiAliasingMode<MSAA_2X_IX, Rgb10a2> = MSAA_2X;

// hackery to avoid managing memory; these are all initialised based on the loaded model
struct SceneBuffers {
    model: Model,
    num_vertices_padded: usize,
    rotation: Cell<f32>,
    shadow_map_xs: RefCell<SimdVec<f32>>,
    shadow_map_ys: RefCell<SimdVec<f32>>,
    shadow_map_zs: RefCell<SimdVec<f32>>,
    shadow_map_iws: RefCell<SimdVec<f32>>,
    xmins: RefCell<SimdVec<f32>>,
    ymins: RefCell<SimdVec<f32>>,
    xmaxs: RefCell<SimdVec<f32>>,
    ymaxs: RefCell<SimdVec<f32>>,
    iareas: RefCell<SimdVec<f32>>,
    tl0s: RefCell<SimdVec<u32>>,
    tl1s: RefCell<SimdVec<u32>>,
    tl2s: RefCell<SimdVec<u32>>,
    // for each binning thread, each tile has a list of triangles
    tile_triangles: RefCell<[Vec<Vec<u32>>; NUM_BIN_THREADS]>,
    shadow_map: RefCell<SimdVec<f32>>,
    xs: RefCell<SimdVec<f32>>,
    ys: RefCell<SimdVec<f32>>,
    zs: RefCell<SimdVec<f32>>,
    iws: RefCell<SimdVec<f32>>,
    diffuse_intensities: RefCell<SimdVec<f32>>,
    colour: RefCell<SimdVec<Rgb10a2>>,
    depth: RefCell<SimdVec<f32>>
}

static SCENE: OnceLock<Mutex<SceneBuffers>> = OnceLock::new();

pub fn init() {
    //let mut model = read_obj(File::open(Path::new("src/cube.obj")).unwrap(), false);
    let mut model = read_obj(File::open(Path::new("src/DinklageLikenessSculpt.obj")).unwrap(), false);
    model.xs.pad_to_mm256();
    model.ys.pad_to_mm256();
    model.zs.pad_to_mm256();
    model.ws.pad_to_mm256();
    model.vertex_normal_xs.pad_to_mm256();
    model.vertex_normal_ys.pad_to_mm256();
    model.vertex_normal_zs.pad_to_mm256();

    let num_vertices_padded = model.xs.len();
    let num_triangles = model.num_triangles as usize;

    let scene = SceneBuffers {
        model,
        rotation: Cell::new(0.0),
        num_vertices_padded,
        shadow_map_xs: RefCell::new(iter::repeat(0f32).take(num_vertices_padded).collect()),
        shadow_map_ys: RefCell::new(iter::repeat(0f32).take(num_vertices_padded).collect()),
        shadow_map_zs: RefCell::new(iter::repeat(0f32).take(num_vertices_padded).collect()),
        shadow_map_iws: RefCell::new(iter::repeat(0f32).take(num_vertices_padded).collect()),
        xmins: RefCell::new(iter::repeat(0f32).take(num_triangles).collect()),
        ymins: RefCell::new(iter::repeat(0f32).take(num_triangles).collect()),
        xmaxs: RefCell::new(iter::repeat(0f32).take(num_triangles).collect()),
        ymaxs: RefCell::new(iter::repeat(0f32).take(num_triangles).collect()),
        iareas: RefCell::new(iter::repeat(0f32).take(num_triangles).collect()),
        tl0s: RefCell::new(iter::repeat(0u32).take(num_triangles).collect()),
        tl1s: RefCell::new(iter::repeat(0u32).take(num_triangles).collect()),
        tl2s: RefCell::new(iter::repeat(0u32).take(num_triangles).collect()),
        tile_triangles: RefCell::new(array::from_fn(|_| Vec::new())),
        shadow_map: RefCell::new(iter::repeat(0f32).take(SHADOW_MAP_SIZE * SHADOW_MAP_SIZE).collect()),
        xs: RefCell::new(iter::repeat(0f32).take(num_vertices_padded).collect()),
        ys: RefCell::new(iter::repeat(0f32).take(num_vertices_padded).collect()),
        zs: RefCell::new(iter::repeat(0f32).take(num_vertices_padded).collect()),
        iws: RefCell::new(iter::repeat(0f32).take(num_vertices_padded).collect()),
        diffuse_intensities: RefCell::new(iter::repeat(0f32).take(num_vertices_padded).collect()),
        colour: RefCell::new(SimdVec::new()),
        depth: RefCell::new(SimdVec::new())
    };

    let _ = SCENE.set(Mutex::new(scene));
}

fn scene_buffers() -> &'static Mutex<SceneBuffers> {
    SCENE.get().unwrap()
}

fn vertices<TV: Send + Copy + Default>(
        xs: &RefCell<SimdVec<f32>>, ys: &RefCell<SimdVec<f32>>, zs: &RefCell<SimdVec<f32>>, iws: &RefCell<SimdVec<f32>>, extras_out: &RefCell<SimdVec<TV>>,
        model: &Model, vertex_shader: &impl AvxVertexShader<TV>, log_prefix: &str) {
    {
        let xs_out = &mut *xs.borrow_mut();
        let ys_out = &mut *ys.borrow_mut();
        let zs_out = &mut *zs.borrow_mut();
        let iws_out = &mut *iws.borrow_mut();
        let extras_out = &mut *extras_out.borrow_mut();
        time(format!("{}Transformed and shaded {} vertices", log_prefix, model.num_vertices), || {
            execute_vertex_shader(xs_out, ys_out, zs_out, iws_out, extras_out.as_mut_slice(), model, vertex_shader)
        });
    }
}

// my machine stops showing improvement above six threads
const NUM_DRAW_THREADS: u32 = 6;
static DRAW_WORKERS: Lazy<Mutex<Pool>> = Lazy::new(|| Mutex::new(Pool::new(NUM_DRAW_THREADS)));

fn draw_tile<TE, const F: bool, TF: Send + Copy, const AA_MODE: i32, C: FragmentColour>(
        colour: &mut [C], depth: &mut [f32], buffer_size: &BufferSize,
        tile_xmin: f32, tile_ymin: f32, tile_xmax: f32, tile_ymax: f32,
        model: &Model, xs: &SimdVec<f32>, ys: &SimdVec<f32>, zs: &SimdVec<f32>, iws: &SimdVec<f32>,
        xmins: &SimdVec<f32>, ymins: &SimdVec<f32>, xmaxs: &SimdVec<f32>, ymaxs: &SimdVec<f32>,
        iareas: &SimdVec<f32>, tl0s: &SimdVec<u32>, tl1s: &SimdVec<u32>, tl2s: &SimdVec<u32>,
        triangles: [&Vec<u32>; NUM_BIN_THREADS], 
        extras: &TE, vertex_extra: fn(&TE, usize) -> TF, 
        aa_mode: AntiAliasingMode<AA_MODE, C>, fragment_shader: &impl AvxFragmentShader<F, TF>) {
    for i in 0..NUM_BIN_THREADS {
        for j in 0..triangles[i].len() {
            let it = triangles[i][j] as usize;
            let xmin = xmins[it];
            let ymin = ymins[it];
            let xmax = xmaxs[it];
            let ymax = ymaxs[it];
            let iarea = iareas[it];
            let tl0 = tl0s[it] != 0;
            let tl1 = tl1s[it] != 0;
            let tl2 = tl2s[it] != 0;

            // clip to the tile
            let xmin = xmin.max(tile_xmin);
            let ymin = ymin.max(tile_ymin);
            let xmax = xmax.min(tile_xmax);
            let ymax = ymax.min(tile_ymax);

            let x0 = xs[model.trianglev0s[it] as usize];
            let y0 = ys[model.trianglev0s[it] as usize];
            let z0 = zs[model.trianglev0s[it] as usize];
            let iw0 = iws[model.trianglev0s[it] as usize];
            let x1 = xs[model.trianglev1s[it] as usize];
            let y1 = ys[model.trianglev1s[it] as usize];
            let z1 = zs[model.trianglev1s[it] as usize];
            let iw1 = iws[model.trianglev1s[it] as usize];
            let x2 = xs[model.trianglev2s[it] as usize];
            let y2 = ys[model.trianglev2s[it] as usize];
            let z2 = zs[model.trianglev2s[it] as usize];
            let iw2 = iws[model.trianglev2s[it] as usize];
            let v0 = model.trianglev0s[it] as usize;
            let v1 = model.trianglev1s[it] as usize;
            let v2 = model.trianglev2s[it] as usize;    
            fill_triangle(colour, depth, buffer_size, it,
                xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2,
                vertex_extra(extras, v0), vertex_extra(extras, v1), vertex_extra(extras, v2), iarea, tl0, tl1, tl2,
                aa_mode, fragment_shader);
        }
    }
}

#[derive(Clone, Copy)]
pub struct BufferSize {
    pub stride: usize,
    pub lines: usize
}

impl BufferSize {
    pub fn len(&self) -> usize {
        self.stride * self.lines
    }
}

fn clear_buffer<T: Default + Copy>(buffer: &mut SimdVec<T>, buffer_size: &BufferSize, zero: T) {
    if buffer.len() > buffer_size.len() {
        buffer.truncate(buffer_size.len());
        buffer.fill(zero);
    }
    else {
        buffer.fill(zero);
        if buffer.len() < buffer_size.len() {
            let extra = buffer_size.len() - buffer.len();
            buffer.reserve_exact(extra);
            buffer.extend(iter::repeat(zero).take(extra));
        }
    }
}

const BG: RGBQUAD = RGBQUAD{rgbBlue: 0, rgbGreen: 0, rgbRed: 0, rgbReserved: 0};

fn draw_triangles<TE : Send + Sync, const CULL_MODE: i32, const AA_MODE: i32, C: FragmentColour, const EXECUTE_FRAGMENT_SHADER: bool, TF: Send + Copy>(
        out: &mut [RGBQUAD], out_buffer_size: &BufferSize, colour: &RefCell<SimdVec<Rgb10a2>>, depth: &RefCell<SimdVec<f32>>, buffer_size: &BufferSize,
        width: usize, height: usize,
        scene: &SceneBuffers, xs: &RefCell<SimdVec<f32>>, ys: &RefCell<SimdVec<f32>>, zs: &RefCell<SimdVec<f32>>, iws: &RefCell<SimdVec<f32>>,
        cull_mode: CullMode<CULL_MODE>, extras: &TE,
        aa_mode: AntiAliasingMode<AA_MODE, C>, vertex_extra: fn(&TE, usize) -> TF, fragment_shader: &impl AvxFragmentShader<EXECUTE_FRAGMENT_SHADER, TF>,
        log_prefix: &str) {
    let model = &scene.model;
    let num_triangles = model.num_triangles;

    let xs = &*xs.borrow();
    let ys = &*ys.borrow();
    let zs = &*zs.borrow();
    let iws = &*iws.borrow();

    {
        let xmins_out = &mut *scene.xmins.borrow_mut();
        let ymins_out = &mut *scene.ymins.borrow_mut();
        let xmaxs_out = &mut *scene.xmaxs.borrow_mut();
        let ymaxs_out = &mut *scene.ymaxs.borrow_mut();
        let iareas_out = &mut *scene.iareas.borrow_mut();
        let tl0s_out = &mut *scene.tl0s.borrow_mut();
        let tl1s_out = &mut *scene.tl1s.borrow_mut();
        let tl2s_out = &mut *scene.tl2s.borrow_mut();
        let tile_triangles_out = &mut *scene.tile_triangles.borrow_mut();
        time(format!("{}Binned {} triangles", log_prefix, num_triangles), || {
            bin_triangles(
                xmins_out, ymins_out, xmaxs_out, ymaxs_out, iareas_out, tl0s_out, tl1s_out, tl2s_out, tile_triangles_out,
                model, xs, ys, width, height, cull_mode);
        })
    }
    let xmins = &*scene.xmins.borrow();
    let ymins = &*scene.ymins.borrow();
    let xmaxs = &*scene.xmaxs.borrow();
    let ymaxs = &*scene.ymaxs.borrow();
    let iareas = &*scene.iareas.borrow();
    let tl0s = &*scene.tl0s.borrow();
    let tl1s = &*scene.tl1s.borrow();
    let tl2s = &*scene.tl2s.borrow();
    let tile_triangles = scene.tile_triangles.borrow();

    let colour = &mut *colour.borrow_mut();
    if EXECUTE_FRAGMENT_SHADER {
        if AA_MODE == NOAA_IX && out_buffer_size.len() > 0 {
            time("Cleared colour buffer", || {
                out.fill(BG);
            });
        }
        else if buffer_size.len() > 0 {
            time("Cleared colour buffer", || {
                clear_buffer(colour, buffer_size, BG.into());
            });
        }
    }

    let depth = &mut *depth.borrow_mut();
    time(format!("{}Cleared depth buffer", log_prefix), ||{
       clear_buffer(depth, buffer_size, 0.0);
    });

    time(format!("{}Filled triangles", log_prefix), || {
        let mut pool = DRAW_WORKERS.lock().unwrap();
        pool.scoped(|scope| {
            let mut ymin = 0;
            let mut i_tile = 0;

            while ymin < height  {
                let mut xmin = 0;
                while xmin < width {
                    // bypass safeness checks when multithreading
                    let colour = unsafe {
                        if AA_MODE == NOAA_IX {
                            from_raw_parts_mut(out.as_mut_ptr() as *mut C, out_buffer_size.len())
                        }
                        else {
                            from_raw_parts_mut(colour.as_mut_ptr() as *mut C, colour.len())
                        }
                    };
                    let depth = unsafe { from_raw_parts_mut(depth.as_mut_ptr(), buffer_size.len()) };
                    let out = unsafe { from_raw_parts_mut(out.as_mut_ptr(), out_buffer_size.len()) };
                    let xmax = (xmin + TILE_WIDTH).min(width);
                    let ymax = (ymin + TILE_HEIGHT).min(height);

                    let triangles = array::from_fn(|i| &tile_triangles[i][i_tile]);

                    scope.execute(move || {
                        draw_tile(
                            colour, depth, buffer_size,
                            xmin as f32, ymin as f32, xmax as f32, ymax as f32,
                            model, xs, ys, zs, iws, xmins, ymins, xmaxs, ymaxs, iareas, tl0s, tl1s, tl2s, triangles, extras, vertex_extra,
                            aa_mode, fragment_shader);
                        resolve(out, out_buffer_size, colour, buffer_size, xmin, ymin, xmax - xmin, ymax - ymin, aa_mode);
                    });

                    xmin += TILE_WIDTH;
                    i_tile += 1;
                }

                ymin += TILE_HEIGHT;
            }
        });
    });
}

const ROTATION_STEP: f32 = 0.01;
const ROTATION_MAX: f32 = std::f32::consts::TAU;

struct VertexShader<'a> {
    model: &'a Model,
    t: &'a Transformation,
    light_direction_x: __m256,
    light_direction_y: __m256,
    light_direction_z: __m256,
    light_intensity: __m256,
    it_world: &'a[[f32; 3]; 3]
}

impl <'a>VertexShader<'a> {
    fn new(model: &'a Model, t: &'a Transformation, light_direction: &'a CartesianVector, light_intensity: f32, it_world: &'a[[f32; 3]; 3]) -> Self {
        let light_direction_x = unsafe { _mm256_set1_ps(light_direction.x) };
        let light_direction_y = unsafe { _mm256_set1_ps(light_direction.y) };
        let light_direction_z = unsafe { _mm256_set1_ps(light_direction.z) };
        let light_intensity = unsafe { _mm256_set1_ps(light_intensity) };
        Self { model, t, light_direction_x, light_direction_y, light_direction_z, light_intensity, it_world }
    }
}

impl AvxVertexShader<f32> for VertexShader<'_> {
    unsafe fn vertex(&self, iv_offset: usize,
            xs_out: &mut [__m256], ys_out: &mut [__m256], zs_out: &mut [__m256], ws_out: &mut [__m256], diffuse_intensities_out: &mut [f32],
            xs: &[__m256], ys: &[__m256], zs: &[__m256], ws: &[__m256]) -> bool {
        let convert_to_cartesian = vertices_chunk_transformed(xs_out, ys_out, zs_out, ws_out, xs, ys, zs, ws, self.t);

        let chunk_start = iv_offset / 8;
        let chunk_end = chunk_start + xs_out.len();
        let vertex_normal_xs = &self.model.vertex_normal_xs.as_m256()[chunk_start..chunk_end];
        let vertex_normal_ys = &self.model.vertex_normal_ys.as_m256()[chunk_start..chunk_end];
        let vertex_normal_zs = &self.model.vertex_normal_zs.as_m256()[chunk_start..chunk_end];

        vectors_chunk_transformed_normalised(vertex_normal_xs, vertex_normal_ys, vertex_normal_zs, self.it_world, |i, xt, yt, zt| {
            let diffuse_intensity = _mm256_mul_ps(_mm256_max_ps(dot_product!(xt, yt, zt, self.light_direction_x, self.light_direction_y, self.light_direction_z), _mm256_setzero_ps()), self.light_intensity);
            _mm256_store_ps(diffuse_intensities_out[i * 8..].as_mut_ptr(), diffuse_intensity);
        });

        return convert_to_cartesian;
    }
}

struct FragmentShader<'a> {
    light_intensity: __m256,
    shadow_intensity: __m256,
    ambient_intensity: __m256,
    shadow_map: &'a [f32]
}

impl<'a> FragmentShader<'a> {
    fn new(light_intensity: f32, shadow_attenuation: f32,ambient_intensity: f32, shadow_map: &'a [f32]) -> Self {
        let light_intensity = unsafe { _mm256_set1_ps(light_intensity) };
        let shadow_intensity = unsafe { _mm256_set1_ps(1.0 - shadow_attenuation) };
        let ambient_intensity = unsafe { _mm256_set1_ps(ambient_intensity) };
        Self{light_intensity, shadow_intensity, ambient_intensity, shadow_map}
    }
}

const COLOUR_LUT_SIZE: usize = 1024;
static COLOUR_LUT: Lazy<[Rgb10a2; COLOUR_LUT_SIZE]> = Lazy::new(|| {
    array::from_fn(|i| {
        let intensity = i as f32 / (COLOUR_LUT_SIZE - 1) as f32;
        let rgb_value = (intensity * 1023.0) as i32;
        Rgb10a2::new(rgb_value, rgb_value, rgb_value, 0)
    })
});

#[derive(Clone, Copy)]
struct FragmentExtra {
    diffuse_intensity: __m256,
    shadow_map_x: __m256,
    shadow_map_y: __m256,
    shadow_map_z: __m256
}

impl AvxFragmentShader<RUN_FRAGMENT_SHADER, FragmentExtra> for FragmentShader<'_> {
    unsafe fn fragment(&self, _it: usize, p_w0: __m256, p_w1: __m256, p_w2: __m256, extra0: FragmentExtra, extra1: FragmentExtra, extra2: FragmentExtra) -> __m256i {
        let shadow_map_x = interpolate!(extra0.shadow_map_x, extra1.shadow_map_x, extra2.shadow_map_x, p_w0, p_w1, p_w2);
        let shadow_map_y = interpolate!(extra0.shadow_map_y, extra1.shadow_map_y, extra2.shadow_map_y, p_w0, p_w1, p_w2);
        let shadow_map_z = interpolate!(extra0.shadow_map_z, extra1.shadow_map_z, extra2.shadow_map_z, p_w0, p_w1, p_w2);
        let diffuse_intensity = interpolate!(extra0.diffuse_intensity, extra1.diffuse_intensity, extra2.diffuse_intensity, p_w0, p_w1, p_w2);

        // bias to avoid shadow acne, scaled by how close the surface is to parallel with the light direction
        // diffuse intensity is a good proxy for this
        let bias = _mm256_max_ps(_mm256_set1_ps(0.001), _mm256_mul_ps(_mm256_set1_ps(0.01), _mm256_sub_ps(self.light_intensity, diffuse_intensity)));
        let shadow_map_z = _mm256_add_ps(shadow_map_z, bias);

        // test the four shadow map texels closest to the shadow map coordinates
        // note that if the coordinates are on the right or bottom they are pushed in slightly; the shadow map should be
        // big enough that this isn't a problem
        let shadow_map_x_int = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvttps_epi32(shadow_map_x), _mm256_setzero_si256()), _mm256_set1_epi32((SHADOW_MAP_SIZE - 2) as i32));
        let shadow_map_y_int = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvttps_epi32(shadow_map_y), _mm256_setzero_si256()), _mm256_set1_epi32((SHADOW_MAP_SIZE - 2) as i32));
        let shadow_map_index0 = _mm256_add_epi32(_mm256_mullo_epi32(shadow_map_y_int, _mm256_set1_epi32(SHADOW_MAP_SIZE as i32)), shadow_map_x_int);
        
        let shadow_map_depth0 = _mm256_i32gather_ps(self.shadow_map.as_ptr(), shadow_map_index0, 4);
        let shadow_map_depth1 = _mm256_i32gather_ps(self.shadow_map.as_ptr().add(1), shadow_map_index0, 4);
        let shadow_map_depth2 = _mm256_i32gather_ps(self.shadow_map.as_ptr().add(SHADOW_MAP_SIZE), shadow_map_index0, 4);
        let shadow_map_depth3 = _mm256_i32gather_ps(self.shadow_map.as_ptr().add(SHADOW_MAP_SIZE + 1), shadow_map_index0, 4);
        
        let lit0 = _mm256_and_ps(_mm256_cmp_ps(shadow_map_z, shadow_map_depth0, _CMP_GE_OQ), _mm256_set1_ps(1.0));
        let lit1 = _mm256_and_ps(_mm256_cmp_ps(shadow_map_z, shadow_map_depth1, _CMP_GE_OQ), _mm256_set1_ps(1.0));
        let lit2 = _mm256_and_ps(_mm256_cmp_ps(shadow_map_z, shadow_map_depth2, _CMP_GE_OQ), _mm256_set1_ps(1.0));
        let lit3 = _mm256_and_ps(_mm256_cmp_ps(shadow_map_z, shadow_map_depth3, _CMP_GE_OQ), _mm256_set1_ps(1.0));
        // allow shadows to retain a small part of the diffuse intensity to give an effect like ambient occlusion
        let lit = _mm256_max_ps(self.shadow_intensity, _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(lit0, lit1), _mm256_add_ps(lit2, lit3)), _mm256_set1_ps(0.25)));
        
        let diffuse_intensity = _mm256_mul_ps(diffuse_intensity, lit);

        let intensity = _mm256_add_ps(diffuse_intensity, self.ambient_intensity);

        let clamped_intensity = _mm256_max_ps(_mm256_min_ps(intensity, _mm256_set1_ps(1.0)), _mm256_setzero_ps());
        let scaled_intensity = _mm256_mul_ps(clamped_intensity, _mm256_set1_ps((COLOUR_LUT_SIZE - 1) as f32));
        let lut_index = _mm256_cvtps_epi32(scaled_intensity);
        _mm256_i32gather_epi32(COLOUR_LUT.as_ptr() as *const i32, lut_index, 4)
    }
}

pub fn get_out_buffer_size(width: usize, height: usize) -> BufferSize {
    rasterisation::get_out_buffer_size(width, height, AA_MODE)
}

pub fn draw(out: &mut [RGBQUAD], out_buffer_size: &BufferSize, width: usize, height: usize) {
    let scene = scene_buffers().lock().unwrap();
    let model = &scene.model;

    // animate by rotating the model
    let rotation = scene.rotation.get();
    let world = Transformation::rotate_y(rotation);
    scene.rotation.set((rotation + ROTATION_STEP) % ROTATION_MAX);
    
    // place the camera above the model's head and look down 30 degrees
    let eye = CartesianCoordinates {x: 0.0, y: 1.0, z: 2.0};
    let centre = CartesianCoordinates {x: 0.0, y: 0.0, z: 0.0};
    let up = CartesianVector {x: 0.0, y: 1.0, z: 0.0};
    let view = Transformation::look_at_rh(&eye, &centre, &up);

    // make the canonical view volume big enough to hold the model and a bit more
    let aspect = height as f32 / width as f32;
    let view_volume_width = 0.5;
    let view_volume_height = view_volume_width * aspect;
    let near = 2.0;
    let far = near + 0.5;
    // flip near and far so we can clear the z-buffer faster; precision isn't really important in this demo
    let projection = Transformation::perspective_rh(view_volume_width, view_volume_height, far, near);

    let buffer_size = get_buffer_size(width, height, AA_MODE);
    let (scale_x, scale_y) = rasterisation::get_scale(AA_MODE);
    let width = (width as f32 * scale_x) as usize;
    let height = (height as f32 * scale_y) as usize;
    let viewport = Transformation::viewport(0, 0, width, height);

    let t = world.then(&view).then(&projection).then(&viewport);

    // one distant light source, coming from top right behind the camera
    let light_direction: CartesianVector = CartesianVector {x: 1.0, y: 1.0, z: 1.0}.normalised();
    let light_intensity = 0.3;
    let shadow_attenuation = 0.8;
    let ambient_intensity = 0.05;
    // to transform vertex normals
    let it_world = world.inverted_transposed_tl_3x3().unwrap();

    // tweaked experimentally to just about contain our model as it rotates; real code would need to
    // base this on the camera frustrum's bounds from the light's point of view
    let light_eye = CartesianCoordinates {x: 1.0, y: 1.0, z: 1.0};
    let light_view = Transformation::look_at_rh(&light_eye, &centre, &up);
    let light_projection = Transformation::orthographic_rh(-0.10, 0.135, 0.10, -0.10, 2.0, 0.0);
    let light_viewport = Transformation::viewport(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
    let light_t = world.then(&light_view).then(&light_projection).then(&light_viewport);

    let shadow_map_vertex_shader = NullVertexShader::new(&light_t);
    let null_extras = RefCell::new(iter::repeat(()).take(scene.num_vertices_padded).collect()); // no allocation
    vertices(&scene.shadow_map_xs, &scene.shadow_map_ys, &scene.shadow_map_zs, &scene.shadow_map_iws, &null_extras, &model, &shadow_map_vertex_shader, "Shadow map: ");
    let null_vertex_extra = |_: &(), _| ();
    draw_triangles(
        &mut [], &BufferSize {stride: 0, lines: 0}, 
        &RefCell::new(SimdVec::new()), &scene.shadow_map, &BufferSize { stride: SHADOW_MAP_SIZE, lines: SHADOW_MAP_SIZE },
        SHADOW_MAP_SIZE, SHADOW_MAP_SIZE,
        &scene, &scene.shadow_map_xs, &scene.shadow_map_ys, &scene.shadow_map_zs, &scene.shadow_map_iws,
        CULL_BACK_FACING, &(),
        NOAA, null_vertex_extra, &NullFragmentShader::INSTANCE,
        "Shadow map: ");
    
    let vertex_shader = VertexShader::new(&model, &t, &light_direction, light_intensity, &it_world);
    vertices(&scene.xs, &scene.ys, &scene.zs, &scene.iws, &scene.diffuse_intensities, &model, &vertex_shader, "");
    let diffuse_intensities = &*scene.diffuse_intensities.borrow();
    let shadow_map_xs = &*scene.shadow_map_xs.borrow();
    let shadow_map_ys = &*scene.shadow_map_ys.borrow();
    let shadow_map_zs = &*scene.shadow_map_zs.borrow();
    let shadow_map = &*scene.shadow_map.borrow();
    let vertex_extra = |extras: &(&[f32], &[f32], &[f32], &[f32]), iv: usize| -> FragmentExtra {
        unsafe { FragmentExtra {
            diffuse_intensity: _mm256_set1_ps(extras.0[iv]),
            shadow_map_x: _mm256_set1_ps(extras.1[iv]),
            shadow_map_y: _mm256_set1_ps(extras.2[iv]),
            shadow_map_z: _mm256_set1_ps(extras.3[iv])
        } }
    };

    let fragment_shader = FragmentShader::new(light_intensity, shadow_attenuation, ambient_intensity, shadow_map.as_slice());
    draw_triangles(
        out, out_buffer_size,
        &scene.colour, &scene.depth, &buffer_size,
        width, height,
        &scene, &scene.xs, &scene.ys, &scene.zs, &scene.iws,
        CULL_BACK_FACING, &(diffuse_intensities.as_slice(), shadow_map_xs.as_slice(), shadow_map_ys.as_slice(), shadow_map_zs.as_slice()), 
        AA_MODE, vertex_extra, &fragment_shader,
        "");
}
