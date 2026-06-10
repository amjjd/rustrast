use core::arch::x86_64::*;
use std::sync::*;
use once_cell::sync::Lazy;
use scoped_threadpool::Pool;

use super::simd_vec::*;
use super::obj::*;
use super::transformation::*;

pub trait AvxVertexShader<T : Send + Copy> : Send + Sync {
    /**
     * Transform the given chunk of vertices and write homogenous coordinates to the output arrays.
     * 
     * # Arguments
     * - iv_offset: Index of the first vertex in the chunk being transformed.
     * - `xs_out`: Transformed vertex x values.
     * - `ys_out`: Transformed vertex y values. Will have the same length as `xs_out`.
     * - `zs_out`: Transformed vertex z values. Will have the same length as `xs_out`.
     * - `ws_out`: Transformed vertex w values. Will have the same length as `xs_out`.
     * - `extras_out`: Extra per-vertex data to pass to the fragment shader. Will be eight times as long as `xs_out`.
     * - `xs`: Input vertex x values. Will have the same length as `xs_out`.
     * - `ys`: Input vertex y values. Will have the same length as `xs_out`.
     * - `zs`: Input vertex z values. Will have the same length as `xs_out`.
     * - `ws`: Input vertex w values. Will have the same length as `xs_out`.
     * 
     * # Returns
     * - `true` if the output coordinates must be divided by w.
     */
    unsafe fn vertex(&self, iv_offset: usize,
        xs_out: &mut [__m256], ys_out: &mut [__m256], zs_out: &mut [__m256], ws_out: &mut [__m256], extras_out: &mut[T],
        xs: &[__m256], ys: &[__m256], zs: &[__m256], ws: &[__m256]) -> bool;
}

pub const RUN_FRAGMENT_SHADER: bool = true;
pub const SKIP_FRAGMENT_SHADER: bool = false;

pub trait AvxFragmentShader<const F: bool, T : Copy> : Send + Sync {
    /**
     * Calculate the colour of the given 8 fragments.
     * 
     * # Arguments
     * - `it`: Index of the triangle being filled.
     * - `w0`, `w1`, `w2`: Barycentric weights for the 8 fragments.
     * - `p_w0`, `p_w1`, `p_w2`: Perspective-corrected barycentric weights for the 8 fragments.
     * - `e0`, `e1`, `e2`: Extra per-vertex data for the three vertices of the triangle, as calculated by the vertex shader.
     * - `mask`: Mask indicating which of the 8 fragments are covered by the triangle and have passed the depth test.
     * 
     * # Returns
     * - RGB0 colour values for the 8 fragments, and the mask of fragments to store.
     */
    unsafe fn fragment(&self, it: usize, w0: __m256, w1: __m256, w2: __m256, p_w0: __m256, p_w1: __m256, p_w2: __m256, e0: T, e1: T, e2: T, mask: __m256i) -> (__m256i, __m256i);
}

// my machine stops showing improvement above four threads
const NUM_VERTEX_SHADER_THREADS: u32 = 4;
static VERTEX_SHADER_WORKERS: Lazy<Mutex<Pool>> = Lazy::new(|| Mutex::new(Pool::new(NUM_VERTEX_SHADER_THREADS)));

// This isn't suitable for production; in particular there's no opportunity to clip
// so it is possible that the perpective divide could lead to infinite values.
// Note that this does not deal with stragglers, so all input and output slices must be padded
// to amultiple of 8 - see SimdVec::pad_to_mm256.
pub fn execute_vertex_shader<T : Send + Copy>(
        xs_out: &mut SimdVec<f32>, ys_out: &mut SimdVec<f32>, zs_out: &mut SimdVec<f32>, iws_out: &mut SimdVec<f32>, extras_out: &mut[T],
        model: &Model, shader: &impl AvxVertexShader<T>) {
    debug_assert!(xs_out.len() % 8 == 0);
    debug_assert!(ys_out.len() % 8 == 0);
    debug_assert!(zs_out.len() % 8 == 0);
    debug_assert!(iws_out.len() % 8 == 0);
    debug_assert!(extras_out.len() % 8 == 0);

    let xs = model.xs.as_m256();
    let ys = model.ys.as_m256();
    let zs = model.zs.as_m256();
    let ws = model.ws.as_m256();
    
    let num_chunks = NUM_VERTEX_SHADER_THREADS;
    // maintain 128 byte alignment for caching
    let chunk_size = (((model.num_vertices / num_chunks) / 32) * 4) as usize;
    if chunk_size == 0 {
        // not worth threading
        let xs_out = xs_out.as_m256_mut();
        let ys_out = ys_out.as_m256_mut();
        let zs_out = zs_out.as_m256_mut();
        let iws_out = iws_out.as_m256_mut();
        unsafe {
            if shader.vertex(0, xs_out, ys_out, zs_out, iws_out, extras_out, xs, ys, zs, ws) {
                vertices_chunk_to_cartesian(xs_out, ys_out, zs_out, iws_out);
            }
        }
        return;
    }
    
    let mut chunk_start = 0;

    // not using built in chunks to allow the last chunk to be a bit bigger
    let mut pool = VERTEX_SHADER_WORKERS.lock().unwrap();
    pool.scoped(|scope| {
        let mut xs_out = xs_out.as_m256_mut();
        let mut ys_out = ys_out.as_m256_mut();
        let mut zs_out = zs_out.as_m256_mut();
        let mut iws_out = iws_out.as_m256_mut();
        let mut extras_out = &mut extras_out[..];

        for i in 0..num_chunks {
            let chunk_end = if i == num_chunks - 1 { xs.len() } else { chunk_start + chunk_size };
            let xs_chunk = &xs[chunk_start..chunk_end];
            let ys_chunk = &ys[chunk_start..chunk_end];
            let zs_chunk = &zs[chunk_start..chunk_end];
            let ws_chunk = &ws[chunk_start..chunk_end];
            let chunk_size = xs_chunk.len();
            let (xs_out_chunk, xs_out_rem) = xs_out.split_at_mut(chunk_size);
            let (ys_out_chunk, ys_out_rem) = ys_out.split_at_mut(chunk_size);
            let (zs_out_chunk, zs_out_rem) = zs_out.split_at_mut(chunk_size);
            let (iws_out_chunk, iws_out_rem) = iws_out.split_at_mut(chunk_size);
            let (extras_out_chunk, extras_out_rem) = extras_out.split_at_mut(chunk_size * 8);
            xs_out = xs_out_rem;
            ys_out = ys_out_rem;
            zs_out = zs_out_rem;
            iws_out = iws_out_rem;
            extras_out = extras_out_rem;
            scope.execute(move || unsafe {
                if shader.vertex(chunk_start * 8, xs_out_chunk, ys_out_chunk, zs_out_chunk, iws_out_chunk, extras_out_chunk, xs_chunk, ys_chunk, zs_chunk, ws_chunk) {
                    vertices_chunk_to_cartesian(xs_out_chunk, ys_out_chunk, zs_out_chunk, iws_out_chunk);
                }
            });
            chunk_start = chunk_end;
        }
    });
}

pub struct NullVertexShader<'a> {
    t: &'a Transformation
}

impl <'a> NullVertexShader<'a> {
    pub fn new(t: &'a Transformation) -> Self {
        Self { t }
    }
}

impl AvxVertexShader<()> for NullVertexShader<'_> {
    unsafe fn vertex(&self, _iv_offset: usize,
            xs_out: &mut [__m256], ys_out: &mut [__m256], zs_out: &mut [__m256], ws_out: &mut [__m256], _extras_out: &mut [()],
            xs: &[__m256], ys: &[__m256], zs: &[__m256], ws: &[__m256]) -> bool {
        vertices_chunk_transformed(xs_out, ys_out, zs_out, ws_out, xs, ys, zs, ws, self.t)
    }
}

pub struct NullFragmentShader {
}

impl NullFragmentShader {
    pub const INSTANCE: Self = NullFragmentShader {};
}

impl AvxFragmentShader<SKIP_FRAGMENT_SHADER, ()> for NullFragmentShader {
    unsafe fn fragment(&self, _it: usize, _w0: __m256, _w1: __m256, _w2: __m256, _p_w0: __m256, _p_w1: __m256, _p_w2: __m256, _extra0: (), _extra1: (), _extra2: (), _mask: __m256i)-> (__m256i, __m256i) {
        (_mm256_setzero_si256(), _mm256_setzero_si256()) // won't be called
    }
}