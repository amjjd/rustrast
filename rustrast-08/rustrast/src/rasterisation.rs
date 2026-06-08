use windows::Win32::Graphics::Gdi::*;
use core::arch::x86_64::*;
use std::sync::*;
use once_cell::sync::Lazy;
use scoped_threadpool::Pool;

use super::simd_vec::*;
use super::obj::*;
use super::shaders::*;

// not-suitable-for-production rasteriser, requires AVX2 and FMA and will fault if they aren't present

#[derive(Clone, Copy)]
pub struct TriangleProperties {
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
    pub iarea: f32,
    pub tl: u8
}

pub struct Buffer<'a, T> {
    pub buffer: &'a mut[T],
    pub left: usize,
    pub top: usize,
    pub stride: usize
}

fn min3(a: f32, b: f32, c: f32) -> f32 {
    a.min(b).min(c)
}

fn max3(a: f32, b: f32, c: f32) -> f32 {
    a.max(b).max(c)
}

fn edge_function(x0: f32, y0: f32, x1: f32, y1: f32, xp: f32, yp: f32) -> f32 {
    // this is backwards from a lot of examples due to our projection inverting the y axis
    (x1-x0)*(y0-yp) - (y0-y1)*(xp-x0)
}

fn is_top_or_left(x0: f32, y0: f32, x1: f32, y1: f32) -> bool {
    // top                   left (assuming counterclockwise, inverted y axis)
    (y0 == y1 && x0 > x1) || (y1 < y0)
}

fn calculate_properties(x0: f32, y0: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> TriangleProperties {
   TriangleProperties {
        xmin: min3(x0, x1, x2).floor(),
        ymin: min3(y0, y1, y2).floor(),
        xmax: max3(x0, x1, x2).ceil(),
        ymax: max3(y0, y1, y2).ceil(),
        iarea: 1.0 / edge_function(x0, y0, x1, y1, x2, y2),
        tl: (is_top_or_left(x1, y1, x2, y2) as u8) << 2 |
            (is_top_or_left(x2, y2, x0, y0) as u8) << 1 |
            (is_top_or_left(x0, y0, x1, y1) as u8)
    }
}

macro_rules! edge_function {
    ($x0:expr, $y0:expr, $x1:expr, $y1:expr, $xp:expr, $yp:expr) => {{
        // (x1-x0)*(y0-yp) - (y0-y1)*(xp-x0)
        _mm256_fmsub_ps(
            _mm256_sub_ps($x1, $x0), _mm256_sub_ps($y0, $yp),
            _mm256_mul_ps(_mm256_sub_ps($y0, $y1), _mm256_sub_ps($xp, $x0)))
    }}
}

macro_rules! is_top_or_left {
    ($x0:expr, $y0:expr, $x1:expr, $y1:expr) => {{
        // (y0 == y1 && x0 > x1) || (y1 < y0)
        let y_equal = _mm256_cmp_ps($y0, $y1, _CMP_EQ_OQ);
        let x0_greater = _mm256_cmp_ps($x0, $x1, _CMP_GT_OQ);
        let y1_less = _mm256_cmp_ps($y1, $y0, _CMP_LT_OQ);

        _mm256_movemask_ps(_mm256_or_ps(_mm256_and_ps(y_equal, x0_greater), y1_less)) as u8
    }};
}

unsafe fn calculate_properties_chunk(
        xmins_out: &mut [__m256], ymins_out: &mut [__m256], xmaxs_out: &mut [__m256], ymaxs_out: &mut [__m256], iareas_out: &mut [__m256], tls_out: &mut [u8],
        v0s: &[__m256i], v1s: &[__m256i], v2s: &[__m256i],
        xs: &SimdVec<f32>, ys: &SimdVec<f32>,
        triangles_offset: usize, chunk_size: usize) {
    let xs_ptr = xs.as_ptr();
    let ys_ptr = ys.as_ptr();

    for i in 0..chunk_size {
        let idx0 = v0s[triangles_offset + i];
        let idx1 = v1s[triangles_offset + i];
        let idx2 = v2s[triangles_offset + i];

        let x0 = _mm256_i32gather_ps(xs_ptr, idx0, 4);
        let y0 = _mm256_i32gather_ps(ys_ptr, idx0, 4);
        let x1 = _mm256_i32gather_ps(xs_ptr, idx1, 4);
        let y1 = _mm256_i32gather_ps(ys_ptr, idx1, 4);
        let x2 = _mm256_i32gather_ps(xs_ptr, idx2, 4);
        let y2 = _mm256_i32gather_ps(ys_ptr, idx2, 4);

        xmins_out[i] = _mm256_floor_ps(_mm256_min_ps(_mm256_min_ps(x0, x1), x2));
        ymins_out[i] = _mm256_floor_ps(_mm256_min_ps(_mm256_min_ps(y0, y1), y2));
        xmaxs_out[i] = _mm256_ceil_ps(_mm256_max_ps(_mm256_max_ps(x0, x1), x2));
        ymaxs_out[i] = _mm256_ceil_ps(_mm256_max_ps(_mm256_max_ps(y0, y1), y2));

        let area = edge_function!(x0, y0, x1, y1, x2, y2);
        iareas_out[i] = _mm256_rcp_ps(area);

        let tl0 = is_top_or_left!(x1, y1, x2, y2);
        let tl1 = is_top_or_left!(x2, y2, x0, y0);
        let tl2 = is_top_or_left!(x0, y0, x1, y1);

        for i in 0..8 {
            tls_out[i] = ((tl0 >> i) & 1) << 2 | ((tl1 >> i) & 1) << 1 | ((tl2 >> i) & 1);
        }
    }
}

// my machine stops showing improvement above four threads
static NUM_BOUNDS_THREADS: u32 = 4;
static BOUNDS_WORKERS: Lazy<Mutex<Pool>> = Lazy::new(|| Mutex::new(Pool::new(NUM_BOUNDS_THREADS)));

pub fn calculate_all_properties(
        xmins_out: &mut SimdVec<f32>, ymins_out: &mut SimdVec<f32>, xmaxs_out: &mut SimdVec<f32>, ymaxs_out: &mut SimdVec<f32>, iareas_out: &mut SimdVec<f32>, tls_out: &mut [u8],
        model: &Model, xs: &SimdVec<f32>, ys: &SimdVec<f32>) {
    let num_chunks = NUM_BOUNDS_THREADS;
    // maintain 128 byte alignment for caching
    let chunk_size = (((model.num_triangles / num_chunks) / 32) * 4) as usize;
    let mut chunk_start = 0;

    if chunk_size > 0 {
        let v0s = model.trianglev0s.as_m256i();
        let v1s = model.trianglev1s.as_m256i();
        let v2s = model.trianglev2s.as_m256i();

        let mut pool = BOUNDS_WORKERS.lock().unwrap();
        pool.scoped(|scope| {
            let xmins_out_chunks = xmins_out.as_m256_mut().chunks_exact_mut(chunk_size);
            let ymins_out_chunks = ymins_out.as_m256_mut().chunks_exact_mut(chunk_size);
            let xmaxs_out_chunks = xmaxs_out.as_m256_mut().chunks_exact_mut(chunk_size);
            let ymaxs_out_chunks = ymaxs_out.as_m256_mut().chunks_exact_mut(chunk_size);
            let iareas_out_chunks = iareas_out.as_m256_mut().chunks_exact_mut(chunk_size);
            let tls_out_chunks = tls_out.chunks_exact_mut(chunk_size * 8);

            for (xmins_out_chunk, (ymins_out_chunk, (xmaxs_out_chunk, (ymaxs_out_chunk, (iareas_chunk, tls_out_chunk))))) in xmins_out_chunks.zip(ymins_out_chunks.zip(xmaxs_out_chunks.zip(ymaxs_out_chunks.zip(iareas_out_chunks.zip(tls_out_chunks))))) {
                let triangles_offset = chunk_start;
                scope.execute(move || unsafe {
                    calculate_properties_chunk(
                        xmins_out_chunk, ymins_out_chunk, xmaxs_out_chunk, ymaxs_out_chunk, iareas_chunk, tls_out_chunk,
                        v0s, v1s, v2s,
                        xs, ys,
                        triangles_offset, chunk_size);
                });

                chunk_start += chunk_size as usize;
            }
        });
    }

    // do any leftovers sequentially
    for i in (chunk_start * 8 as usize)..(model.num_triangles as usize) {
        let bounds = calculate_properties(xs[model.trianglev0s[i] as usize], ys[model.trianglev0s[i] as usize], xs[model.trianglev1s[i] as usize], ys[model.trianglev1s[i] as usize], xs[model.trianglev2s[i] as usize], ys[model.trianglev2s[i] as usize]);
        xmins_out[i] = bounds.xmin;
        ymins_out[i] = bounds.ymin;
        xmaxs_out[i] = bounds.xmax;
        ymaxs_out[i] = bounds.ymax;
        iareas_out[i] = bounds.iarea;
        tls_out[i] = bounds.tl;
    }
}

macro_rules! interpolate {
    // s0*w0 + s1*w1 + s2*w2
    ($s0:expr, $s1:expr, $s2:expr, $w0:expr, $w1:expr, $w2:expr) => {
        _mm256_fmadd_ps($s2, $w2, _mm256_fmadd_ps($s1, $w1, _mm256_mul_ps($s0, $w0)))
    }
}

fn fill_triangle_generic<const C0: i32, const C1: i32, const C2: i32, const F: bool, T : Copy>(
        colour: &mut Buffer<RGBQUAD>, depth: &mut Buffer<f32>,
        it: usize,
        xmin: f32, ymin: f32, xmax: f32, ymax: f32,
        x0: f32, y0: f32, z0: f32, iw0: f32,
        x1: f32, y1: f32, z1: f32, iw1: f32,
        x2: f32, y2: f32, z2: f32, iw2: f32,
        e0: T, e1: T, e2: T,
        iarea: f32,
        fragment_shader: &impl AvxFragmentShader<F, T>) {
    debug_assert!(colour.buffer.as_ptr().align_offset(32) == 0);
    debug_assert!(colour.stride % 8 == 0);
    debug_assert!(colour.left % 8 == 0);
    debug_assert!(depth.buffer.as_ptr().align_offset(32) == 0);
    debug_assert!(depth.stride % 8 == 0);
    debug_assert!(depth.left % 8 == 0);

    // draw 8 aligned pixels at once
    let xmin = (xmin / 8.0).floor() * 8.0;
    let xmax = (xmax / 8.0).ceil() * 8.0;

    unsafe {
        // barycentric coordinates of the first 8 pixels on the first row of the bounding box
        let x0_v = _mm256_set1_ps(x0);
        let y0_v = _mm256_set1_ps(y0);
        let x1_v = _mm256_set1_ps(x1);
        let y1_v = _mm256_set1_ps(y1);
        let x2_v = _mm256_set1_ps(x2);
        let y2_v = _mm256_set1_ps(y2);
        let xp = _mm256_add_ps(_mm256_set1_ps(xmin), _mm256_setr_ps(0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5));
        let yp = _mm256_set1_ps(ymin + 0.5);
        let iarea = _mm256_set1_ps(iarea);
        let mut row_w0 = _mm256_mul_ps(edge_function!(x1_v, y1_v, x2_v, y2_v, xp, yp), iarea);
        let mut row_w1 = _mm256_mul_ps(edge_function!(x2_v, y2_v, x0_v, y0_v, xp, yp), iarea);
        let mut row_w2 = _mm256_mul_ps(edge_function!(x0_v, y0_v, x1_v, y1_v, xp, yp), iarea);

        // if you substitute `xp + 1` for `xp` into the edge function you can see that
        // for a given edge, the value of the function for `xp + 1, yp` is the value for `xp, yp` minus `y0-y1`
        let iarea_times_8 = _mm256_mul_ps(iarea, _mm256_set1_ps(8.0));
        let xstep0 = _mm256_mul_ps(_mm256_set1_ps(y1-y2), iarea_times_8);
        let xstep1 = _mm256_mul_ps(_mm256_set1_ps(y2-y0), iarea_times_8);
        let xstep2 = _mm256_mul_ps(_mm256_set1_ps(y0-y1), iarea_times_8);

        // as above, the value of the edge function for `xp, yp + 1` is the value for `xp,yp` minus `x1-x0`. 
        let ystep0 = _mm256_mul_ps(_mm256_set1_ps(x2-x1), iarea);
        let ystep1 = _mm256_mul_ps(_mm256_set1_ps(x0-x2), iarea);
        let ystep2 = _mm256_mul_ps(_mm256_set1_ps(x1-x0), iarea);

        let zero = _mm256_setzero_ps();
        let one = _mm256_set1_ps(1.0);
        let c_buffer = colour.buffer.as_mut_ptr() as *mut i32;
        let d_buffer = depth.buffer.as_mut_ptr() as *mut f32;
        let iw0 = _mm256_set1_ps(iw0);
        let iw1 = _mm256_set1_ps(iw1);
        let iw2 = _mm256_set1_ps(iw2);
        let z0 = _mm256_set1_ps(z0);
        let z1 = _mm256_set1_ps(z1);
        let z2 = _mm256_set1_ps(z2);

        let mut yp = ymin as isize;
        let mut c_row = c_buffer.offset((((ymin as usize - colour.top) * colour.stride) - colour.left) as isize);
        let mut d_row = d_buffer.offset((((ymin as usize - depth.top) * depth.stride) - depth.left) as isize);
        while yp < ymax as isize {
            let mut w0 = row_w0;
            let mut w1 = row_w1;
            let mut w2 = row_w2;
            let mut xp = xmin as isize;
            while xp < xmax as isize {
                let inside0 = _mm256_castps_si256(_mm256_cmp_ps::<C0>(w0, zero));
                let inside1 = _mm256_castps_si256(_mm256_cmp_ps::<C1>(w1, zero));
                let inside2 = _mm256_castps_si256(_mm256_cmp_ps::<C2>(w2, zero));
                let inside_mask = _mm256_and_si256(inside0, _mm256_and_si256(inside1, inside2));

                // skip spans that are fully outside the triangle
                if _mm256_movemask_epi8(inside_mask) != 0 {
                    // adjust for perspective correct interpolation
                    let mut p_w0 = _mm256_mul_ps(w0, iw0);
                    let mut p_w1 = _mm256_mul_ps(w1, iw1);
                    let mut p_w2 = _mm256_mul_ps(w2, iw2);

                    let t = _mm256_rcp_ps(_mm256_add_ps(p_w0, _mm256_add_ps(p_w1, p_w2)));
                    p_w0 = _mm256_mul_ps(p_w0, t);
                    p_w1 = _mm256_mul_ps(p_w1, t);
                    p_w2 = _mm256_mul_ps(p_w2, t);

                    let z = interpolate!(z0, z1, z2, p_w0, p_w1, p_w2);

                    // this near test isn't really enough, we really need to clip geometry against the near plane
                    let near_mask = _mm256_castps_si256(_mm256_cmp_ps(z, one, _CMP_LE_OQ));

                    let existing_z = _mm256_loadu_ps(d_row.offset(xp));
                    let depth_mask = _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(z, existing_z, _CMP_GT_OQ)), near_mask);
                    let mask = _mm256_and_si256(inside_mask, depth_mask);

                    if _mm256_movemask_epi8(mask) != 0 {
                        if F {
                            let (filled_span, c_mask) = fragment_shader.fragment(it, w0, w1, w2, p_w0, p_w1, p_w2, e0, e1, e2, mask);
                            _mm256_maskstore_epi32(c_row.offset(xp) as *mut i32, c_mask, filled_span);
                        }
                        _mm256_maskstore_ps(d_row.offset(xp), mask, z);
                    }
                }

                xp += 8;

                w0 = _mm256_sub_ps(w0, xstep0);
                w1 = _mm256_sub_ps(w1, xstep1);
                w2 = _mm256_sub_ps(w2, xstep2);
            }

            yp += 1;
            c_row = c_row.offset(colour.stride as isize);
            d_row = d_row.offset(depth.stride as isize);

            row_w0 = _mm256_sub_ps(row_w0, ystep0);
            row_w1 = _mm256_sub_ps(row_w1, ystep1);
            row_w2 = _mm256_sub_ps(row_w2, ystep2);
        }
    }
}

pub fn fill_triangle<const F: bool, T : Copy>(
        colour: &mut Buffer<RGBQUAD>, depth: &mut Buffer<f32>,
        it: usize,
        xmin: f32, ymin: f32, xmax: f32, ymax: f32,
        x0: f32, y0: f32, z0: f32, iw0: f32,
        x1: f32, y1: f32, z1: f32, iw1: f32,
        x2: f32, y2: f32, z2: f32, iw2: f32,
        e0: T, e1: T, e2: T,
        iarea: f32, tl: u8,
        fragment_shader: &impl AvxFragmentShader<F, T>) {

    // NB - no TL or all TL edges are impossible, but are here for completeness
    match tl {
        0b000 => fill_triangle_generic::<_CMP_GT_OQ, _CMP_GT_OQ, _CMP_GT_OQ, F, T>(colour, depth, it, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, e0, e1, e2, iarea, fragment_shader),
        0b001 => fill_triangle_generic::<_CMP_GT_OQ, _CMP_GT_OQ, _CMP_GE_OQ, F, T>(colour, depth, it, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, e0, e1, e2, iarea, fragment_shader),
        0b010 => fill_triangle_generic::<_CMP_GT_OQ, _CMP_GE_OQ, _CMP_GT_OQ, F, T>(colour, depth, it, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, e0, e1, e2, iarea, fragment_shader),
        0b011 => fill_triangle_generic::<_CMP_GT_OQ, _CMP_GE_OQ, _CMP_GE_OQ, F, T>(colour, depth, it, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, e0, e1, e2, iarea, fragment_shader),
        0b100 => fill_triangle_generic::<_CMP_GE_OQ, _CMP_GT_OQ, _CMP_GT_OQ, F, T>(colour, depth, it, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, e0, e1, e2, iarea, fragment_shader),        
        0b101 => fill_triangle_generic::<_CMP_GE_OQ, _CMP_GT_OQ, _CMP_GE_OQ, F, T>(colour, depth, it, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, e0, e1, e2, iarea, fragment_shader),
        0b110 => fill_triangle_generic::<_CMP_GE_OQ, _CMP_GE_OQ, _CMP_GT_OQ, F, T>(colour, depth, it, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, e0, e1, e2, iarea, fragment_shader),
        0b111 => fill_triangle_generic::<_CMP_GE_OQ, _CMP_GE_OQ, _CMP_GE_OQ, F, T>(colour, depth, it, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, e0, e1, e2, iarea, fragment_shader),
        _ => unreachable!(),
    }
}