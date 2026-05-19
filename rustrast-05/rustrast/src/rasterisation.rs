use windows::Win32::Graphics::Gdi::*;
use core::arch::x86_64::*;
use std::sync::*;
use once_cell::sync::Lazy;
use scoped_threadpool::Pool;

use super::simd_vec::*;
use super::obj::*;

#[derive(Clone, Copy)]
pub struct Bounds {
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32
}

impl Bounds {
    pub const NONE: Bounds = Bounds {xmin: 0.0, ymin: 0.0, xmax: -1.0, ymax: 0.0};
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

pub fn calculate_bounds(xmin: f32, ymin: f32, xmax: f32, ymax: f32, x0: f32, y0: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> Bounds {
    // cull backwards facing triangles
    if edge_function(x0, y0, x1, y1, x2, y2) <= 0.0 {
        return Bounds::NONE;
    }
    
    Bounds {
        xmin: min3(x0, x1, x2).max(xmin as f32).floor(),
        ymin: min3(y0, y1, y2).max(ymin as f32).floor(),
        xmax: max3(x0, x1, x2).min(xmax as f32).ceil(),
        ymax: max3(y0, y1, y2).min(ymax as f32).ceil()
    }
}

fn is_top_or_left(x0: f32, y0: f32, x1: f32, y1: f32) -> bool {
    // top                   left (assuming counterclockwise, inverted y axis)
    (y0 == y1 && x0 > x1) || (y1 < y0)
}

#[allow(dead_code)]
pub unsafe fn fill_triangle(buffer: *mut RGBQUAD, stride: usize, xmin: f32, ymin: f32, xmax: f32, ymax: f32, x0: f32, y0: f32, x1: f32, y1: f32, x2: f32, y2: f32, colour: RGBQUAD) {
    // what edges are top or left?
    let tl0 = is_top_or_left(x0, y0, x1, y1);
    let tl1 = is_top_or_left(x1, y1, x2, y2);
    let tl2 = is_top_or_left(x2, y2, x0, y0);

    // values of the edge functions for the first pixel on the first row of the bounding box
    let mut row_w0 = edge_function(x0, y0, x1, y1, xmin + 0.5, ymin + 0.5);
    let mut row_w1 = edge_function(x1, y1, x2, y2, xmin + 0.5, ymin + 0.5);
    let mut row_w2 = edge_function(x2, y2, x0, y0, xmin + 0.5, ymin + 0.5);

    let mut yp = ymin;
    while yp <= ymax {
        let mut w0 = row_w0;
        let mut w1 = row_w1;
        let mut w2 = row_w2;
        let mut xp = xmin;
        while xp <= xmax {
            if ((tl0 && w0 >= 0.0) || w0 > 0.0) && ((tl1 && w1 >= 0.0) || w1 > 0.0) && ((tl2 && w2 >= 0.0) || w2 > 0.0) {
                *buffer.offset((yp as isize * stride as isize) + xp as isize) = colour;
            }

            xp += 1.0; 

            // if you substitute `xp + 1` for `xp` into the edge function you can see that
            // for a given edge, the value of the function for `xp + 1, yp` is the value for `xp, yp` minus `y0-y1`
            w0 -= y0-y1;
            w1 -= y1-y2;
            w2 -= y2-y0;
        }
        
        yp += 1.0;

        // as above, the value for `xp, yp + 1` is the value for `yp` minus `x1-x0`. 
        row_w0 -= x1-x0;
        row_w1 -= x2-x1;
        row_w2 -= x0-x2;
    }
}

// not-suitable-for-production SIMD implementations; these will only work on processors that support AVX2

#[target_feature(enable = "avx,avx2")]
unsafe fn avx2_calculate_bounds_chunk(
        xmins_out: &mut [__m256], ymins_out: &mut [__m256], xmaxs_out: &mut [__m256], ymaxs_out: &mut [__m256],
        v0s: &[__m256i], v1s: &[__m256i], v2s: &[__m256i],
        xs: &SimdVec<f32>, ys: &SimdVec<f32>,
        xmin: f32, ymin: f32, xmax: f32, ymax: f32,
        triangles_offset: usize, chunk_size: usize) {
    for i in 0..chunk_size {
        let x0 = _mm256_i32gather_ps(xs.as_ptr(), v0s[triangles_offset + i], 4);
        let y0 = _mm256_i32gather_ps(ys.as_ptr(), v0s[triangles_offset + i], 4);
        let x1 = _mm256_i32gather_ps(xs.as_ptr(), v1s[triangles_offset + i], 4);
        let y1 = _mm256_i32gather_ps(ys.as_ptr(), v1s[triangles_offset + i], 4);
        let x2 = _mm256_i32gather_ps(xs.as_ptr(), v2s[triangles_offset + i], 4);
        let y2 = _mm256_i32gather_ps(ys.as_ptr(), v2s[triangles_offset + i], 4);

        // cull backwards facing triangles
        let backface = _mm256_cmp_ps(_mm256_sub_ps(
            _mm256_mul_ps(_mm256_sub_ps(x1, x0), _mm256_sub_ps(y0, y2)),
            _mm256_mul_ps(_mm256_sub_ps(y0, y1), _mm256_sub_ps(x2, x0))), _mm256_setzero_ps(), _CMP_LE_OQ);
        let backface_xmax = _mm256_and_ps(backface, _mm256_set1_ps(-1.0));

        xmins_out[i] = _mm256_andnot_ps(backface, _mm256_floor_ps(_mm256_max_ps(_mm256_min_ps(_mm256_min_ps(x0, x1), x2), _mm256_set1_ps(xmin))));
        ymins_out[i] = _mm256_andnot_ps(backface, _mm256_floor_ps(_mm256_max_ps(_mm256_min_ps(_mm256_min_ps(y0, y1), y2), _mm256_set1_ps(ymin))));
        xmaxs_out[i] = _mm256_or_ps(backface_xmax, _mm256_andnot_ps(backface, _mm256_ceil_ps(_mm256_min_ps(_mm256_max_ps(_mm256_max_ps(x0, x1), x2), _mm256_set1_ps(xmax)))));
        ymaxs_out[i] = _mm256_andnot_ps(backface, _mm256_ceil_ps(_mm256_min_ps(_mm256_max_ps(_mm256_max_ps(y0, y1), y2), _mm256_set1_ps(ymax))));
    }
}

// my machine stops showing improvement above four threads
static NUM_BOUNDS_THREADS: u32 = 4;
static BOUNDS_WORKERS: Lazy<Mutex<Pool>> = Lazy::new(|| Mutex::new(Pool::new(NUM_BOUNDS_THREADS)));

pub fn avx2_calculate_all_bounds(
        xmins_out: &mut SimdVec<f32>, ymins_out: &mut SimdVec<f32>, xmaxs_out: &mut SimdVec<f32>, ymaxs_out: &mut SimdVec<f32>,
        model: &Model, xs: &SimdVec<f32>, ys: &SimdVec<f32>, 
        xmin: f32, ymin: f32, width: f32, height: f32) {
    let num_chunks = NUM_BOUNDS_THREADS;
    // maintain 128 byte alignment for caching
    let chunk_size = ((model.num_triangles / num_chunks) / 32) * 4;

    let v0s = model.trianglev0s.as_m256i();
    let v1s = model.trianglev1s.as_m256i();
    let v2s = model.trianglev2s.as_m256i();

    let mut pool = BOUNDS_WORKERS.lock().unwrap();
    let mut chunk_start = 0;
    pool.scoped(|scope| {
        let xmins_out_chunks = xmins_out.as_m256_mut().chunks_exact_mut(chunk_size as usize);
        let ymins_out_chunks = ymins_out.as_m256_mut().chunks_exact_mut(chunk_size as usize);
        let xmaxs_out_chunks = xmaxs_out.as_m256_mut().chunks_exact_mut(chunk_size as usize);
        let ymaxs_out_chunks = ymaxs_out.as_m256_mut().chunks_exact_mut(chunk_size as usize);

        for (xmins_out_chunk, (ymins_out_chunk, (xmaxs_out_chunk, ymaxs_out_chunk))) in xmins_out_chunks.zip(ymins_out_chunks.zip(xmaxs_out_chunks.zip(ymaxs_out_chunks))) {
            let triangles_offset = chunk_start;
            scope.execute(move || unsafe {
                avx2_calculate_bounds_chunk(
                    xmins_out_chunk, ymins_out_chunk, xmaxs_out_chunk, ymaxs_out_chunk,
                    v0s, v1s, v2s,
                    xs, ys,
                    xmin, ymin, width, height,
                    triangles_offset, chunk_size as usize);
            });

            chunk_start += chunk_size as usize;
        }
    });

    // do any leftovers sequentially
    for i in (chunk_start * 8 as usize)..(model.num_triangles as usize) {
        let bounds = calculate_bounds(xmin, ymin, width, height, xs[model.trianglev0s[i] as usize], ys[model.trianglev0s[i] as usize], xs[model.trianglev1s[i] as usize], ys[model.trianglev1s[i] as usize], xs[model.trianglev2s[i] as usize], ys[model.trianglev2s[i] as usize]);
        xmins_out[i] = bounds.xmin;
        ymins_out[i] = bounds.ymin;
        xmaxs_out[i] = bounds.xmax;
        ymaxs_out[i] = bounds.ymax;
    }
}

#[target_feature(enable = "avx,avx2")]
pub unsafe fn avx2_fill_triangle(buffer: *mut RGBQUAD, stride: usize, xmin: f32, ymin: f32, xmax: f32, ymax: f32, x0: f32, y0: f32, x1: f32, y1: f32, x2: f32, y2: f32, colour: RGBQUAD) {
    debug_assert!(buffer.align_offset(32) == 0);
    debug_assert!(stride % 8 == 0);

    // what edges are top or left?
    let tl0 = is_top_or_left(x0, y0, x1, y1);
    let tl1 = is_top_or_left(x1, y1, x2, y2);
    let tl2 = is_top_or_left(x2, y2, x0, y0);

    // draw 8 pixels at once
    let xmin = (xmin / 8.0).floor();
    let xmax = (xmax / 8.0).ceil();

    // values of the edge functions for the first pixel on the first row of the bounding box
    let mut row_w0 = _mm256_set1_ps(edge_function(x0, y0, x1, y1, xmin * 8.0 + 0.5, ymin + 0.5));
    let mut row_w1 = _mm256_set1_ps(edge_function(x1, y1, x2, y2, xmin * 8.0 + 0.5, ymin + 0.5));
    let mut row_w2 = _mm256_set1_ps(edge_function(x2, y2, x0, y0, xmin * 8.0 + 0.5, ymin + 0.5));

    // if you substitute `xp + 1` for `xp` into the edge function you can see that
    // for a given edge, the value of the function for `xp + 1, yp` is the value for `xp, yp` minus `y0-y1`
    let mut xstep0 = _mm256_set1_ps(y0-y1);
    let mut xstep1 = _mm256_set1_ps(y1-y2);
    let mut xstep2 = _mm256_set1_ps(y2-y0);

    // adjust to the values for the first eight pixels on the first row
    let zero_to_seven = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    row_w0 = _mm256_sub_ps(row_w0, _mm256_mul_ps(xstep0, zero_to_seven));
    row_w1 = _mm256_sub_ps(row_w1, _mm256_mul_ps(xstep1, zero_to_seven));
    row_w2 = _mm256_sub_ps(row_w2, _mm256_mul_ps(xstep2, zero_to_seven));

    // step to the next span of 8
    let eight = _mm256_set1_ps(8.0);
    xstep0 = _mm256_mul_ps(xstep0, eight);
    xstep1 = _mm256_mul_ps(xstep1, eight);
    xstep2 = _mm256_mul_ps(xstep2, eight);

    // as above, the value of the edge function for `xp, yp + 1` is the value for `xp,yp` minus `x1-x0`. 
    let ystep0 = _mm256_set1_ps(x1-x0);
    let ystep1 = _mm256_set1_ps(x2-x1);
    let ystep2 = _mm256_set1_ps(x0-x2);

    let zero = _mm256_setzero_ps();
    let filled_span = _mm256_set1_epi32(*(((&colour) as *const RGBQUAD) as *const i32));
    let buffer = buffer as *mut __m256i;
    let stride = stride as isize / 8;
    let xmin = xmin as isize;
    let xmax = xmax as isize;

    let mut yp = ymin as isize;
    while yp <= ymax as isize {
        let mut w0 = row_w0;
        let mut w1 = row_w1;
        let mut w2 = row_w2;
        let mut xp = xmin;
        while xp < xmax { // not <= because of (xmax / 8.0).ceil() above
            let inside0 = if tl0 {
                _mm256_castps_si256(_mm256_cmp_ps(w0, zero, _CMP_GE_OQ))
            }
            else {
                _mm256_castps_si256(_mm256_cmp_ps(w0, zero, _CMP_GT_OQ))
            };
            let inside1 = if tl1 {
                _mm256_castps_si256(_mm256_cmp_ps(w1, zero, _CMP_GE_OQ))
            }
            else {
                _mm256_castps_si256(_mm256_cmp_ps(w1, zero, _CMP_GT_OQ))
            };
            let inside2 = if tl2 {
                _mm256_castps_si256(_mm256_cmp_ps(w2, zero, _CMP_GE_OQ))
            }
            else {
                _mm256_castps_si256(_mm256_cmp_ps(w2, zero, _CMP_GT_OQ))
            };
            let inside = _mm256_and_si256(inside0, _mm256_and_si256(inside1, inside2));

            let filled_pixels = _mm256_and_si256(inside, filled_span);
            let bg_pixels = _mm256_andnot_si256(inside, *buffer.offset((yp * stride) + xp));
            *buffer.offset((yp * stride) + xp) = _mm256_or_si256(bg_pixels, filled_pixels);

            xp += 1;

            w0 = _mm256_sub_ps(w0, xstep0);
            w1 = _mm256_sub_ps(w1, xstep1);
            w2 = _mm256_sub_ps(w2, xstep2);
        }
        
        yp += 1;

        row_w0 = _mm256_sub_ps(row_w0, ystep0);
        row_w1 = _mm256_sub_ps(row_w1, ystep1);
        row_w2 = _mm256_sub_ps(row_w2, ystep2);
    }
}