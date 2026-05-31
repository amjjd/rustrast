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
    pub ymax: f32,
    pub iarea: f32,
}

pub struct Buffer<'a, T> {
    pub buffer: &'a mut[T],
    pub left: usize,
    pub top: usize,
    pub stride: usize
}

impl <T> Buffer<'_, T> where T : Copy {
    fn get(&self, x: usize, y: usize) -> T {
        self.buffer[((y - self.top) * self.stride) + x - self.left]
    }

    fn set(&mut self, x: usize, y: usize, value: T) {
        self.buffer[((y - self.top) * self.stride) + x - self.left] = value
    }
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

fn calculate_bounds(xmin: f32, ymin: f32, xmax: f32, ymax: f32, x0: f32, y0: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> Bounds {
   Bounds {
        xmin: min3(x0, x1, x2).max(xmin as f32).floor(),
        ymin: min3(y0, y1, y2).max(ymin as f32).floor(),
        xmax: max3(x0, x1, x2).min(xmax as f32).ceil(),
        ymax: max3(y0, y1, y2).min(ymax as f32).ceil(),
        iarea: 1.0 / edge_function(x0, y0, x1, y1, x2, y2),
    }
}

fn is_top_or_left(x0: f32, y0: f32, x1: f32, y1: f32) -> bool {
    // top                   left (assuming counterclockwise, inverted y axis)
    (y0 == y1 && x0 > x1) || (y1 < y0)
}

#[allow(dead_code)]
fn simple_fill_triangle(
        colour: &mut Buffer<RGBQUAD>, depth: &mut Buffer<f32>,
        xmin: f32, ymin: f32, xmax: f32, ymax: f32,
        x0: f32, y0: f32, z0: f32, iw0: f32,
        x1: f32, y1: f32, z1: f32, iw1: f32,
        x2: f32, y2: f32, z2: f32, iw2: f32,
        iarea: f32,
        fill_colour: RGBQUAD) {
    // cull backwards-facing triangles
    if iarea <= 0.0 {
        return;
    }

    // what edges are top or left?
    let tl0 = is_top_or_left(x1, y1, x2, y2);
    let tl1 = is_top_or_left(x2, y2, x0, y0);
    let tl2 = is_top_or_left(x0, y0, x1, y1);

    // barycentric coordinates of the first pixel on the first row of the bounding box
    let mut row_w0 = edge_function(x1, y1, x2, y2, xmin + 0.5, ymin + 0.5) * iarea;
    let mut row_w1 = edge_function(x2, y2, x0, y0, xmin + 0.5, ymin + 0.5) * iarea;
    let mut row_w2 = edge_function(x0, y0, x1, y1, xmin + 0.5, ymin + 0.5) * iarea;

    let mut yp = ymin as usize;
    while yp < ymax as usize {
        let mut w0 = row_w0;
        let mut w1 = row_w1;
        let mut w2 = row_w2;
        let mut xp = xmin as usize;
        while xp < xmax as usize {
            if ((tl0 && w0 >= 0.0) || w0 > 0.0) && ((tl1 && w1 >= 0.0) || w1 > 0.0) && ((tl2 && w2 >= 0.0) || w2 > 0.0) {
                // adjust for perspective correct interpolation
                let mut p_w0 = w0 * iw0;
                let mut p_w1 = w1 * iw1;
                let mut p_w2 = w2 * iw2;

                let t = 1.0 / (p_w0 + p_w1 + p_w2);
                p_w0 *= t;
                p_w1 *= t;
                p_w2 *= t;

                let z = z0 * p_w0 + z1 * p_w1 + z2 * p_w2;

                // this near test isn't really enough, we really need to clip geometry against the near plane
                if z >= 0.0 && z < depth.get(xp, yp) {
                    colour.set(xp, yp, fill_colour);
                    depth.set(xp, yp, z);
                }
            }

            xp += 1; 

            // if you substitute `xp + 1` for `xp` into the edge function you can see that
            // for a given edge, the value of the function for `xp + 1, yp` is the value for `xp, yp` minus `y0-y1`
            w0 -= (y1-y2) * iarea;
            w1 -= (y2-y0) * iarea;
            w2 -= (y0-y1) * iarea;
        }
        
        yp += 1;

        // as above, the value for `xp, yp + 1` is the value for `yp` minus `x1-x0`. 
        row_w0 -= (x2-x1) * iarea;
        row_w1 -= (x0-x2) * iarea;
        row_w2 -= (x1-x0) * iarea;
    }
}

// not-suitable-for-production SIMD implementations; these will only work on processors that support AVX2

#[target_feature(enable = "avx,avx2,fma")]
unsafe fn avx2_calculate_bounds_chunk(
        xmins_out: &mut [__m256], ymins_out: &mut [__m256], xmaxs_out: &mut [__m256], ymaxs_out: &mut [__m256], iareas_out: &mut [__m256],
        v0s: &[__m256i], v1s: &[__m256i], v2s: &[__m256i],
        xs: &SimdVec<f32>, ys: &SimdVec<f32>,
        xmin: f32, ymin: f32, xmax: f32, ymax: f32,
        triangles_offset: usize, chunk_size: usize) {
    let xs_ptr = xs.as_ptr();
    let ys_ptr = ys.as_ptr();
    let t_xmin = _mm256_set1_ps(xmin);
    let t_ymin = _mm256_set1_ps(ymin);
    let t_xmax = _mm256_set1_ps(xmax);
    let t_ymax = _mm256_set1_ps(ymax);

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

        // interleave for pipelining
        // area calc is (x1-x0)*(y0-y2) - (y0-y1)*(x2-x0)

        let area2 = _mm256_sub_ps(y0, y1);
        let area3 = _mm256_sub_ps(x2, x0);
        
        let mut xmin = _mm256_min_ps(x0, x1);
        let mut ymin = _mm256_min_ps(y0, y1);
        let mut xmax = _mm256_max_ps(x0, x1);
        let mut ymax = _mm256_max_ps(y0, y1);

        let area23 = _mm256_mul_ps(area2, area3);

        xmin = _mm256_min_ps(xmin, x2);
        ymin = _mm256_min_ps(ymin, y2);
        xmax = _mm256_max_ps(xmax, x2);
        ymax = _mm256_max_ps(ymax, y2);

        let area0 = _mm256_sub_ps(x1, x0);
        let area1 = _mm256_sub_ps(y0, y2);

        xmin = _mm256_max_ps(xmin, t_xmin);
        ymin = _mm256_max_ps(ymin, t_ymin);
        xmax = _mm256_min_ps(xmax, t_xmax);
        ymax = _mm256_min_ps(ymax, t_ymax);

        let area = _mm256_fmsub_ps(area0, area1, area23);

        xmins_out[i] = _mm256_floor_ps(xmin);
        ymins_out[i] = _mm256_floor_ps(ymin);
        xmaxs_out[i] = _mm256_ceil_ps(xmax);
        ymaxs_out[i] = _mm256_ceil_ps(ymax);
        iareas_out[i] = _mm256_rcp_ps(area);
    }
}

// my machine stops showing improvement above four threads
static NUM_BOUNDS_THREADS: u32 = 4;
static BOUNDS_WORKERS: Lazy<Mutex<Pool>> = Lazy::new(|| Mutex::new(Pool::new(NUM_BOUNDS_THREADS)));

pub fn calculate_all_bounds(
        xmins_out: &mut SimdVec<f32>, ymins_out: &mut SimdVec<f32>, xmaxs_out: &mut SimdVec<f32>, ymaxs_out: &mut SimdVec<f32>, iareas_out: &mut SimdVec<f32>,
        model: &Model, xs: &SimdVec<f32>, ys: &SimdVec<f32>, 
        xmin: f32, ymin: f32, width: f32, height: f32) {
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

            for (xmins_out_chunk, (ymins_out_chunk, (xmaxs_out_chunk, (ymaxs_out_chunk, iareas_chunk)))) in xmins_out_chunks.zip(ymins_out_chunks.zip(xmaxs_out_chunks.zip(ymaxs_out_chunks.zip(iareas_out_chunks)))) {
                let triangles_offset = chunk_start;
                scope.execute(move || unsafe {
                    avx2_calculate_bounds_chunk(
                        xmins_out_chunk, ymins_out_chunk, xmaxs_out_chunk, ymaxs_out_chunk, iareas_chunk,
                        v0s, v1s, v2s,
                        xs, ys,
                        xmin, ymin, width, height,
                        triangles_offset, chunk_size);
                });

                chunk_start += chunk_size as usize;
            }
        });
    }

    // do any leftovers sequentially
    for i in (chunk_start * 8 as usize)..(model.num_triangles as usize) {
        let bounds = calculate_bounds(xmin, ymin, width, height, xs[model.trianglev0s[i] as usize], ys[model.trianglev0s[i] as usize], xs[model.trianglev1s[i] as usize], ys[model.trianglev1s[i] as usize], xs[model.trianglev2s[i] as usize], ys[model.trianglev2s[i] as usize]);
        xmins_out[i] = bounds.xmin;
        ymins_out[i] = bounds.ymin;
        xmaxs_out[i] = bounds.xmax;
        ymaxs_out[i] = bounds.ymax;
        iareas_out[i] = bounds.iarea;
    }
}

#[allow(dead_code)]
#[target_feature(enable = "avx,avx2,fma")]
pub unsafe fn avx2_fill_triangle(
        colour: &mut Buffer<RGBQUAD>, depth: &mut Buffer<f32>,
        xmin: f32, ymin: f32, xmax: f32, ymax: f32,
        x0: f32, y0: f32, z0: f32, iw0: f32,
        x1: f32, y1: f32, z1: f32, iw1: f32,
        x2: f32, y2: f32, z2: f32, iw2: f32,
        iarea: f32,
        fill_colour: RGBQUAD) {
    debug_assert!(colour.buffer.as_ptr().align_offset(32) == 0);
    debug_assert!(colour.stride % 8 == 0);
    debug_assert!(colour.left % 8 == 0);
    debug_assert!(depth.buffer.as_ptr().align_offset(32) == 0);
    debug_assert!(depth.stride % 8 == 0);
    debug_assert!(depth.left % 8 == 0);

    // cull backwards-facing triangles
    if iarea <= 0.0 {
        return;
    }

    // what edges are top or left?
    let tl0 = is_top_or_left(x1, y1, x2, y2);
    let tl1 = is_top_or_left(x2, y2, x0, y0);
    let tl2 = is_top_or_left(x0, y0, x1, y1);

    // draw 8 aligned pixels at once
    let xmin = (xmin / 8.0).floor() * 8.0;
    let xmax = (xmax / 8.0).ceil() * 8.0;

    // barycentric coordinates of the first pixel on the first row of the bounding box
    let iarea = _mm256_set1_ps(iarea);
    let mut row_w0 = _mm256_mul_ps(_mm256_set1_ps(edge_function(x1, y1, x2, y2, xmin + 0.5, ymin + 0.5)), iarea);
    let mut row_w1 = _mm256_mul_ps(_mm256_set1_ps(edge_function(x2, y2, x0, y0, xmin + 0.5, ymin + 0.5)), iarea);
    let mut row_w2 = _mm256_mul_ps(_mm256_set1_ps(edge_function(x0, y0, x1, y1, xmin + 0.5, ymin + 0.5)), iarea);

    // if you substitute `xp + 1` for `xp` into the edge function you can see that
    // for a given edge, the value of the function for `xp + 1, yp` is the value for `xp, yp` minus `y0-y1`
    let mut xstep0 = _mm256_mul_ps(_mm256_set1_ps(y1-y2), iarea);
    let mut xstep1 = _mm256_mul_ps(_mm256_set1_ps(y2-y0), iarea);
    let mut xstep2 = _mm256_mul_ps(_mm256_set1_ps(y0-y1), iarea);

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
    let ystep0 = _mm256_mul_ps(_mm256_set1_ps(x2-x1), iarea);
    let ystep1 = _mm256_mul_ps(_mm256_set1_ps(x0-x2), iarea);
    let ystep2 = _mm256_mul_ps(_mm256_set1_ps(x1-x0), iarea);

    let zero = _mm256_setzero_ps();
    let filled_span = _mm256_set1_epi32(*(((&fill_colour) as *const RGBQUAD) as *const i32));
    let c_buffer = colour.buffer.as_mut_ptr() as *mut i32;
    let d_buffer = depth.buffer.as_mut_ptr() as *mut f32;
    let iw0 = _mm256_set1_ps(iw0);
    let iw1 = _mm256_set1_ps(iw1);
    let iw2 = _mm256_set1_ps(iw2);
    let z0 = _mm256_set1_ps(z0);
    let z1 = _mm256_set1_ps(z1);
    let z2 = _mm256_set1_ps(z2);

    macro_rules! fill_with_tl {
        ($cmp0:expr, $cmp1:expr, $cmp2:expr) => {{
            let mut yp = ymin as isize;
            let mut c_row = c_buffer.offset((((ymin as usize - colour.top) * colour.stride) - colour.left) as isize);
            let mut d_row = d_buffer.offset((((ymin as usize - depth.top) * depth.stride) - depth.left) as isize);
            while yp < ymax as isize {
                let mut w0 = row_w0;
                let mut w1 = row_w1;
                let mut w2 = row_w2;
                let mut xp = xmin as isize;
                while xp < xmax as isize {
                    let inside0 = _mm256_castps_si256(_mm256_cmp_ps(w0, zero, $cmp0));
                    let inside1 = _mm256_castps_si256(_mm256_cmp_ps(w1, zero, $cmp1));
                    let inside2 = _mm256_castps_si256(_mm256_cmp_ps(w2, zero, $cmp2));
                    let inside_mask = _mm256_and_si256(inside0, _mm256_and_si256(inside1, inside2));

                    // avoid interpolation/depth work for spans that are fully outside this triangle
                    // not currently an advantage
                    //if _mm256_testz_si256(inside_mask, inside_mask) == 0 {
                        // adjust for perspective correct interpolation
                        let mut p_w0 = _mm256_mul_ps(w0, iw0);
                        let mut p_w1 = _mm256_mul_ps(w1, iw1);
                        let mut p_w2 = _mm256_mul_ps(w2, iw2);

                        let t = _mm256_rcp_ps(_mm256_add_ps(p_w0, _mm256_add_ps(p_w1, p_w2)));
                        p_w0 = _mm256_mul_ps(p_w0, t);
                        p_w1 = _mm256_mul_ps(p_w1, t);
                        p_w2 = _mm256_mul_ps(p_w2, t);

                        let mut z = _mm256_mul_ps(z0, p_w0);
                        z = _mm256_fmadd_ps(z1, p_w1, z);
                        z = _mm256_fmadd_ps(z2, p_w2, z);

                        // this near test isn't really enough, we really need to clip geometry against the near plane
                        let near_mask = _mm256_castps_si256(_mm256_cmp_ps(z, zero, _CMP_GE_OQ));

                        let existing_z = _mm256_loadu_ps(d_row.offset(xp));
                        let depth_mask = _mm256_castps_si256(_mm256_cmp_ps(z, existing_z, _CMP_LT_OQ));

                        let mask = _mm256_and_si256(_mm256_and_si256(inside_mask, near_mask), depth_mask);

                        _mm256_maskstore_epi32(c_row.offset(xp) as *mut i32, mask, filled_span);
                        _mm256_maskstore_ps(d_row.offset(xp), mask, z);
                    //}

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
        }};
    }

    // run a version of the loop with the correct comparisons for this triangle's combination of top-left edges
    match ((tl0 as u8) << 2) | ((tl1 as u8) << 1) | (tl2 as u8) {
        0b000 => fill_with_tl!(_CMP_GT_OQ, _CMP_GT_OQ, _CMP_GT_OQ), // impossible?
        0b001 => fill_with_tl!(_CMP_GT_OQ, _CMP_GT_OQ, _CMP_GE_OQ),
        0b010 => fill_with_tl!(_CMP_GT_OQ, _CMP_GE_OQ, _CMP_GT_OQ),
        0b011 => fill_with_tl!(_CMP_GT_OQ, _CMP_GE_OQ, _CMP_GE_OQ),
        0b100 => fill_with_tl!(_CMP_GE_OQ, _CMP_GT_OQ, _CMP_GT_OQ),
        0b101 => fill_with_tl!(_CMP_GE_OQ, _CMP_GT_OQ, _CMP_GE_OQ),
        0b110 => fill_with_tl!(_CMP_GE_OQ, _CMP_GE_OQ, _CMP_GT_OQ),
        0b111 => fill_with_tl!(_CMP_GE_OQ, _CMP_GE_OQ, _CMP_GE_OQ), // impossible?
        _ => unreachable!(),
    }
}

pub fn fill_triangle(
        colour: &mut Buffer<RGBQUAD>, depth: &mut Buffer<f32>,
        xmin: f32, ymin: f32, xmax: f32, ymax: f32,
        x0: f32, y0: f32, z0: f32, iw0: f32,
        x1: f32, y1: f32, z1: f32, iw1: f32,
        x2: f32, y2: f32, z2: f32, iw2: f32,
        iarea: f32,
        fill_colour: RGBQUAD) {
    unsafe {
        avx2_fill_triangle(colour, depth, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, iarea, fill_colour);
    }
}