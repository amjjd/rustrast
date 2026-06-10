use core::arch::x86_64::*;
use std::sync::*;
use once_cell::sync::Lazy;
use scoped_threadpool::Pool;

use crate::TILE_HEIGHT;
use crate::TILE_WIDTH;

use super::simd_vec::*;
use super::obj::*;
use super::rasterisation::*;

fn min3(a: f32, b: f32, c: f32) -> f32 {
    a.min(b).min(c)
}

fn max3(a: f32, b: f32, c: f32) -> f32 {
    a.max(b).max(c)
}

fn is_top_or_left(x0: f32, y0: f32, x1: f32, y1: f32) -> u8 {
    // top                   left (assuming counterclockwise, inverted y axis)
    if (y0 == y1 && x0 > x1) || (y1 < y0) { u8::MAX } else { 0 }
}

macro_rules! is_top_or_left {
    ($x0:expr, $y0:expr, $x1:expr, $y1:expr) => {{
        // (y0 == y1 && x0 > x1) || (y1 < y0)
        let y_equal = _mm256_cmp_ps($y0, $y1, _CMP_EQ_OQ);
        let x0_greater = _mm256_cmp_ps($x0, $x1, _CMP_GT_OQ);
        let y1_less = _mm256_cmp_ps($y1, $y0, _CMP_LT_OQ);

        _mm256_castps_si256(_mm256_or_ps(_mm256_and_ps(y_equal, x0_greater), y1_less))
    }};
}

fn add_to_bins(tile_triangles: &mut Vec<Vec<u32>>, left: usize, top: usize, right: usize, bottom: usize, it: u32, num_tiles_x: usize) {
    let mut row_start = top * num_tiles_x;
    for _ in top..=bottom {
        let l = row_start + left;
        let r = row_start + right;
        for t in l..=r {
            tile_triangles[t].push(it);
        }

        row_start += num_tiles_x;
    }
}

unsafe fn bin_triangles_chunk(
        xmins_out: &mut [__m256], ymins_out: &mut [__m256], xmaxs_out: &mut [__m256], ymaxs_out: &mut [__m256],
        iareas_out: &mut [__m256], tl0s_out: &mut [u8], tl1s_out: &mut [u8], tl2s_out: &mut [u8],
        tile_triangles_out: &mut Vec<Vec<u32>>,
        v0s: &[__m256i], v1s: &[__m256i], v2s: &[__m256i],
        xs: &SimdVec<f32>, ys: &SimdVec<f32>,
        vs_offset: usize, chunk_size: usize,
        num_tiles_x: usize, num_tiles_y: usize) {
    let xs_ptr = xs.as_ptr();
    let ys_ptr = ys.as_ptr();

    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let itile_width = _mm256_div_ps(one, _mm256_set1_ps(super::TILE_WIDTH as f32));
    let itile_height = _mm256_div_ps(one, _mm256_set1_ps(super::TILE_HEIGHT as f32));
    let zero_i32 = _mm256_set1_epi32(0);
    let max_tile_x = _mm256_set1_epi32(num_tiles_x as i32 - 1);
    let max_tile_y = _mm256_set1_epi32(num_tiles_y as i32 - 1);

    for i in 0..chunk_size {
        let idx0 = v0s[vs_offset + i];
        let idx1 = v1s[vs_offset + i];
        let idx2 = v2s[vs_offset + i];

        let x0 = _mm256_i32gather_ps(xs_ptr, idx0, 4);
        let y0 = _mm256_i32gather_ps(ys_ptr, idx0, 4);
        let x1 = _mm256_i32gather_ps(xs_ptr, idx1, 4);
        let y1 = _mm256_i32gather_ps(ys_ptr, idx1, 4);
        let x2 = _mm256_i32gather_ps(xs_ptr, idx2, 4);
        let y2 = _mm256_i32gather_ps(ys_ptr, idx2, 4);

        let xmin = _mm256_floor_ps(_mm256_min_ps(_mm256_min_ps(x0, x1), x2));
        let ymin = _mm256_floor_ps(_mm256_min_ps(_mm256_min_ps(y0, y1), y2));
        let xmax = _mm256_ceil_ps(_mm256_max_ps(_mm256_max_ps(x0, x1), x2));
        let ymax = _mm256_ceil_ps(_mm256_max_ps(_mm256_max_ps(y0, y1), y2));
        xmins_out[i] = xmin;
        ymins_out[i] = ymin;
        xmaxs_out[i] = xmax;
        ymaxs_out[i] = ymax;

        let area = edge_function!(x0, y0, x1, y1, x2, y2);
        iareas_out[i] = _mm256_rcp_ps(area);
        let cull = _mm256_castps_si256(_mm256_cmp_ps(area, zero, _CMP_LE_OQ));

        let tl0 = is_top_or_left!(x1, y1, x2, y2);
        let tl1 = is_top_or_left!(x2, y2, x0, y0);
        let tl2 = is_top_or_left!(x0, y0, x1, y1);

        let left = _mm256_max_epi32(_mm256_cvttps_epi32(_mm256_mul_ps(xmin, itile_width)), zero_i32);
        let top = _mm256_max_epi32(_mm256_cvttps_epi32(_mm256_mul_ps(ymin, itile_height)), zero_i32);
        let right = _mm256_min_epi32(_mm256_cvttps_epi32(_mm256_mul_ps(xmax, itile_width)), max_tile_x);
        let bottom = _mm256_min_epi32(_mm256_cvttps_epi32(_mm256_mul_ps(ymax, itile_height)), max_tile_y);

        macro_rules! per_triangle {
            ($j:expr) => {
                let ot = i * 8 + $j;

                tl0s_out[ot] = _mm256_extract_epi32(tl0, $j as i32) as u8;
                tl1s_out[ot] = _mm256_extract_epi32(tl1, $j as i32) as u8;
                tl2s_out[ot] = _mm256_extract_epi32(tl2, $j as i32) as u8;

                if (_mm256_extract_epi32(cull, $j as i32) == 0) {
                    let left = _mm256_extract_epi32(left, $j as i32) as usize;
                    let top = _mm256_extract_epi32(top, $j as i32) as usize;
                    let right = _mm256_extract_epi32(right, $j as i32) as usize;
                    let bottom = _mm256_extract_epi32(bottom, $j as i32) as usize;
                    add_to_bins(tile_triangles_out, left, top, right, bottom, (vs_offset * 8 + ot) as u32, num_tiles_x);
                }
            };
        }

        per_triangle!(0);
        per_triangle!(1);
        per_triangle!(2);
        per_triangle!(3);
        per_triangle!(4);
        per_triangle!(5);
        per_triangle!(6);
        per_triangle!(7);
    }
}

fn bin_triangle(
        xmins_out: &mut SimdVec<f32>, ymins_out: &mut SimdVec<f32>, xmaxs_out: &mut SimdVec<f32>, ymaxs_out: &mut SimdVec<f32>,
        iareas_out: &mut SimdVec<f32>, tl0s_out: &mut Vec<u8>, tl1s_out: &mut Vec<u8>, tl2s_out: &mut Vec<u8>,
        tile_triangles_out: &mut Vec<Vec<u32>>,
        it: usize, x0: f32, y0: f32, x1: f32, y1: f32, x2: f32, y2: f32,
        num_tiles_x: usize, num_tiles_y: usize) {
    let xmin =  min3(x0, x1, x2).floor();
    let ymin = min3(y0, y1, y2).floor();
    let xmax = max3(x0, x1, x2).ceil();
    let ymax = max3(y0, y1, y2).ceil();
    xmins_out[it] = xmin;
    ymins_out[it] = ymin;
    xmaxs_out[it] = xmax;
    ymaxs_out[it] = ymax;

    let area = edge_function(x0, y0, x1, y1, x2, y2);
    iareas_out[it] = 1.0 / area;
    
    tl0s_out[it] = is_top_or_left(x1, y1, x2, y2);
    tl1s_out[it] = is_top_or_left(x2, y2, x0, y0);
    tl2s_out[it] = is_top_or_left(x0, y0, x1, y1);

    if area > 0.0 {
        let left = ((xmin / TILE_WIDTH as f32) as usize).max(0);
        let top = ((ymin / TILE_HEIGHT as f32) as usize).max(0);
        let right = ((xmax / TILE_WIDTH as f32) as usize).min(num_tiles_x - 1);
        let bottom = ((ymax / TILE_HEIGHT as f32) as usize).min(num_tiles_y - 1);
        add_to_bins(tile_triangles_out, left, top, right, bottom, it as u32, num_tiles_x);
    }
}

// my machine stops showing improvement above four threads
pub const NUM_BIN_THREADS: usize = 4;
static BIN_WORKERS: Lazy<Mutex<Pool>> = Lazy::new(|| Mutex::new(Pool::new(NUM_BIN_THREADS as u32)));

pub fn bin_triangles(
        xmins_out: &mut SimdVec<f32>, ymins_out: &mut SimdVec<f32>, xmaxs_out: &mut SimdVec<f32>, ymaxs_out: &mut SimdVec<f32>,
        iareas_out: &mut SimdVec<f32>, tl0s_out: &mut Vec<u8>, tl1s_out: &mut Vec<u8>, tl2s_out: &mut Vec<u8>,
        tile_triangles_out: &mut [Vec<Vec<u32>>; NUM_BIN_THREADS],
        model: &Model, xs: &SimdVec<f32>, ys: &SimdVec<f32>, stride: usize, height: usize) {

    // this should only allocate heavily during the first few frames
    let num_tiles_x = (stride + super::TILE_WIDTH - 1) / super::TILE_WIDTH;
    let num_tiles_y = (height + super::TILE_HEIGHT - 1) / super::TILE_HEIGHT;
    let num_tiles = num_tiles_x * num_tiles_y;
    for i in 0..NUM_BIN_THREADS {
        if tile_triangles_out[i].len() > num_tiles {
            tile_triangles_out[i].truncate(num_tiles);
        }
        else {
            // somewhat pessimistic guess
            let initial_capacity = (model.num_triangles as usize / num_tiles) * 4;
            for _ in tile_triangles_out[i].len()..num_tiles {
                tile_triangles_out[i].push(Vec::with_capacity(initial_capacity));
            }
        }

        for j in 0..num_tiles {
            // doesn't affect capacity
            tile_triangles_out[i][j].truncate(0);
        }
    }

    let v0s = model.trianglev0s.as_m256i();
    let v1s = model.trianglev1s.as_m256i();
    let v2s = model.trianglev2s.as_m256i();

    let num_chunks = NUM_BIN_THREADS;
    // maintain 128 byte alignment on xmins, etc., for caching
    let chunk_size = ((model.num_triangles as usize / num_chunks) / 32) * 4;
    let mut chunk_start = 0;

    if chunk_size == 0 {
        // do the SIMD part in one thread
        let chunk_size = model.num_triangles as usize / 8;
        unsafe { 
            bin_triangles_chunk(
                xmins_out.as_m256_mut(), ymins_out.as_m256_mut(), xmaxs_out.as_m256_mut(), ymaxs_out.as_m256_mut(),
                iareas_out.as_m256_mut(), tl0s_out.as_mut_slice(), tl1s_out.as_mut_slice(), tl2s_out.as_mut_slice(),
                &mut tile_triangles_out[0],
                v0s, v1s, v2s,
                xs, ys,
                0, chunk_size,
                num_tiles_x, num_tiles_y);
        }
        chunk_start += chunk_size;
    }
    else {
        let mut pool = BIN_WORKERS.lock().unwrap();
        pool.scoped(|scope| {
            let mut xmins_out_chunks = xmins_out.as_m256_mut().chunks_exact_mut(chunk_size);
            let mut ymins_out_chunks = ymins_out.as_m256_mut().chunks_exact_mut(chunk_size);
            let mut xmaxs_out_chunks = xmaxs_out.as_m256_mut().chunks_exact_mut(chunk_size);
            let mut ymaxs_out_chunks = ymaxs_out.as_m256_mut().chunks_exact_mut(chunk_size);
            let mut iareas_out_chunks = iareas_out.as_m256_mut().chunks_exact_mut(chunk_size);
            let mut tl0s_out_chunks = tl0s_out.chunks_exact_mut(chunk_size * 8);
            let mut tl1s_out_chunks = tl1s_out.chunks_exact_mut(chunk_size * 8);
            let mut tl2s_out_chunks = tl2s_out.chunks_exact_mut(chunk_size * 8);
            let mut tile_triangles_out_chunks = tile_triangles_out.iter_mut();

            for _ in 0..num_chunks {
                let xmins_out_chunk = xmins_out_chunks.next().unwrap();
                let ymins_out_chunk = ymins_out_chunks.next().unwrap();
                let xmaxs_out_chunk = xmaxs_out_chunks.next().unwrap();
                let ymaxs_out_chunk = ymaxs_out_chunks.next().unwrap();
                let iareas_out_chunk = iareas_out_chunks.next().unwrap();
                let tl0s_out_chunk = tl0s_out_chunks.next().unwrap();
                let tl1s_out_chunk = tl1s_out_chunks.next().unwrap();
                let tl2s_out_chunk = tl2s_out_chunks.next().unwrap();
                let tile_triangles_out_chunk = tile_triangles_out_chunks.next().unwrap();
                let vs_offset = chunk_start;
                scope.execute(move || unsafe {
                    bin_triangles_chunk(
                        xmins_out_chunk, ymins_out_chunk, xmaxs_out_chunk, ymaxs_out_chunk,
                        iareas_out_chunk, tl0s_out_chunk, tl1s_out_chunk, tl2s_out_chunk,
                        tile_triangles_out_chunk,
                        v0s, v1s, v2s,
                        xs, ys,
                        vs_offset, chunk_size,
                        num_tiles_x, num_tiles_y);
                });

                chunk_start += chunk_size;
            }
        });
    }

    // do any leftovers sequentially
    for it in (chunk_start * 8 as usize)..(model.num_triangles as usize) {
        bin_triangle(
            xmins_out, ymins_out, xmaxs_out, ymaxs_out, iareas_out, tl0s_out, tl1s_out, tl2s_out, &mut tile_triangles_out[NUM_BIN_THREADS - 1],
            it, xs[model.trianglev0s[it] as usize], ys[model.trianglev0s[it] as usize], xs[model.trianglev1s[it] as usize], ys[model.trianglev1s[it] as usize], xs[model.trianglev2s[it] as usize], ys[model.trianglev2s[it] as usize],
            num_tiles_x, num_tiles_y);
    }
}