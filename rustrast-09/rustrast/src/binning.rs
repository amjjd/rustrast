use core::arch::x86_64::*;
use std::sync::*;
use once_cell::sync::Lazy;
use scoped_threadpool::Pool;

use crate::TILE_HEIGHT;
use crate::TILE_WIDTH;
use crate::LOG_TILE_WIDTH;
use crate::LOG_TILE_HEIGHT;

use super::simd_vec::*;
use super::obj::*;
use super::rasterisation::*;

fn min3(a: f32, b: f32, c: f32) -> f32 {
    a.min(b).min(c)
}

fn max3(a: f32, b: f32, c: f32) -> f32 {
    a.max(b).max(c)
}

fn is_top_or_left(x0: f32, y0: f32, x1: f32, y1: f32) -> u32 {
    // top                   left (assuming counterclockwise, inverted y axis)
    if (y0 == y1 && x0 > x1) || (y1 < y0) { u32::MAX } else { 0 }
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

unsafe fn bin_triangles_chunk<const CULL_MODE: i32>(
        xmins_out: &mut [__m256], ymins_out: &mut [__m256], xmaxs_out: &mut [__m256], ymaxs_out: &mut [__m256],
        iareas_out: &mut [__m256], tl0s_out: &mut [__m256i], tl1s_out: &mut [__m256i], tl2s_out: &mut [__m256i],
        tile_triangles_out: &mut Vec<Vec<u32>>,
        v0s: &[__m256i], v1s: &[__m256i], v2s: &[__m256i],
        xs: &SimdVec<f32>, ys: &SimdVec<f32>,
        vs_offset: usize, chunk_size: usize,
        width: usize, height: usize, num_tiles_x: usize, num_tiles_y: usize) {
    let xs_ptr = xs.as_ptr();
    let ys_ptr = ys.as_ptr();

    let zero = _mm256_setzero_ps();
    let width = _mm256_set1_ps(width as f32);
    let height = _mm256_set1_ps(height as f32);
    let zero_i32 = _mm256_set1_epi32(0);
    let max_tile_x = _mm256_set1_epi32(num_tiles_x as i32 - 1);
    let max_tile_y = _mm256_set1_epi32(num_tiles_y as i32 - 1);

    let mut left = [0i32; 8];
    let mut top= [0i32; 8];
    let mut right = [0i32; 8];
    let mut bottom = [0i32; 8];

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

        // remove triangles that are completely outside the screen bounds
        let cull = _mm256_cmp_ps(xmax, zero, _CMP_LT_OQ);
        let cull = _mm256_or_ps(cull, _mm256_cmp_ps(ymax, zero, _CMP_LT_OQ));
        let cull = _mm256_or_ps(cull, _mm256_cmp_ps(xmin, width, _CMP_GE_OQ));
        let cull = _mm256_or_ps(cull, _mm256_cmp_ps(ymin, height, _CMP_GE_OQ));

        let area = edge_function!(x0, y0, x1, y1, x2, y2);
        iareas_out[i] = _mm256_rcp_ps(area);
        let cull = _mm256_or_ps(cull, _mm256_cmp_ps(area, zero, CULL_MODE));

        tl0s_out[i] = is_top_or_left!(x1, y1, x2, y2);
        tl1s_out[i] = is_top_or_left!(x2, y2, x0, y0);
        tl2s_out[i] = is_top_or_left!(x0, y0, x1, y1);

        let cull = _mm256_movemask_ps(cull);

        _mm256_storeu_epi32(left.as_mut_ptr(), _mm256_max_epi32(_mm256_srai_epi32(_mm256_cvttps_epi32(xmin), LOG_TILE_WIDTH as i32), zero_i32));
        _mm256_storeu_epi32(top.as_mut_ptr(), _mm256_max_epi32(_mm256_srai_epi32(_mm256_cvttps_epi32(ymin), LOG_TILE_HEIGHT as i32), zero_i32));
        _mm256_storeu_epi32(right.as_mut_ptr(), _mm256_min_epi32(_mm256_srai_epi32(_mm256_cvttps_epi32(xmax), LOG_TILE_WIDTH as i32), max_tile_x));
        _mm256_storeu_epi32(bottom.as_mut_ptr(), _mm256_min_epi32(_mm256_srai_epi32(_mm256_cvttps_epi32(ymax), LOG_TILE_HEIGHT as i32), max_tile_y));

        macro_rules! per_triangle {
            ($j:expr) => {
                let ot = i * 8 + $j;

                if (cull & (1 << $j)) == 0 {
                    add_to_bins(tile_triangles_out, left[$j] as usize, top[$j] as usize, right[$j] as usize, bottom[$j] as usize, (vs_offset * 8 + ot) as u32, num_tiles_x);
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

fn bin_triangle<const CULL_MODE: i32>(
        xmins_out: &mut SimdVec<f32>, ymins_out: &mut SimdVec<f32>, xmaxs_out: &mut SimdVec<f32>, ymaxs_out: &mut SimdVec<f32>,
        iareas_out: &mut SimdVec<f32>, tl0s_out: &mut SimdVec<u32>, tl1s_out: &mut SimdVec<u32>, tl2s_out: &mut SimdVec<u32>,
        tile_triangles_out: &mut Vec<Vec<u32>>,
        it: usize, x0: f32, y0: f32, x1: f32, y1: f32, x2: f32, y2: f32,
        width: usize, height: usize, num_tiles_x: usize, num_tiles_y: usize) {
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

    if xmax >= 0.0 && ymax >= 0.0 && xmin < width as f32 && ymin < height as f32  &&
    (CULL_MODE == _CMP_EQ_OQ || (CULL_MODE == _CMP_LE_OQ && area > 0.0) || (CULL_MODE == _CMP_GE_OQ && area < 0.0)) {
        let left = ((xmin as usize) >> LOG_TILE_WIDTH).max(0);
        let top = ((ymin as usize) >> LOG_TILE_HEIGHT).max(0);
        let right = ((xmax as usize) >> LOG_TILE_WIDTH).min(num_tiles_x - 1);
        let bottom = ((ymax as usize) >> LOG_TILE_HEIGHT).min(num_tiles_y - 1);
        add_to_bins(tile_triangles_out, left, top, right, bottom, it as u32, num_tiles_x);
    }
}

// my machine stops showing improvement above four threads
pub const NUM_BIN_THREADS: usize = 4;
static BIN_WORKERS: Lazy<Mutex<Pool>> = Lazy::new(|| Mutex::new(Pool::new(NUM_BIN_THREADS as u32)));

#[derive(Clone, Copy)]
pub struct CullMode<const T: i32> {
    // prevent construction by others
    _private: ()
}
#[allow(dead_code)]
pub const CULL_NONE: CullMode<_CMP_EQ_OQ> = CullMode{_private: ()};
#[allow(dead_code)]
pub const CULL_BACK_FACING: CullMode<_CMP_LE_OQ> = CullMode{_private: ()};
#[allow(dead_code)]
pub const CULL_FRONT_FACING: CullMode<_CMP_GE_OQ> = CullMode{_private: ()};

pub fn bin_triangles<const CULL_MODE: i32>(
        xmins_out: &mut SimdVec<f32>, ymins_out: &mut SimdVec<f32>, xmaxs_out: &mut SimdVec<f32>, ymaxs_out: &mut SimdVec<f32>,
        iareas_out: &mut SimdVec<f32>, tl0s_out: &mut SimdVec<u32>, tl1s_out: &mut SimdVec<u32>, tl2s_out: &mut SimdVec<u32>,
        tile_triangles_out: &mut [Vec<Vec<u32>>; NUM_BIN_THREADS],
        model: &Model, xs: &SimdVec<f32>, ys: &SimdVec<f32>, width: usize, height: usize, _cull_mode: CullMode<CULL_MODE>) {

    // this should only allocate heavily during the first few frames
    let num_tiles_x = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    let num_tiles_y = (height + TILE_HEIGHT - 1) / TILE_HEIGHT;
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
            bin_triangles_chunk::<CULL_MODE>(
                xmins_out.as_m256_mut(), ymins_out.as_m256_mut(), xmaxs_out.as_m256_mut(), ymaxs_out.as_m256_mut(),
                iareas_out.as_m256_mut(), tl0s_out.as_m256i_mut(), tl1s_out.as_m256i_mut(), tl2s_out.as_m256i_mut(),
                &mut tile_triangles_out[0],
                v0s, v1s, v2s,
                xs, ys,
                0, chunk_size,
                width, height, num_tiles_x, num_tiles_y);
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
            let mut tl0s_out_chunks = tl0s_out.as_m256i_mut().chunks_exact_mut(chunk_size);
            let mut tl1s_out_chunks = tl1s_out.as_m256i_mut().chunks_exact_mut(chunk_size);
            let mut tl2s_out_chunks = tl2s_out.as_m256i_mut().chunks_exact_mut(chunk_size);
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
                    bin_triangles_chunk::<CULL_MODE>(
                        xmins_out_chunk, ymins_out_chunk, xmaxs_out_chunk, ymaxs_out_chunk,
                        iareas_out_chunk, tl0s_out_chunk, tl1s_out_chunk, tl2s_out_chunk,
                        tile_triangles_out_chunk,
                        v0s, v1s, v2s,
                        xs, ys,
                        vs_offset, chunk_size,
                        width, height, num_tiles_x, num_tiles_y);
                });

                chunk_start += chunk_size;
            }
        });
    }

    // do any leftovers sequentially
    for it in (chunk_start * 8 as usize)..(model.num_triangles as usize) {
        bin_triangle::<CULL_MODE>(
            xmins_out, ymins_out, xmaxs_out, ymaxs_out, iareas_out, tl0s_out, tl1s_out, tl2s_out, &mut tile_triangles_out[NUM_BIN_THREADS - 1],
            it, xs[model.trianglev0s[it] as usize], ys[model.trianglev0s[it] as usize], xs[model.trianglev1s[it] as usize], ys[model.trianglev1s[it] as usize], xs[model.trianglev2s[it] as usize], ys[model.trianglev2s[it] as usize],
            width, height, num_tiles_x, num_tiles_y);
    }
}