use windows::Win32::Graphics::Gdi::*;
use core::arch::x86_64::*;

use super::shaders::*;

// not-suitable-for-production rasteriser, requires AVX2 and FMA and will fault if they aren't present

pub struct Buffer<'a, T> {
    pub buffer: &'a mut[T],
    pub left: usize,
    pub top: usize,
    pub stride: usize
}

pub fn edge_function(x0: f32, y0: f32, x1: f32, y1: f32, xp: f32, yp: f32) -> f32 {
    // this is backwards from a lot of examples due to our projection inverting the y axis
    (x1-x0)*(y0-yp) - (y0-y1)*(xp-x0)
}

macro_rules! edge_function {
    ($x0:expr, $y0:expr, $x1:expr, $y1:expr, $xp:expr, $yp:expr) => {{
        // (x1-x0)*(y0-yp) - (y0-y1)*(xp-x0)
        _mm256_fmsub_ps(
            _mm256_sub_ps($x1, $x0), _mm256_sub_ps($y0, $yp),
            _mm256_mul_ps(_mm256_sub_ps($y0, $y1), _mm256_sub_ps($xp, $x0)))
    }}
}

macro_rules! interpolate {
    // s0*w0 + s1*w1 + s2*w2
    ($s0:expr, $s1:expr, $s2:expr, $w0:expr, $w1:expr, $w2:expr) => {
        _mm256_fmadd_ps($s2, $w2, _mm256_fmadd_ps($s1, $w1, _mm256_mul_ps($s0, $w0)))
    }
}

fn fill_triangle_generic<const C0: i32, const C1: i32, const C2: i32, const EXECUTE_FRAGMENT_SHADER: bool, T : Copy>(
        colour: &mut Buffer<RGBQUAD>, depth: &mut Buffer<f32>,
        it: usize,
        xmin: f32, ymin: f32, xmax: f32, ymax: f32,
        x0: f32, y0: f32, z0: f32, iw0: f32,
        x1: f32, y1: f32, z1: f32, iw1: f32,
        x2: f32, y2: f32, z2: f32, iw2: f32,
        e0: T, e1: T, e2: T,
        iarea: f32,
        fragment_shader: &impl AvxFragmentShader<EXECUTE_FRAGMENT_SHADER, T>) {
    debug_assert!(colour.buffer.as_ptr().align_offset(32) == 0);
    debug_assert!(colour.stride % 4 == 0);
    debug_assert!(colour.left % 4 == 0);
    debug_assert!(depth.buffer.as_ptr().align_offset(32) == 0);
    debug_assert!(depth.stride % 4 == 0);
    debug_assert!(depth.left % 4 == 0);

    // draw 4x2 aligned pixels at once
    let xmin = (xmin / 4.0).floor() * 4.0;
    let xmax = (xmax / 4.0).ceil() * 4.0;

    unsafe {
        // barycentric coordinates of the first 4x2 pixels block on the first rows of the bounding box
        let x0_v = _mm256_set1_ps(x0);
        let y0_v = _mm256_set1_ps(y0);
        let x1_v = _mm256_set1_ps(x1);
        let y1_v = _mm256_set1_ps(y1);
        let x2_v = _mm256_set1_ps(x2);
        let y2_v = _mm256_set1_ps(y2);
        let xp = _mm256_add_ps(_mm256_set1_ps(xmin), _mm256_setr_ps(0.5, 1.5, 2.5, 3.5, 0.5, 1.5, 2.5, 3.5));
        let yp = _mm256_add_ps(_mm256_set1_ps(ymin), _mm256_setr_ps(0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5));
        let iarea = _mm256_set1_ps(iarea);
        let mut row_w0 = _mm256_mul_ps(edge_function!(x1_v, y1_v, x2_v, y2_v, xp, yp), iarea);
        let mut row_w1 = _mm256_mul_ps(edge_function!(x2_v, y2_v, x0_v, y0_v, xp, yp), iarea);
        let mut row_w2 = _mm256_mul_ps(edge_function!(x0_v, y0_v, x1_v, y1_v, xp, yp), iarea);

        // if you substitute `xp + 1` for `xp` into the edge function you can see that
        // for a given edge, the value of the function for `xp + 1, yp` is the value for `xp, yp` minus `y0-y1`
        let iarea_times_4 = _mm256_mul_ps(iarea, _mm256_set1_ps(4.0));
        let xstep0 = _mm256_mul_ps(_mm256_set1_ps(y1-y2), iarea_times_4);
        let xstep1 = _mm256_mul_ps(_mm256_set1_ps(y2-y0), iarea_times_4);
        let xstep2 = _mm256_mul_ps(_mm256_set1_ps(y0-y1), iarea_times_4);

        // as above, the value of the edge function for `xp, yp + 1` is the value for `xp,yp` minus `x1-x0`.
        let iarea_times_2 = _mm256_mul_ps(iarea, _mm256_set1_ps(2.0));
        let ystep0 = _mm256_mul_ps(_mm256_set1_ps(x2-x1), iarea_times_2);
        let ystep1 = _mm256_mul_ps(_mm256_set1_ps(x0-x2), iarea_times_2);
        let ystep2 = _mm256_mul_ps(_mm256_set1_ps(x1-x0), iarea_times_2);

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

        let mut yp = ymin as usize;
        let mut c_row0 = c_buffer.add(((ymin as usize - colour.top) * colour.stride) - colour.left);
        let mut c_row1 = c_row0.add(colour.stride);
        let mut d_row0 = d_buffer.add(((ymin as usize - depth.top) * depth.stride) - depth.left);
        let mut d_row1 = d_row0.add(depth.stride);
        let xmin = xmin as usize;
        let xmax = xmax as usize;
        while yp < ymax as usize {
            let mut w0 = row_w0;
            let mut w1 = row_w1;
            let mut w2 = row_w2;
            let mut xp = xmin;
            while xp < xmax {
                let inside0 = _mm256_castps_si256(_mm256_cmp_ps::<C0>(w0, zero));
                let inside1 = _mm256_castps_si256(_mm256_cmp_ps::<C1>(w1, zero));
                let inside2 = _mm256_castps_si256(_mm256_cmp_ps::<C2>(w2, zero));
                let inside_mask = _mm256_and_si256(inside0, _mm256_and_si256(inside1, inside2));

                // skip blocks that are fully outside the triangle
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

                    let existing_z = _mm256_loadu2_m128(d_row1.add(xp), d_row0.add(xp));
                    let depth_mask = _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(z, existing_z, _CMP_GT_OQ)), near_mask);
                    let mask = _mm256_and_si256(inside_mask, depth_mask);

                    if _mm256_movemask_epi8(mask) != 0 {
                        if EXECUTE_FRAGMENT_SHADER {
                            let (filled_span, c_mask) = fragment_shader.fragment(it, w0, w1, w2, p_w0, p_w1, p_w2, e0, e1, e2, mask);
                            _mm_maskstore_epi32(c_row0.add(xp) as *mut i32, _mm256_castsi256_si128(c_mask), _mm256_castsi256_si128(filled_span));
                            _mm_maskstore_epi32(c_row1.add(xp) as *mut i32, _mm256_extracti128_si256(c_mask, 1), _mm256_extracti128_si256(filled_span, 1));
                        }
                        let blended_z = _mm256_blendv_ps(existing_z, z, _mm256_castsi256_ps(mask));
                        _mm256_storeu2_m128(d_row1.add(xp), d_row0.add(xp), blended_z);
                    }
                }

                xp += 4;

                w0 = _mm256_sub_ps(w0, xstep0);
                w1 = _mm256_sub_ps(w1, xstep1);
                w2 = _mm256_sub_ps(w2, xstep2);
            }

            yp += 2;
            c_row0 = c_row0.add(colour.stride * 2);
            c_row1 = c_row1.add(colour.stride * 2);
            d_row0 = d_row0.add(depth.stride * 2);
            d_row1 = d_row1.add(depth.stride * 2);

            row_w0 = _mm256_sub_ps(row_w0, ystep0);
            row_w1 = _mm256_sub_ps(row_w1, ystep1);
            row_w2 = _mm256_sub_ps(row_w2, ystep2);
        }
    }
}

pub fn fill_triangle<const EXECUTE_FRAGMENT_SHADER: bool, T : Copy>(
        colour: &mut Buffer<RGBQUAD>, depth: &mut Buffer<f32>,
        it: usize,
        xmin: f32, ymin: f32, xmax: f32, ymax: f32,
        x0: f32, y0: f32, z0: f32, iw0: f32,
        x1: f32, y1: f32, z1: f32, iw1: f32,
        x2: f32, y2: f32, z2: f32, iw2: f32,
        e0: T, e1: T, e2: T,
        iarea: f32, tl0: bool, tl1: bool, tl2: bool,
        fragment_shader: &impl AvxFragmentShader<EXECUTE_FRAGMENT_SHADER, T>) {

    // NB - no TL or all TL edges are impossible, but are here for completeness
    match (tl0, tl1, tl2) {
        (false, false, false) => fill_triangle_generic::<_CMP_GT_OQ, _CMP_GT_OQ, _CMP_GT_OQ, EXECUTE_FRAGMENT_SHADER, T>(colour, depth, it, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, e0, e1, e2, iarea, fragment_shader),
        (false, false, true) => fill_triangle_generic::<_CMP_GT_OQ, _CMP_GT_OQ, _CMP_GE_OQ, EXECUTE_FRAGMENT_SHADER, T>(colour, depth, it, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, e0, e1, e2, iarea, fragment_shader),
        (false, true, false) => fill_triangle_generic::<_CMP_GT_OQ, _CMP_GE_OQ, _CMP_GT_OQ, EXECUTE_FRAGMENT_SHADER, T>(colour, depth, it, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, e0, e1, e2, iarea, fragment_shader),
        (false, true, true) => fill_triangle_generic::<_CMP_GT_OQ, _CMP_GE_OQ, _CMP_GE_OQ, EXECUTE_FRAGMENT_SHADER, T>(colour, depth, it, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, e0, e1, e2, iarea, fragment_shader),
        (true, false, false) => fill_triangle_generic::<_CMP_GE_OQ, _CMP_GT_OQ, _CMP_GT_OQ, EXECUTE_FRAGMENT_SHADER, T>(colour, depth, it, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, e0, e1, e2, iarea, fragment_shader),
        (true, false, true) => fill_triangle_generic::<_CMP_GE_OQ, _CMP_GT_OQ, _CMP_GE_OQ, EXECUTE_FRAGMENT_SHADER, T>(colour, depth, it, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, e0, e1, e2, iarea, fragment_shader),
        (true, true, false) => fill_triangle_generic::<_CMP_GE_OQ, _CMP_GE_OQ, _CMP_GT_OQ, EXECUTE_FRAGMENT_SHADER, T>(colour, depth, it, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, e0, e1, e2, iarea, fragment_shader),
        (true, true, true) => fill_triangle_generic::<_CMP_GE_OQ, _CMP_GE_OQ, _CMP_GE_OQ, EXECUTE_FRAGMENT_SHADER, T>(colour, depth, it, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, e0, e1, e2, iarea, fragment_shader),
    }
}