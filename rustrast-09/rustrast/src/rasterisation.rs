use std::{array, marker::PhantomData};
use windows::Win32::Graphics::Gdi::*;
use core::arch::x86_64::*;
use core::panic;
use once_cell::sync::Lazy;

use super::shaders::*;
use super::BufferSize;

// not-suitable-for-production rasteriser, requires AVX2 and FMA and will fault if they aren't present

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

// NB - Into<i32> is not actually used; all colours must be 4 bytes
pub trait FragmentColour: From<RGBQUAD> + Into<i32> + Clone + Copy + Default + Send + Sync {
}

#[derive(Clone, Copy, Default)]
pub struct Rgb10a2 {
    _0: i32
}

impl Rgb10a2 {
    pub const fn new(red: i32, green: i32, blue: i32, alpha: i32) -> Self {
        Rgb10a2{_0: ((alpha & 0x3) << 30) | ((red & 0x3FF) << 20) | ((green & 0x3FF) << 10) | (blue & 0x3FF)}
    }
}

impl FragmentColour for Rgb10a2 {}

impl From<Rgb10a2> for i32 {
    fn from(rgb10_a2: Rgb10a2) -> Self {
        rgb10_a2._0
    }
}

impl From<RGBQUAD> for Rgb10a2 {
    fn from(rgbquad: RGBQUAD) -> Self {
        Rgb10a2::new(rgbquad.rgbRed as i32, rgbquad.rgbGreen as i32, rgbquad.rgbBlue as i32, rgbquad.rgbReserved as i32)
    }
}

#[derive(Clone, Copy, Default)]
pub struct Rgba8 {
    _0: RGBQUAD
}

impl Rgba8 {
    pub const fn new(red: u8, green: u8, blue: u8, alpha: u8) -> Self {
        Rgba8{_0: RGBQUAD{rgbRed: red, rgbGreen: green, rgbBlue: blue, rgbReserved: alpha}}
    }
}

impl FragmentColour for Rgba8 {}

impl From<Rgba8> for i32 {
    fn from(rgba8: Rgba8) -> Self {
        ((rgba8._0.rgbReserved as i32 & 0xFF) << 24) | ((rgba8._0.rgbRed as i32 & 0xFF) << 16) | ((rgba8._0.rgbGreen as i32 & 0xFF) << 8) | (rgba8._0.rgbBlue as i32 & 0xFF)
    }
}

impl From<RGBQUAD> for Rgba8 {
    fn from(rgbquad: RGBQUAD) -> Self {
        Rgba8{_0: rgbquad}
    }
}

#[derive(Clone, Copy)]
pub struct AntiAliasingMode<const T: i32, C: FragmentColour> {
    _marker: PhantomData<C>
}

pub const NOAA_IX: i32 = 0;
#[allow(dead_code)]
pub const NOAA: AntiAliasingMode<NOAA_IX, Rgba8> = AntiAliasingMode{_marker: PhantomData};

pub const SSAA_IX: i32 = 1;
#[allow(dead_code)]
pub const SSAA: AntiAliasingMode<SSAA_IX, Rgb10a2> = AntiAliasingMode{_marker: PhantomData};

pub const MSAA_2X_IX: i32 = 2;
#[allow(dead_code)]
pub const MSAA_2X: AntiAliasingMode<MSAA_2X_IX, Rgb10a2> = AntiAliasingMode{_marker: PhantomData};

pub const MSAA_4X_IX: i32 = 3;
#[allow(dead_code)]
pub const MSAA_4X: AntiAliasingMode<MSAA_4X_IX, Rgb10a2> = AntiAliasingMode{_marker: PhantomData};

fn get_aligned_buffer_size(width: usize, height: usize, x_alignment: usize, y_alignment: usize) -> BufferSize {
    let stride = (width + x_alignment - 1) / x_alignment * x_alignment;
    let lines = ((height + y_alignment - 1) / y_alignment) * y_alignment;
    BufferSize {stride, lines}
}

pub fn get_buffer_size<const AA_MODE: i32, C: FragmentColour>(width: usize, height: usize, _aa_mode: AntiAliasingMode<AA_MODE, C>) -> BufferSize {
    match AA_MODE {
        NOAA_IX => get_aligned_buffer_size(width, height, 4, 2),
        SSAA_IX => get_aligned_buffer_size(width * 2, height * 2, 8, 2),
        MSAA_2X_IX => get_aligned_buffer_size(width * 2, height, 8, 2),
        MSAA_4X_IX => get_aligned_buffer_size(width * 2, height * 2, 8, 4),
        _ => panic!("Unknown AA_MODE {}", AA_MODE)
    }
}

pub fn get_out_buffer_size<const AA_MODE: i32, C: FragmentColour>(width: usize, height: usize, _aa_mode: AntiAliasingMode<AA_MODE, C>) -> BufferSize {
    let (x_alignment, y_alignment) = match AA_MODE {
        NOAA_IX => (4, 2),
        SSAA_IX => (4, 1),
        MSAA_2X_IX => (4, 2),
        MSAA_4X_IX => (4, 2),
        _ => panic!("Unknown AA_MODE {}", AA_MODE)
    };
    get_aligned_buffer_size(width, height, x_alignment, y_alignment)
}

pub fn get_scale<const AA_MODE: i32, _C: FragmentColour>(_aa_mode: AntiAliasingMode<AA_MODE, _C>) -> (f32, f32) {
    match AA_MODE {
        SSAA_IX => (2.0, 2.0),
        NOAA_IX | MSAA_2X_IX | MSAA_4X_IX => (1.0, 1.0),
        _ => panic!("Unknown AA_MODE {}", AA_MODE)
    }
}

const GAMMA: f32 = 2.2;
const COMPONENT_LUT_SIZE: usize = 1024;
fn init_colour_lut(shift: usize) -> [i32; COMPONENT_LUT_SIZE] {
    array::from_fn(|i| {
        let intensity = i as f32 / (COMPONENT_LUT_SIZE - 1) as f32;
        let gamma_corrected_intensity = intensity.powf(1.0 / GAMMA);
        ((gamma_corrected_intensity * 255.0) as i32) << shift
    })
}
static RED_LUT: Lazy<[i32; COMPONENT_LUT_SIZE]> = Lazy::new(|| init_colour_lut(16));
static GREEN_LUT: Lazy<[i32; COMPONENT_LUT_SIZE]> = Lazy::new(|| init_colour_lut(8));
static BLUE_LUT: Lazy<[i32; COMPONENT_LUT_SIZE]> = Lazy::new(|| init_colour_lut(0));

macro_rules! gamma_correct_rgb10_a2_m256i {
    ($rgb10_a2:expr) => {{
        let mask = _mm256_set1_epi32(0x3FF);
        let red = _mm256_i32gather_epi32(RED_LUT.as_ptr(), _mm256_and_si256(_mm256_srli_epi32($rgb10_a2, 20), mask), 4);
        let green = _mm256_i32gather_epi32(GREEN_LUT.as_ptr(), _mm256_and_si256(_mm256_srli_epi32($rgb10_a2, 10), mask), 4);
        let blue = _mm256_i32gather_epi32(BLUE_LUT.as_ptr(), _mm256_and_si256($rgb10_a2, mask), 4);
        _mm256_or_si256(_mm256_or_si256(red, green), blue)
    }};
}

macro_rules! gamma_correct_rgb10_a2_m128i {
    ($rgb10_a2:expr) => {{
        let mask = _mm_set1_epi32(0x3FF);
        let red = _mm_i32gather_epi32(RED_LUT.as_ptr(), _mm_and_si128(_mm_srli_epi32($rgb10_a2, 20), mask), 4);
        let green = _mm_i32gather_epi32(GREEN_LUT.as_ptr(), _mm_and_si128(_mm_srli_epi32($rgb10_a2, 10), mask), 4);
        let blue = _mm_i32gather_epi32(BLUE_LUT.as_ptr(), _mm_and_si128($rgb10_a2, mask), 4);
        _mm_or_si128(_mm_or_si128(red, green), blue)
    }};
}

fn fill_triangle_generic<const AA_MODE: i32, C: FragmentColour, const C0: i32, const C1: i32, const C2: i32, const EXECUTE_FRAGMENT_SHADER: bool, T: Copy>(
        colour: &mut [C], depth: &mut [f32], buffer_size: &BufferSize, it: usize,
        xmin: f32, ymin: f32, xmax: f32, ymax: f32,
        x0: f32, y0: f32, z0: f32, iw0: f32,
        x1: f32, y1: f32, z1: f32, iw1: f32,
        x2: f32, y2: f32, z2: f32, iw2: f32,
        e0: T, e1: T, e2: T,
        iarea: f32,
        _aa_mode: AntiAliasingMode<AA_MODE, C>, fragment_shader: &impl AvxFragmentShader<EXECUTE_FRAGMENT_SHADER, T>) {
    debug_assert!(colour.as_ptr().align_offset(32) == 0);
    debug_assert!(depth.as_ptr().align_offset(32) == 0);
    debug_assert!(buffer_size.stride % if AA_MODE == MSAA_2X_IX || AA_MODE == MSAA_4X_IX {8} else {4} == 0);
    debug_assert!(buffer_size.lines % if AA_MODE == MSAA_4X_IX {4} else {2}  == 0);
    debug_assert!(iarea != 0.0);

    // draw 4x2 pixels at once; horizontal alignment required for SIMD ...
    let xmin = (xmin / 4.0).floor() * 4.0;
    let xmax = (xmax / 4.0).ceil() * 4.0;

    // ... vertical for MSAA_4X
    let y_alignment = match AA_MODE {
        NOAA_IX | SSAA_IX => 1,
        MSAA_2X_IX | MSAA_4X_IX => 2,
        _ => panic!("Unknown AA_MODE {}", AA_MODE)
    };
    let ymin = (ymin / y_alignment as f32).floor() * y_alignment as f32;
    let ymax = (ymax / y_alignment as f32).ceil() * y_alignment as f32;
   
    let yscale = match AA_MODE {
        NOAA_IX | SSAA_IX | MSAA_2X_IX => 1,
        MSAA_4X_IX => 2,
        _ => panic!("Unknown AA_MODE {}", AA_MODE)
    };

    unsafe {
        // barycentric coordinates of the first 4x2 pixels block on the first rows of the bounding box
        let x0_v = _mm256_set1_ps(x0);
        let y0_v = _mm256_set1_ps(y0);
        let x1_v = _mm256_set1_ps(x1);
        let y1_v = _mm256_set1_ps(y1);
        let x2_v = _mm256_set1_ps(x2);
        let y2_v = _mm256_set1_ps(y2);

        // the first sampling position within each pixel in the 4x2 output block
        let (block_offsets_x, block_offsets_y) = match AA_MODE {
            NOAA_IX | SSAA_IX =>
                (_mm256_setr_ps(0.5, 1.5, 2.5, 3.5, 0.5, 1.5, 2.5, 3.5),
                 _mm256_setr_ps(0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5)),
            MSAA_2X_IX =>
                (_mm256_setr_ps(0.25, 1.75, 2.25, 3.75, 0.75, 1.25, 2.75, 3.25),
                 _mm256_setr_ps(0.25, 0.25, 0.25, 0.25, 1.25, 1.25, 1.25, 1.25)),
            MSAA_4X_IX =>
                (_mm256_setr_ps(0.375, 1.375, 2.375, 3.375, 0.375, 1.375, 2.375, 3.375),
                 _mm256_setr_ps(0.125, 0.125, 0.125, 0.125, 1.125, 1.125, 1.125, 1.125)),
            _ => panic!("Unknown AA_MODE {}", AA_MODE)
        };

        let x_sample0 = _mm256_add_ps(_mm256_set1_ps(xmin), block_offsets_x);
        let y_sample0 = _mm256_add_ps(_mm256_set1_ps(ymin), block_offsets_y);

        let iarea = _mm256_set1_ps(iarea);
        let mut row_w0_0 = _mm256_mul_ps(edge_function!(x1_v, y1_v, x2_v, y2_v, x_sample0, y_sample0), iarea);
        let mut row_w1_0 = _mm256_mul_ps(edge_function!(x2_v, y2_v, x0_v, y0_v, x_sample0, y_sample0), iarea);
        let mut row_w2_0 = _mm256_mul_ps(edge_function!(x0_v, y0_v, x1_v, y1_v, x_sample0, y_sample0), iarea);

        // if you substitute `xp + 1` for `xp` into the edge function you can see that
        // for a given edge, the value of the function for `xp + 1, yp` is the value for `xp, yp` minus `y0-y1`
        let xstep0 = _mm256_set1_ps(y1-y2);
        let xstep1 = _mm256_set1_ps(y2-y0);
        let xstep2 = _mm256_set1_ps(y0-y1);

        // as above, the value of the edge function for `xp, yp + 1` is the value for `xp,yp` minus `x1-x0`.
        let ystep0 = _mm256_set1_ps(x2-x1);
        let ystep1 = _mm256_set1_ps(x0-x2);
        let ystep2 = _mm256_set1_ps(x1-x0);

        macro_rules! sample_offsets {
            ($xoffset:expr, $yoffset:expr) => {{
                let xoffset = _mm256_mul_ps(iarea, $xoffset);
                let yoffset = _mm256_mul_ps(iarea, $yoffset);
                (_mm256_add_ps(_mm256_mul_ps(xstep0, xoffset), _mm256_mul_ps(ystep0, yoffset)),
                 _mm256_add_ps(_mm256_mul_ps(xstep1, xoffset), _mm256_mul_ps(ystep1, yoffset)),
                 _mm256_add_ps(_mm256_mul_ps(xstep2, xoffset), _mm256_mul_ps(ystep2, yoffset)))
            }}
        }

        // offsets from the first sample position to other samples
        let mut sample_offsets1 = (_mm256_undefined_ps(), _mm256_undefined_ps(), _mm256_undefined_ps());
        let mut sample_offsets2 = (_mm256_undefined_ps(), _mm256_undefined_ps(), _mm256_undefined_ps());
        let mut sample_offsets3 = (_mm256_undefined_ps(), _mm256_undefined_ps(), _mm256_undefined_ps());
        match AA_MODE {
            NOAA_IX | SSAA_IX => {},
            MSAA_2X_IX =>
                sample_offsets1 = sample_offsets!(_mm256_setr_ps(0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5), _mm256_set1_ps(0.5)),
            MSAA_4X_IX => {
                sample_offsets1 = sample_offsets!(_mm256_set1_ps(0.5), _mm256_set1_ps(0.25));
                sample_offsets2 = sample_offsets!(_mm256_set1_ps(-0.25), _mm256_set1_ps(0.5));
                sample_offsets3 = sample_offsets!(_mm256_set1_ps(0.25), _mm256_set1_ps(0.75));
            },
            _ => panic!("Unknown AA_MODE {}", AA_MODE)
        };

        // advance by 4x2 blocks
        let iarea_times_4 = _mm256_mul_ps(iarea, _mm256_set1_ps(4.0));
        let xstep0 = _mm256_mul_ps(xstep0, iarea_times_4);
        let xstep1 = _mm256_mul_ps(xstep1, iarea_times_4);
        let xstep2 = _mm256_mul_ps(xstep2, iarea_times_4);

        let iarea_times_2 = _mm256_mul_ps(iarea, _mm256_set1_ps(2.0));
        let ystep0 = _mm256_mul_ps(ystep0, iarea_times_2);
        let ystep1 = _mm256_mul_ps(ystep1, iarea_times_2);
        let ystep2 = _mm256_mul_ps(ystep2, iarea_times_2);

        let zero = _mm256_setzero_ps();
        let one = _mm256_set1_ps(1.0);
        let iw0 = _mm256_set1_ps(iw0);
        let iw1 = _mm256_set1_ps(iw1);
        let iw2 = _mm256_set1_ps(iw2);
        let z0 = _mm256_set1_ps(z0);
        let z1 = _mm256_set1_ps(z1);
        let z2 = _mm256_set1_ps(z2);

        let mut yp = ymin as usize;
        let mut c_row = (colour.as_mut_ptr() as *mut i32).add(yp * buffer_size.stride * yscale);
        let mut d_row = depth.as_mut_ptr().add(yp * buffer_size.stride * yscale);
        let xmin = xmin as usize;
        let xmax = xmax as usize;
        while yp < ymax as usize {
            let mut w0_0 = row_w0_0;
            let mut w1_0 = row_w1_0;
            let mut w2_0 = row_w2_0;
            let mut xp = xmin;
            while xp < xmax {
                let any_inside;
                let inside_mask_0;
                let mut inside_mask_1 = _mm256_undefined_si256();
                let mut inside_mask_2 = _mm256_undefined_si256();
                let mut inside_mask_3 = _mm256_undefined_si256();
                let mut w0_1 = _mm256_undefined_ps();
                let mut w1_1 = _mm256_undefined_ps();
                let mut w2_1 = _mm256_undefined_ps();
                let mut w0_2 = _mm256_undefined_ps();
                let mut w1_2 = _mm256_undefined_ps();
                let mut w2_2 = _mm256_undefined_ps();
                let mut w0_3 = _mm256_undefined_ps();
                let mut w1_3 = _mm256_undefined_ps();
                let mut w2_3 = _mm256_undefined_ps();

                macro_rules! inside {
                    ($w0:expr, $w1:expr, $w2:expr) => {{
                        let inside0 = _mm256_castps_si256(_mm256_cmp_ps::<C0>($w0, zero));
                        let inside1 = _mm256_castps_si256(_mm256_cmp_ps::<C1>($w1, zero));
                        let inside2 = _mm256_castps_si256(_mm256_cmp_ps::<C2>($w2, zero));
                        _mm256_and_si256(inside0, _mm256_and_si256(inside1, inside2))
                    }};
                }
                match AA_MODE {
                    NOAA_IX | SSAA_IX => {
                        inside_mask_0 = inside!(w0_0, w1_0, w2_0);
                        any_inside = inside_mask_0;
                    },
                    MSAA_2X_IX => {
                        inside_mask_0 = inside!(w0_0, w1_0, w2_0);
                        w0_1 = _mm256_sub_ps(w0_0, sample_offsets1.0);
                        w1_1 = _mm256_sub_ps(w1_0, sample_offsets1.1);
                        w2_1 = _mm256_sub_ps(w2_0, sample_offsets1.2);
                        inside_mask_1 = inside!(w0_1, w1_1, w2_1);
                        any_inside = _mm256_or_si256(inside_mask_0, inside_mask_1);
                    },
                    MSAA_4X_IX => {
                        inside_mask_0 = inside!(w0_0, w1_0, w2_0);
                        w0_1 = _mm256_sub_ps(w0_0, sample_offsets1.0);
                        w1_1 = _mm256_sub_ps(w1_0, sample_offsets1.1);
                        w2_1 = _mm256_sub_ps(w2_0, sample_offsets1.2);
                        inside_mask_1 = inside!(w0_1, w1_1, w2_1);
                        w0_2 = _mm256_sub_ps(w0_0, sample_offsets2.0);
                        w1_2 = _mm256_sub_ps(w1_0, sample_offsets2.1);
                        w2_2 = _mm256_sub_ps(w2_0, sample_offsets2.2);
                        inside_mask_2 = inside!(w0_2, w1_2, w2_2);
                        w0_3 = _mm256_sub_ps(w0_0, sample_offsets3.0);
                        w1_3 = _mm256_sub_ps(w1_0, sample_offsets3.1);
                        w2_3 = _mm256_sub_ps(w2_0, sample_offsets3.2);
                        inside_mask_3 = inside!(w0_3, w1_3, w2_3);
                        any_inside = _mm256_or_si256(inside_mask_0, _mm256_or_si256(inside_mask_1, _mm256_or_si256(inside_mask_2, inside_mask_3)));
                    },
                    _ => panic!("Unknown AA_MODE {}", AA_MODE)
                }

                // skip blocks that are fully outside the triangle
                if _mm256_movemask_epi8(any_inside) != 0 {
                    // adjust for perspective correct interpolation
                    macro_rules! test_sample {
                        ($w0:expr, $w1:expr, $w2:expr, $inside_mask:expr, $existing_z:expr) => {{
                            // adjust for perspective correct interpolation
                            let p_w0 = _mm256_mul_ps($w0, iw0);
                            let p_w1 = _mm256_mul_ps($w1, iw1);
                            let p_w2 = _mm256_mul_ps($w2, iw2);

                            let t = _mm256_rcp_ps(_mm256_add_ps(p_w0, _mm256_add_ps(p_w1, p_w2)));
                            let p_w0 = _mm256_mul_ps(p_w0, t);
                            let p_w1 = _mm256_mul_ps(p_w1, t);
                            let p_w2 = _mm256_mul_ps(p_w2, t);

                            let z = interpolate!(z0, z1, z2, p_w0, p_w1, p_w2);

                            // this near test isn't really enough, we really need to clip geometry against the near plane
                            let near_mask = _mm256_castps_si256(_mm256_cmp_ps(z, one, _CMP_LE_OQ));
                            let depth_mask = _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(z, $existing_z, _CMP_GT_OQ)), near_mask);
                            let mask = _mm256_and_si256($inside_mask, depth_mask);

                            let blended_z = _mm256_blendv_ps($existing_z, z, _mm256_castsi256_ps(mask));
                            (p_w0, p_w1, p_w2, mask, blended_z)
                        }};
                    }
                    
                    match AA_MODE {
                        NOAA_IX | SSAA_IX => {
                            let existing_z = _mm256_loadu2_m128(d_row.add(buffer_size.stride + xp), d_row.add(xp));
                            let (p_w0, p_w1, p_w2, mask, blended_z) = test_sample!(w0_0, w1_0, w2_0, inside_mask_0, existing_z);

                            if EXECUTE_FRAGMENT_SHADER {
                                // shadow map is quicker with this test here
                                if _mm256_movemask_epi8(mask) != 0 {
                                    let mut filled_span = fragment_shader.fragment(it, p_w0, p_w1, p_w2, e0, e1, e2);
                                    if AA_MODE == NOAA_IX {
                                        filled_span = gamma_correct_rgb10_a2_m256i!(filled_span);
                                    }
                                    _mm_maskstore_epi32(c_row.add(xp), _mm256_castsi256_si128(mask), _mm256_castsi256_si128(filled_span));
                                    _mm_maskstore_epi32(c_row.add(buffer_size.stride + xp), _mm256_extracti128_si256(mask, 1), _mm256_extracti128_si256(filled_span, 1));
                                }
                            }
                            _mm256_storeu2_m128(d_row.add(buffer_size.stride + xp), d_row.add(xp), blended_z);
                        },
                        MSAA_2X_IX => {
                            // double width and height; depths and colours are stored with the 4x2 block flattened to 8x1;
                            // this is masked according to each sample point and stored vertically and the downsampler
                            // eventually converts each 8x2 back into 4x2
                            let existing_z_0 = _mm256_load_ps(d_row.add(xp * 2));
                            let (p_w0_0, p_w1_0, p_w2_0, mask_0, blended_z_0) = test_sample!(w0_0, w1_0, w2_0, inside_mask_0, existing_z_0);

                            let existing_z_1 = _mm256_load_ps(d_row.add(buffer_size.stride + xp * 2));
                            let (p_w0_1, p_w1_1, p_w2_1, mask_1, blended_z_1) = test_sample!(w0_1, w1_1, w2_1, inside_mask_1, existing_z_1);
                            
                            if EXECUTE_FRAGMENT_SHADER {
                                let any_pass = _mm256_or_si256(mask_0, mask_1);
                                if _mm256_movemask_epi8(any_pass) != 0 {
                                    // select a passing set of interpolants for each pixel to give to the fragment shader
                                    let p_w0 = _mm256_blendv_ps(p_w0_0, p_w0_1, _mm256_castsi256_ps(mask_1));
                                    let p_w1 = _mm256_blendv_ps(p_w1_0, p_w1_1, _mm256_castsi256_ps(mask_1));
                                    let p_w2 = _mm256_blendv_ps(p_w2_0, p_w2_1, _mm256_castsi256_ps(mask_1));
                                    let colour = fragment_shader.fragment(it, p_w0, p_w1, p_w2, e0, e1, e2);
                                    _mm256_maskstore_epi32(c_row.add(xp * 2), mask_0, colour);
                                    _mm256_maskstore_epi32(c_row.add(buffer_size.stride + xp * 2), mask_1, colour);
                                }
                            }

                            _mm256_store_ps(d_row.add(xp * 2), blended_z_0);
                            _mm256_store_ps(d_row.add(buffer_size.stride + xp * 2), blended_z_1);
                        },
                        MSAA_4X_IX => {
                            // double width and height; depths and colours are stored with the 4x2 block flattened to 8x1;
                            // this is masked according to each sample point and stored vertically and the downsampler
                            // eventually converts each 8x4 back into 4x2
                            let existing_z_0 = _mm256_load_ps(d_row.add(xp * 2));
                            let (p_w0_0, p_w1_0, p_w2_0, mask_0, blended_z_0) = test_sample!(w0_0, w1_0, w2_0, inside_mask_0, existing_z_0);

                            let existing_z_1 = _mm256_load_ps(d_row.add(buffer_size.stride + xp * 2));
                            let (p_w0_1, p_w1_1, p_w2_1, mask_1, blended_z_1) = test_sample!(w0_1, w1_1, w2_1, inside_mask_1, existing_z_1);

                            let existing_z_2 = _mm256_load_ps(d_row.add(buffer_size.stride * 2 + xp * 2));
                            let (p_w0_2, p_w1_2, p_w2_2, mask_2, blended_z_2) = test_sample!(w0_2, w1_2, w2_2, inside_mask_2, existing_z_2);

                            let existing_z_3 = _mm256_load_ps(d_row.add(buffer_size.stride * 3 + xp * 2));
                            let (p_w0_3, p_w1_3, p_w2_3, mask_3, blended_z_3) = test_sample!(w0_3, w1_3, w2_3, inside_mask_3, existing_z_3);

                            if EXECUTE_FRAGMENT_SHADER {
                                let any_pass = _mm256_or_si256(mask_0, _mm256_or_si256(mask_1, _mm256_or_si256(mask_2, mask_3)));
                                if _mm256_movemask_epi8(any_pass) != 0 {
                                    // select a passing set of interpolants for each pixel to give to the fragment shader
                                    let p_w0 = _mm256_blendv_ps(p_w0_0, p_w0_1, _mm256_castsi256_ps(mask_1));
                                    let p_w1 = _mm256_blendv_ps(p_w1_0, p_w1_1, _mm256_castsi256_ps(mask_1));
                                    let p_w2 = _mm256_blendv_ps(p_w2_0, p_w2_1, _mm256_castsi256_ps(mask_1));

                                    let p_w0 = _mm256_blendv_ps(p_w0, p_w0_2, _mm256_castsi256_ps(mask_2));
                                    let p_w1 = _mm256_blendv_ps(p_w1, p_w1_2, _mm256_castsi256_ps(mask_2));
                                    let p_w2 = _mm256_blendv_ps(p_w2, p_w2_2, _mm256_castsi256_ps(mask_2));

                                    let p_w0 = _mm256_blendv_ps(p_w0, p_w0_3, _mm256_castsi256_ps(mask_3));
                                    let p_w1 = _mm256_blendv_ps(p_w1, p_w1_3, _mm256_castsi256_ps(mask_3));
                                    let p_w2 = _mm256_blendv_ps(p_w2, p_w2_3, _mm256_castsi256_ps(mask_3));

                                    let colour = fragment_shader.fragment(it, p_w0, p_w1, p_w2, e0, e1, e2);
                                    _mm256_maskstore_epi32(c_row.add(xp * 2), mask_0, colour);
                                    _mm256_maskstore_epi32(c_row.add(buffer_size.stride + xp * 2), mask_1, colour);
                                    _mm256_maskstore_epi32(c_row.add(buffer_size.stride * 2 + xp * 2), mask_2, colour);
                                    _mm256_maskstore_epi32(c_row.add(buffer_size.stride * 3 + xp * 2), mask_3, colour);
                                }
                            }

                            _mm256_store_ps(d_row.add(xp * 2), blended_z_0);
                            _mm256_store_ps(d_row.add(buffer_size.stride + xp * 2), blended_z_1);
                            _mm256_store_ps(d_row.add(buffer_size.stride * 2 + xp * 2), blended_z_2);
                            _mm256_store_ps(d_row.add(buffer_size.stride * 3 + xp * 2), blended_z_3);
                        },
                        _ => panic!("Unknown AA_MODE {}", AA_MODE)
                    }
                }

                xp += 4;

                w0_0 = _mm256_sub_ps(w0_0, xstep0);
                w1_0 = _mm256_sub_ps(w1_0, xstep1);
                w2_0 = _mm256_sub_ps(w2_0, xstep2);
            }

            yp += 2;
            c_row = c_row.add(buffer_size.stride * yscale * 2);
            d_row = d_row.add(buffer_size.stride * yscale * 2);
            row_w0_0 = _mm256_sub_ps(row_w0_0, ystep0);
            row_w1_0 = _mm256_sub_ps(row_w1_0, ystep1);
            row_w2_0 = _mm256_sub_ps(row_w2_0, ystep2);
        }
    }
}

pub fn fill_triangle<const AA_MODE: i32, C: FragmentColour, const EXECUTE_FRAGMENT_SHADER: bool, T: Copy>(
        colour: &mut [C], depth: &mut [f32], buffer_size: &BufferSize, it: usize,
        xmin: f32, ymin: f32, xmax: f32, ymax: f32,
        x0: f32, y0: f32, z0: f32, iw0: f32,
        x1: f32, y1: f32, z1: f32, iw1: f32,
        x2: f32, y2: f32, z2: f32, iw2: f32,
        e0: T, e1: T, e2: T,
        iarea: f32, tl0: bool, tl1: bool, tl2: bool,
        aa_mode: AntiAliasingMode<AA_MODE, C>, fragment_shader: &impl AvxFragmentShader<EXECUTE_FRAGMENT_SHADER, T>) {

    // NB - no TL or all TL edges are impossible, but are here for completeness
    match (tl0, tl1, tl2) {
        (false, false, false) => fill_triangle_generic::<AA_MODE, C, _CMP_GT_OQ, _CMP_GT_OQ, _CMP_GT_OQ, EXECUTE_FRAGMENT_SHADER, T>(colour, depth, buffer_size, it, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, e0, e1, e2, iarea, aa_mode, fragment_shader),
        (false, false, true) => fill_triangle_generic::<AA_MODE, C, _CMP_GT_OQ, _CMP_GT_OQ, _CMP_GE_OQ, EXECUTE_FRAGMENT_SHADER, T>(colour, depth, buffer_size, it, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, e0, e1, e2, iarea, aa_mode, fragment_shader),
        (false, true, false) => fill_triangle_generic::<AA_MODE, C, _CMP_GT_OQ, _CMP_GE_OQ, _CMP_GT_OQ, EXECUTE_FRAGMENT_SHADER, T>(colour, depth, buffer_size, it, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, e0, e1, e2, iarea, aa_mode, fragment_shader),
        (false, true, true) => fill_triangle_generic::<AA_MODE, C, _CMP_GT_OQ, _CMP_GE_OQ, _CMP_GE_OQ, EXECUTE_FRAGMENT_SHADER, T>(colour, depth, buffer_size, it, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, e0, e1, e2, iarea, aa_mode, fragment_shader),
        (true, false, false) => fill_triangle_generic::<AA_MODE, C, _CMP_GE_OQ, _CMP_GT_OQ, _CMP_GT_OQ, EXECUTE_FRAGMENT_SHADER, T>(colour, depth, buffer_size, it, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, e0, e1, e2, iarea, aa_mode, fragment_shader),
        (true, false, true) => fill_triangle_generic::<AA_MODE, C, _CMP_GE_OQ, _CMP_GT_OQ, _CMP_GE_OQ, EXECUTE_FRAGMENT_SHADER, T>(colour, depth, buffer_size, it, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, e0, e1, e2, iarea, aa_mode, fragment_shader),
        (true, true, false) => fill_triangle_generic::<AA_MODE, C, _CMP_GE_OQ, _CMP_GE_OQ, _CMP_GT_OQ, EXECUTE_FRAGMENT_SHADER, T>(colour, depth, buffer_size, it, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, e0, e1, e2, iarea, aa_mode, fragment_shader),
        (true, true, true) => fill_triangle_generic::<AA_MODE, C, _CMP_GE_OQ, _CMP_GE_OQ, _CMP_GE_OQ, EXECUTE_FRAGMENT_SHADER, T>(colour, depth, buffer_size, it, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, e0, e1, e2, iarea, aa_mode,fragment_shader),
    }
}

const LSBS: i32 = 0b00000000_00010000_00000100_00000001;
macro_rules! avg_rgb10_a2 {
    ($a:expr, $b:expr) => {{
        let sums = _mm256_xor_epi32($a, $b);
        let half_sums = _mm256_srli_epi32(_mm256_and_si256(sums, _mm256_set1_epi32(!LSBS)), 1);
        let carries = _mm256_and_si256($a, $b);
        let floor_avg = _mm256_add_epi32(half_sums, carries);
        let rounded_avg = _mm256_add_epi32(floor_avg, _mm256_and_si256(sums, _mm256_set1_epi32(LSBS)));
        rounded_avg
    }};
}

// these are less generic than the type signature: they assume Rgb10a2 in the colour buffer
fn resolve_ssaa<C: FragmentColour>(out: &mut [RGBQUAD], out_buffer_size: &BufferSize, colour: &[C], colour_buffer_size: &BufferSize, left: usize, top: usize, width: usize, height: usize) {
    debug_assert!(out.as_ptr().align_offset(32) == 0);
    debug_assert!(out_buffer_size.stride % 4 == 0);
    debug_assert!(colour.as_ptr().align_offset(32) == 0);
    debug_assert!(colour_buffer_size.stride % 8 == 0);
    debug_assert!(colour_buffer_size.lines % 2 == 0);
    debug_assert!(left % 8 == 0);
    debug_assert!(top % 2 == 0);

    unsafe {
        let out_top = top / 2;
        let out_left = left / 2;
        let mut out_row = out.as_mut_ptr().add(out_top * out_buffer_size.stride + out_left) as *mut __m128i;
        let mut in_row0 = colour.as_ptr().add(top * colour_buffer_size.stride + left) as *mut __m256i;
        let mut in_row1 = in_row0.add(colour_buffer_size.stride / 8);
        for _ in (0..height).step_by(2) {
            for x in 0..((width + 7) / 8) {
                let vavg = avg_rgb10_a2!(*in_row0.add(x), *in_row1.add(x));

                // 0, 0, 1, 1, 2, 2, 3, _ 
                let avg = avg_rgb10_a2!(vavg, _mm256_srli_si256(vavg, 4));

                // 0, 1, _, _, 2, 3, _, _ 
                let avg = _mm256_shuffle_epi32(avg, 0b0000_1000);

                //                 0, 1, _ _                    2, 3, _ _
                // =>
                // 0, 1, 2, 3
                let avg = _mm_unpacklo_epi64(_mm256_castsi256_si128(avg), _mm256_extracti128_si256(avg, 1));

                let gamma_corrected = gamma_correct_rgb10_a2_m128i!(avg);

                _mm_store_si128(out_row.add(x), gamma_corrected);
            }
            out_row = out_row.add(out_buffer_size.stride / 4);
            in_row0 = in_row0.add(colour_buffer_size.stride / 8 * 2);
            in_row1 = in_row1.add(colour_buffer_size.stride / 8 * 2);
        }
    }
}

fn resolve_msaa_2x<C: FragmentColour>(out: &mut [RGBQUAD], out_buffer_size: &BufferSize, colour: &[C], colour_buffer_size: &BufferSize, left: usize, top: usize, width: usize, height: usize) {
    debug_assert!(out.as_ptr().align_offset(32) == 0);
    debug_assert!(out_buffer_size.stride % 4 == 0);
    debug_assert!(out_buffer_size.lines % 2 == 0);
    debug_assert!(colour.as_ptr().align_offset(32) == 0);
    debug_assert!(colour_buffer_size.stride % 8 == 0);
    debug_assert!(colour_buffer_size.lines % 2 == 0);
    debug_assert!(left % 8 == 0);
    debug_assert!(top % 2 == 0);

    unsafe {
        let out_left = left / 2;
        let mut out_row0 = out.as_mut_ptr().add(top * out_buffer_size.stride + out_left) as *mut __m128i;
        let mut out_row1 = out_row0.add(out_buffer_size.stride / 4);
        let mut in_row0 = colour.as_ptr().add(top * colour_buffer_size.stride + left) as *mut __m256i;
        let mut in_row1 = in_row0.add(colour_buffer_size.stride / 8);
        for _ in (0..height).step_by(2) {
            for x in 0..((width + 7) / 8) {
                let avg = avg_rgb10_a2!(*in_row0.add(x), *in_row1.add(x));
                let gamma_corrected = gamma_correct_rgb10_a2_m256i!(avg);
                _mm256_storeu2_m128i(out_row1.add(x), out_row0.add(x), gamma_corrected);
            }
            out_row0 = out_row0.add(out_buffer_size.stride / 4 * 2);
            out_row1 = out_row1.add(out_buffer_size.stride / 4 * 2);
            in_row0 = in_row0.add(colour_buffer_size.stride / 8 * 2);
            in_row1 = in_row1.add(colour_buffer_size.stride / 8 * 2);
        }
    }
}

fn resolve_msaa_4x<C: FragmentColour>(out: &mut [RGBQUAD], out_buffer_size: &BufferSize, colour: &[C], colour_buffer_size: &BufferSize, left: usize, top: usize, width: usize, height: usize) {
    debug_assert!(out.as_ptr().align_offset(32) == 0);
    debug_assert!(out_buffer_size.stride % 4 == 0);
    debug_assert!(out_buffer_size.lines % 2 == 0);
    debug_assert!(colour.as_ptr().align_offset(32) == 0);
    debug_assert!(colour_buffer_size.stride % 8 == 0);
    debug_assert!(colour_buffer_size.lines % 4 == 0);
    debug_assert!(left % 8 == 0);
    debug_assert!(top % 4 == 0);

    unsafe {
        let out_top = top / 2;
        let out_left = left / 2;
        let mut out_row0 = out.as_mut_ptr().add(out_top * out_buffer_size.stride + out_left) as *mut __m128i;
        let mut out_row1 = out_row0.add(out_buffer_size.stride / 4);
        let mut in_row0 = colour.as_ptr().add(top * colour_buffer_size.stride + left) as *mut __m256i;
        let mut in_row1 = in_row0.add(colour_buffer_size.stride / 8);
        let mut in_row2 = in_row1.add(colour_buffer_size.stride / 8);
        let mut in_row3 = in_row2.add(colour_buffer_size.stride / 8);
        for _ in (0..height).step_by(4) {
            for x in 0..((width + 7) / 8) {
                let avg01 = avg_rgb10_a2!(*in_row0.add(x), *in_row1.add(x));
                let avg23 = avg_rgb10_a2!(*in_row2.add(x), *in_row3.add(x));
                let avg = avg_rgb10_a2!(avg01, avg23);
                let gamma_corrected = gamma_correct_rgb10_a2_m256i!(avg);
                _mm256_storeu2_m128i(out_row1.add(x), out_row0.add(x), gamma_corrected);
            }
            out_row0 = out_row0.add(out_buffer_size.stride / 4 * 2);
            out_row1 = out_row1.add(out_buffer_size.stride / 4 * 2);
            in_row0 = in_row0.add(colour_buffer_size.stride / 8 * 4);
            in_row1 = in_row1.add(colour_buffer_size.stride / 8 * 4);
            in_row2 = in_row2.add(colour_buffer_size.stride / 8 * 4);
            in_row3 = in_row3.add(colour_buffer_size.stride / 8 * 4);
        }
    }
}

pub fn resolve<const AA_MODE: i32, C: FragmentColour>(out: &mut [RGBQUAD], out_buffer_size: &BufferSize, colour: &[C], colour_buffer_size: &BufferSize, left: usize, top: usize, width: usize, height: usize, _aa_mode: AntiAliasingMode<AA_MODE, C>) {
    match AA_MODE {
        NOAA_IX => {}, // must write directly to out for NOAA
        SSAA_IX => resolve_ssaa(out, out_buffer_size, colour, colour_buffer_size, left, top, width, height),
        MSAA_2X_IX => resolve_msaa_2x(out, out_buffer_size, colour, colour_buffer_size, left*2, top, width*2, height),
        MSAA_4X_IX => resolve_msaa_4x(out, out_buffer_size, colour, colour_buffer_size, left*2, top*2, width*2, height*2),
        _ => panic!("Unknown AA_MODE {}", AA_MODE)
    }
}