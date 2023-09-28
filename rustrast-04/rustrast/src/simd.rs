use core::arch::x86_64::*;
use std::{ops::*, sync::mpsc::channel};
use aligned_vec::*;
use lazy_static::*;
use super::obj::*;
use super::maths::*;
use super::threadpool::*;

// not-suitable-for-production SIMD operations; these will only work on processors that support AVX2

// hides the mechanics of alignment from calling code and would allow different alignment for different SIMD tech
pub struct CoordinateComponents {
    vs: AVec<f32, ConstAlign<128>>
}

impl CoordinateComponents {
    pub fn new() -> Self {
        CoordinateComponents {vs: AVec::new(128) }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        CoordinateComponents {vs: AVec::with_capacity(128, capacity) }
    }

    pub fn from_iter<I: IntoIterator<Item = f32>>(iter: I) -> Self {
        CoordinateComponents {vs: AVec::from_iter(128, iter) }
    }
 
    pub fn push(&mut self, v: f32) {
        self.vs.push(v)
    }

    pub fn len(&self) -> usize {
        self.vs.len()
    }

    fn as_ptr (&self) -> *const f32 {
        self.vs.as_ptr()
    }

    fn as_mut_ptr (&mut self) -> *mut f32 {
        self.vs.as_mut_ptr()
    }
}

impl Index<usize> for CoordinateComponents
{
    type Output = f32;

    fn index(&self, ix: usize) -> &Self::Output {
        self.vs.index(ix)
    }
}

impl IndexMut<usize> for CoordinateComponents
{
    fn index_mut(&mut self, ix: usize) -> &mut Self::Output {
        self.vs.index_mut(ix)
    }
}

// enables bypassing safeness checks when multithreading
struct Buffers {
    d: [*mut __m256; 3],
    xs: *const __m256,
    ys: *const __m256,
    zs: *const __m256,
    ws: *const __m256
}

unsafe impl Send for Buffers {}

#[target_feature(enable = "fma,avx,avx2")]
unsafe fn simd_slice_transformed_to_cartesian(buffers: Buffers, count: isize, t: Transformation) {
    debug_assert!(buffers.d[0].align_offset(32) == 0);
    debug_assert!(buffers.d[1].align_offset(32) == 0);
    debug_assert!(buffers.d[2].align_offset(32) == 0);
    debug_assert!(buffers.xs.align_offset(32) == 0);
    debug_assert!(buffers.ys.align_offset(32) == 0);
    debug_assert!(buffers.zs.align_offset(32) == 0);
    debug_assert!(buffers.ws.align_offset(32) == 0);

    for row in 0..3 {
        let c0 = _mm256_set1_ps(t.matrix[0][row]);
        let c1 = _mm256_set1_ps(t.matrix[1][row]);
        let c2 = _mm256_set1_ps(t.matrix[2][row]);
        let c3 = _mm256_set1_ps(t.matrix[3][row]);
        
        for i in 0..count {
            let mut r = _mm256_mul_ps(*buffers.xs.offset(i), c0);
            r = _mm256_fmadd_ps(*buffers.ys.offset(i), c1, r);
            r = _mm256_fmadd_ps(*buffers.zs.offset(i), c2, r);
            *buffers.d[row].offset(i) = _mm256_fmadd_ps(*buffers.ws.offset(i), c3, r);
        }
    }

    let c03 = _mm256_set1_ps(t.matrix[0][3]);
    let c13 = _mm256_set1_ps(t.matrix[1][3]);
    let c23 = _mm256_set1_ps(t.matrix[2][3]);
    let c33 = _mm256_set1_ps(t.matrix[3][3]);        

    for i in 0..count {
        let mut r = _mm256_mul_ps(*buffers.xs.offset(i), c03);
        r = _mm256_fmadd_ps(*buffers.ys.offset(i), c13, r);
        r = _mm256_fmadd_ps(*buffers.zs.offset(i), c23, r);
        r = _mm256_fmadd_ps(*buffers.ws.offset(i), c33, r);
        
        *buffers.d[0].offset(i) = _mm256_div_ps(*buffers.d[0].offset(i), r);
        *buffers.d[1].offset(i) = _mm256_div_ps(*buffers.d[1].offset(i), r);
        *buffers.d[2].offset(i) = _mm256_div_ps(*buffers.d[2].offset(i), r);
    }
}

lazy_static! {
    static ref SIMD_WORKERS: ThreadPool = ThreadPool::new(3);
}

pub unsafe fn simd_transformed_to_cartesian(dest: &mut [CoordinateComponents; 3], model: &Model, t: &Transformation) {
    let num_slices = 3;
    // maintain 128 byte alignment for caching
    let slice_size = ((model.num_vertices / num_slices) / 32) * 32;

    let (tx, rx) = channel();

    for i in 0..num_slices {
        let start = (i*slice_size) as isize;
        let buffers = Buffers {
            d: [dest[0].as_mut_ptr().offset(start) as *mut __m256,
                dest[1].as_mut_ptr().offset(start) as *mut __m256,
                dest[2].as_mut_ptr().offset(start) as *mut __m256],
            xs: model.xs.as_ptr().offset(start) as *const __m256,
            ys: model.ys.as_ptr().offset(start) as *const __m256,
            zs: model.zs.as_ptr().offset(start) as *const __m256,
            ws: model.ws.as_ptr().offset(start) as *const __m256
        };
        let t = *t;
        let tx = tx.clone();

        SIMD_WORKERS.execute(move || {
            simd_slice_transformed_to_cartesian(buffers, (slice_size / 8) as isize, t);
            tx.send(1).unwrap();
        })
    }

    // do any leftovers sequentially in parallel with the workers
    for i in (slice_size * num_slices)..(model.num_vertices) {
        let r = HomogenousCoordinates {
            x: model.xs[i], 
            y: model.ys[i], 
            z: model.zs[i], 
            w: model.ws[i]}.transformed(t).to_cartesian();
        dest[0][i] = r.x;
        dest[1][i] = r.y;
        dest[2][i] = r.z;
    }

    // wait for completion
    let _ = rx.iter().take(num_slices).last();
}