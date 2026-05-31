use core::arch::x86_64::*;
use std::sync::*;
use once_cell::sync::Lazy;
use scoped_threadpool::Pool;

use super::simd_vec::*;
use super::obj::*;

#[derive(Clone, Copy)]
pub struct CartesianVector {
    pub x: f32,
    pub y: f32,
    pub z: f32
}

impl CartesianVector {
    pub fn cross_product(self, other: &CartesianVector) -> Self {
        CartesianVector {
            x: self.y*other.z - self.z*other.y,
            y: self.z*other.x - self.x*other.z,
            z: self.x*other.y - self.y*other.x
        }
    }

    pub fn dot_product(self, other: &CartesianVector) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn magnitude(self) -> f32 {
        f32::sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    }

    pub fn normalised(self) -> Self {
        let magnitude = self.magnitude();
        CartesianVector {
            x: self.x / magnitude,
            y: self.y / magnitude,
            z: self.z / magnitude
        }
    }

    pub fn transformed(self, it_t: &[[f32; 3]; 3]) -> Self {
        let mut r = [0.0; 3];

        for row in 0..3 {
            r[row] = it_t[0][row] * self.x + it_t[1][row] * self.y + it_t[2][row] * self.z;
        }

        CartesianVector {x: r[0], y: r[1], z: r[2]}
    }
}

impl std::ops::Add<CartesianVector> for CartesianVector {
    type Output = CartesianVector;

    fn add(self, other: CartesianVector) -> Self {
        CartesianVector {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z
        }
    }
}

impl std::ops::Sub<CartesianVector> for CartesianVector {
    type Output = CartesianVector;

    fn sub(self, other: CartesianVector) -> Self {
        CartesianVector {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z
        }
    }
}

#[derive(Clone, Copy)]
pub struct CartesianCoordinates {
    pub x: f32,
    pub y: f32,
    pub z: f32
}

impl CartesianCoordinates {
    #[allow(dead_code)]
    pub fn to_homogenous(self) -> HomogenousCoordinates {
        HomogenousCoordinates {
            x: self.x,
            y: self.y,
            z: self.z,
            w: 1.0
        }
    }
}

impl std::ops::Sub<CartesianCoordinates> for CartesianCoordinates {
    type Output = CartesianVector;

    fn sub(self, other: CartesianCoordinates) -> CartesianVector {
        CartesianVector {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z
        }
    }
}

#[derive(Clone, Copy)]
pub struct HomogenousCoordinates {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32
}

impl HomogenousCoordinates {
    pub fn transformed(&self, t: &Transformation) -> Self {
        let mut r = [0.0; 4];

        let a = &t.matrix;
        for row in 0..4 {
            r[row] = a[0][row] * self.x + a[1][row] * self.y + a[2][row] * self.z + a[3][row] * self.w;
        }

        HomogenousCoordinates {x: r[0], y: r[1], z: r[2], w: r[3]}
    }

    pub fn to_cartesian(&self) -> (CartesianCoordinates, f32) {
        let iw = 1.0 / self.w;
        (CartesianCoordinates {
            x: self.x * iw,
            y: self.y * iw,
            z: self.z * iw
        }, iw)
    }
}

#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct Transformation {
    // 4 columns of 4 rows
    pub matrix: [[f32; 4]; 4],
    
    // prevent construction by others
    _private: ()
}

impl Transformation {
    #[allow(dead_code)]
    pub const IDENTITY: Self = Transformation { matrix: [
        [1.0, 0.0, 0.0, 0.0], 
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]],
        _private: ()
    };

    pub fn translate(dx: f32, dy: f32, dz: f32) -> Self {
        Transformation { matrix: [
            [1.0, 0.0, 0.0,  1.0], 
            [0.0, 1.0, 0.0,  0.0],
            [0.0, 0.0, 1.0,  0.0],
            [ dx,  dy,  dz,  1.0]],
            _private: ()
        }
    }

    #[allow(dead_code)]
    pub fn scale(sx: f32, sy: f32, sz: f32) -> Self {
        Transformation { matrix: [
            [ sx, 0.0, 0.0, 0.0], 
            [0.0,  sy, 0.0, 0.0],
            [0.0, 0.0,  sz, 0.0],
            [0.0, 0.0, 0.0, 1.0]],
            _private: ()
        }
    }

    #[allow(dead_code)]
    pub fn rotate_x(radians: f32) -> Self {
        let (sin, cos) = radians.sin_cos();
        Transformation { matrix: [
            [1.0, 0.0, 0.0, 0.0], 
            [0.0, cos, sin, 0.0],
            [0.0,-sin, cos, 0.0],
            [0.0, 0.0, 0.0, 1.0]],
            _private: ()
        }
    }

    pub fn rotate_y(radians: f32) -> Self {
        let (sin, cos) = radians.sin_cos();
        Transformation { matrix: [
            [cos, 0.0,-sin, 0.0], 
            [0.0, 1.0, 0.0, 0.0],
            [sin, 0.0, cos, 0.0],
            [0.0, 0.0, 0.0, 1.0]],
            _private: ()
        }
    }

    #[allow(dead_code)]
    pub fn rotate_z(radians: f32) -> Self {
        let (sin, cos) = radians.sin_cos();
        Transformation { matrix: [
            [ cos, sin, 0.0, 0.0], 
            [-sin, cos, 0.0, 0.0],
            [ 0.0, 0.0, 1.0, 0.0],
            [ 0.0, 0.0, 0.0, 1.0]],
            _private: ()
        }
    }

    fn det_2x2(&self, r0: usize, r1: usize, c0: usize, c1: usize) -> f32 {
        let m = &self.matrix;
        m[c0][r0] * m[c1][r1] - m[c1][r0] * m[c0][r1]
    }

    pub fn inverted_transposed_tl_3x3(&self) -> Option<[[f32; 3]; 3]> {
        let m = &self.matrix;

        let det = m[0][0] * self.det_2x2(1, 2, 1, 2)
            - m[1][0] * self.det_2x2(1, 2, 0, 2)
            + m[2][0] * self.det_2x2(1, 2, 0, 1);
        if det == 0.0 {
            return None;
        }
        let idet = 1.0 / det;

        let im00 =  self.det_2x2(1, 2, 1, 2) * idet;
        let im01 = -self.det_2x2(1, 2, 0, 2) * idet;
        let im02 =  self.det_2x2(1, 2, 0, 1) * idet;

        let im10 = -self.det_2x2(0, 2, 1, 2) * idet;
        let im11 =  self.det_2x2(0, 2, 0, 2) * idet;
        let im12 = -self.det_2x2(0, 2, 0, 1) * idet;

        let im20 =  self.det_2x2(0, 1, 1, 2) * idet;
        let im21 = -self.det_2x2(0, 1, 0, 2) * idet;
        let im22 =  self.det_2x2(0, 1, 0, 1) * idet;

        Some([
            [im00, im10, im20],
            [im01, im11, im21],
            [im02, im12, im22]
        ])
    }

    // assumes premultiplication so returns t*self
    pub fn then(&self, t: &Transformation) -> Self {
        let mut matrix: [[f32; 4]; 4] = [[0.0; 4]; 4];

        let a = t.matrix;
        let b = self.matrix;

        for row in 0..4 {
            for col in 0..4 {
                matrix[col][row] = a[0][row] * b[col][0] + a[1][row] * b[col][1] + a[2][row] * b[col][2] + a[3][row] * b[col][3];
            }
        }

        Transformation {matrix, _private: ()}
    }

    pub fn look_at_rh(eye: &CartesianCoordinates, centre: &CartesianCoordinates, up: &CartesianVector) -> Self {
        let z = (*eye - *centre).normalised();
        let x = up.cross_product(&z).normalised();
        let y = z.cross_product(&x).normalised();

        Transformation::translate(-eye.x, -eye.y, -eye.z).then(&Transformation { matrix: [
            [x.x, y.x, z.x, 0.0], 
            [x.y, y.y, z.y, 0.0],
            [x.z, y.z, z.z, 0.0],
            [0.0, 0.0, 0.0, 1.0]],
            _private: ()
        })
    }

    pub fn perspective_rh(width: f32, height: f32, near: f32, far: f32) -> Self {
        Transformation { matrix: [
            [2.0*near/width,             0.0,                 0.0,  0.0],
            [           0.0, 2.0*near/height,                 0.0,  0.0],
            [           0.0,             0.0,      far/(near-far), -1.0],
            [           0.0,             0.0, near*far/(near-far),  0.0]],
            _private: ()
        }
    }

    pub fn viewport(x: usize, y: usize, width: usize, height: usize) -> Self {
        let hw = width as f32 / 2.0;
        let hh = height as f32 / 2.0;
        Transformation { matrix: [
            [           hw,           0.0, 0.0, 0.0],
            [          0.0,           -hh, 0.0, 0.0],
            [          0.0,           0.0, 1.0, 0.0],
            [(x as f32)+hw, (y as f32)+hh, 0.0, 1.0]],
            _private: ()
        }
    }
}

// not-suitable-for-production SIMD operations; these will only work on processors that support AVX2

// this isn't suitable for production; in particular there's no opportunity to clip
// so it is possible that the perpective divide could lead to infinite values
#[target_feature(enable = "fma,avx,avx2")]
unsafe fn avx2_chunk_transformed_to_cartesian(
        vs_out: [&mut [__m256]; 4], 
        xs: &[__m256], ys: &[__m256], zs: &[__m256], ws: &[__m256], t: &Transformation,
        source_offset: usize, chunk_size: usize) {
    // transformations are stored in columns to benefit the simple compiled version; variables here are named row/column
    let t00 = _mm256_set1_ps(t.matrix[0][0]);
    let t01 = _mm256_set1_ps(t.matrix[1][0]);
    let t02 = _mm256_set1_ps(t.matrix[2][0]);
    let t03 = _mm256_set1_ps(t.matrix[3][0]);

    let t10 = _mm256_set1_ps(t.matrix[0][1]);
    let t11 = _mm256_set1_ps(t.matrix[1][1]);
    let t12 = _mm256_set1_ps(t.matrix[2][1]);
    let t13 = _mm256_set1_ps(t.matrix[3][1]);

    let t20 = _mm256_set1_ps(t.matrix[0][2]);
    let t21 = _mm256_set1_ps(t.matrix[1][2]);
    let t22 = _mm256_set1_ps(t.matrix[2][2]);
    let t23 = _mm256_set1_ps(t.matrix[3][2]);

    let t30 = _mm256_set1_ps(t.matrix[0][3]);
    let t31 = _mm256_set1_ps(t.matrix[1][3]);
    let t32 = _mm256_set1_ps(t.matrix[2][3]);
    let t33 = _mm256_set1_ps(t.matrix[3][3]);

    for i in 0..chunk_size {
        // compute w first so it's ready for conversion to cartesian; interleave for better pipelining
        let w = ws[source_offset + i];
        let x = xs[source_offset + i];
        let y = ys[source_offset + i];
        let z = zs[source_offset + i];

        let mut wh = _mm256_mul_ps(x, t30);
        let mut xh = _mm256_mul_ps(x, t00);
        let mut yh = _mm256_mul_ps(x, t10);
        let mut zh = _mm256_mul_ps(x, t20);
        
        wh = _mm256_fmadd_ps(y, t31, wh);
        xh = _mm256_fmadd_ps(y, t01, xh);
        yh = _mm256_fmadd_ps(y, t11, yh);
        zh = _mm256_fmadd_ps(y, t21, zh);
        
        wh = _mm256_fmadd_ps(z, t32, wh);
        xh = _mm256_fmadd_ps(z, t02, xh);
        yh = _mm256_fmadd_ps(z, t12, yh);
        zh = _mm256_fmadd_ps(z, t22, zh);

        wh = _mm256_fmadd_ps(w, t33, wh);
        xh = _mm256_fmadd_ps(w, t03, xh);
        yh = _mm256_fmadd_ps(w, t13, yh);
        zh = _mm256_fmadd_ps(w, t23, zh);
       
        let iw = _mm256_rcp_ps(wh);

        vs_out[0][i] = _mm256_mul_ps(xh, iw);
        vs_out[1][i] = _mm256_mul_ps(yh, iw);
        vs_out[2][i] = _mm256_mul_ps(zh, iw);
        vs_out[3][i] = iw;
    }
}

// my machine stops showing improvement above four threads
static NUM_PROJECTION_THREADS: u32 = 4;
static PROJECTION_WORKERS: Lazy<Mutex<Pool>> = Lazy::new(|| Mutex::new(Pool::new(NUM_PROJECTION_THREADS)));

pub fn avx2_transformed_to_cartesian(xs_out: &mut SimdVec<f32>, ys_out: &mut SimdVec<f32>, zs_out: &mut SimdVec<f32>, iws_out: &mut SimdVec<f32>, model: &Model, t: &Transformation) {
    let num_chunks = NUM_PROJECTION_THREADS;
    // maintain 128 byte alignment for caching
    let chunk_size = ((model.num_vertices / num_chunks) / 32) * 4;
    let mut chunk_start = 0;

    if chunk_size > 0 {
        let xs = model.xs.as_m256();
        let ys = model.ys.as_m256();
        let zs = model.zs.as_m256();
        let ws = model.ws.as_m256();

        let mut pool = PROJECTION_WORKERS.lock().unwrap();
        pool.scoped(|scope| {
            let xs_out_chunks = xs_out.as_m256_mut().chunks_exact_mut(chunk_size as usize);
            let ys_out_chunks = ys_out.as_m256_mut().chunks_exact_mut(chunk_size as usize);
            let zs_out_chunks = zs_out.as_m256_mut().chunks_exact_mut(chunk_size as usize);
            let iws_out_chunks = iws_out.as_m256_mut().chunks_exact_mut(chunk_size as usize);

            for (xs_out_chunk, (ys_out_chunk, (zs_out_chunk, iws_out_chunk))) in xs_out_chunks.zip(ys_out_chunks.zip(zs_out_chunks.zip(iws_out_chunks))) {
                let vs_out_chunk = [xs_out_chunk, ys_out_chunk, zs_out_chunk, iws_out_chunk];
                let source_offset = chunk_start;
                scope.execute(move || unsafe {
                    avx2_chunk_transformed_to_cartesian(vs_out_chunk, xs, ys, zs, ws, t, source_offset, chunk_size as usize);
                });

                chunk_start += chunk_size as usize;
            }
        });
    }

    // do any leftovers sequentially
    for i in (chunk_start * 8 as usize)..(model.num_vertices as usize) {
        let (r, iw) = HomogenousCoordinates {
            x: model.xs[i], 
            y: model.ys[i], 
            z: model.zs[i], 
            w: model.ws[i]}.transformed(t).to_cartesian();
        xs_out[i] = r.x;
        ys_out[i] = r.y;
        zs_out[i] = r.z;
        iws_out[i] = iw;
    }
}