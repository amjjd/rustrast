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

    #[allow(dead_code)]
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

    pub fn to_cartesian(&self) -> CartesianCoordinates {
        CartesianCoordinates {
            x: self.x / self.w,
            y: self.y / self.w,
            z: self.z / self.w
        }
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

#[target_feature(enable = "fma,avx,avx2")]
unsafe fn avx2_chunk_transformed_to_cartesian(
        vs_out: [&mut [__m256]; 3], 
        xs: &[__m256], ys: &[__m256], zs: &[__m256], ws: &[__m256], t: &Transformation,
        source_offset: usize, chunk_size: usize) {
    for row in 0..3 {
        let c0 = _mm256_set1_ps(t.matrix[0][row]);
        let c1 = _mm256_set1_ps(t.matrix[1][row]);
        let c2 = _mm256_set1_ps(t.matrix[2][row]);
        let c3 = _mm256_set1_ps(t.matrix[3][row]);
        
        for i in 0..chunk_size {
            let mut r = _mm256_mul_ps(xs[source_offset + i], c0);
            r = _mm256_fmadd_ps(ys[source_offset + i], c1, r);
            r = _mm256_fmadd_ps(zs[source_offset + i], c2, r);
            vs_out[row][i] = _mm256_fmadd_ps(ws[source_offset + i], c3, r);
        }
    }

    let c03 = _mm256_set1_ps(t.matrix[0][3]);
    let c13 = _mm256_set1_ps(t.matrix[1][3]);
    let c23 = _mm256_set1_ps(t.matrix[2][3]);
    let c33 = _mm256_set1_ps(t.matrix[3][3]);        

    for i in 0..chunk_size {
        let mut r = _mm256_mul_ps(xs[source_offset + i], c03);
        r = _mm256_fmadd_ps(ys[source_offset + i], c13, r);
        r = _mm256_fmadd_ps(zs[source_offset + i], c23, r);
        r = _mm256_fmadd_ps(ws[source_offset + i], c33, r);
        
        vs_out[0][i] = _mm256_div_ps(vs_out[0][i], r);
        vs_out[1][i] = _mm256_div_ps(vs_out[1][i], r);
        vs_out[2][i] = _mm256_div_ps(vs_out[2][i], r);
    }
}

// my machine stops showing improvement above three threads
static NUM_PROJECTION_THREADS: u32 = 3;
static PROJECTION_WORKERS: Lazy<Mutex<Pool>> = Lazy::new(|| Mutex::new(Pool::new(NUM_PROJECTION_THREADS)));

pub fn avx2_transformed_to_cartesian(xs_out: &mut SimdVec<f32>, ys_out: &mut SimdVec<f32>, zs_out: &mut SimdVec<f32>, model: &Model, t: &Transformation) {
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

            for (xs_out_chunk, (ys_out_chunk, zs_out_chunk)) in xs_out_chunks.zip(ys_out_chunks.zip(zs_out_chunks)) {
                let vs_out_chunk = [xs_out_chunk, ys_out_chunk, zs_out_chunk];
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
        let r = HomogenousCoordinates {
            x: model.xs[i], 
            y: model.ys[i], 
            z: model.zs[i], 
            w: model.ws[i]}.transformed(t).to_cartesian();
        xs_out[i] = r.x;
        ys_out[i] = r.y;
        zs_out[i] = r.z;
    }
}