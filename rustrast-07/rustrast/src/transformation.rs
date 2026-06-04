use core::arch::x86_64::*;

#[derive(Clone, Copy)]
pub struct CartesianVector {
    pub x: f32,
    pub y: f32,
    pub z: f32
}

impl CartesianVector {
    #[allow(dead_code)]
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

    #[allow(dead_code)]
    pub fn magnitude(self) -> f32 {
        f32::sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    }

    #[allow(dead_code)]
    pub fn normalised(self) -> Self {
        let magnitude = self.magnitude();
        CartesianVector {
            x: self.x / magnitude,
            y: self.y / magnitude,
            z: self.z / magnitude
        }
    }

    #[allow(dead_code)]
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
    #[allow(dead_code)]
    pub fn transformed(&self, t: &Transformation) -> Self {
        let mut r = [0.0; 4];

        let a = &t.matrix;
        for row in 0..4 {
            r[row] = a[0][row] * self.x + a[1][row] * self.y + a[2][row] * self.z + a[3][row] * self.w;
        }

        HomogenousCoordinates {x: r[0], y: r[1], z: r[2], w: r[3]}
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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
    #[allow(dead_code)]
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

    #[allow(dead_code)]
    pub fn look_at_rh(eye: &CartesianCoordinates, centre: &CartesianCoordinates, up: &CartesianVector) -> Self {
        let z = (*eye - *centre).normalised();
        let x = up.cross_product(&z).normalised();
        let y = z.cross_product(&x).normalised();

        let eye_v = CartesianVector {x: eye.x, y: eye.y, z: eye.z};
        let dx = -x.dot_product(&eye_v);
        let dy = -y.dot_product(&eye_v);
        let dz = -z.dot_product(&eye_v);

        Transformation { matrix: [
            [x.x, y.x, z.x, 0.0], 
            [x.y, y.y, z.y, 0.0],
            [x.z, y.z, z.z, 0.0],
            [ dx,  dy,  dz, 1.0]],
            _private: ()
        }
    }

    #[allow(dead_code)]
    pub fn perspective_rh(width: f32, height: f32, near: f32, far: f32) -> Self {
        Transformation { matrix: [
            [2.0*near/width,             0.0,                 0.0,  0.0],
            [           0.0, 2.0*near/height,                 0.0,  0.0],
            [           0.0,             0.0,      far/(near-far), -1.0],
            [           0.0,             0.0, near*far/(near-far),  0.0]],
            _private: ()
        }
    }

    #[allow(dead_code)]
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

pub unsafe fn vertices_chunk_transformed(
        xs_out: &mut [__m256], ys_out: &mut [__m256], zs_out: &mut [__m256], ws_out: &mut [__m256],
        xs: &[__m256], ys: &[__m256], zs: &[__m256], ws: &[__m256], t: &Transformation) {
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

    let chunk_size = xs_out.len();
    for i in 0..chunk_size {
        let x = xs[i];
        let y = ys[i];
        let z = zs[i];
        let w = ws[i];

        let mut xh = _mm256_mul_ps(x, t00);
        let mut yh = _mm256_mul_ps(x, t10);
        let mut zh = _mm256_mul_ps(x, t20);
        let mut wh = _mm256_mul_ps(x, t30);
        
        xh = _mm256_fmadd_ps(y, t01, xh);
        yh = _mm256_fmadd_ps(y, t11, yh);
        zh = _mm256_fmadd_ps(y, t21, zh);
        wh = _mm256_fmadd_ps(y, t31, wh);
        
        xh = _mm256_fmadd_ps(z, t02, xh);
        yh = _mm256_fmadd_ps(z, t12, yh);
        zh = _mm256_fmadd_ps(z, t22, zh);
        wh = _mm256_fmadd_ps(z, t32, wh);

        xs_out[i] = _mm256_fmadd_ps(w, t03, xh);
        ys_out[i] = _mm256_fmadd_ps(w, t13, yh);
        zs_out[i] = _mm256_fmadd_ps(w, t23, zh);
        ws_out[i] = _mm256_fmadd_ps(w, t33, wh);
    }
}

pub unsafe fn vertices_chunk_to_cartesian(
    xs_in_out: &mut [__m256], ys_in_out: &mut [__m256], zs_in_out: &mut [__m256], ws_in_iws_out: &mut [__m256]) {
    let chunk_size = xs_in_out.len();
    for i in 0..chunk_size {
        let iw = _mm256_rcp_ps(ws_in_iws_out[i]);
        xs_in_out[i] = _mm256_mul_ps(xs_in_out[i], iw);
        ys_in_out[i] = _mm256_mul_ps(ys_in_out[i], iw);
        zs_in_out[i] = _mm256_mul_ps(zs_in_out[i], iw);
        ws_in_iws_out[i] = iw;
    }
}

macro_rules! dot_product {
    ($x0:expr, $y0:expr, $z0:expr, $x1:expr, $y1:expr, $z1:expr) => {{
        // x0*x1 + y0*y1 + z0*z1
        _mm256_fmadd_ps($x0, $x1, _mm256_fmadd_ps($y0, $y1, _mm256_mul_ps($z0, $z1)))
    }}
}

macro_rules! vectors_chunk_transformed_normalised {
    ($xs:expr, $ys:expr, $zs:expr, $it_t:expr, $closure:expr) => {{
        // transformations are stored in columns to benefit the simple compiled version; variables here are named row/column
        let t00 = _mm256_set1_ps($it_t[0][0]);
        let t01 = _mm256_set1_ps($it_t[1][0]);
        let t02 = _mm256_set1_ps($it_t[2][0]);

        let t10 = _mm256_set1_ps($it_t[0][1]);
        let t11 = _mm256_set1_ps($it_t[1][1]);
        let t12 = _mm256_set1_ps($it_t[2][1]);

        let t20 = _mm256_set1_ps($it_t[0][2]);
        let t21 = _mm256_set1_ps($it_t[1][2]);
        let t22 = _mm256_set1_ps($it_t[2][2]);

        let chunk_size = $xs.len();
        for i in 0..chunk_size {
            let x = $xs[i];
            let y = $ys[i];
            let z = $zs[i];
            
            let mut xt = _mm256_mul_ps(x, t00);
            let mut yt = _mm256_mul_ps(x, t10);
            let mut zt = _mm256_mul_ps(x, t20);
            
            xt = _mm256_fmadd_ps(y, t01, xt);
            yt = _mm256_fmadd_ps(y, t11, yt);
            zt = _mm256_fmadd_ps(y, t21, zt);
            
            xt = _mm256_fmadd_ps(z, t02, xt);
            yt = _mm256_fmadd_ps(z, t12, yt);
            zt = _mm256_fmadd_ps(z, t22, zt);

            // normalise
            let imag = _mm256_rsqrt_ps(dot_product!(xt, yt, zt, xt, yt, zt));
            xt = _mm256_mul_ps(xt, imag);
            yt = _mm256_mul_ps(yt, imag);
            zt = _mm256_mul_ps(zt, imag);

            $closure(i, xt, yt, zt);
        }
    }}
}

