// fundamental 3D maths utilities
use std::ops;

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
}

impl ops::Add<CartesianVector> for CartesianVector {
    type Output = CartesianVector;

    fn add(self, other: CartesianVector) -> Self {
        CartesianVector {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z
        }
    }
}

impl ops::Sub<CartesianVector> for CartesianVector {
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
    pub fn to_homogenous(self) -> HomogenousCoordinates {
        HomogenousCoordinates {
            x: self.x,
            y: self.y,
            z: self.z,
            w: 1.0
        }
    }
}

impl ops::Sub<CartesianCoordinates> for CartesianCoordinates {
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
    #[target_feature(enable = "fma,avx2")]
    pub unsafe fn transformed(&self, t: &Transformation) -> Self {
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

    pub fn scale(sx: f32, sy: f32, sz: f32) -> Self {
        Transformation { matrix: [
            [ sx, 0.0, 0.0, 0.0], 
            [0.0,  sy, 0.0, 0.0],
            [0.0, 0.0,  sz, 0.0],
            [0.0, 0.0, 0.0, 1.0]],
            _private: ()
        }
    }

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

    pub fn viewport(x: u16, y: u16, width: u16, height: u16) -> Self {
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