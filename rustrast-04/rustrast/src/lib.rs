use std::{fs::*, path::*, sync::*, iter};
use simd::CoordinateComponents;
use windows::Win32::Graphics::Gdi::*;

mod time;
mod maths;
mod obj;
mod simd;
mod threadpool;

use time::*;
use maths::*;
use obj::*;
use simd::*;

static MODEL: OnceLock<Model> = OnceLock::new();

static mut TRANSFORMED_COORDS: OnceLock<[CoordinateComponents; 3]> = OnceLock::new();

pub unsafe fn init() {
    let model = read_obj(File::open(Path::new("src/DinklageLikenessSculpt.obj")).unwrap());
    let n = model.num_vertices;

    // NB: OnceLock uses the contained type as the error type in its Result, meaning most niceties
    // like unwrap require the contained type to derive Debug
    if MODEL.set(model).is_err() {
        panic!("Couldn't set OnceLock");
    }

    if TRANSFORMED_COORDS.set([
        CoordinateComponents::from_iter(iter::repeat(0f32).take(n)),
        CoordinateComponents::from_iter(iter::repeat(0f32).take(n)),
        CoordinateComponents::from_iter(iter::repeat(0f32).take(n))
    ]).is_err() {
        panic!("Couldn't set OnceLock");
    }
}

static mut ROTATION: f32 = 0.0;
const ROTATION_STEP: f32 = 0.05;
const ROTATION_MAX: f32 = std::f32::consts::TAU;

pub unsafe fn draw(buffer: *mut RGBQUAD, width: u16, height: u16) {
    let model = MODEL.get().unwrap();

    // animate by rotating the model
    let world = Transformation::rotate_y(ROTATION);
    ROTATION = (ROTATION + ROTATION_STEP) % ROTATION_MAX;
    
    // place the camera above the model's head and look down 30 degrees
    let eye = CartesianCoordinates {x: 0.0, y: 1.0, z: 2.0};
    let centre = CartesianCoordinates {x: 0.0, y: 0.0, z: 0.0};
    let up = CartesianVector {x: 0.0, y: 1.0, z: -0.5};
    let view = Transformation::look_at_rh(&eye, &centre, &up);

    // make the canonical view volume big enough to hold the model and a bit more
    let aspect = height as f32 / width as f32;
    let view_volume_width = 0.4;
    let view_volume_height = view_volume_width * aspect;
    let near = 2.0;
    let far = near + 0.5;
    let projection = Transformation::perspective_rh(view_volume_width, view_volume_height, near, far);

    let viewport = Transformation::viewport(0, 0, width, height);

    let t = world.then(&view).then(&projection).then(&viewport);

    let n = model.num_vertices;
    let mut vs_out = TRANSFORMED_COORDS.get_mut().unwrap();

    time(format!("Transformed {} vertices", n), || simd_transformed_to_cartesian(&mut vs_out, &model, &t));

    let vertices = time(format!("Sorted {} vertices", n), || {
        let mut vertices: Vec<CartesianCoordinates> = (0..n).map(|i| CartesianCoordinates {x: vs_out[0][i], y: vs_out[1][i], z: vs_out[2][i]}).collect();
        vertices.sort_unstable_by(|a, b| b.z.total_cmp(&a.z));
        vertices
    });

    for v in vertices {
        if v.x >= 0.0 && v.x < (width as f32) && v.y >= 0.0 && v.y < (height as f32) && v.z >= 0.0 && v.z < 1.0 {
            let intensity =  255 - (256.0 * v.z) as u8;
            *buffer.offset(((v.y) as isize * (width as isize)) + (v.x as isize)) = RGBQUAD{rgbRed: intensity, rgbGreen: intensity, rgbBlue: intensity, rgbReserved: 0};
        }
    }
}
