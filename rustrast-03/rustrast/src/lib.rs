use std::{fs::*, path::*, sync::*};
use windows::Win32::{System::Performance::*, System::SystemInformation::*, Graphics::Gdi::*};

mod obj;
use obj::*;

static MODEL: OnceLock<Model> = OnceLock::new();

pub fn init() {
    // NB: OnceLock uses the contained type as the error type in its Result, meaning most niceties
    // like unwrap require the contained type to derive Debug
    if MODEL.set(read_obj(File::open(Path::new("src/DinklageLikenessSculpt.obj")).unwrap())).is_err() {
        panic!("Couldn't set OnceLock");
    }
}

static mut PULSE: f32 = 1.0;
static mut PULSE_STEP: f32 = 0.05;

pub unsafe fn draw(buffer: *mut RGBQUAD, width: u16, height: u16) {
    let model = MODEL.get().unwrap();

    // sort vertices by z, so we can draw back-to-front and colour by depth
    let vertices = time(format!("Sorted {} vertices", model.vertices.len()), || {
        let mut vertices = model.vertices.clone();
        vertices.sort_unstable_by(|a, b| a.z.total_cmp(&b.z));
        vertices
    });

    // fit the larger of the model's width and height onto the smaller of the screen's
    let model_width = model.max_x - model.min_x;
    let model_height = model.max_y - model.min_y;
    let scale = u16::min(width, height) as f32 / f32::max(model_width, model_height) * PULSE;

    // pulse the scale to get some animation
    PULSE = PULSE + PULSE_STEP;
    if (PULSE_STEP > 0.0 && PULSE > 1.25) || (PULSE_STEP < 0.0 && PULSE < 0.75) {
        PULSE_STEP = -PULSE_STEP;
    }

    // centre the model's extents (so its origin may not be centred on screen)
    let offset_x = (0.0 - model.min_x) * scale + ((width as f32 - (model_width * scale)) / 2.0);
    let offset_y = (0.0 - model.min_y) * scale + ((height as f32 - (model_height * scale)) / 2.0);

    // pixels go from dark grey to near-white based on z
    let intensity_scale = 200.0 / (model.max_z - model.min_z);

    for vertex in vertices {
        let x = f32::round((vertex.x * scale) + offset_x) as isize;
        let y = f32::round((vertex.y * scale) + offset_y) as isize;
        if x >= 0 && x < (width as isize) && y >= 0 && y < (height as isize) {
            let intensity = ((vertex.z - model.min_z) * intensity_scale) as u8;
            *buffer.offset((y * (width as isize)) + x) = RGBQUAD{rgbRed: intensity, rgbGreen: intensity, rgbBlue: intensity, rgbReserved: 0};
        }
    }
}

pub unsafe fn time<S: AsRef<str>, T, F: FnOnce() -> T> (desc: S, f: F) -> T {
    let mut start: i64 = 0;
    QueryPerformanceCounter(&mut start);

    let ret = f();

    let mut end: i64 = 0;
    QueryPerformanceCounter(&mut end);

    let mut ticks_per_second: i64 = 0;
    QueryPerformanceFrequency(&mut ticks_per_second);
    let ticks_to_ms = 1000.0 / (ticks_per_second as f64);

    println!("{}: {} in {:.2}ms", GetTickCount(), {desc.as_ref()}, ((end - start) as f64) * ticks_to_ms);

    ret
}