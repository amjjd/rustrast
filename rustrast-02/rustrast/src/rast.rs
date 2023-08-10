use windows::Win32::Graphics::Gdi::*;

static mut START: usize = 0;
static PATTERN: [[RGBQUAD; 2]; 2] = [
    [RGBQUAD { rgbRed: 255, rgbBlue: 255, rgbGreen: 255, rgbReserved: 0 }, RGBQUAD { rgbRed: 0, rgbBlue: 0, rgbGreen: 0, rgbReserved: 0 }],
    [RGBQUAD { rgbRed: 0, rgbBlue: 0, rgbGreen: 0, rgbReserved: 0 }, RGBQUAD { rgbRed: 255, rgbBlue: 255, rgbGreen: 255, rgbReserved: 0 }]];

pub unsafe fn draw(buffer: *mut RGBQUAD, width: u16, height: u16) {
    let pattern64 = *((&PATTERN[START] as *const RGBQUAD) as *const u64);
    let buffer64 = buffer as *mut u64;

    for offset in 0..((width as isize) * (height as isize) / 2) {
        *buffer64.offset(offset) = pattern64;
    }

    START = (START + 1) % 2;
}