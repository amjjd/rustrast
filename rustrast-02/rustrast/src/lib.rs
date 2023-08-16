use windows::Win32::Graphics::Gdi::*;

static mut START: usize = 0;
static BLACK: RGBQUAD = RGBQUAD { rgbRed: 255, rgbBlue: 255, rgbGreen: 255, rgbReserved: 0 };
static WHITE: RGBQUAD = RGBQUAD { rgbRed: 0, rgbBlue: 0, rgbGreen: 0, rgbReserved: 0 };

pub unsafe fn draw(buffer: *mut RGBQUAD, width: u16, height: u16) {
    if START == 0 {
        for offset in 0..((width as isize) * (height as isize) / 2) {
            *buffer.offset(offset * 2) = BLACK;
            *buffer.offset(offset * 2 + 1) = WHITE;
        }
    }
    else {
        for offset in 0..((width as isize) * (height as isize) / 2) {
            *buffer.offset(offset * 2) = WHITE;
            *buffer.offset(offset * 2 + 1) = BLACK;
        }
    }

    START = (START + 1) % 2;
}