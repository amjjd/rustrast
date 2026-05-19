use std::{mem::*, ptr::*, sync::*};
use core::ffi::*;
use windows::{
    core::*,
    Win32::{
        Foundation::*,
        System::LibraryLoader::*,
        UI::WindowsAndMessaging::*,
        Graphics::Gdi::*,
    }
};
use lazy_static::*;

use rustrast::*;

mod time;
use time::*;

fn main() -> Result<()> {
    unsafe {
        // Register the window class.
        let h_instance = GetModuleHandleW(None)?;
        let class_name = w!("rustrast");

        let wc = WNDCLASSW {
            style: CS_OWNDC | CS_HREDRAW | CS_VREDRAW,
            lpfnWndProc: Some(window_proc),
            hInstance: h_instance,
            lpszClassName: class_name,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };

        RegisterClassW(&wc);

        // Create the window.

        let hwnd = CreateWindowExW(
            WINDOW_EX_STYLE::default(), // Optional window styles.
            class_name, // Window class
            w!("rustrast"), // Window text
            WS_OVERLAPPEDWINDOW,    // Window style

            // Size and position
            CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,

            None,    // Parent window    
            None,   // Menu
            h_instance, // Instance handle
            None    // Additional application data
        );

        if hwnd.0 == 0 {
            panic!("Failed to create window.");
        }

        ShowWindow(hwnd, SW_SHOW);

        // Run the message loop.

        let mut msg = MSG::default();
        while GetMessageW(&mut msg, HWND(0), 0, 0).into() {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }

        Ok(())
    }
}

struct BackBuffer {
    client_area_width: usize,
    client_area_height: usize,

    dc: CreatedHDC,
    width: usize,
    height: usize,
    bitmap: HBITMAP,
    buffer: *mut RGBQUAD
}

unsafe impl Send for BackBuffer {}

lazy_static! {
    static ref BACK_BUFFER: Mutex<BackBuffer> = Mutex::new(BackBuffer{
        client_area_width: 0,
        client_area_height: 0,
        dc: CreatedHDC(0),
        width: 0,
        height: 0,
        bitmap: HBITMAP(0),
        buffer: null_mut()
    });
}

static BG: RGBQUAD = RGBQUAD { rgbRed: 0, rgbBlue: 0, rgbGreen: 0, rgbReserved: 0 };

unsafe extern "system" fn window_proc(hwnd: HWND, msg: u32, w_param: WPARAM, l_param: LPARAM) -> LRESULT {
    match msg {
        WM_CREATE => {
            time(format!("initialised"), || {init()});
            LRESULT(0)
        }

        WM_DESTROY => {
            PostQuitMessage(0);
            LRESULT(0)
        }

        WM_SIZE => {
            // we get WM_SIZE before the initial paint so we can create the back buffer here
            let mut back_buffer = BACK_BUFFER.lock().unwrap();

            if back_buffer.bitmap.0 != 0 {
                DeleteObject(back_buffer.bitmap);
            }
            if back_buffer.dc.0 != 0 {
                DeleteDC(back_buffer.dc);
            }

            back_buffer.dc = CreateCompatibleDC(None);

            // so the start of each row is aligned for easier SIMD
            back_buffer.client_area_width = (l_param.0 & 0xffff) as usize;
            back_buffer.client_area_height = ((l_param.0 >> 16) & 0xffff) as usize;
            back_buffer.width = ((back_buffer.client_area_width + BACK_BUFFER_ALIGNMENT - 1) / BACK_BUFFER_ALIGNMENT) * BACK_BUFFER_ALIGNMENT;
            back_buffer.height = back_buffer.client_area_height;

            let bitmap_info = BITMAPINFO {
                bmiHeader: BITMAPINFOHEADER {
                    biSize: size_of::<BITMAPINFOHEADER>() as u32,
                    biWidth: back_buffer.width as i32,
                    biHeight: -(back_buffer.height as i32),
                    biPlanes: 1,
                    biBitCount: 32,
                    biCompression: BI_RGB.0 as u32,                    
                    ..Default::default()
                },
                ..Default::default()
            };

            // CreateDIBSection seems to return a 4K page aligned buffer; for production code
            // it would be better to allocate our own and use SetDIBitsToDevice or StretchDIBits
            // instead of BitBlt
            let mut bits: *mut c_void = null_mut();
            back_buffer.bitmap = CreateDIBSection(
                back_buffer.dc,
                &bitmap_info,
                DIB_RGB_COLORS,
                &mut bits,
                None, 
                0
            ).unwrap();
            back_buffer.buffer = bits as *mut RGBQUAD;

            SelectObject(back_buffer.dc, back_buffer.bitmap);

            LRESULT(0)
        }

        WM_PAINT => {
            let mut ps = PAINTSTRUCT::default();
            let hdc = BeginPaint(hwnd, &mut ps);

            // clear the buffer
            let back_buffer = BACK_BUFFER.lock().unwrap();
            let buffer_slice = std::slice::from_raw_parts_mut(back_buffer.buffer, back_buffer.width * back_buffer.height);
            time(format!("cleared {}x{}", back_buffer.width + 0, back_buffer.height + 0), || buffer_slice.fill(BG));

            // draw
            time(format!("drew {}x{}", back_buffer.client_area_width, back_buffer.client_area_height), || {
                draw(back_buffer.buffer, back_buffer.client_area_width, back_buffer.client_area_height, back_buffer.width)
            });

            // copy to screen
            time(format!("BitBlted {}x{}", back_buffer.client_area_width, back_buffer.client_area_height), || {
                BitBlt(
                    hdc,
                    0, 0,
                    back_buffer.client_area_width as i32, back_buffer.client_area_height as i32,
                    back_buffer.dc,
                    0, 0,
                    SRCCOPY
                );
            });

            EndPaint(hwnd, &ps);

            // paint the full window again as soon as we can
            InvalidateRect(hwnd, None, FALSE);
        
            LRESULT(0)
        }

        WM_DPICHANGED => {
            let rect = &*(l_param.0 as *const RECT);

            SetWindowPos(
                hwnd,
                None,
                rect.left,
                rect.top,
                rect.right - rect.left,
                rect.bottom - rect.top,
                SWP_NOZORDER | SWP_NOACTIVATE
            );

            LRESULT(0)
        }

        _ => DefWindowProcW(hwnd, msg, w_param, l_param),
    }
}
