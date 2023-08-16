use std::mem::*;
use std::ptr::*;
use core::ffi::*;
use windows::{
    core::*,
    Win32::{
        Foundation::*,
        System::LibraryLoader::*,
        System::Performance::*,
        System::SystemInformation::*,
        UI::WindowsAndMessaging::*,
        Graphics::Gdi::*,
    }
};
use rustrast::*;

static mut TICKS_TO_MS: f64 = 0.0;

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

        let mut ticks_per_second: i64 = 0;
        QueryPerformanceFrequency(&mut ticks_per_second);
        TICKS_TO_MS = 1000.0 / (ticks_per_second as f64);

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

static mut BACK_BUFFER_DC: CreatedHDC = CreatedHDC(0);
static mut BACK_BUFFER_WIDTH: u16 = 0;
static mut BACK_BUFFER_HEIGHT: u16 = 0;
static mut BACK_BUFFER_BITMAP: HBITMAP = HBITMAP(0);
static mut BACK_BUFFER: *mut RGBQUAD = null_mut();

unsafe extern "system" fn window_proc(hwnd: HWND, msg: u32, w_param: WPARAM, l_param: LPARAM) -> LRESULT {
    match msg {
        WM_DESTROY => {
            PostQuitMessage(0);
            LRESULT(0)
        }

        WM_SIZE => {
            // we get WM_SIZE before the initial paint so we can create the back buffer here

            if BACK_BUFFER_BITMAP.0 != 0 {
                DeleteObject(BACK_BUFFER_BITMAP);
            }
            if BACK_BUFFER_DC.0 != 0 {
                DeleteDC(BACK_BUFFER_DC);
            }

            BACK_BUFFER_DC = CreateCompatibleDC(None);

            BACK_BUFFER_WIDTH = (l_param.0 & 0xffff) as u16;
            BACK_BUFFER_HEIGHT = ((l_param.0 >> 16) & 0xffff) as u16;

            let bitmap_info = BITMAPINFO {
                bmiHeader: BITMAPINFOHEADER {
                    biSize: size_of::<BITMAPINFOHEADER>() as u32,
                    biWidth: BACK_BUFFER_WIDTH as i32,
                    biHeight: BACK_BUFFER_HEIGHT as i32,
                    biPlanes: 1,
                    biBitCount: 32,
                    biCompression: BI_RGB.0 as u32,                    
                    ..Default::default()
                },
                ..Default::default()
            };

            let mut bits: *mut c_void = null_mut();
            BACK_BUFFER_BITMAP = CreateDIBSection(
                BACK_BUFFER_DC,
                &bitmap_info,
                DIB_RGB_COLORS,
                &mut bits,
                None, 
                0
            ).unwrap();
            BACK_BUFFER = bits as *mut RGBQUAD;

            SelectObject(BACK_BUFFER_DC, BACK_BUFFER_BITMAP);

            LRESULT(0)
        }

        WM_PAINT => {
            let mut ps = PAINTSTRUCT::default();
            let hdc = BeginPaint(hwnd, &mut ps);

            let mut start: i64 = 0;
            QueryPerformanceCounter(&mut start);

            draw(BACK_BUFFER, BACK_BUFFER_WIDTH, BACK_BUFFER_HEIGHT);

            let mut end: i64 = 0;
            QueryPerformanceCounter(&mut end);

            println!("{}: drew {}x{} in {:.2}ms", GetTickCount(), BACK_BUFFER_WIDTH, BACK_BUFFER_HEIGHT, ((end - start) as f64) * TICKS_TO_MS);
            
            QueryPerformanceCounter(&mut start);

            BitBlt(
                hdc,
                0, 0,
                BACK_BUFFER_WIDTH as i32, BACK_BUFFER_HEIGHT as i32,
                BACK_BUFFER_DC,
                0, 0,
                SRCCOPY
            );

            QueryPerformanceCounter(&mut end);

            println!("{}: BitBlted {}x{} in {:.2}ms", GetTickCount(), BACK_BUFFER_WIDTH, BACK_BUFFER_HEIGHT, ((end - start) as f64) * TICKS_TO_MS);

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
