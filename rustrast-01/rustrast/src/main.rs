#![windows_subsystem = "windows"]

use windows::{
    core::*,
    Win32::{
        Foundation::*,
        System::LibraryLoader::*,
        UI::WindowsAndMessaging::*,
        Graphics::Gdi::*,
    }
};

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

unsafe extern "system" fn window_proc(hwnd: HWND, msg: u32, w_param: WPARAM, l_param: LPARAM) -> LRESULT {
    match msg {
        WM_DESTROY => {
            PostQuitMessage(0);
            LRESULT(0)
        }

        WM_PAINT => {
            let mut ps = PAINTSTRUCT::default();
            let hdc = BeginPaint(hwnd, &mut ps);

            // All painting occurs here, between BeginPaint and EndPaint.

            FillRect(hdc, &ps.rcPaint, GetSysColorBrush(COLOR_WINDOW));

            EndPaint(hwnd, &ps);
        
            LRESULT(0)
        }

        _ => DefWindowProcW(hwnd, msg, w_param, l_param),
    }
}
