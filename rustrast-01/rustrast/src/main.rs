use windows::{
    core::*,
    Win32::{
        Foundation::*,
        System::LibraryLoader::*,
        UI::WindowsAndMessaging::*,
        Graphics::Gdi::*,
    }
};

fn main() {
    // Register the window class.
    let h_instance = unsafe { GetModuleHandleW(PCWSTR::null()).unwrap() };
    let class_name = w!("rustrast");

    let wc = WNDCLASSW {
        style: CS_OWNDC | CS_HREDRAW | CS_VREDRAW,
        lpfnWndProc: Some(window_proc),
        hInstance: h_instance,
        lpszClassName: class_name,
        cbClsExtra: 0,
        cbWndExtra: 0,
        hIcon: HICON(0),
        hCursor: unsafe { LoadCursorW(HMODULE(0), IDC_ARROW).unwrap() },
        hbrBackground: HBRUSH(0),
        lpszMenuName: PCWSTR::null(),
    };

    unsafe { RegisterClassW(&wc) };

    // Create the window.

    let hwnd = unsafe { CreateWindowExW(
        WINDOW_EX_STYLE(0), // Optional window styles.
        class_name, // Window class
        w!("rustrast"), // Window text
        WS_OVERLAPPEDWINDOW,    // Window style

        // Size and position
        CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,

        HWND(0),    // Parent window    
        HMENU(0),   // Menu
        h_instance, // Instance handle
        None    // Additional application data
    ) };

    if hwnd == HWND(0) {
        panic!("Failed to create window.");
    }

    unsafe { ShowWindow(hwnd, SW_SHOW) };

    // Run the message loop.

    let mut msg = MSG::default();
    while unsafe { GetMessageW(&mut msg, HWND(0), 0, 0).as_bool() } {
        unsafe {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
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
