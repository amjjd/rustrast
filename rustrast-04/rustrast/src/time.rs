use windows::Win32::System::Performance::*;
use lazy_static::*;

unsafe fn get_ticks_per_ms() -> f64 {
    let mut ticks_per_second: i64 = 0;
    QueryPerformanceFrequency(&mut ticks_per_second);
    1000.0 / (ticks_per_second as f64)
}

lazy_static! {
    pub static ref TICKS_TO_MS: f64 = unsafe { get_ticks_per_ms() };
}

pub fn time<S: AsRef<str>, T, F: FnOnce() -> T> (desc: S, f: F) -> T {
    let (start, end, ret) = time_silently(f);
    println!("{:.2}-{:.2}: {} in {:.2}ms", start as f64 * *TICKS_TO_MS, end as f64 * *TICKS_TO_MS, {desc.as_ref()}, ((end - start) as f64) * *TICKS_TO_MS);
    ret
}

pub fn time_silently<T, F: FnOnce() -> T> (f: F) -> (i64, i64, T) {
    let start = timestamp();
    let ret = f();
    let end = timestamp();
    (start, end, ret)
}

pub fn timestamp() -> i64 {
    let mut ts: i64 = 0;
    unsafe {
        QueryPerformanceCounter(&mut ts);
    }
    ts
}