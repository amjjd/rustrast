use windows::Win32::System::Performance::*;
use std::{collections::{HashMap, VecDeque}, sync::RwLock};
use lazy_static::*;

unsafe fn get_ticks_per_ms() -> f64 {
    let mut ticks_per_second: i64 = 0;
    QueryPerformanceFrequency(&mut ticks_per_second);
    1000.0 / (ticks_per_second as f64)
}

lazy_static! {
    static ref TICKS_TO_MS: f64 = unsafe { get_ticks_per_ms() };
}

lazy_static! {
    static ref TIMES: RwLock<HashMap<String, VecDeque<i64>>> = RwLock::new(HashMap::new());
}

pub fn time<S: AsRef<str>, T, F: FnOnce() -> T> (desc: S, f: F) -> T {
    let (start, end, ret) = time_silently(f);
    println!("{:.2}-{:.2}: {} in {:.2}ms", start as f64 * *TICKS_TO_MS, end as f64 * *TICKS_TO_MS, {desc.as_ref()}, ((end - start) as f64) * *TICKS_TO_MS);
    let mut all_times = TIMES.write().unwrap();
    let times = all_times.entry(desc.as_ref().to_string()).or_insert(VecDeque::with_capacity(100));
    if times.len() == 100 {
        times.pop_front();
    }
    times.push_back(end - start);
    ret
}

pub fn print_average_times() {
    let all_times = TIMES.read().unwrap();
    for (desc, times) in all_times.iter() {
        let total: i64 = times.iter().sum();
        let average = total as f64 / (times.len() as f64);
        println!("{} in {:.2}ms on average for the last {} runs", desc, average * *TICKS_TO_MS, times.len());
    }
}

pub fn time_silently<T, F: FnOnce() -> T> (f: F) -> (i64, i64, T) {
    let start = timestamp();
    let ret = f();
    let end = timestamp();
    (start, end, ret)
}

fn timestamp() -> i64 {
    let mut ts: i64 = 0;
    unsafe {
        QueryPerformanceCounter(&mut ts);
    }
    ts
}