extern crate winres;

fn main() {
    let mut res = winres::WindowsResource::new();
    res.set_manifest_file("src/app.manifest");
    res.compile().unwrap();
}