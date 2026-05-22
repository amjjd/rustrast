use std::{fs::*, path::*, sync::*, cell::*, iter, array};
use windows::Win32::Graphics::Gdi::*;
use once_cell::sync::Lazy;
use scoped_threadpool::Pool;

mod time;
mod simd_vec;
mod obj;
mod transformation;
mod rasterisation;

use time::*;
use simd_vec::*;
use obj::*;
use transformation::*;
use rasterisation::*;

// used by main to ensure the buffer is big enough for whatever SIMD operations we use
pub const BACK_BUFFER_ALIGNMENT: usize = 8;

const TILE_WIDTH: usize = 128; // must be a multiple of BACK_BUFFER_ALIGNMENT
const TILE_HEIGHT: usize = 128;

// my machine stops showing improvement above 4 threads
static NUM_BIN_THREADS: usize = 4;

// more hackery to avoid managing memory; these are all initialised based on the loaded model
struct SceneBuffers {
    model: Model,
    rotation: Cell<f32>,
    xs: RefCell<SimdVec<f32>>,
    ys: RefCell<SimdVec<f32>>,
    zs: RefCell<SimdVec<f32>>,
    xmins: RefCell<SimdVec<f32>>,
    ymins: RefCell<SimdVec<f32>>,
    xmaxs: RefCell<SimdVec<f32>>,
    ymaxs: RefCell<SimdVec<f32>>,
    // for each binning thread, each tile has a list of triangles
    tile_triangles: RefCell<[Vec<Vec<u32>>; NUM_BIN_THREADS]>
}

static SCENE: OnceLock<Mutex<SceneBuffers>> = OnceLock::new();

pub fn init() {
    let model = read_obj(File::open(Path::new("src/DinklageLikenessSculpt.obj")).unwrap());
    let num_vertices = model.num_vertices as usize;
    let num_triangles = model.num_triangles as usize;

    let scene = SceneBuffers {
        model,
        rotation: Cell::new(0.0),
        xs: RefCell::new(iter::repeat(0f32).take(num_vertices).collect()),
        ys: RefCell::new(iter::repeat(0f32).take(num_vertices).collect()),
        zs: RefCell::new(iter::repeat(0f32).take(num_vertices).collect()),
        xmins: RefCell::new(iter::repeat(0f32).take(num_triangles).collect()),
        ymins: RefCell::new(iter::repeat(0f32).take(num_triangles).collect()),
        xmaxs: RefCell::new(iter::repeat(0f32).take(num_triangles).collect()),
        ymaxs: RefCell::new(iter::repeat(0f32).take(num_triangles).collect()),
        tile_triangles: RefCell::new(array::from_fn(|_| Vec::new()))
    };

    let _ = SCENE.set(Mutex::new(scene));
}

fn scene_buffers() -> &'static Mutex<SceneBuffers> {
    SCENE.get().unwrap()
}

static BIN_WORKERS: Lazy<Mutex<Pool>> = Lazy::new(|| Mutex::new(Pool::new(NUM_BIN_THREADS as u32)));

fn bin_triangles(tile_triangles_out: &mut [Vec<Vec<u32>>; NUM_BIN_THREADS], num_triangles: u32, bounds: [&SimdVec<f32>; 4], num_tiles: usize, num_tiles_x: usize) {
    // this should only allocate heavily during the first few frames
    for i in 0..NUM_BIN_THREADS {
        if tile_triangles_out[i].len() > num_tiles {
            tile_triangles_out[i].truncate(num_tiles);
        }
        else {
            // somewhat pessimistic guess
            let initial_capacity = (num_triangles as usize / num_tiles) * 4;
            for _ in tile_triangles_out[i].len()..num_tiles {
                tile_triangles_out[i].push(Vec::with_capacity(initial_capacity));
            }
        }

        for j in 0..num_tiles {
            // doesn't affect capacity
            tile_triangles_out[i][j].truncate(0);
        }
    }

    let num_chunks = NUM_BIN_THREADS as u32;
    let chunk_size = (num_triangles + num_chunks - 1) / num_chunks;

    let mut pool = BIN_WORKERS.lock().unwrap();
    pool.scoped(|scope| {
        let out_chunks = tile_triangles_out.chunks_mut(1);

        let mut chunk_start = 0;
        for out_chunk in out_chunks {
            let out = &mut out_chunk[0];
            let start = chunk_start;
            scope.execute(move || {
                for i in start..((start + chunk_size).min(num_triangles)) {
                    if bounds[2][i as usize] < 0.0 {
                        // backwards-facing
                        continue;
                    }

                    // a triangle is in the tile(s) each of the corners of its bounding box is in
                    let left = bounds[0][i as usize] as usize / TILE_WIDTH;
                    let top = bounds[1][i as usize] as usize / TILE_HEIGHT;
                    // bounds are integers, so casting is OK
                    let right = bounds[2][i as usize] as usize / TILE_WIDTH;
                    let bottom = bounds[3][i as usize] as usize / TILE_HEIGHT;

                    let tl = (top * num_tiles_x) + left;
                    out[tl].push(i);

                    if right != left {
                        let tr = (top * num_tiles_x) + right;
                        out[tr].push(i);
                    }

                    if bottom != top {
                        let bl = (bottom * num_tiles_x) + left;
                        out[bl].push(i);
                    }

                    if right != left || bottom != top {
                        let br = (bottom * num_tiles_x) + right;
                        out[br].push(i);
                    }
                }
            });

            chunk_start += chunk_size;
        }
    });
}

// enables bypassing safeness checks when multithreading
struct Tile {
    buffer: *mut RGBQUAD,
    stride: usize,
    xmin: usize,
    ymin: usize,
    xmax: usize,
    ymax: usize,
}

unsafe impl Send for Tile {}

// my machine stops showing improvement above 4 threads
static NUM_DRAW_THREADS: u32 = 4;
static DRAW_WORKERS: Lazy<Mutex<Pool>> = Lazy::new(|| Mutex::new(Pool::new(NUM_DRAW_THREADS)));

fn draw_tile(tile: Tile, model: &Model, xs: &SimdVec<f32>, ys: &SimdVec<f32>, bounds: [&SimdVec<f32>; 4], triangles: [&Vec<u32>; NUM_BIN_THREADS]) {
    let tile_xmin = tile.xmin as f32;
    let tile_ymin = tile.ymin as f32;
    let tile_xmax = tile.xmax as f32;
    let tile_ymax = tile.ymax as f32;
    let xmins = &bounds[0];
    let ymins = &bounds[1];
    let xmaxs = &bounds[2];
    let ymaxs = &bounds[3];

    for i in 0..NUM_BIN_THREADS {
        for j in 0..triangles[i].len() {
            let it = triangles[i][j] as usize;
            let mut xmin = xmins[it];
            let mut ymin = ymins[it];
            let mut xmax = xmaxs[it];
            let mut ymax = ymaxs[it];

            // clip to the tile
            xmin = xmin.max(tile_xmin);
            ymin = ymin.max(tile_ymin);
            xmax = xmax.min(tile_xmax);
            ymax = ymax.min(tile_ymax);

            let colour = RGBQUAD {rgbRed: ((it + 128) % 255) as u8, rgbGreen: ((it + 128) % 255) as u8, rgbBlue: ((it + 128) % 255) as u8, rgbReserved: 0};
            let x0 = xs[model.trianglev0s[it] as usize];
            let y0 = ys[model.trianglev0s[it] as usize];
            let x1 = xs[model.trianglev1s[it] as usize];
            let y1 = ys[model.trianglev1s[it] as usize];
            let x2 = xs[model.trianglev2s[it] as usize];
            let y2 = ys[model.trianglev2s[it] as usize];
            
            unsafe {
                avx2_fill_triangle(tile.buffer, tile.stride, xmin, ymin, xmax, ymax, x0, y0, x1, y1, x2, y2, colour);
            }
        }
    }
}

const ROTATION_STEP: f32 = 0.05;
const ROTATION_MAX: f32 = std::f32::consts::TAU;

pub fn draw(buffer: *mut RGBQUAD, width: usize, height: usize, stride: usize) {
    let scene = scene_buffers().lock().unwrap();
    let model = &scene.model;

    // animate by rotating the model
    let rotation = scene.rotation.get();
    let world = Transformation::rotate_y(rotation);
    scene.rotation.set((rotation + ROTATION_STEP) % ROTATION_MAX);
    
    // place the camera above the model's head and look down 30 degrees
    let eye = CartesianCoordinates {x: 0.0, y: 1.0, z: 2.0};
    let centre = CartesianCoordinates {x: 0.0, y: 0.0, z: 0.0};
    let up = CartesianVector {x: 0.0, y: 1.0, z: -0.5};
    let view = Transformation::look_at_rh(&eye, &centre, &up);

    // make the canonical view volume big enough to hold the model and a bit more
    let aspect = height as f32 / width as f32;
    let view_volume_width = 0.4;
    let view_volume_height = view_volume_width * aspect;
    let near = 2.0;
    let far = near + 0.5;
    let projection = Transformation::perspective_rh(view_volume_width, view_volume_height, near, far);

    let viewport = Transformation::viewport(0, 0, width, height);

    let t = world.then(&view).then(&projection).then(&viewport);

    let num_vertices = model.num_vertices;
    let num_triangles = model.num_triangles;

    {
        let xs_out = &mut *scene.xs.borrow_mut();
        let ys_out = &mut *scene.ys.borrow_mut();
        let zs_out = &mut *scene.zs.borrow_mut();
        time(format!("Transformed {} vertices", num_vertices), || {
            avx2_transformed_to_cartesian(xs_out, ys_out, zs_out, model, &t)
        });
    }
    let xs = &*scene.xs.borrow();
    let ys = &*scene.ys.borrow();

    {
        let xmins_out = &mut *scene.xmins.borrow_mut();
        let ymins_out = &mut *scene.ymins.borrow_mut();
        let xmaxs_out = &mut *scene.xmaxs.borrow_mut();
        let ymaxs_out = &mut *scene.ymaxs.borrow_mut();
        time(format!("Calculated {} bounding boxes", num_triangles), || {
            avx2_calculate_all_bounds(xmins_out, ymins_out,xmaxs_out, ymaxs_out, model, xs, ys, 0.0, 0.0, width as f32, height as f32)
        });
    }
    let bounds = [&*scene.xmins.borrow(), &*scene.ymins.borrow(), &*scene.xmaxs.borrow(), &*scene.ymaxs.borrow()];

    let num_tiles_x = (stride + TILE_WIDTH - 1) / TILE_WIDTH;
    let num_tiles = num_tiles_x * ((height + TILE_HEIGHT - 1) / TILE_HEIGHT);

    {
        let tile_triangles_out = &mut *scene.tile_triangles.borrow_mut();
        time(format!("Binned {} triangles", num_triangles), || {
            bin_triangles(tile_triangles_out, num_triangles, bounds, num_tiles, num_tiles_x);
        });
    }
    let tile_triangles = scene.tile_triangles.borrow();

    time("Filled triangles", || {
        let mut pool = DRAW_WORKERS.lock().unwrap();
        pool.scoped(|scope| {
            let mut ymin = 0;
            let mut i_tile = 0;
            
            while ymin < height  {
                let mut xmin = 0;
                while xmin < stride {
                    let tile = Tile {
                        buffer: buffer,
                        stride,
                        xmin,
                        ymin,
                        xmax: (xmin + TILE_WIDTH - 1).min(stride),
                        ymax: (ymin + TILE_HEIGHT - 1).min(height)
                    };

                    let triangles = array::from_fn(|i| &tile_triangles[i][i_tile]);

                    scope.execute(move || {
                        draw_tile(tile, model, xs, ys, bounds, triangles);
                    });

                    xmin += TILE_WIDTH;
                    i_tile += 1;
                }

                ymin += TILE_HEIGHT;
            }
        });
    });
}
