use std::{fs::*, path::*, sync::*, cell::*, slice::*, iter, array};
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
    iws: RefCell<SimdVec<f32>>,
    xmins: RefCell<SimdVec<f32>>,
    ymins: RefCell<SimdVec<f32>>,
    xmaxs: RefCell<SimdVec<f32>>,
    ymaxs: RefCell<SimdVec<f32>>,
    iareas: RefCell<SimdVec<f32>>,
    intensities: RefCell<SimdVec<u8>>,
    // for each binning thread, each tile has a list of triangles
    tile_triangles: RefCell<[Vec<Vec<u32>>; NUM_BIN_THREADS]>,
    depth: RefCell<Vec<f32>>
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
        iws: RefCell::new(iter::repeat(0f32).take(num_vertices).collect()),
        xmins: RefCell::new(iter::repeat(0f32).take(num_triangles).collect()),
        ymins: RefCell::new(iter::repeat(0f32).take(num_triangles).collect()),
        xmaxs: RefCell::new(iter::repeat(0f32).take(num_triangles).collect()),
        ymaxs: RefCell::new(iter::repeat(0f32).take(num_triangles).collect()),
        iareas: RefCell::new(iter::repeat(0f32).take(num_triangles).collect()),
        intensities: RefCell::new(iter::repeat(0u8).take(num_triangles).collect()),
        tile_triangles: RefCell::new(array::from_fn(|_| Vec::new())),
        depth: RefCell::new(Vec::new())
    };

    let _ = SCENE.set(Mutex::new(scene));
}

fn scene_buffers() -> &'static Mutex<SceneBuffers> {
    SCENE.get().unwrap()
}

static BIN_WORKERS: Lazy<Mutex<Pool>> = Lazy::new(|| Mutex::new(Pool::new(NUM_BIN_THREADS as u32)));

fn bin_triangles(tile_triangles_out: &mut [Vec<Vec<u32>>; NUM_BIN_THREADS], num_triangles: u32, bounds: [&SimdVec<f32>; 5], num_tiles: usize, num_tiles_x: usize) {
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
    let xmins = bounds[0];
    let ymins = bounds[1];
    let xmaxs = bounds[2];
    let ymaxs = bounds[3];
    let iareas = bounds[4];

    let mut pool = BIN_WORKERS.lock().unwrap();
    pool.scoped(|scope| {
        let mut chunk_start = 0;
        for out in tile_triangles_out.iter_mut() {
            let start = chunk_start;
            scope.execute(move || {
                for i in start..((start + chunk_size).min(num_triangles)) {
                    let it = i as usize;
                    if iareas[it] < 0.0 {
                        // cull backwards-facing triangles
                        continue;
                    }

                    // a triangle is in the tile(s) between each of the corners of its bounding box is in
                    let left = xmins[it] as usize / TILE_WIDTH;
                    let top = ymins[it] as usize / TILE_HEIGHT;
                    // bounds are integers, so casting is OK
                    let right = xmaxs[it] as usize / TILE_WIDTH;
                    let bottom = ymaxs[it] as usize / TILE_HEIGHT;

                    let mut row_start = top * num_tiles_x;
                    for _ in top..=bottom {
                        let l = row_start + left;
                        let r = row_start + right;
                        for t in l..=r {
                            out[t].push(i);
                        }

                        row_start += num_tiles_x;
                    }
                }
            });

            chunk_start += chunk_size;
        }
    });
}

// enables bypassing safeness checks when multithreading
struct Tile<'a> {
    colour: Buffer<'a, RGBQUAD>,
    depth: Buffer<'a, f32>,
    xmin: usize,
    ymin: usize,
    xmax: usize,
    ymax: usize,
}

// my machine stops showing improvement above 4 threads
static NUM_DRAW_THREADS: u32 = 4;
static DRAW_WORKERS: Lazy<Mutex<Pool>> = Lazy::new(|| Mutex::new(Pool::new(NUM_DRAW_THREADS)));

fn draw_tile(tile: &mut Tile, model: &Model, xs: &SimdVec<f32>, ys: &SimdVec<f32>, zs: &SimdVec<f32>, iws: &SimdVec<f32>, bounds: [&SimdVec<f32>; 5], intensities: &SimdVec<u8>, triangles: [&Vec<u32>; NUM_BIN_THREADS]) {
    let tile_xmin = tile.xmin as f32;
    let tile_ymin = tile.ymin as f32;
    let tile_xmax = tile.xmax as f32;
    let tile_ymax = tile.ymax as f32;
    let xmins = &bounds[0];
    let ymins = &bounds[1];
    let xmaxs = &bounds[2];
    let ymaxs = &bounds[3];
    let iareas = &bounds[4];

    for i in 0..NUM_BIN_THREADS {
        for j in 0..triangles[i].len() {
            let it = triangles[i][j] as usize;
            let mut xmin = xmins[it];
            let mut ymin = ymins[it];
            let mut xmax = xmaxs[it];
            let mut ymax = ymaxs[it];
            let iarea = iareas[it];

            // clip to the tile
            xmin = xmin.max(tile_xmin);
            ymin = ymin.max(tile_ymin);
            xmax = xmax.min(tile_xmax);
            ymax = ymax.min(tile_ymax);

            let x0 = xs[model.trianglev0s[it] as usize];
            let y0 = ys[model.trianglev0s[it] as usize];
            let z0 = zs[model.trianglev0s[it] as usize];
            let iw0 = iws[model.trianglev0s[it] as usize];
            let x1 = xs[model.trianglev1s[it] as usize];
            let y1 = ys[model.trianglev1s[it] as usize];
            let z1 = zs[model.trianglev1s[it] as usize];
            let iw1 = iws[model.trianglev1s[it] as usize];
            let x2 = xs[model.trianglev2s[it] as usize];
            let y2 = ys[model.trianglev2s[it] as usize];
            let z2 = zs[model.trianglev2s[it] as usize];
            let iw2 = iws[model.trianglev2s[it] as usize];

            let intensity = intensities[it];
            let colour = RGBQUAD {rgbRed: intensity, rgbGreen: intensity, rgbBlue: intensity, rgbReserved: 0};
            
            fill_triangle(&mut tile.colour, &mut tile.depth, xmin, ymin, xmax, ymax, x0, y0, z0, iw0, x1, y1, z1, iw1, x2, y2, z2, iw2, iarea, colour);
        }
    }
}

const ROTATION_STEP: f32 = 0.005;
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

    // one distant light source, coming from top right behind the camera
    let light_direction = CartesianVector {x: 1.0, y: 1.0, z: 1.0}.normalised();
    // to transform surface normals
    let it_world = world.inverted_transposed_tl_3x3().unwrap();
    let gamma = 2.2;

    let num_vertices = model.num_vertices;
    let num_triangles = model.num_triangles;

    {
        let xs_out = &mut *scene.xs.borrow_mut();
        let ys_out = &mut *scene.ys.borrow_mut();
        let zs_out = &mut *scene.zs.borrow_mut();
        let iws_out = &mut *scene.iws.borrow_mut();
        time(format!("Transformed {} vertices", num_vertices), || {
            avx2_transformed_to_cartesian(xs_out, ys_out, zs_out, iws_out, model, &t)
        });
    }
    let xs = &*scene.xs.borrow();
    let ys = &*scene.ys.borrow();
    let zs = &*scene.zs.borrow();
    let iws = &*scene.iws.borrow();

    time(format!("Lit {} triangles", num_triangles), || {
        let intensities_out = &mut *scene.intensities.borrow_mut();
        for i in 0..num_triangles as usize {
            let surface_normal = model.surface_normal(i as u32).transformed(&it_world).normalised();
            let diffuse = surface_normal.dot_product(&light_direction).max(0.0) * 0.3;
            let ambient = 0.05;
            let intensity = ((diffuse + ambient).min(1.0).powf(1.0 / gamma) * 255.0) as u8;
            intensities_out[i] = intensity;
        }
    });
    let intensities = &*scene.intensities.borrow();

    {
        let xmins_out = &mut *scene.xmins.borrow_mut();
        let ymins_out = &mut *scene.ymins.borrow_mut();
        let xmaxs_out = &mut *scene.xmaxs.borrow_mut();
        let ymaxs_out = &mut *scene.ymaxs.borrow_mut();
        let iareas_out = &mut *scene.iareas.borrow_mut();
        time(format!("Calculated {} bounding boxes", num_triangles), || {
            calculate_all_bounds(xmins_out, ymins_out,xmaxs_out, ymaxs_out, iareas_out, model, xs, ys, 0.0, 0.0, width as f32, height as f32)
        });
    }
    let bounds = [&*scene.xmins.borrow(), &*scene.ymins.borrow(), &*scene.xmaxs.borrow(), &*scene.ymaxs.borrow(), &*scene.iareas.borrow()];

    let num_tiles_x = (stride + TILE_WIDTH - 1) / TILE_WIDTH;
    let num_tiles = num_tiles_x * ((height + TILE_HEIGHT - 1) / TILE_HEIGHT);

    {
        let tile_triangles_out = &mut *scene.tile_triangles.borrow_mut();
        time(format!("Binned {} triangles", num_triangles), || {
            bin_triangles(tile_triangles_out, num_triangles, bounds, num_tiles, num_tiles_x);
        });
    }
    let tile_triangles = scene.tile_triangles.borrow();

    let depth = &mut *scene.depth.borrow_mut();
    time("Cleared depth buffer", ||{
        if depth.len() > stride * height {
            depth.truncate(stride * height);
            depth.fill(1.0);
        }
        else {
            depth.fill(1.0);
            if depth.len() < (stride * height) {
                depth.reserve_exact((stride * height) - depth.len());
                depth.extend(iter::repeat(1.0).take((stride * height) - depth.len()));
            }
        }
    });

    time("Filled triangles", || {
        let mut pool = DRAW_WORKERS.lock().unwrap();
        pool.scoped(|scope| {
            let mut ymin = 0;
            let mut i_tile = 0;
            let mut depth = depth.as_mut_slice();

            while ymin < height  {
                let mut xmin = 0;
                while xmin < stride {
                    let (tile_depth, rem_depth) = depth.split_at_mut(TILE_WIDTH.min(stride - xmin) * TILE_HEIGHT.min(height - ymin));
                    depth = rem_depth;
                    let mut tile = Tile {
                        colour: Buffer {
                            buffer: unsafe { from_raw_parts_mut(buffer, stride * height) },
                            left: 0,
                            top: 0,
                            stride
                        },
                        depth: Buffer {
                            buffer: tile_depth,
                            left: xmin,
                            top: ymin,
                            stride: TILE_WIDTH.min(stride - xmin)
                        },
                        xmin,
                        ymin,
                        xmax: (xmin + TILE_WIDTH).min(stride),
                        ymax: (ymin + TILE_HEIGHT).min(height)
                    };

                    let triangles = array::from_fn(|i| &tile_triangles[i][i_tile]);

                    scope.execute(move || {
                        draw_tile(&mut tile, model, xs, ys, zs, iws, bounds, intensities, triangles);
                    });

                    xmin += TILE_WIDTH;
                    i_tile += 1;
                }

                ymin += TILE_HEIGHT;
            }
        });
    });
}
