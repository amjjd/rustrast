use lazy_static::*;
use regex::*;
use std::io::*;

use super::simd_vec::*;
use super::transformation::*;

// not-suitable-for-production Wavefront .obj parsing; panics on any error
// https://en.wikipedia.org/wiki/Wavefront_.obj_file

#[derive(Clone, Copy)]
struct TextureCoordinates {
    u: f32,
    v: f32,
    w: f32
}

impl TextureCoordinates {
    fn from_texture_line<S: AsRef<str>>(line: S) -> TextureCoordinates {
        let captures = TEXTURE_LINE.captures(line.as_ref()).unwrap();
        let u = captures["u"].parse::<f32>().unwrap();
        let v = captures.name("v").map(|m| m.as_str().parse::<f32>().unwrap()).unwrap_or(0.0);
        let w = captures.name("w").map(|m| m.as_str().parse::<f32>().unwrap()).unwrap_or(0.0);

        TextureCoordinates { u, v, w }
    }
}
#[derive(Clone, Copy)]
struct FaceVertex {
    v: usize,
    t: Option<usize>,
    n: Option<usize>,
}

impl FaceVertex {
    fn from_face_line_component<S: AsRef<str>>(component: S, num_vertices: u32) -> FaceVertex {
        let mut parts = component.as_ref().split('/');
        let v = match parts.next().unwrap().parse::<isize>().unwrap() {
            v if v > 0 => v as u32 - 1,
            v => num_vertices + 1 - (v.abs() as u32)
        };

        let t = parts.next().map(|s| s.parse::<usize>().unwrap() - 1);
        let n = parts.next().map(|s| s.parse::<usize>().unwrap() - 1);
        FaceVertex { v: v as usize, t, n }
    }
}

#[derive(Clone, Copy)]
struct Triangle {
    v0: FaceVertex,
    v1: FaceVertex,
    v2: FaceVertex
}

impl Triangle {
    fn from_face_line<S: AsRef<str>>(line: S, num_vertices: u32) -> Vec<Triangle> {
        let vs: Vec<FaceVertex> = line.as_ref().split(' ').skip(1).map(|component| FaceVertex::from_face_line_component(component, num_vertices)).collect();

        let mut triangles = Vec::new();

        // fan triangulation, so requires convex polygons
        let v0 = vs[0];
        for iv1 in 1..(vs.len()-1) {
            let v1 = vs[iv1];
            let v2 = vs[iv1 + 1];
            triangles.push(Triangle {v0, v1, v2});
        }

        triangles       
    }

    fn surface_normal(&self, xs: &SimdVec<f32>, ys: &SimdVec<f32>, zs: &SimdVec<f32>, ws: &SimdVec<f32>) -> CartesianVector {
        let (v0, _) = HomogenousCoordinates{x: xs[self.v0.v], y: ys[self.v0.v], z: zs[self.v0.v], w: ws[self.v0.v]}.to_cartesian();
        let (v1, _) = HomogenousCoordinates{x: xs[self.v1.v], y: ys[self.v1.v], z: zs[self.v1.v], w: ws[self.v1.v]}.to_cartesian();
        let (v2, _) = HomogenousCoordinates{x: xs[self.v2.v], y: ys[self.v2.v], z: zs[self.v2.v], w: ws[self.v2.v]}.to_cartesian();

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;

        edge1.cross_product(&edge2)
    }
}


#[allow(dead_code)]
pub struct Model {
    pub num_vertices: u32,
    pub xs: SimdVec<f32>,
    pub ys: SimdVec<f32>,
    pub zs: SimdVec<f32>,
    pub ws: SimdVec<f32>,
    pub vertex_normal_xs: SimdVec<f32>,
    pub vertex_normal_ys: SimdVec<f32>,
    pub vertex_normal_zs: SimdVec<f32>,
    pub texture_us: SimdVec<f32>,
    pub texture_vs: SimdVec<f32>,
    pub texture_ws: SimdVec<f32>,
    pub num_triangles: u32,
    pub trianglev0s: SimdVec<u32>,
    pub trianglev1s: SimdVec<u32>,
    pub trianglev2s: SimdVec<u32>,
    pub surface_normal_xs: SimdVec<f32>,
    pub surface_normal_ys: SimdVec<f32>,
    pub surface_normal_zs: SimdVec<f32>
}

#[allow(dead_code)]
impl Model {
    pub fn vertex(&self, i: u32) -> HomogenousCoordinates {
        HomogenousCoordinates { x: self.xs[i as usize], y: self.ys[i as usize], z: self.zs[i as usize], w: self.ws[i as usize] }
    }

    pub fn vertex_normal(&self, it: u32) -> CartesianVector {
        CartesianVector { x: self.vertex_normal_xs[it as usize], y: self.vertex_normal_ys[it as usize], z: self.vertex_normal_zs[it as usize] }
    }

    pub fn surface_normal(&self, it: u32) -> CartesianVector {
        CartesianVector { x: self.surface_normal_xs[it as usize], y: self.surface_normal_ys[it as usize], z: self.surface_normal_zs[it as usize] }
    }
}

lazy_static! {
    static ref LINE: Regex = Regex::new(r"(\S+).*").unwrap();
    static ref VERTEX_LINE: Regex = Regex::new(r"v\s+(?P<x>\S+)\s+(?P<y>\S+)\s+(?P<z>\S+)(?:\s+(?P<w>\S+))?\s*").unwrap();
    static ref TEXTURE_LINE: Regex = Regex::new(r"vt\s+(?P<u>\S+)?(?:\s+(?P<v>\S+))?\s*").unwrap();
    static ref NORMAL_LINE: Regex = Regex::new(r"vn\s+(?P<x>\S+)\s+(?P<y>\S+)\s+(?P<z>\S+)\s*").unwrap();
}

impl HomogenousCoordinates {
    fn from_vertex_line<S: AsRef<str>>(line: S) -> HomogenousCoordinates {
        let captures = VERTEX_LINE.captures(line.as_ref()).unwrap();
        let x = captures["x"].parse::<f32>().unwrap();
        let y = captures["y"].parse::<f32>().unwrap();
        let z = captures["z"].parse::<f32>().unwrap();
        let w = captures.name("w").map(|m| m.as_str().parse::<f32>().unwrap());

        HomogenousCoordinates { x, y, z, w: w.unwrap_or(1.0) }
    }
}

impl CartesianVector {
    fn from_normal_line<S: AsRef<str>>(line: S) -> CartesianVector {
        let captures = NORMAL_LINE.captures(line.as_ref()).unwrap();
        let x = captures["x"].parse::<f32>().unwrap();
        let y = captures["y"].parse::<f32>().unwrap();
        let z = captures["z"].parse::<f32>().unwrap();

        CartesianVector { x, y, z }
    }
}

pub fn read_obj<R: Read>(file: R, read_uvw: bool) -> Model {
    let mut xs = SimdVec::new();
    let mut ys = SimdVec::new();
    let mut zs = SimdVec::new();
    let mut ws = SimdVec::new();
    let mut texture_us = SimdVec::new();
    let mut texture_vs = SimdVec::new();
    let mut texture_ws = SimdVec::new();
    let mut texture_coordinates = Vec::new();
    let mut vertex_normal_xs = SimdVec::new();
    let mut vertex_normal_ys = SimdVec::new();
    let mut vertex_normal_zs = SimdVec::new();
    let mut vertex_normals = Vec::new();
    let mut triangles = Vec::new();
    let mut surface_normal_xs = SimdVec::new();
    let mut surface_normal_ys = SimdVec::new();
    let mut surface_normal_zs = SimdVec::new();

    for line in BufReader::new(file).lines() {
        if let Ok(line) = line {
            if let Some(captures) = LINE.captures(&line) {
                match &captures[1] {
                    "v" => {
                        let vertex = HomogenousCoordinates::from_vertex_line(&line);
                        xs.push(vertex.x);
                        ys.push(vertex.y);
                        zs.push(vertex.z);
                        ws.push(vertex.w);
                        texture_us.push(-1.0);
                        texture_vs.push(-1.0);
                        texture_ws.push(-1.0);
                        vertex_normal_xs.push(0.0);
                        vertex_normal_ys.push(0.0);
                        vertex_normal_zs.push(0.0);
                    }
                    "vt" if read_uvw => {
                        let texture = TextureCoordinates::from_texture_line(&line);
                        texture_coordinates.push(texture);
                    }
                    "vn" => {
                        let normal = CartesianVector::from_normal_line(&line);
                        vertex_normals.push(normal);
                    }
                    "f" => {
                        triangles.extend(Triangle::from_face_line(&line, xs.len() as u32));
                    }
                    _ => ()
                }
            }
        }
    }

    let mut trianglev0s = SimdVec::new();
    let mut trianglev1s = SimdVec::new();
    let mut trianglev2s = SimdVec::new();

    for triangle in triangles {
        let surface_normal = triangle.surface_normal(&xs, &ys, &zs, &ws);
        surface_normal_xs.push(surface_normal.x);
        surface_normal_ys.push(surface_normal.y);
        surface_normal_zs.push(surface_normal.z);

        macro_rules! tv {
            ($v:expr) => {{
                let v = $v.v;
                let t = match $v.t {
                    Some(n) if read_uvw => texture_coordinates[n],
                    _ => TextureCoordinates { u: -1.0, v: -1.0, w: -1.0 }
                };
                let n = match $v.n {
                    Some(n) => vertex_normals[n],
                    None => surface_normal
                };

                if texture_us[v] == -1.0 && texture_vs[v] == -1.0 && texture_ws[v] == -1.0 &&vertex_normal_xs[v] == 0.0 && vertex_normal_ys[v] == 0.0 && vertex_normal_zs[v] == 0.0 {
                    texture_us[v] = t.u;
                    texture_vs[v] = t.v;
                    texture_ws[v] = t.w;
                    vertex_normal_xs[v] = n.x;
                    vertex_normal_ys[v] = n.y;
                    vertex_normal_zs[v] = n.z;
                    v
                }
                else if texture_us[v] != t.u || texture_vs[v] != t.v || texture_ws[v] != t.w || vertex_normal_xs[v] != n.x || vertex_normal_ys[v] != n.y || vertex_normal_zs[v] != n.z {
                    // need to duplicate the vertex to support different texture coordinates and/or normals for different triangles sharing the vertex
                    let new_v = xs.len();
                    xs.push(xs[v]);
                    ys.push(ys[v]);
                    zs.push(zs[v]);
                    ws.push(ws[v]);
                    texture_us.push(t.u);
                    texture_vs.push(t.v);
                    texture_ws.push(t.w);
                    vertex_normal_xs.push(n.x);
                    vertex_normal_ys.push(n.y);
                    vertex_normal_zs.push(n.z);
                    new_v
                }
                else {
                    v
                }
            }}
        }
        
        trianglev0s.push(tv!(triangle.v0) as u32);
        trianglev1s.push(tv!(triangle.v1) as u32);
        trianglev2s.push(tv!(triangle.v2) as u32);
    }

    Model { num_vertices: xs.len() as u32, xs, ys, zs, ws, texture_us, texture_vs, texture_ws, vertex_normal_xs, vertex_normal_ys, vertex_normal_zs, num_triangles: trianglev0s.len() as u32, trianglev0s, trianglev1s, trianglev2s, surface_normal_xs, surface_normal_ys, surface_normal_zs }
}