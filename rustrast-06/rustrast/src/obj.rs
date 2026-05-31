use lazy_static::*;
use regex::*;
use std::io::*;

use super::simd_vec::*;
use super::transformation::*;

// not-suitable-for-production Wavefront .obj parsing; panics on any error
// https://en.wikipedia.org/wiki/Wavefront_.obj_file

pub struct FaceVertex {
    pub v: isize
}

impl FaceVertex {
    fn from_face_line_component<S: AsRef<str>>(component: S) -> FaceVertex {
        FaceVertex { v: component.as_ref().split('/').next().unwrap().parse::<isize>().unwrap() }
    }

    fn vertex_index(&self, num_vertices: u32) -> u32 {
        // vertex indices are 1-based and can be from the start or end of the vertex list
        if self.v > 0 {
            self.v as u32 - 1
        }
        else {
            num_vertices + 1 - (self.v.abs() as u32)
        }
    }
}

#[derive(Clone, Copy)]
struct Triangle {
    pub v0: u32,
    pub v1: u32,
    pub v2: u32
}

impl Triangle {
    fn from_face_line<S: AsRef<str>>(line: S, num_vertices: u32) -> Vec<Triangle> {
        let vs: Vec<FaceVertex> = line.as_ref().split(' ').skip(1).map(FaceVertex::from_face_line_component).collect();

        let mut triangles = Vec::new();

        // fan triangulation, so requires convex polygons
        let v0 = vs[0].vertex_index(num_vertices);
        for iv1 in 1..(vs.len()-1) {
            let v1 = vs[iv1].vertex_index(num_vertices);
            let v2 = vs[iv1 + 1].vertex_index(num_vertices);
            triangles.push(Triangle {v0, v1, v2});
        }

        triangles       
    }

    fn surface_normal(&self, xs: &SimdVec<f32>, ys: &SimdVec<f32>, zs: &SimdVec<f32>, ws: &SimdVec<f32>) -> CartesianVector {
        let (v0, _) = HomogenousCoordinates{x: xs[self.v0 as usize], y: ys[self.v0 as usize], z: zs[self.v0 as usize], w: ws[self.v0 as usize]}.to_cartesian();
        let (v1, _) = HomogenousCoordinates{x: xs[self.v1 as usize], y: ys[self.v1 as usize], z: zs[self.v1 as usize], w: ws[self.v1 as usize]}.to_cartesian();
        let (v2, _) = HomogenousCoordinates{x: xs[self.v2 as usize], y: ys[self.v2 as usize], z: zs[self.v2 as usize], w: ws[self.v2 as usize]}.to_cartesian();

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;

        edge1.cross_product(&edge2)
    }
}

pub struct Model {
    pub num_vertices: u32,
    pub xs: SimdVec<f32>,
    pub ys: SimdVec<f32>,
    pub zs: SimdVec<f32>,
    pub ws: SimdVec<f32>,
    pub num_triangles: u32,
    pub trianglev0s: SimdVec<u32>,
    pub trianglev1s: SimdVec<u32>,
    pub trianglev2s: SimdVec<u32>,
    pub surface_normal_xs: SimdVec<f32>,
    pub surface_normal_ys: SimdVec<f32>,
    pub surface_normal_zs: SimdVec<f32>
}

impl Model {
    #[allow(dead_code)]
    pub fn homogenous_coordinates(&self, i: u32) -> HomogenousCoordinates {
        HomogenousCoordinates { x: self.xs[i as usize], y: self.ys[i as usize], z: self.zs[i as usize], w: self.ws[i as usize] }
    }

    #[allow(dead_code)]
    pub fn surface_normal(&self, it: u32) -> CartesianVector {
        CartesianVector { x: self.surface_normal_xs[it as usize], y: self.surface_normal_ys[it as usize], z: self.surface_normal_zs[it as usize] }
    }
}

lazy_static! {
    static ref LINE: Regex = Regex::new(r"(\S+).*").unwrap();
    static ref VERTEX_LINE: Regex = Regex::new(r"v\s+(\S+)\s+(\S+)\s+(\S+)(?:\s+(\S+))?\s*").unwrap();
}

impl HomogenousCoordinates {
    fn from_vertex_line<S: AsRef<str>>(line: S) -> HomogenousCoordinates {
        let captures = VERTEX_LINE.captures(line.as_ref()).unwrap();
        let x = captures[1].parse::<f32>().unwrap();
        let y = captures[2].parse::<f32>().unwrap();
        let z = captures[3].parse::<f32>().unwrap();
        let w = captures.get(4).map(|m| m.as_str().parse::<f32>().unwrap());

        HomogenousCoordinates { x, y, z, w: w.unwrap_or(1.0) }
    }
}

pub fn read_obj<R: Read>(file: R) -> Model {
    let mut xs = SimdVec::new();
    let mut ys = SimdVec::new();
    let mut zs = SimdVec::new();
    let mut ws = SimdVec::new();
    let mut triangles = Vec::new();

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
                    }
                    "f" => {
                        triangles.extend(Triangle::from_face_line(&line, xs.len() as u32));
                    }
                    _ => ()
                }
            }
        }
    }

    let trianglev0s = triangles.iter().map(|t| t.v0).collect();
    let trianglev1s = triangles.iter().map(|t| t.v1).collect();
    let trianglev2s = triangles.iter().map(|t| t.v2).collect();

    let mut surface_normal_xs = SimdVec::with_capacity(triangles.len());
    let mut surface_normal_ys = SimdVec::with_capacity(triangles.len());
    let mut surface_normal_zs = SimdVec::with_capacity(triangles.len());

    for triangle in &triangles {
        let surface_normal = triangle.surface_normal(&xs, &ys, &zs, &ws);
        surface_normal_xs.push(surface_normal.x);
        surface_normal_ys.push(surface_normal.y);
        surface_normal_zs.push(surface_normal.z);
    }
    
    Model { num_vertices: xs.len() as u32, xs, ys, zs, ws, num_triangles: triangles.len() as u32, trianglev0s, trianglev1s, trianglev2s, surface_normal_xs, surface_normal_ys, surface_normal_zs }
}