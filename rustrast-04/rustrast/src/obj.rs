use lazy_static::*;
use regex::*;
use std::io::*;
use super::simd::CoordinateComponents;
use super::maths::HomogenousCoordinates;

// not-suitable-for-production Wavefront .obj parsing; panics on any error

pub struct Model {
    pub num_vertices: usize,
    pub xs: CoordinateComponents,
    pub ys: CoordinateComponents,
    pub zs: CoordinateComponents,
    pub ws: CoordinateComponents
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
    let mut xs = CoordinateComponents::new();
    let mut ys = CoordinateComponents::new();
    let mut zs = CoordinateComponents::new();
    let mut ws = CoordinateComponents::new();

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
                    _ => ()
                }
            }
        }
    }
    
    Model { num_vertices: xs.len(), xs, ys, zs, ws }
}