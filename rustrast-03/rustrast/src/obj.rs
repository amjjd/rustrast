use lazy_static::*;
use regex::*;
use std::io::*;

// not-suitable-for-production Wavefront .obj parsing; panics on any error

#[derive(Clone)]
pub struct Vertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32
}

pub struct Model {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
    pub min_z: f32,
    pub max_z: f32,
    pub vertices: Vec<Vertex>
}

lazy_static! {
    static ref LINE: Regex = Regex::new(r"(\S+).*").unwrap();
    static ref VERTEX_LINE: Regex = Regex::new(r"v\s+(\S+)\s+(\S+)\s+(\S+)(?:\s+(\S+))?\s*").unwrap();
}

impl Vertex {
    fn from_line<S: AsRef<str>>(line: S) -> Vertex {
        let captures = VERTEX_LINE.captures(line.as_ref()).unwrap();
        let x = captures[1].parse::<f32>().unwrap();
        let y = captures[2].parse::<f32>().unwrap();
        let z = captures[3].parse::<f32>().unwrap();
        let w = captures.get(4).map(|m| m.as_str().parse::<f32>().unwrap());

        Vertex { x, y, z, w: w.unwrap_or(1.0) }
    }
}

pub fn read_obj<R: Read>(file: R) -> Model {
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    let mut min_z = f32::MAX;
    let mut max_z = f32::MIN;
    
    let mut vertices: Vec<Vertex> = Vec::new();

    for line in BufReader::new(file).lines() {
        if let Ok(line) = line {
            if let Some(captures) = LINE.captures(&line) {
                match &captures[1] {
                    "v" => {
                        let vertex = Vertex::from_line(&line);
                        min_x = f32::min(min_x, vertex.x);
                        max_x = f32::max(max_x, vertex.x);
                        min_y = f32::min(min_y, vertex.y);
                        max_y = f32::max(max_y, vertex.y);
                        min_z = f32::min(min_z, vertex.z);
                        max_z = f32::max(max_z, vertex.z);
                        vertices.push(vertex);
                    }
                    _ => ()
                }
            }
        }
    }

    Model {min_x, max_x, min_y, max_y, min_z, max_z, vertices}
}