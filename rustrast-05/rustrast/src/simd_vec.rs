use core::arch::x86_64::*;
use std::{ops::*, slice::*};
use aligned_vec::*;
use safe_transmute::trivial::*;

// needs to be as high as that required by the widest SIMD tech in use
const ALIGNMENT: usize = 128;

// hides the mechanics of alignment  and conversion to from calling code
pub struct SimdVec<T> where T : TriviallyTransmutable {
    vs: AVec<T, ConstAlign<ALIGNMENT>>
}

impl<T> SimdVec<T> where T : TriviallyTransmutable {
    pub fn new() -> Self {
        SimdVec {vs: AVec::new(ALIGNMENT) }
    }

    #[allow(dead_code)]
    pub fn with_capacity(capacity: usize) -> Self {
        SimdVec {vs: AVec::with_capacity(ALIGNMENT, capacity) }
    }
 
    pub fn push(&mut self, v: T) {
        self.vs.push(v)
    }

    pub fn len(&self) -> usize {
        self.vs.len()
    }

    pub fn as_ptr(&self) -> *const T {
        self.vs.as_ptr()
    }

    // these ignore any trailing values; alignment ensures there are no leading ones
    // can't figure out how to mark SIMD types as TriviallyTransmutable
    pub fn as_m256(&self) -> &[__m256] {
        unsafe {
            let (_, mid, _) = self.vs.align_to();
            return mid;
        }
    }

    pub fn as_m256_mut(&mut self) -> &mut [__m256] {
        unsafe {
            let (_, mid, _) = self.vs.align_to_mut();
            return mid;
        }
    }

    pub fn as_m256i(&self) -> &[__m256i] {
        unsafe {
            let (_, mid, _) = self.vs.align_to();
            return mid;
        }
    }

    #[allow(dead_code)]
    pub fn as_m256i_mut(&mut self) -> &mut [__m256i] {
        unsafe {
            let (_, mid, _) = self.vs.align_to_mut();
            return mid;
        }
    }
}

impl<T, Idx> Index<Idx> for SimdVec<T> where T : TriviallyTransmutable, Idx: SliceIndex<[T]> {
    type Output = Idx::Output;

    fn index(&self, ix: Idx) -> &Self::Output {
        self.vs.index(ix)
    }
}

impl<T, Idx> IndexMut<Idx> for SimdVec<T> where T : TriviallyTransmutable, Idx: SliceIndex<[T]> {
    fn index_mut(&mut self, ix: Idx) -> &mut Self::Output {
        self.vs.index_mut(ix)
    }
}

impl<T> FromIterator<T> for SimdVec<T> where T : TriviallyTransmutable {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        SimdVec {vs: AVec::from_iter(ALIGNMENT, iter) }
    }
}