use core::arch::x86_64::*;
use std::{ops::*, slice::*};
use aligned_vec::*;
use safe_transmute::trivial::*;

// needs to be as high as that required by the widest SIMD tech in use; here it's 128 for caching
const ALIGNMENT: usize = 128;

// hides the mechanics of alignment  and conversion to from calling code
pub struct SimdVec<T> where T : Default {
    vs: AVec<T, ConstAlign<ALIGNMENT>>
}

impl<T> SimdVec<T> where T : Default {
    pub fn new() -> Self {
        SimdVec {vs: AVec::new(ALIGNMENT) }
    }

    #[allow(dead_code)]
    pub fn with_capacity(capacity: usize) -> Self {
        SimdVec {vs: AVec::with_capacity(ALIGNMENT, capacity) }
    }

    pub fn reserve_exact(&mut self, additional: usize) {
        self.vs.reserve_exact(additional)
    }
 
    pub fn push(&mut self, v: T) {
        self.vs.push(v)
    }

    pub fn fill(&mut self, v: T) where T : Copy {
        self.vs.fill(v)
    }

    pub fn extend(&mut self, iter: impl IntoIterator<Item = T>) {
        for elem in iter {
            self.push(elem);
        }
    }

    pub fn len(&self) -> usize {
        self.vs.len()
    }

    pub fn truncate(&mut self, new_len: usize) {
        self.vs.truncate(new_len)
    }

    pub fn as_slice(&self) -> &[T] {
        self.vs.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.vs.as_mut_slice()
    }

    pub fn as_ptr(&self) -> *const T {
        self.vs.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.vs.as_mut_ptr()
    }
}

impl <T> SimdVec<T> where T : TriviallyTransmutable + Default {
    pub fn pad_to_mm256(&mut self) {
        let padding_bytes = (32 - ((self.vs.len() * size_of::<T>()) % 32)) % 32;
        let num_padding_elements = padding_bytes / size_of::<T>();
        for _ in 0..num_padding_elements {
            self.vs.push(T::default());
        }
    }

    // these all ignore traling elements; call pad_to_mm256 first if required
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

impl<T, Idx> Index<Idx> for SimdVec<T> where T : Default, Idx: SliceIndex<[T]> {
    type Output = Idx::Output;

    fn index(&self, ix: Idx) -> &Self::Output {
        self.vs.index(ix)
    }
}

impl<T, Idx> IndexMut<Idx> for SimdVec<T> where T : Default, Idx: SliceIndex<[T]> {
    fn index_mut(&mut self, ix: Idx) -> &mut Self::Output {
        self.vs.index_mut(ix)
    }
}

impl<T> FromIterator<T> for SimdVec<T> where T : Default {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        SimdVec {vs: AVec::from_iter(ALIGNMENT, iter) }
    }
}