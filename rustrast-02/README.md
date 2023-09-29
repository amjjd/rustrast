rustrast 02 - Animation
=======================

For context, see the [main README](../).

In this chapter, I animate a basic pattern on screen and compare the performance of some options for doing that.

Options
-------

Windows has a few ways of drawing pixels on the screen. The simplest is
[SetPixel](https://learn.microsoft.com/en-us/windows/win32/api/wingdi/nf-wingdi-setpixel), but given that a full HD
display has over 2 million pixels, full frame animation using SetPixel would require calling into the Windows kernel
at least 120 million times per second. That is unlikely to be possible!

A much better way is to set colour values in a memory buffer and copy it to the screen when it's ready. This means just
a few kernel calls per frame. There are a few ways to do this; you can allocate your own memory and use either
[SetDIBitsToDevice](https://learn.microsoft.com/en-us/windows/win32/api/wingdi/nf-wingdi-setdibitstodevice) or
[StretchDIBits](https://learn.microsoft.com/en-us/windows/win32/api/wingdi/nf-wingdi-stretchdibits). Alternatively,
use [CreateDIBSection](https://learn.microsoft.com/en-us/windows/win32/api/wingdi/nf-wingdi-createdibsection) to
allocate the buffer and [BitBlt](https://learn.microsoft.com/en-us/windows/win32/api/wingdi/nf-wingdi-bitblt) to copy
it to the screen. The second method is [documented as being
faster](https://learn.microsoft.com/en-us/windows/win32/api/wingdi/nf-wingdi-bitblt#remarks) but this may or may not be
true given differences between display drivers.

When I last did this kind of thing, Windows had just added [DirectDraw](https://en.wikipedia.org/wiki/DirectDraw) to
provide lower-level access to the video card, including things like page flipping between buffers in video memory and
synchronising with the vertical refresh. DirectDraw has been pretty thoroughly deprecated in favour of
[Direct2D](https://learn.microsoft.com/en-us/windows/win32/direct2d/direct2d-portal), and using that to draw pixels
while staying hardware accelerated seemed far more complicated than the basic GDI functions above.

Measure, Measure, Code
----------------------

To keep things simple, I decided to start with `CreateDIBSection` and `BitBlt` and measure how long it takes to copy
the buffer to the screen. For timing I used
[QueryPerformanceCounter](https://learn.microsoft.com/en-us/windows/win32/api/profileapi/nf-profileapi-queryperformancecounter)
and I copied as many frames as I could by requesting a complete repaint of the window after every paint, with no
attempt to cap the frame rate or synchronise with the vertical refesh. I also ensured that the application declares
itself as [DPI
aware](https://learn.microsoft.com/en-us/windows/win32/hidpi/high-dpi-desktop-application-development-on-windows) so it
doesn't get bitmap scaled by the OS.

The result? It takes between 0.5 and 2ms to `BitBlt` an entire maximized 1920x1200 window on my Dell XPS 9500. It has a
GTX 1650 Ti in it, but looking at Task Manager shows that GDI is using the integrated Intel UHD 630. On my Surface
Laptop Studio it takes about 2ms when connected to a 2560x1440 display, and similar on the laptop's own 2400x1600
screen. In both cases GDI uses the integrated Iris Xe graphics rather than the discrete RTX 3050 Ti. I though that was
good enough - it would support 500+ frames per second which is far beyond any of the screens' refresh rates - so there
was no need to test DirectX.

Playing with the Surface Laptop Studio also let me discover that when you move a window between screens with different
DPI, Windows will happily suggest a new windows size that is bigger than the target screen.

The performance of my first attempt at an animated pattern left a lot to be desired: about 13ms per frame on the Dell.
It was slow enough that my first guess was the Windows compositor was introducing a vertical sync pause; however
commenting out the call to `draw` verified that you can BitBlt at far above the screen's refresh rate.

I knew that the right thing to do is attach a profiler, but looking at the code to draw the pattern:

```rust
static mut START: usize = 0;
static PATTERN: [RGBQUAD; 3] = [
    RGBQUAD { rgbRed: 255, rgbBlue: 0, rgbGreen: 0, rgbReserved: 0 },
    RGBQUAD { rgbRed: 0, rgbBlue: 255, rgbGreen: 0, rgbReserved: 0 },
    RGBQUAD { rgbRed: 0, rgbBlue: 0, rgbGreen: 255, rgbReserved: 0 }];

pub unsafe fn draw(buffer: *mut RGBQUAD, width: u16, height: u16) {
    for offset in 0..((width as usize) * (height as usize)) {
        *buffer.offset(offset as isize) = PATTERN[(START + offset) % 3];
    }

    START = (START + 1) % 3;
}
```

... that call to `offset` for every pixel made me suspicious. I tried `cargo build -r` to get a release build, assuming
that the call would be inlined. There was a big improvement: the release build takes about 1.6ms to draw a maximised
frame. However, I wanted to get decent performance in debug mode, and even 1.6ms seems a bit slower than it should be
possible to fill memory, so I tried a [slice](https://doc.rust-lang.org/std/primitive.slice.html):

```rust
let buffer_slice = from_raw_parts_mut(buffer, (width as usize) * (height as usize));

for offset in 0..buffer_slice.len() {
    buffer_slice[offset] = PATTERN[(START + offset) % 3];
}
```

That is significantly slower in debug mode at about 24ms per frame, but identical in release mode. Still not wanting to
use a profiler or disassembler, my next thought was to use a lower-level loop construct:

```rust
let mut offset: usize = 0;
let len = (width as usize) * (height as usize);
while offset < len {
	*buffer.offset(offset as isize) = PATTERN[(START + offset) % 3];
	offset = offset + 1;
}
```

No difference, or maybe even slightly slower in debug mode. My next suspect was the copy itself: it was copying structs
into an array; maybe copying bytes would be faster? However, I will need to be able to set individual pixels to draw
triangles later, so I didn't bother testing `fill`. Instead, I treated the buffer as a block of `u32` values, planning
to use `u64` for more speed later:

```rust
static PATTERN: [u32; 3] = [0x000000ff, 0x0000ff00, 0x00ff0000];

pub unsafe fn draw(buffer: *mut RGBQUAD, width: u16, height: u16) {
    let buffer32 = buffer as *mut u32;
    for offset in 0..((width as usize) * (height as usize)) {
        *buffer32.offset(offset as isize) = PATTERN[(START + offset) % 3];
    }

    START = (START + 1) % 3;
}
```

This version is as slow as the version that uses a slice in both modes. I decided to test `fill` after all, to make
sure I wasn't missing something. This meant changing the pattern to a two-pixel one:

```rust
// assume little-endian
static PATTERN: [u64; 2] =[0x00ffffff00000000, 0x0000000000ffffff];

pub unsafe fn draw(buffer: *mut RGBQUAD, width: u16, height: u16) {
    let buffer64_slice = from_raw_parts_mut(buffer as *mut u64, (width as usize) * (height as usize) / 2);
    buffer64_slice.fill(PATTERN[START]);

    START = (START + 1) % 2;
}
```

That's an improvement: 8ms in debug, about 0.6ms in release. 0.6ms for about 8MB of pixel data is not far off half the
peak memory bandwidth of the DDR4-4000 RAM in my machine (I think, I don't know exactly how this stuff works but there
is [sales material](https://www.crucial.com/support/memory-speeds-compatability) online), so I don't think I can do a
lot better than that. At this point I realised that I had a divide-modulo-3 per pixel in all the other versions, which
is probably bad, so I went back to the very first version and got rid of it. Again, the simplest way was to use a 
two-pixel pattern:

```
static PATTERN: [RGBQUAD; 2] = [
    RGBQUAD { rgbRed: 255, rgbBlue: 255, rgbGreen: 255, rgbReserved: 0 },
    RGBQUAD { rgbRed: 0, rgbBlue: 0, rgbGreen: 0, rgbReserved: 0 }];

pub unsafe fn draw(buffer: *mut RGBQUAD, width: u16, height: u16) {
    for offset in 0..((width as usize) * (height as usize)) {
        *buffer.offset(offset as isize) = PATTERN[(START + offset) & 1];
    }

    START = (START + 1) % 2;
}
```

That makes no difference to performance in debug but reduces the draw time to about 1ms in release. So, I got rid of
the array lookup to fetch the pixel colour, and removed some unnecessary casts:

```rust
static BLACK: RGBQUAD = RGBQUAD { rgbRed: 255, rgbBlue: 255, rgbGreen: 255, rgbReserved: 0 };
static WHITE: RGBQUAD = RGBQUAD { rgbRed: 0, rgbBlue: 0, rgbGreen: 0, rgbReserved: 0 };

pub unsafe fn draw(buffer: *mut RGBQUAD, width: u16, height: u16) {
    if START == 0 {
        for offset in 0..((width as isize) * (height as isize) / 2) {
            *buffer.offset(offset * 2) = BLACK;
            *buffer.offset(offset * 2 + 1) = WHITE;
        }
    }
    else {
        for offset in 0..((width as isize) * (height as isize) / 2) {
            *buffer.offset(offset * 2) = WHITE;
            *buffer.offset(offset * 2 + 1) = BLACK;
        }
    }

    START = (START + 1) % 2;
}
```

This is about the same speed as `fill` in release mode and a bit slower than it at 12ms per frame in debug mode. The
pointer offset calculations are capable of being compiled directly into an i86 `MOV` instruction. I would need to
disassemble the function to see if the compiler is actually doing that, but I did check if it's combining the two
writes into one 64-bit one by doing that explicitly:

```rust
static PATTERN: [[RGBQUAD; 2]; 2] = [
    [RGBQUAD { rgbRed: 255, rgbBlue: 255, rgbGreen: 255, rgbReserved: 0 }, RGBQUAD { rgbRed: 0, rgbBlue: 0, rgbGreen: 0, rgbReserved: 0 }],
    [RGBQUAD { rgbRed: 0, rgbBlue: 0, rgbGreen: 0, rgbReserved: 0 }, RGBQUAD { rgbRed: 255, rgbBlue: 255, rgbGreen: 255, rgbReserved: 0 }]];

pub unsafe fn draw(buffer: *mut RGBQUAD, width: u16, height: u16) {
    let pattern64 = *((&PATTERN[START] as *const RGBQUAD) as *const u64);
    let buffer64 = buffer as *mut u64;

    for offset in 0..((width as isize) * (height as isize) / 2) {
        *buffer64.offset(offset) = pattern64;
    }

    START = (START + 1) % 2;
}
```

Debug mode takes about 9ms per frame with no change to release mode, suggesting that the compiler does in fact combine
the writes in release mode.

Finally doing the right thing
-----------------------------

I decided that I need to disassemble to see exactly what it does. I used `dumpbin` that comes with the C compiler tools
you need to install for Rust. The Rust compiler does heavy inlining so it's a bit tricky to find, but the inner loop of
the version that separately sets two pixels at a time is in fact optimised more than I expected:

```
  0000000140001660: 42 0F 11 04 C1     movups      xmmword ptr [rcx+r8*8],xmm0
  0000000140001665: 42 0F 11 44 C1 10  movups      xmmword ptr [rcx+r8*8+10h],xmm0
  000000014000166B: 49 83 C0 04        add         r8,4
  000000014000166F: 4C 39 C2           cmp         rdx,r8
  0000000140001672: 75 EC              jne         0000000140001660
```

This is processing 256 bits at a time using two calls to `movups`, an SSE instruction. The inner loop of the version
that copies 64 bits at a time is similar, but uses two calls to `movdqu` from SSE2:

```
  0000000140001640: F3 42 0F 7F 04 D0  movdqu      xmmword ptr [rax+r10*8],xmm0
  0000000140001646: F3 42 0F 7F 44 D0  movdqu      xmmword ptr [rax+r10*8+10h],xmm0
                    10
  000000014000164D: 49 83 C2 04        add         r10,4
  0000000140001651: 4D 39 D1           cmp         r9,r10
  0000000140001654: 75 EA              jne         0000000140001640
```

Unrolling the loop like that is only possible because I'm setting the same 64-bit value throughout the array. However,
it does suggest a few things:

* There's no penalty to copying `RGBQUAD`s rather than converting them to `u32` or larger.
* Those are unaligned instructions, although they perform the same as the aligned versions if the memory is aligned.
  While the locations I see from `CreateDIBSection` do seem to be at least 16-byte aligned (more probably, 4KB page
  aligned), I would need to allocate my own buffer and use `SetDIBitsToDevice` or `StretchDIBits` to ensure it.
* When I'm drawing polygons I may need to write 4 pixels at once to give the optimiser a chance. But I will look at the
  disassembly first!

I committed the second-last version of the draw loop as dealing with `RGBQUAD`s is a lot easier than packing pairs of
pixels into `i64`s, and I presume windows-rs has handled the endianness properly.

How am I feeling about Rust?
----------------------------

I like how opinionated Rust is: there is a standard build and dependency system; you get warnings if you don't use
idiomatic casing.

I still have a constant feeling that I'm not doing things idiomatically, which is not helped by my not having read much
of the learning material yet. I know I don't understand how casting works yet, as I just keep trying things until it
compiles. I definitely don't fully understand modules yet either.

Next, [it's time to load a 3D model](../rustrast-03/).
