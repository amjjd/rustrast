rustrast
========

My journey through learning both modern 3d rendering techniques and Rust by implementing a software renderer.

Background
----------

I recently read John Romero's excellent biography, [DOOM Guy](https://romero.com/shop/p/doomguy) and became nostalgic
for the mid-90s, when seeing Quake's early tech demo, [Qtest](https://quake.fandom.com/wiki/Qtest) absolutely blew my
mind. Like so many others, I found everything I could online, devoured as much of
[The Book](https://www.amazon.com/Computer-Graphics-Principles-Practice-2nd/dp/0201848406) as I could understand, and
soon had the same rotating flat-shaded cube in VGA mode 13h as, I'm sure, ten thousand others.

The nostalgia rabbit hole led me to Fabien Sanglard's
[Quake Code Review](https://fabiensanglard.net/quakeSource/index.php) and from there, Michael Abrash's
[Graphics Programming Black Book](https://archive.org/details/michaelabrashsgr00abra), a legendary book that I never
managed to acquire back then (although I treasured the few Dr. Dobb's Journal articles of his that I did have).

However, things have changed in the last 25+ years. GPUs are ubiquitous. While I know pretty much nothing about how
modern 3d works, I am aware that modern rendering involves writing shaders (that do more than just choose the colour
of a pixel!) for the GPU to run on a massively parallel engine. CPUs have changed enormously: there is almost certainly
no point in writing cycle-optimised code like I did for the Pentium given high levels of superscalar parallelism and
out-of-order execution. Memory has changed too, with the penalty of a memory access being on the order of thousands of
instructions.

A plan formed.

The plan
--------

I'm going to roughly follow Dmitry V. Sokolov's excellent [tinyrenderer](https://github.com/ssloy/tinyrenderer) course
and build a software renderer. I'm not interested in writing a game, so I'm not planning on figuring out Vulkan or
Metal or Direct3D 12; I want to figure out how the things that were too slow or too hard to understand in the 90s like
surface mapping and shadows work.

However, Dmitry's course isn't particularly concerned with making a fast renderer, so I'm going to aim to make one that
can handle animation at a high frame rate all the way through. That will mean learning things like how to use the
modern SIMD instructions - I did play with MMX way back when, but everything after that passed me by.

Finally, C++ just doesn't appeal to me. I last used it in the early 2000s, using MFC to write some Windows apps. I
never actually properly learned how the most advanced parts of it work and haven't kept up to date. Instead, I'm
planning on using Rust for the first time.
[It's John Carmack approved](https://twitter.com/ID_AA_Carmack/status/1094419108781789184)!

Next up, [let's get some pixels on the screen using Rust](rustrast-01/).