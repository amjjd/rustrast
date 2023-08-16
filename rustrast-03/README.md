rustrast 03 - Loading a 3D model
================================

For context, see the [main README](../).

In this chapter, I load a 3D model and draw the vertices on screen.

Format
------

Wavefront's [.obj](https://en.wikipedia.org/wiki/Wavefront_.obj_file) format is simple and doesn't really need a parser
that would distract from simply rendering to screen. For this chapter at least I don't need any textures, so I decided
to use this amazing free model of actor
[Peter Dinklage](https://www.turbosquid.com/3d-models/3d-peter-dinklage-likeness-portrait-2092437) because the fact
that someone put so much time into creating it makes me happy. As mentioned, I'm not writing a proper parser for the
format, so I went a simple parser that will panic if anything is wrong with the file, and which only loads vertices for
now.

Just get to the point(s)!
-------------------------

I don't want to actually do the 3D geometry calculations at this step, so decided to just scale the model to fit the
window, with a small pulse to the scaling factor so there is some animation, and draw just the vertices, shaded
according to their z coordinates to emulate some form of diminished lighting. Conveniently the model I chose is y up,
and has higher y values at the bottom, so matches the screen well.

The result is as expected:

![The vertices](./screenshot.png)

While the drawing code in this chapter is throwaway work and therefore performance doesn't matter, I decided to use
`f32` as that should allow for the best performance with SIMD instructions later. Performance is disappointing, at
about 8.5ms per frame in release mode, 60ms in debug mode. Window size doesn't matter as the draw loop always writes
O(number of vertices) pixels. Half the time in release mode and almost all of it in debug mode is spent sorting the
vector of vertices. I chose to do this every frame as it's necessary if the model is rotating, say, or if the camera
moves. Once I start redering polygons the sort will go away.

Rust?
-----

I've used a few more language features, notably closures to avoid repeating the timing code.

Lifetimes make the learning curve steep, and I think in particular `String` vs `str`: it's really not obvious to a
beginner how to write a function that accepts a string.

It's a bit annoying that the support for sorting by floating point numbers is incomplete. Java defines an
[odering for floats](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Float.html#compareTo(java.lang.Float))
that is inconsistent with the comparison operators and I have never heard of anyone complaining. `NaN` is conceptually
similar to an SQL `NULL`, and SQL sorts (and groups) `NULL` inconsistently with the comparison operators. Perhaps a
`sort_by_float_key` to go with the experimental `sort_floats` on `Vec` would help.

Next, [I'll draw the polygons](../rustrast-03/).
