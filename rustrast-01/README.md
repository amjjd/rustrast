rustrast 01 - Pixels on a screen
================================

For context, see the [main README](../).

In this chapter, I get a Rust development environment up and running, and figure out how to get some pixels on the
screen.

Environment
-----------

I've used Windows extensively and MacOS quite a lot. I currently don't have a Mac. I haven't done any modern mobile
development, but I don't want to learn three new things at the same time and I expect that the overhead of
cross-platform development would be a distraction. So, we're going with Windows 11. However, while I am curious as to
how much (or little) has changed, I want to keep the boilerplate to an absolute minimum, and get to having a flat
off-screen buffer I can write pixels to and copy to the screen as quickly as possible. I'll keep the actual rendering
code separate as I go, so porting to another OS should be straightforward.

Development environment
-----------------------

Microsoft has a [Rust for Windows tutorial](https://learn.microsoft.com/en-us/windows/dev-environment/rust/setup) I'll
follow. Oddly, it seems to recommend installing Visual Studio but then states that it will use Visual Studio Code for
the examples. I like Visual Studio Code, so I decided to use that.

First step was to install the [C++ build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). Is Rust
not self-hosting yet? Anyway, the tutorial recommends installing support for C++, .NET desktop and UWP, rather
charmingly suggesting that "some dependency will arise where they're required". This is on microsoft.com!

Here's the first big thing that's changed in the last 25 years: the stub installer that downloads the real installers
is 3.5MB. [That's about the size of Qtest](https://archive.org/details/qtest), the demo that so impressed me as a
teenager. The development tools themselves take up _20GB_.

Next was Rust itself, the 64-bit installer from [rust-lang.org](https://www.rust-lang.org/tools/install). This is a
console installer, press enter after it tells you the path it's going to install to. The installer updates your user
`PATH` variable so you can start a fresh console (I recommend [Cmder](https://cmder.app/) in
[Windows Terminal](https://medium.com/talpor/windows-terminal-cmder-%EF%B8%8F-573e6890d143)) and immediately run
`rustc` to get usage instructions.

Finally, I respect text-mode programmers, but I like graphical IDEs. I already had
[Visual Studio Code](https://code.visualstudio.com/) installed. Click the settings icon in the bottom left for the
extensions module, then install `rust-analyzer` and `CodeLLDB` as suggested in the tutorial. I needed to add
`%USERPROFILE%\AppData\Local\Programs\Microsoft VS Code` to my path; I'm not sure if the installer does this for you.

Hello world!
------------

Let's create a Rust project and add the Microsoft-provided library ("crate") for the Windows features I think we'll
need to create a basic application:

    cargo new rustrast
	cd rustrast
	cargo add --features Win32_Foundation,Win32_UI_WindowsAndMessaging,Win32_Graphics_Gdi windows
	code .

As mentioned, it's been a while since I've written a native Windows application. Microsoft has a
[basic example](https://learn.microsoft.com/en-us/windows/win32/learnwin32/your-first-windows-program) and it's
reassuringly familiar: there's a window class, a window procedure, and a message pump. To help me convert this to
Rust I found a nice
[example](https://friendlyuser.github.io/posts/tech/rust/Creating_a_Basic_Windows_Application_with_WinAPI_and_Rust/)
that uses a different, non-Microsoft crate called `winapi`. The Rust for Windows tutorial links to
[another example](https://github.com/robmikh/minesweeper-rs) but that uses
[WinRT](https://en.wikipedia.org/wiki/Windows_Runtime) which feels to me like it would be another new thing to learn
at the same time as Rust and modern 3D rendering.

About 30 minutes of combining the two examples above, and scanning the first few chapters of the
[Rust book](https://doc.rust-lang.org/book/title-page.html) later, I got a working app. I felt I was not writing
idiomatic Rust; this is the very first thing I've written in the language. I was particularly unsure that how I dealt
with the various `null` handles and structure creations was idiomatic. Then I found the `windows-rs` sample
[create_window](https://github.com/microsoft/windows-rs/blob/0.48.0/crates/samples/windows/create_window/src/main.rs)
and did some cleaning up to get the code in the repo. The main changes I made were to `None` instead of several other
approaches for nulls; the question mark operator instead of `unwrap`; and I moved `unsafe` to the entire main function.

Next, [let's do some animation](rustrast-02/).

