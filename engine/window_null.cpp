// Window stub for builds without X11 (make HEADLESS_ONLY=1, e.g. cluster compute nodes
// that lack the X11 development headers). Headless runs never open a window; a windowed
// run fails cleanly in Window::open.
#include <cstdio>

#include "window.hpp"

namespace spark {

struct Window::Impl {};

Window::Window() : impl_(std::make_unique<Impl>()) {}
Window::~Window() = default;

bool Window::open(const char *, int, int, int, bool) {
    std::fprintf(stderr, "spark: built with HEADLESS_ONLY=1 (no X11): use --headless\n");
    return false;
}

bool Window::poll(Input &) { return false; }

void Window::present(const Canvas &) {}

} // namespace spark
