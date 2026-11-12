// Platform window (Xlib). The X11 headers stay inside window_x11.cpp so their
// macros (None, Bool, Status, ...) never leak into engine or game code.
#pragma once

#include <memory>

#include "canvas.hpp"
#include "input.hpp"

namespace spark {

class Window {
public:
    Window();
    ~Window();
    Window(const Window &) = delete;
    Window &operator=(const Window &) = delete;

    /** Open a window sized logicalW*scale x logicalH*scale. */
    bool open(const char *title, int logicalW, int logicalH, int scale, bool showCursor);
    /** Process pending events into `input`. Returns false once the window is closed. */
    bool poll(Input &input);
    /** Scale the canvas to the window (letterboxed, nearest-neighbor) and show it. */
    void present(const Canvas &canvas);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace spark
