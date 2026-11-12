// Xlib platform layer. Presents the software framebuffer with MIT-SHM
// (shared memory, no copy through the socket) when available, falling back
// to plain XPutImage otherwise.

// Engine headers first: X11 defines macros like `None` that would clash.
#include "window.hpp"

#include <sys/ipc.h>
#include <sys/shm.h>

#include <X11/XKBlib.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XShm.h>
#include <X11/keysym.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace spark {

namespace {

bool gShmError = false;
int shmErrorHandler(Display *, XErrorEvent *) {
    gShmError = true;
    return 0;
}

Key mapKey(KeySym ks) {
    if (ks >= XK_a && ks <= XK_z) return Key(int(Key::A) + int(ks - XK_a));
    if (ks >= XK_A && ks <= XK_Z) return Key(int(Key::A) + int(ks - XK_A));
    if (ks >= XK_0 && ks <= XK_9) return Key(int(Key::Num0) + int(ks - XK_0));
    switch (ks) {
    case XK_space: return Key::Space;
    case XK_Return: case XK_KP_Enter: return Key::Enter;
    case XK_Escape: return Key::Escape;
    case XK_Tab: return Key::Tab;
    case XK_BackSpace: return Key::Backspace;
    case XK_Up: return Key::Up;
    case XK_Down: return Key::Down;
    case XK_Left: return Key::Left;
    case XK_Right: return Key::Right;
    case XK_Shift_L: case XK_Shift_R: return Key::Shift;
    case XK_Control_L: case XK_Control_R: return Key::Ctrl;
    case XK_Alt_L: case XK_Alt_R: return Key::Alt;
    case XK_F1: return Key::F1;
    case XK_F2: return Key::F2;
    case XK_F3: return Key::F3;
    case XK_F4: return Key::F4;
    default: return Key(0); // Key::None (X11 #defines None)
    }
}

} // namespace

struct Window::Impl {
    Display *dpy = nullptr;
    ::Window win = 0;
    GC gc = nullptr;
    Visual *visual = nullptr;
    int depth = 0;
    Atom wmDelete = 0;
    bool shmAvailable = false;

    // Backbuffer the size of the window.
    XImage *image = nullptr;
    XShmSegmentInfo shm{};
    bool usingShm = false;
    int winW = 0, winH = 0;

    // Logical -> window mapping (letterbox).
    int logicalW = 0, logicalH = 0;
    float scale = 1;
    int offX = 0, offY = 0, dstW = 0, dstH = 0;
    std::vector<int> xmap;
    int xmapSrcW = 0; // canvas width xmap was built for (changes with the render scale)

    void destroyImage() {
        if (!image) return;
        if (usingShm) {
            XShmDetach(dpy, &shm);
            XDestroyImage(image); // does not free shm memory
            shmdt(shm.shmaddr);
        } else {
            XDestroyImage(image); // frees the malloc'd data
        }
        image = nullptr;
        usingShm = false;
    }

    void createImage(int w, int h) {
        destroyImage();
        winW = std::max(1, w);
        winH = std::max(1, h);

        if (shmAvailable) {
            image = XShmCreateImage(dpy, visual, unsigned(depth), ZPixmap, nullptr, &shm, unsigned(winW), unsigned(winH));
            if (image) {
                shm.shmid = shmget(IPC_PRIVATE, size_t(image->bytes_per_line) * image->height, IPC_CREAT | 0600);
                if (shm.shmid >= 0) {
                    shm.shmaddr = image->data = static_cast<char *>(shmat(shm.shmid, nullptr, 0));
                    shm.readOnly = False;
                    gShmError = false;
                    auto old = XSetErrorHandler(shmErrorHandler);
                    XShmAttach(dpy, &shm);
                    XSync(dpy, False);
                    XSetErrorHandler(old);
                    shmctl(shm.shmid, IPC_RMID, nullptr); // freed once both sides detach
                    if (!gShmError) {
                        usingShm = true;
                    } else {
                        shmdt(shm.shmaddr);
                        image->data = nullptr;
                        XDestroyImage(image);
                        image = nullptr;
                        shmAvailable = false; // e.g. remote display; stop trying
                    }
                } else {
                    XDestroyImage(image);
                    image = nullptr;
                }
            }
        }
        if (!image) {
            char *data = static_cast<char *>(std::calloc(size_t(winW) * winH, 4));
            image = XCreateImage(dpy, visual, unsigned(depth), ZPixmap, 0, data, unsigned(winW), unsigned(winH), 32, 0);
        }
        std::memset(image->data, 0, size_t(image->bytes_per_line) * image->height);

        // Letterbox: largest scale that fits, centered.
        scale = std::min(float(winW) / float(logicalW), float(winH) / float(logicalH));
        dstW = std::max(1, int(float(logicalW) * scale));
        dstH = std::max(1, int(float(logicalH) * scale));
        offX = (winW - dstW) / 2;
        offY = (winH - dstH) / 2;
        buildXmap(logicalW);
    }

    void buildXmap(int srcW) {
        xmapSrcW = srcW;
        xmap.resize(size_t(dstW));
        for (int x = 0; x < dstW; ++x) xmap[size_t(x)] = std::min(srcW - 1, x * srcW / dstW);
    }

    Vec2 toLogical(int x, int y) const { return {float(x - offX) / scale, float(y - offY) / scale}; }
};

Window::Window() : impl_(std::make_unique<Impl>()) {}

Window::~Window() {
    if (!impl_->dpy) return;
    impl_->destroyImage();
    if (impl_->gc) XFreeGC(impl_->dpy, impl_->gc);
    if (impl_->win) XDestroyWindow(impl_->dpy, impl_->win);
    XCloseDisplay(impl_->dpy);
}

bool Window::open(const char *title, int logicalW, int logicalH, int scale, bool showCursor) {
    Impl &m = *impl_;
    m.dpy = XOpenDisplay(nullptr);
    if (!m.dpy) {
        std::fprintf(stderr, "spark: cannot open X display (is DISPLAY set?)\n");
        return false;
    }
    int screen = DefaultScreen(m.dpy);
    m.visual = DefaultVisual(m.dpy, screen);
    m.depth = DefaultDepth(m.dpy, screen);
    if (m.depth < 24) {
        std::fprintf(stderr, "spark: need a 24/32-bit TrueColor display (got depth %d)\n", m.depth);
        return false;
    }
    m.logicalW = logicalW;
    m.logicalH = logicalH;
    int w = logicalW * std::max(1, scale), h = logicalH * std::max(1, scale);

    m.win = XCreateSimpleWindow(m.dpy, RootWindow(m.dpy, screen), 0, 0, unsigned(w), unsigned(h), 0, 0,
                                BlackPixel(m.dpy, screen));
    XStoreName(m.dpy, m.win, title);
    XSelectInput(m.dpy, m.win,
                 KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask |
                     StructureNotifyMask | FocusChangeMask | ExposureMask);
    m.wmDelete = XInternAtom(m.dpy, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(m.dpy, m.win, &m.wmDelete, 1);
    XkbSetDetectableAutoRepeat(m.dpy, True, nullptr); // no fake key-release on auto-repeat
    m.gc = XCreateGC(m.dpy, m.win, 0, nullptr);

    if (!showCursor) {
        static char blank[8] = {};
        Pixmap pm = XCreateBitmapFromData(m.dpy, m.win, blank, 8, 8);
        XColor black{};
        Cursor c = XCreatePixmapCursor(m.dpy, pm, pm, &black, &black, 0, 0);
        XDefineCursor(m.dpy, m.win, c);
        XFreeCursor(m.dpy, c);
        XFreePixmap(m.dpy, pm);
    }

    int major, minor;
    Bool pixmaps;
    m.shmAvailable = XShmQueryVersion(m.dpy, &major, &minor, &pixmaps) && !std::getenv("SPARK_NO_SHM");

    XMapWindow(m.dpy, m.win);
    m.createImage(w, h);
    XFlush(m.dpy);
    return true;
}

bool Window::poll(Input &input) {
    Impl &m = *impl_;
    while (XPending(m.dpy)) {
        XEvent ev;
        XNextEvent(m.dpy, &ev);
        switch (ev.type) {
        case KeyPress:
        case KeyRelease: {
            Key k = mapKey(XLookupKeysym(&ev.xkey, 0));
            if (k != Key(0)) input.setKey(k, ev.type == KeyPress);
            break;
        }
        case ButtonPress:
        case ButtonRelease: {
            Key k = ev.xbutton.button == Button1   ? Key::MouseLeft
                    : ev.xbutton.button == Button2 ? Key::MouseMiddle
                    : ev.xbutton.button == Button3 ? Key::MouseRight
                                                   : Key(0);
            input.setMouse(m.toLogical(ev.xbutton.x, ev.xbutton.y));
            if (k != Key(0)) input.setKey(k, ev.type == ButtonPress);
            break;
        }
        case MotionNotify: input.setMouse(m.toLogical(ev.xmotion.x, ev.xmotion.y)); break;
        case ConfigureNotify:
            if (ev.xconfigure.width != m.winW || ev.xconfigure.height != m.winH)
                m.createImage(ev.xconfigure.width, ev.xconfigure.height);
            break;
        case FocusOut: input.releaseAll(); break;
        case ClientMessage:
            if (Atom(ev.xclient.data.l[0]) == m.wmDelete) return false;
            break;
        default: break;
        }
    }
    return true;
}

void Window::present(const Canvas &cv) {
    Impl &m = *impl_;
    if (!m.image) return;
    const uint32_t *src = cv.pixels();
    const int lw = cv.width(), lh = cv.height();
    if (lw != m.xmapSrcW) m.buildXmap(lw);
    char *base = m.image->data;
    const int stride = m.image->bytes_per_line;

    // Nearest-neighbor scale into the letterboxed area. Rows that map to the
    // same source row are copied with memcpy instead of re-sampled.
    int prevSy = -1;
    uint32_t *prevRow = nullptr;
    for (int y = 0; y < m.dstH; ++y) {
        int sy = std::min(lh - 1, y * lh / m.dstH);
        auto *row = reinterpret_cast<uint32_t *>(base + size_t(y + m.offY) * stride) + m.offX;
        if (sy == prevSy) {
            std::memcpy(row, prevRow, size_t(m.dstW) * 4);
        } else {
            const uint32_t *s = src + size_t(sy) * lw;
            for (int x = 0; x < m.dstW; ++x) row[x] = s[m.xmap[size_t(x)]];
        }
        prevSy = sy;
        prevRow = row;
    }

    if (m.usingShm) XShmPutImage(m.dpy, m.win, m.gc, m.image, 0, 0, 0, 0, unsigned(m.winW), unsigned(m.winH), False);
    else XPutImage(m.dpy, m.win, m.gc, m.image, 0, 0, 0, 0, unsigned(m.winW), unsigned(m.winH));
    XSync(m.dpy, False); // the server must be done reading before we draw again
}

} // namespace spark
