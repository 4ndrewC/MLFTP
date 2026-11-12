// Software renderer. Everything is drawn by the CPU into a 32-bit ARGB
// framebuffer that the platform layer then copies to the window.
//
// Shapes are anti-aliased by computing per-pixel coverage from a signed
// distance (circles, rings, capsule-shaped lines). Polygons use an even-odd
// scanline fill; outlines drawn on top give them smooth edges.
#pragma once

#include <span>
#include <string_view>
#include <vector>

#include "math.hpp"

namespace spark {

enum class Blend { Alpha, Add };
enum class Align { Left, Center, Right };

/** Primitive classes counted by CanvasStats (frame-time research instrumentation). */
enum class Prim : uint8_t { Clear, Rect, Circle, Ring, Glow, Line, Poly, Count };

/**
 * Work done by the canvas since the last resetStats(). Pixel counts are
 * physical pixels in the clipped bounding box each primitive iterates over
 * (spans for polygons), i.e. a proxy for per-pixel work. Counting costs a few
 * integer adds per primitive call, never per pixel.
 */
struct CanvasStats {
    uint32_t calls[size_t(Prim::Count)] = {};
    uint64_t px[size_t(Prim::Count)] = {};
    uint64_t glowReqPx = 0; // glow area requested at full quality (clipped), even if glow is reduced/off
    uint32_t textChars = 0;
};

class Canvas {
public:
    /** A canvas with logical size w x h and pixel scale 1. */
    Canvas(int w, int h);

    /** Pixel buffer size (logical size times the pixel scale). */
    int width() const { return w_; }
    int height() const { return h_; }
    /** Logical size: the coordinate space all drawing calls use. */
    int logicalWidth() const { return lw_; }
    int logicalHeight() const { return lh_; }

    /**
     * Render at `s` physical pixels per logical pixel (internal resolution).
     * Resizes the pixel buffer to round(logical * s); drawing coordinates stay
     * logical. s = 1 is the default and draws exactly as before.
     */
    void setPixelScale(float s);
    float pixelScale() const { return s_; }
    /** Pixel buffer size setPixelScale(s) would produce. */
    static int scaledSize(int logical, float s) { return std::max(1, int(float(logical) * s + 0.5f)); }

    /**
     * Anti-aliasing quality lever. On (default): per-pixel coverage from a signed
     * distance over each primitive's bounding box. Off: circles, rings and lines are
     * rasterised as aliased per-row spans (pixel centres inside the shape), which
     * skips the sqrt and, for diagonal lines, most of the bounding box.
     */
    void setAntialias(bool on) { aa_ = on; }
    bool antialias() const { return aa_; }

    /** Glow quality: radius multiplier for glow() (1 = full, 0 = glow off). */
    void setGlowScale(float g) { glowScale_ = g; }
    float glowScale() const { return glowScale_; }

    const CanvasStats &stats() const { return stats_; }
    void resetStats() { stats_ = {}; }
    const uint32_t *pixels() const { return px_.data(); }
    uint32_t pixel(int x, int y) const { return px_[size_t(y) * w_ + x]; }

    void setBlend(Blend b) { blend_ = b; }
    Blend blend() const { return blend_; }

    /** Offset applied to all drawing (camera position / screen shake). */
    void setOffset(Vec2 o) { off_ = o; }
    Vec2 offset() const { return off_; }

    void clear(Color c);
    void rect(float x, float y, float w, float h, Color c);
    void rectLine(float x, float y, float w, float h, float thick, Color c);
    void circle(Vec2 p, float r, Color c);               // filled
    void ring(Vec2 p, float r, float thick, Color c);    // outline
    void glow(Vec2 p, float r, Color c);                 // soft additive light
    void line(Vec2 a, Vec2 b, float thick, Color c);
    void polygon(std::span<const Vec2> pts, Color c);    // filled
    void polyline(std::span<const Vec2> pts, float thick, Color c, bool closed = true);
    /** Draw local-space points transformed by position, rotation and scale. */
    void shape(std::span<const Vec2> pts, Vec2 pos, float angle, float scale, Color fill, Color outline,
               float thick);
    /** Tint the whole screen, ignoring the offset. */
    void overlay(Color c);

    /** 5x7 bitmap font; `scale` is an integer pixel multiplier. */
    void text(float x, float y, std::string_view s, int scale, Color c, Align align = Align::Left);
    /** printf-style text. */
    void textf(float x, float y, int scale, Color c, Align align, const char *fmt, ...)
        __attribute__((format(printf, 7, 8)));
    static int textWidth(std::string_view s, int scale);

    /** Save as binary PPM (screenshots, tests). */
    bool savePPM(const char *path) const;

private:
    void plot(int x, int y, Color c, int cov);
    bool clipBox(float fx0, float fy0, float fx1, float fy1, int &x0, int &y0, int &x1, int &y1) const;
    /** Logical point -> pixel space (applies the offset, then the pixel scale). */
    Vec2 toPx(Vec2 p) const { return (p + off_) * s_; }
    void count(Prim p, int x0, int y0, int x1, int y1) {
        ++stats_.calls[size_t(p)];
        stats_.px[size_t(p)] += uint64_t(x1 - x0 + 1) * uint64_t(y1 - y0 + 1);
    }

    int lw_, lh_; // logical size
    int w_, h_;   // pixel buffer size
    float s_ = 1; // pixels per logical unit
    float glowScale_ = 1;
    bool aa_ = true;
    void spanRow(int y, float xa, float xb, Color c); // fill pixel centres in [xa, xb] on row y
    void lineAliased(Vec2 a, Vec2 b, float half, Color c);
    std::vector<uint32_t> px_;
    Blend blend_ = Blend::Alpha;
    Vec2 off_;
    CanvasStats stats_;
};

} // namespace spark
