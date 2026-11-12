#include "canvas.hpp"

#include <cstdarg>
#include <cstdio>

namespace spark {

const uint8_t *fontGlyph(char ch); // font.cpp: 7 rows, low 5 bits each

namespace {
/** Coverage (0..256) from signed distance d (negative = inside), 1px AA band. */
inline int coverage(float d) {
    float c = 0.5f - d;
    if (c <= 0) return 0;
    if (c >= 1) return 256;
    return int(c * 256);
}
constexpr size_t kMaxPoly = 64;
} // namespace

Canvas::Canvas(int w, int h) : lw_(w), lh_(h), w_(w), h_(h), px_(size_t(w) * h, 0xff000000u) {}

void Canvas::setPixelScale(float s) {
    s_ = s;
    w_ = scaledSize(lw_, s);
    h_ = scaledSize(lh_, s);
    // The buffer only ever grows: switching back and forth between render scales
    // reuses it (the image occupies its first w_*h_ pixels), so a quality change
    // costs no allocation or page faults after the first visit to the largest scale.
    if (px_.size() < size_t(w_) * h_) px_.resize(size_t(w_) * h_, 0xff000000u);
}

void Canvas::plot(int x, int y, Color c, int cov) {
    if (unsigned(x) >= unsigned(w_) || unsigned(y) >= unsigned(h_)) return;
    int a = c.a() * cov >> 8; // 0..255
    if (a <= 0) return;
    uint32_t &d = px_[size_t(y) * w_ + x];
    int dr = (d >> 16) & 255, dg = (d >> 8) & 255, db = d & 255;
    if (blend_ == Blend::Add) {
        int a2 = a + (a >> 7); // map 0..255 to 0..256 so full alpha adds the full color
        dr = std::min(255, dr + (c.r() * a2 >> 8));
        dg = std::min(255, dg + (c.g() * a2 >> 8));
        db = std::min(255, db + (c.b() * a2 >> 8));
    } else {
        dr += (c.r() - dr) * a / 255;
        dg += (c.g() - dg) * a / 255;
        db += (c.b() - db) * a / 255;
    }
    d = 0xff000000u | uint32_t(dr) << 16 | uint32_t(dg) << 8 | uint32_t(db);
}

bool Canvas::clipBox(float fx0, float fy0, float fx1, float fy1, int &x0, int &y0, int &x1, int &y1) const {
    x0 = std::max(0, int(std::floor(fx0)));
    y0 = std::max(0, int(std::floor(fy0)));
    x1 = std::min(w_ - 1, int(std::ceil(fx1)));
    y1 = std::min(h_ - 1, int(std::ceil(fy1)));
    return x0 <= x1 && y0 <= y1;
}

void Canvas::clear(Color c) {
    count(Prim::Clear, 0, 0, w_ - 1, h_ - 1);
    std::fill(px_.begin(), px_.begin() + long(w_) * h_, c.argb | 0xff000000u);
}

void Canvas::rect(float x, float y, float w, float h, Color c) {
    x = (x + off_.x) * s_;
    y = (y + off_.y) * s_;
    w *= s_;
    h *= s_;
    // Pixel (px, py) is covered if its center lies inside the rect.
    int x0 = std::max(0, int(std::ceil(x - 0.5f))), x1 = std::min(w_ - 1, int(std::ceil(x + w - 0.5f)) - 1);
    int y0 = std::max(0, int(std::ceil(y - 0.5f))), y1 = std::min(h_ - 1, int(std::ceil(y + h - 0.5f)) - 1);
    if (x0 > x1 || y0 > y1) return;
    count(Prim::Rect, x0, y0, x1, y1);
    if (blend_ == Blend::Alpha && c.a() == 255) { // fast path: opaque fill
        for (int py = y0; py <= y1; ++py) std::fill(&px_[size_t(py) * w_ + x0], &px_[size_t(py) * w_ + x1] + 1, c.argb);
        return;
    }
    for (int py = y0; py <= y1; ++py)
        for (int px = x0; px <= x1; ++px) plot(px, py, c, 256);
}

void Canvas::rectLine(float x, float y, float w, float h, float t, Color c) {
    rect(x, y, w, t, c);
    rect(x, y + h - t, w, t, c);
    rect(x, y + t, t, h - 2 * t, c);
    rect(x + w - t, y + t, t, h - 2 * t, c);
}

void Canvas::overlay(Color c) {
    // Covers the whole pixel buffer whatever the pixel scale.
    Vec2 saved = off_;
    float savedScale = s_;
    off_ = {};
    s_ = 1;
    rect(0, 0, float(w_), float(h_), c);
    off_ = saved;
    s_ = savedScale;
}

void Canvas::spanRow(int y, float xa, float xb, Color c) {
    int x0 = std::max(0, int(std::ceil(xa - 0.5f))), x1 = std::min(w_ - 1, int(std::floor(xb - 0.5f)));
    if (x0 > x1) return;
    for (int x = x0; x <= x1; ++x) plot(x, y, c, 256);
}

void Canvas::circle(Vec2 p, float r, Color c) {
    if (r <= 0) return;
    p = toPx(p);
    r *= s_;
    if (!aa_) { // aliased: one span per row. Stats count the AA bounding box, so px_* mean the same in both modes.
        int bx0, by0, bx1, by1;
        if (!clipBox(p.x - r - 1, p.y - r - 1, p.x + r + 1, p.y + r + 1, bx0, by0, bx1, by1)) return;
        count(Prim::Circle, bx0, by0, bx1, by1);
        int y0 = std::max(0, int(std::ceil(p.y - r - 0.5f))), y1 = std::min(h_ - 1, int(std::floor(p.y + r - 0.5f)));
        for (int y = y0; y <= y1; ++y) {
            float dy = float(y) + 0.5f - p.y, hw2 = r * r - dy * dy;
            if (hw2 < 0) continue;
            float hw = std::sqrt(hw2);
            spanRow(y, p.x - hw, p.x + hw, c);
        }
        return;
    }
    int x0, y0, x1, y1;
    if (!clipBox(p.x - r - 1, p.y - r - 1, p.x + r + 1, p.y + r + 1, x0, y0, x1, y1)) return;
    count(Prim::Circle, x0, y0, x1, y1);
    float inner2 = r > 1 ? (r - 1) * (r - 1) : 0;
    for (int y = y0; y <= y1; ++y) {
        float dy = float(y) + 0.5f - p.y;
        for (int x = x0; x <= x1; ++x) {
            float dx = float(x) + 0.5f - p.x;
            float d2 = dx * dx + dy * dy;
            plot(x, y, c, d2 <= inner2 ? 256 : coverage(std::sqrt(d2) - r)); // no sqrt in the interior
        }
    }
}

void Canvas::ring(Vec2 p, float r, float thick, Color c) {
    if (r <= 0) return;
    p = toPx(p);
    r *= s_;
    thick *= s_;
    if (!aa_) { // aliased: up to two spans per row between the inner and outer circles
        float ro = r + thick * 0.5f, ri = std::max(0.0f, r - thick * 0.5f), outer = ro + 1;
        int bx0, by0, bx1, by1;
        if (!clipBox(p.x - outer, p.y - outer, p.x + outer, p.y + outer, bx0, by0, bx1, by1)) return;
        count(Prim::Ring, bx0, by0, bx1, by1);
        int y0 = std::max(0, int(std::ceil(p.y - ro - 0.5f))), y1 = std::min(h_ - 1, int(std::floor(p.y + ro - 0.5f)));
        for (int y = y0; y <= y1; ++y) {
            float dy = float(y) + 0.5f - p.y, o2 = ro * ro - dy * dy;
            if (o2 < 0) continue;
            float wo = std::sqrt(o2), i2 = ri * ri - dy * dy;
            if (i2 <= 0) {
                spanRow(y, p.x - wo, p.x + wo, c);
            } else {
                float wi = std::sqrt(i2);
                spanRow(y, p.x - wo, p.x - wi, c);
                spanRow(y, p.x + wi, p.x + wo, c);
            }
        }
        return;
    }
    float half = thick * 0.5f, outer = r + half + 1;
    int x0, y0, x1, y1;
    if (!clipBox(p.x - outer, p.y - outer, p.x + outer, p.y + outer, x0, y0, x1, y1)) return;
    count(Prim::Ring, x0, y0, x1, y1);
    float in = r - half - 1, in2 = in > 0 ? in * in : -1, out2 = outer * outer;
    for (int y = y0; y <= y1; ++y) {
        float dy = float(y) + 0.5f - p.y;
        for (int x = x0; x <= x1; ++x) {
            float dx = float(x) + 0.5f - p.x;
            float d2 = dx * dx + dy * dy;
            if (d2 < in2 || d2 > out2) continue;
            plot(x, y, c, coverage(std::fabs(std::sqrt(d2) - r) - half));
        }
    }
}

void Canvas::glow(Vec2 p, float r, Color c) {
    if (r <= 0) return;
    p = toPx(p);
    r *= s_;
    int x0, y0, x1, y1;
    if (!clipBox(p.x - r, p.y - r, p.x + r, p.y + r, x0, y0, x1, y1)) return;
    stats_.glowReqPx += uint64_t(x1 - x0 + 1) * uint64_t(y1 - y0 + 1);
    if (glowScale_ != 1) { // reduced glow quality: smaller light, or none
        r *= glowScale_;
        if (r <= 0 || !clipBox(p.x - r, p.y - r, p.x + r, p.y + r, x0, y0, x1, y1)) {
            ++stats_.calls[size_t(Prim::Glow)];
            return;
        }
    }
    count(Prim::Glow, x0, y0, x1, y1);
    Blend prev = blend_;
    blend_ = Blend::Add;
    float inv = 1.0f / (r * r);
    for (int y = y0; y <= y1; ++y) {
        float dy = float(y) + 0.5f - p.y;
        for (int x = x0; x <= x1; ++x) {
            float dx = float(x) + 0.5f - p.x;
            float t = 1.0f - (dx * dx + dy * dy) * inv; // 1 at center, 0 at edge
            if (t > 0) plot(x, y, c, int(t * t * 256));
        }
    }
    blend_ = prev;
}

void Canvas::line(Vec2 a, Vec2 b, float thick, Color c) {
    a = toPx(a);
    b = toPx(b);
    thick *= s_;
    float half = thick * 0.5f, pad = half + 1;
    if (!aa_) {
        int bx0, by0, bx1, by1;
        if (!clipBox(std::min(a.x, b.x) - pad, std::min(a.y, b.y) - pad, std::max(a.x, b.x) + pad,
                     std::max(a.y, b.y) + pad, bx0, by0, bx1, by1))
            return;
        count(Prim::Line, bx0, by0, bx1, by1);
        lineAliased(a, b, std::max(0.5f, half), c);
        return;
    }
    int x0, y0, x1, y1;
    if (!clipBox(std::min(a.x, b.x) - pad, std::min(a.y, b.y) - pad, std::max(a.x, b.x) + pad,
                 std::max(a.y, b.y) + pad, x0, y0, x1, y1))
        return;
    count(Prim::Line, x0, y0, x1, y1);
    Vec2 e = b - a;
    float len2 = e.lengthSq(), inv = len2 > 0 ? 1.0f / len2 : 0;
    for (int y = y0; y <= y1; ++y) {
        float py = float(y) + 0.5f - a.y;
        for (int x = x0; x <= x1; ++x) {
            float px = float(x) + 0.5f - a.x;
            // distance from the pixel center to the segment (capsule SDF)
            float t = std::clamp((px * e.x + py * e.y) * inv, 0.0f, 1.0f);
            float dx = px - e.x * t, dy = py - e.y * t;
            float d = std::sqrt(dx * dx + dy * dy) - half;
            if (d < 0.5f) plot(x, y, c, coverage(d));
        }
    }
}

void Canvas::lineAliased(Vec2 a, Vec2 b, float half, Color c) {
    // Capsule (segment a-b, radius `half`) as per-row spans: the union of the two end
    // discs and the swept rectangle. The capsule is convex, so each row is one span.
    Vec2 e = b - a;
    float len = e.length();
    Vec2 n = len > 1e-6f ? Vec2{-e.y, e.x} * (half / len) : Vec2{0, 0};
    const Vec2 quad[4] = {a + n, b + n, b - n, a - n};
    float ymin = std::min(a.y, b.y) - half, ymax = std::max(a.y, b.y) + half;
    int y0 = std::max(0, int(std::ceil(ymin - 0.5f))), y1 = std::min(h_ - 1, int(std::floor(ymax - 0.5f)));
    for (int y = y0; y <= y1; ++y) {
        float py = float(y) + 0.5f, lo = 1e30f, hi = -1e30f;
        for (Vec2 cpt : {a, b}) { // end discs
            float dy = py - cpt.y, w2 = half * half - dy * dy;
            if (w2 >= 0) {
                float w = std::sqrt(w2);
                lo = std::min(lo, cpt.x - w);
                hi = std::max(hi, cpt.x + w);
            }
        }
        if (len > 1e-6f)
            for (int i = 0, j = 3; i < 4; j = i++) { // rectangle edges crossing this row
                float yi = quad[i].y, yj = quad[j].y;
                if ((yi <= py) != (yj <= py)) {
                    float x = quad[i].x + (py - yi) / (yj - yi) * (quad[j].x - quad[i].x);
                    lo = std::min(lo, x);
                    hi = std::max(hi, x);
                }
            }
        if (lo <= hi) spanRow(y, lo, hi, c);
    }
}

void Canvas::polygon(std::span<const Vec2> pts, Color c) {
    size_t n = pts.size();
    if (n < 3 || n > kMaxPoly) return;
    Vec2 tp[kMaxPoly]; // pixel space
    float miny = 1e9f, maxy = -1e9f;
    for (size_t i = 0; i < n; ++i) {
        tp[i] = toPx(pts[i]);
        miny = std::min(miny, tp[i].y);
        maxy = std::max(maxy, tp[i].y);
    }
    int y0 = std::max(0, int(std::floor(miny))), y1 = std::min(h_ - 1, int(std::ceil(maxy)));
    float xs[kMaxPoly];
    uint64_t spanPx = 0;
    for (int y = y0; y <= y1; ++y) {
        float sy = float(y) + 0.5f;
        size_t k = 0;
        for (size_t i = 0, j = n - 1; i < n; j = i++) {
            float yi = tp[i].y, yj = tp[j].y;
            if ((yi <= sy) != (yj <= sy)) {
                float xi = tp[i].x, xj = tp[j].x;
                xs[k++] = xi + (sy - yi) / (yj - yi) * (xj - xi);
            }
        }
        std::sort(xs, xs + k);
        for (size_t i = 0; i + 1 < k; i += 2) {
            int xa = std::max(0, int(std::ceil(xs[i] - 0.5f)));
            int xb = std::min(w_ - 1, int(std::floor(xs[i + 1] - 0.5f)));
            if (xb >= xa) spanPx += uint64_t(xb - xa + 1);
            for (int x = xa; x <= xb; ++x) plot(x, y, c, 256);
        }
    }
    ++stats_.calls[size_t(Prim::Poly)];
    stats_.px[size_t(Prim::Poly)] += spanPx;
}

void Canvas::polyline(std::span<const Vec2> pts, float thick, Color c, bool closed) {
    size_t n = pts.size();
    if (n < 2) return;
    for (size_t i = 1; i < n; ++i) line(pts[i - 1], pts[i], thick, c);
    if (closed && n > 2) line(pts[n - 1], pts[0], thick, c);
}

void Canvas::shape(std::span<const Vec2> pts, Vec2 pos, float angle, float scale, Color fill, Color outline,
                   float thick) {
    if (pts.size() > kMaxPoly) return;
    Vec2 tp[kMaxPoly];
    float cs = std::cos(angle) * scale, sn = std::sin(angle) * scale;
    for (size_t i = 0; i < pts.size(); ++i)
        tp[i] = {pos.x + pts[i].x * cs - pts[i].y * sn, pos.y + pts[i].x * sn + pts[i].y * cs};
    std::span<const Vec2> s(tp, pts.size());
    if (fill.a()) polygon(s, fill);
    if (outline.a() && thick > 0) polyline(s, thick, outline);
}

int Canvas::textWidth(std::string_view s, int scale) {
    return s.empty() ? 0 : (int(s.size()) * 6 - 1) * std::max(1, scale);
}

void Canvas::text(float x, float y, std::string_view s, int scale, Color c, Align align) {
    scale = std::max(1, scale);
    float w = float(textWidth(s, scale));
    if (align == Align::Center) x -= w / 2;
    else if (align == Align::Right) x -= w;
    x = std::floor(x);
    y = std::floor(y);
    float sc = float(scale);
    stats_.textChars += uint32_t(s.size());
    for (char ch : s) {
        if (const uint8_t *g = fontGlyph(ch))
            for (int row = 0; row < 7; ++row)
                for (int col = 0; col < 5; ++col)
                    if (g[row] & (0x10 >> col)) rect(x + float(col) * sc, y + float(row) * sc, sc, sc, c);
        x += 6 * sc;
    }
}

void Canvas::textf(float x, float y, int scale, Color c, Align align, const char *fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    text(x, y, buf, scale, c, align);
}

bool Canvas::savePPM(const char *path) const {
    FILE *f = std::fopen(path, "wb");
    if (!f) return false;
    std::fprintf(f, "P6\n%d %d\n255\n", w_, h_);
    std::vector<unsigned char> row(size_t(w_) * 3);
    for (int y = 0; y < h_; ++y) {
        for (int x = 0; x < w_; ++x) {
            uint32_t p = px_[size_t(y) * w_ + x];
            row[x * 3 + 0] = (p >> 16) & 255;
            row[x * 3 + 1] = (p >> 8) & 255;
            row[x * 3 + 2] = p & 255;
        }
        std::fwrite(row.data(), 1, row.size(), f);
    }
    return std::fclose(f) == 0;
}

} // namespace spark
