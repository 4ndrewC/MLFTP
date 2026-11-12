#include "r3d.hpp"

#include <algorithm>
#include <cmath>

namespace r3d {

// ---------------------------------------------------------------- Vec3 / Rgb

float Vec3::length() const { return std::sqrt(dot(*this)); }

Vec3 Vec3::normalized() const {
    float l = length();
    return l > 0 ? *this * (1.0f / l) : Vec3{};
}

Rgb Rgb::lerp(Rgb a, Rgb b, float t) { return {a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t}; }

// ---------------------------------------------------------------- Mat4

Mat4 Mat4::identity() {
    Mat4 r{};
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
    return r;
}

Mat4 Mat4::translate(Vec3 t) {
    Mat4 r = identity();
    r.m[3] = t.x;
    r.m[7] = t.y;
    r.m[11] = t.z;
    return r;
}

Mat4 Mat4::scale(float s) {
    Mat4 r = identity();
    r.m[0] = r.m[5] = r.m[10] = s;
    return r;
}

Mat4 Mat4::rotateX(float a) {
    Mat4 r = identity();
    float c = std::cos(a), s = std::sin(a);
    r.m[5] = c;
    r.m[6] = -s;
    r.m[9] = s;
    r.m[10] = c;
    return r;
}

Mat4 Mat4::rotateY(float a) {
    Mat4 r = identity();
    float c = std::cos(a), s = std::sin(a);
    r.m[0] = c;
    r.m[2] = s;
    r.m[8] = -s;
    r.m[10] = c;
    return r;
}

Mat4 Mat4::rotateZ(float a) {
    Mat4 r = identity();
    float c = std::cos(a), s = std::sin(a);
    r.m[0] = c;
    r.m[1] = -s;
    r.m[4] = s;
    r.m[5] = c;
    return r;
}

Mat4 Mat4::lookAt(Vec3 eye, Vec3 target, Vec3 up) {
    Vec3 f = (target - eye).normalized();
    Vec3 s = f.cross(up).normalized();
    Vec3 u = s.cross(f);
    Mat4 r = identity();
    r.m[0] = s.x;  r.m[1] = s.y;  r.m[2] = s.z;  r.m[3] = -s.dot(eye);
    r.m[4] = u.x;  r.m[5] = u.y;  r.m[6] = u.z;  r.m[7] = -u.dot(eye);
    r.m[8] = -f.x; r.m[9] = -f.y; r.m[10] = -f.z; r.m[11] = f.dot(eye);
    return r;
}

Mat4 Mat4::perspective(float fovY, float aspect, float zNear, float zFar) {
    Mat4 r{};
    float f = 1.0f / std::tan(fovY * 0.5f);
    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = (zFar + zNear) / (zNear - zFar);
    r.m[11] = 2.0f * zFar * zNear / (zNear - zFar);
    r.m[14] = -1.0f;
    return r;
}

Mat4 Mat4::operator*(const Mat4 &o) const {
    Mat4 r{};
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col) {
            float s = 0;
            for (int k = 0; k < 4; ++k) s += m[row * 4 + k] * o.m[k * 4 + col];
            r.m[row * 4 + col] = s;
        }
    return r;
}

Vec3 Mat4::point(Vec3 v) const {
    return {m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3],
            m[4] * v.x + m[5] * v.y + m[6] * v.z + m[7],
            m[8] * v.x + m[9] * v.y + m[10] * v.z + m[11]};
}

Vec3 Mat4::dir(Vec3 v) const {
    return {m[0] * v.x + m[1] * v.y + m[2] * v.z, m[4] * v.x + m[5] * v.y + m[6] * v.z,
            m[8] * v.x + m[9] * v.y + m[10] * v.z};
}

// ---------------------------------------------------------------- Mesh

void Mesh::addTri(Vec3 a, Vec3 b, Vec3 c, Rgb col, Vec3 inside) {
    Vec3 n = (b - a).cross(c - a);
    float len = n.length();
    if (len < 1e-12f) return;
    Vec3 centroid = (a + b + c) * (1.0f / 3.0f);
    if (n.dot(centroid - inside) < 0) {
        std::swap(b, c);
        n = -n;
    }
    n = n * (1.0f / len);
    v_.push_back({a, n, col});
    v_.push_back({b, n, col});
    v_.push_back({c, n, col});
}

void Mesh::addQuad(Vec3 a, Vec3 b, Vec3 c, Vec3 d, Rgb col, Vec3 inside) {
    addTri(a, b, c, col, inside);
    addTri(a, c, d, col, inside);
}

void Mesh::addBox(Vec3 o, Vec3 h, Rgb col) {
    // 8 corners: bit 0 = +x, bit 1 = +y, bit 2 = +z.
    auto corner = [&](int i) {
        return Vec3(o.x + ((i & 1) ? h.x : -h.x), o.y + ((i & 2) ? h.y : -h.y), o.z + ((i & 4) ? h.z : -h.z));
    };
    // Each face in ring order, seen from outside.
    static constexpr int faces[6][4] = {{1, 5, 7, 3}, {0, 2, 6, 4}, {2, 3, 7, 6},
                                        {0, 4, 5, 1}, {4, 6, 7, 5}, {0, 1, 3, 2}};
    for (const auto &f : faces) addQuad(corner(f[0]), corner(f[1]), corner(f[2]), corner(f[3]), col, o);
}

// ---------------------------------------------------------------- clipping

namespace {
ClipVert lerpVert(const ClipVert &a, const ClipVert &b, float t) {
    ClipVert r;
    r.p = a.p + (b.p - a.p) * t;
    r.c = Rgb::lerp(a.c, b.c, t);
    r.fog = a.fog + (b.fog - a.fog) * t;
    return r;
}

// Clip-space to pixel coordinates, without the matrix product cost of a full Vec4.
void projectRaw(const Mat4 &P, Vec3 p, float &cx, float &cy, float &cw) {
    cx = P.m[0] * p.x + P.m[1] * p.y + P.m[2] * p.z + P.m[3];
    cy = P.m[4] * p.x + P.m[5] * p.y + P.m[6] * p.z + P.m[7];
    cw = P.m[12] * p.x + P.m[13] * p.y + P.m[14] * p.z + P.m[15];
}

uint32_t pack(float r, float g, float b) {
    auto q = [](float v) { return uint32_t(int(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f)); };
    return 0xff000000u | (q(r) << 16) | (q(g) << 8) | q(b);
}
} // namespace

int clipPolygonNear(const ClipVert *in, int n, float zNear, ClipVert *out) {
    // Inside means z <= -zNear, i.e. distance d = -z - zNear >= 0.
    int m = 0;
    for (int i = 0; i < n; ++i) {
        const ClipVert &a = in[i];
        const ClipVert &b = in[(i + 1) % n];
        float da = -a.p.z - zNear, db = -b.p.z - zNear;
        if (da >= 0) out[m++] = a;
        if ((da >= 0) != (db >= 0)) {
            float t = da / (da - db);
            out[m++] = lerpVert(a, b, t);
        }
    }
    return m;
}

// ---------------------------------------------------------------- Renderer

Renderer::Renderer(int width, int height) : w_(width), h_(height), zbuf_(size_t(width) * size_t(height), 0.0f) {
    lightView_ = light_.dir.normalized();
    updateProjection();
}

void Renderer::updateProjection() { proj_ = Mat4::perspective(fovY_, float(w_) / float(h_), near_, 1000.0f); }

void Renderer::setCamera(Vec3 eye, Vec3 target, float fovY) {
    view_ = Mat4::lookAt(eye, target, Vec3(0, 1, 0));
    fovY_ = fovY;
    lightView_ = view_.dir(light_.dir).normalized();
    updateProjection();
}

void Renderer::setLight(const Light &l) {
    light_ = l;
    light_.dir = l.dir.normalized();
    lightView_ = view_.dir(light_.dir).normalized();
}

void Renderer::beginFrame(spark::Canvas &cv, Rgb top, Rgb horizon) {
    if (cv.width() != w_ || cv.height() != h_) {
        w_ = cv.width();
        h_ = cv.height();
        zbuf_.assign(size_t(w_) * size_t(h_), 0.0f);
        updateProjection();
    }
    // The canvas owns its pixel buffer; the renderer writes into it in place.
    dst_ = const_cast<uint32_t *>(cv.pixels());
    std::fill(zbuf_.begin(), zbuf_.end(), 0.0f);
    st_ = {};

    const int horizonRows = std::max(1, int(float(h_) * 0.42f));
    for (int y = 0; y < h_; ++y) {
        float t = std::min(1.0f, (float(y) + 0.5f) / float(horizonRows));
        Rgb c = Rgb::lerp(top, horizon, t);
        uint32_t p = pack(c.r, c.g, c.b);
        std::fill(dst_ + size_t(y) * size_t(w_), dst_ + size_t(y + 1) * size_t(w_), p);
    }
    // Static stars: positions come from an integer hash, so they never move or flicker.
    for (uint32_t i = 1; i <= 70; ++i) {
        uint32_t h = i * 2654435761u;
        uint32_t x = (h >> 4) % uint32_t(w_);
        uint32_t y = (h >> 13) % uint32_t(horizonRows);
        float b = 0.35f + 0.5f * float((h >> 7) & 7) / 7.0f;
        dst_[size_t(y) * size_t(w_) + x] = pack(b, b * 1.05f, b * 1.2f);
    }
}

bool Renderer::project(Vec3 p, float &sx, float &sy) const {
    if (-p.z < near_) return false;
    float cx, cy, cw;
    projectRaw(proj_, p, cx, cy, cw);
    sx = (cx / cw * 0.5f + 0.5f) * float(w_);
    sy = (0.5f - cy / cw * 0.5f) * float(h_);
    return true;
}

void Renderer::draw(const Mesh &mesh, const DrawOpts &o) {
    if (!dst_) return;
    const Mat4 mv = view_ * o.model;
    const auto &vs = mesh.vertices();
    const int nt = mesh.triangles();
    for (int t = 0; t < nt; ++t) {
        ClipVert v[3];
        for (int k = 0; k < 3; ++k) {
            const Vertex &src = vs[size_t(t) * 3 + size_t(k)];
            ClipVert &cv = v[k];
            cv.p = mv.point(src.pos);
            Vec3 n = mv.dir(src.nrm).normalized();
            float ndl = std::max(0.0f, n.dot(lightView_));
            float shade = o.unlit ? 1.25f : light_.ambient + light_.diffuse * ndl;
            cv.c = (src.col * o.tint) * shade;
            cv.fog = std::clamp((-cv.p.z - fog_.start) / (fog_.end - fog_.start), 0.0f, 1.0f);
        }
        ++st_.submitted;

        // Back-face test in view space: the eye is at the origin, so a front face has n . p0 < 0.
        Vec3 fn = (v[1].p - v[0].p).cross(v[2].p - v[0].p);
        if (fn.dot(v[0].p) >= 0) {
            ++st_.culled;
            continue;
        }

        int inside = 0;
        for (const ClipVert &cv : v) inside += cv.p.z <= -near_ ? 1 : 0;
        if (inside == 3) {
            emit(v);
        } else if (inside > 0) {
            ClipVert poly[4];
            int n = clipPolygonNear(v, 3, near_, poly);
            ++st_.clipped;
            for (int i = 1; i + 1 < n; ++i) {
                ClipVert tri[3] = {poly[0], poly[i], poly[i + 1]};
                emit(tri);
            }
        }
    }
}

void Renderer::emit(const ClipVert *tri) {
    Screen s[3];
    for (int k = 0; k < 3; ++k) {
        float cx, cy, cw;
        projectRaw(proj_, tri[k].p, cx, cy, cw);
        float iw = 1.0f / cw; // cw = -z > 0 after clipping
        s[k].x = (cx * iw * 0.5f + 0.5f) * float(w_);
        s[k].y = (0.5f - cy * iw * 0.5f) * float(h_);
        s[k].iz = iw;
        s[k].r = tri[k].c.r * iw;
        s[k].g = tri[k].c.g * iw;
        s[k].b = tri[k].c.b * iw;
        s[k].f = tri[k].fog * iw;
    }
    rasterize(s);
}

void Renderer::rasterize(const Screen (&v)[3]) {
    float minx = std::min({v[0].x, v[1].x, v[2].x}), maxx = std::max({v[0].x, v[1].x, v[2].x});
    float miny = std::min({v[0].y, v[1].y, v[2].y}), maxy = std::max({v[0].y, v[1].y, v[2].y});
    float fx0 = std::max(0.0f, std::floor(minx)), fx1 = std::min(float(w_ - 1), std::ceil(maxx));
    float fy0 = std::max(0.0f, std::floor(miny)), fy1 = std::min(float(h_ - 1), std::ceil(maxy));
    if (fx0 > fx1 || fy0 > fy1) return;

    const float area = (v[1].x - v[0].x) * (v[2].y - v[0].y) - (v[1].y - v[0].y) * (v[2].x - v[0].x);
    if (std::fabs(area) < 1e-9f) return;
    const float inv = 1.0f / area;
    ++st_.drawn;

    const int x0 = int(fx0), x1 = int(fx1), y0 = int(fy0), y1 = int(fy1);
    const Rgb fogc = fog_.color;
    for (int py = y0; py <= y1; ++py) {
        const float fy = float(py) + 0.5f;
        for (int px = x0; px <= x1; ++px) {
            const float fx = float(px) + 0.5f;
            // Barycentric weights from edge functions (1 at the opposite vertex).
            const float b0 = ((v[2].x - v[1].x) * (fy - v[1].y) - (v[2].y - v[1].y) * (fx - v[1].x)) * inv;
            const float b1 = ((v[0].x - v[2].x) * (fy - v[2].y) - (v[0].y - v[2].y) * (fx - v[2].x)) * inv;
            const float b2 = 1.0f - b0 - b1;
            if (b0 < 0 || b1 < 0 || b2 < 0) continue;

            const size_t idx = size_t(py) * size_t(w_) + size_t(px);
            const float iz = b0 * v[0].iz + b1 * v[1].iz + b2 * v[2].iz; // 1/depth, linear in screen space
            if (iz <= zbuf_[idx]) continue;
            zbuf_[idx] = iz;

            const float z = 1.0f / iz;
            const float r = (b0 * v[0].r + b1 * v[1].r + b2 * v[2].r) * z;
            const float g = (b0 * v[0].g + b1 * v[1].g + b2 * v[2].g) * z;
            const float b = (b0 * v[0].b + b1 * v[1].b + b2 * v[2].b) * z;
            const float f = std::clamp((b0 * v[0].f + b1 * v[1].f + b2 * v[2].f) * z, 0.0f, 1.0f);
            dst_[idx] = pack(r + (fogc.r - r) * f, g + (fogc.g - g) * f, b + (fogc.b - b) * f);
            ++st_.pixels;
        }
    }
}

} // namespace r3d
