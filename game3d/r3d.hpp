// r3d: a small CPU software renderer for triangle meshes. It draws into the
// engine's spark::Canvas pixel buffer, so 2D text and rects drawn afterwards
// (HUD) sit on top of the 3D view with the usual engine calls.
//
// Pipeline, per draw call:
//   1. model, then view: vertices go from mesh space to view space (camera at
//      the origin looking down -z). Normals use the same matrix (uniform scale
//      only), are normalised, and lit per vertex: ambient + one directional
//      light (Gouraud). Fog is a per-vertex factor from view depth.
//   2. back-face culling: a triangle whose face normal points away from the eye
//      is dropped (meshes are closed/convex and wound outward, see Mesh).
//   3. near-plane clipping: triangles crossing z = -near are cut with
//      Sutherland-Hodgman into one or two triangles; colour and fog are
//      interpolated along the cut.
//   4. projection: perspective matrix, divide by w, viewport transform.
//   5. rasterisation: bounding box + edge functions over pixel centres. A
//      z-buffer holds 1/w (larger = nearer). Colour and fog are interpolated
//      perspective-correctly, fog is blended towards the fog colour, and the
//      pixel is written as opaque ARGB.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "canvas.hpp"

namespace r3d {

struct Vec3 {
    float x = 0, y = 0, z = 0;

    constexpr Vec3() = default;
    constexpr Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    constexpr Vec3 operator+(Vec3 o) const { return {x + o.x, y + o.y, z + o.z}; }
    constexpr Vec3 operator-(Vec3 o) const { return {x - o.x, y - o.y, z - o.z}; }
    constexpr Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    constexpr Vec3 operator-() const { return {-x, -y, -z}; }
    constexpr float dot(Vec3 o) const { return x * o.x + y * o.y + z * o.z; }
    constexpr Vec3 cross(Vec3 o) const { return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x}; }
    float length() const;
    Vec3 normalized() const;
};
constexpr Vec3 operator*(float s, Vec3 v) { return v * s; }

/** Linear RGB in 0..1 (may exceed 1 before the final clamp). */
struct Rgb {
    float r = 0, g = 0, b = 0;

    constexpr Rgb() = default;
    constexpr Rgb(float r_, float g_, float b_) : r(r_), g(g_), b(b_) {}
    constexpr Rgb operator+(Rgb o) const { return {r + o.r, g + o.g, b + o.b}; }
    constexpr Rgb operator*(float s) const { return {r * s, g * s, b * s}; }
    constexpr Rgb operator*(Rgb o) const { return {r * o.r, g * o.g, b * o.b}; }
    static Rgb lerp(Rgb a, Rgb b, float t);
    /** From 0xRRGGBB (sRGB-ish values, used as-is). */
    static constexpr Rgb hex(uint32_t rgb) {
        return {float((rgb >> 16) & 255) / 255.0f, float((rgb >> 8) & 255) / 255.0f, float(rgb & 255) / 255.0f};
    }
};

/** Row-major 4x4 matrix; points are transformed as column vectors (M * v). */
struct Mat4 {
    float m[16];

    static Mat4 identity();
    static Mat4 translate(Vec3 t);
    static Mat4 scale(float s);
    static Mat4 rotateX(float a);
    static Mat4 rotateY(float a);
    static Mat4 rotateZ(float a);
    /** Rigid view matrix: camera at `eye` looking at `target`, camera looks down -z. */
    static Mat4 lookAt(Vec3 eye, Vec3 target, Vec3 up);
    /** OpenGL-style perspective: clip.w = -z_view. */
    static Mat4 perspective(float fovY, float aspect, float zNear, float zFar);

    Mat4 operator*(const Mat4 &o) const;
    /** Affine point transform (w = 1). */
    Vec3 point(Vec3 v) const;
    /** Direction transform (w = 0): the upper 3x3 only. */
    Vec3 dir(Vec3 v) const;
};

struct Vertex {
    Vec3 pos;
    Vec3 nrm; // unit normal (flat per face for the procedural meshes)
    Rgb col;
};

/** Triangle list (3 vertices per triangle). Procedural meshes are built from convex faces. */
class Mesh {
public:
    /**
     * Adds a triangle with a flat normal. The winding is chosen so the normal points away
     * from `inside`, a point on the back side of the triangle (the centre of a closed
     * convex object, or a point below a ground tile). Degenerate triangles are skipped.
     */
    void addTri(Vec3 a, Vec3 b, Vec3 c, Rgb col, Vec3 inside);
    /** Quad with corners in ring order (a, b, c, d); split into two triangles. */
    void addQuad(Vec3 a, Vec3 b, Vec3 c, Vec3 d, Rgb col, Vec3 inside);
    /** Axis-aligned box centred on `center` with half sizes `half`: 12 triangles. */
    void addBox(Vec3 center, Vec3 half, Rgb col);

    void clear() { v_.clear(); }
    int triangles() const { return int(v_.size() / 3); }
    const std::vector<Vertex> &vertices() const { return v_; }

private:
    std::vector<Vertex> v_;
};

struct Light {
    Vec3 dir{-0.35f, 0.8f, 0.5f}; // world space, towards the light (normalised by setLight)
    float ambient = 0.35f;
    float diffuse = 0.75f;
};

struct Fog {
    Rgb color;
    float start = 14, end = 40; // view depth range; fully fogged at `end`
};

struct DrawOpts {
    Mat4 model = Mat4::identity();
    Rgb tint{1, 1, 1};   // multiplies the vertex colours
    bool unlit = false;  // emissive: colour * 1.25 instead of lighting
};

/** A vertex after view transform, with its lit colour and fog factor (for clipping). */
struct ClipVert {
    Vec3 p;   // view space
    Rgb c;
    float fog = 0;
};

struct FrameStats {
    long submitted = 0; // triangles that went through the transform
    long culled = 0;    // back-facing
    long clipped = 0;   // crossed the near plane and were cut
    long drawn = 0;     // reached the rasteriser with a non-empty bounding box
    long pixels = 0;    // pixels written (passed the z-test)
};

/**
 * Near-plane clip of a convex polygon (Sutherland-Hodgman). `in` has n vertices
 * (n <= 3); `out` must hold at least 4 entries. Returns the output vertex count,
 * which is 0 (fully outside), or at least 3 (fan-triangulate it with n - 2 triangles).
 */
int clipPolygonNear(const ClipVert *in, int n, float zNear, ClipVert *out);

class Renderer {
public:
    Renderer(int width, int height);

    /** Camera for the coming frame. fovY in radians (vertical). */
    void setCamera(Vec3 eye, Vec3 target, float fovY);
    void setLight(const Light &l);
    void setFog(const Fog &f) { fog_ = f; }

    /**
     * Start a frame on `cv`: fills the sky (a vertical gradient from `top` down to
     * `horizon`, with stars), clears the z-buffer and resets the frame stats.
     */
    void beginFrame(spark::Canvas &cv, Rgb top, Rgb horizon);
    void draw(const Mesh &mesh, const DrawOpts &opts);

    /** Project a view-space point to pixel coordinates. False if it is not in front of the camera. */
    bool project(Vec3 viewPos, float &sx, float &sy) const;
    Mat4 view() const { return view_; }
    const FrameStats &stats() const { return st_; }
    int width() const { return w_; }
    int height() const { return h_; }

private:
    struct Screen {
        float x, y, iz; // pixel position and 1/w
        float r, g, b, f; // colour and fog, already multiplied by 1/w
    };
    void emit(const ClipVert *tri);
    void rasterize(const Screen (&v)[3]);
    void updateProjection();

    int w_, h_;
    float fovY_ = 1.0f;
    float near_ = 0.1f;
    Mat4 view_ = Mat4::identity(), proj_ = Mat4::identity();
    Vec3 lightView_;
    Light light_;
    Fog fog_;
    std::vector<float> zbuf_;
    uint32_t *dst_ = nullptr;
    FrameStats st_;
};

} // namespace r3d
