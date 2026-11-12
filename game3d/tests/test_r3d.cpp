// Renderer checks for Spark Drift: projection, near-plane clipping, z-buffer, culling and mesh counts.
// Run with `make test3d`. Prints one line per failure and a summary; exit code 1 on any failure.
#include <cmath>
#include <cstdio>

#include "../meshes.hpp"
#include "../r3d.hpp"

using namespace r3d;
using spark::Canvas;

namespace {

int gChecks = 0, gFailures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        ++gChecks;                                                        \
        if (!(cond)) {                                                    \
            ++gFailures;                                                  \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                 \
    } while (0)

bool near(float a, float b, float eps = 1e-3f) { return std::fabs(a - b) <= eps; }

// A triangle whose normal faces +z, i.e. towards a camera at the origin looking down -z.
Mesh frontTri(Vec3 a, Vec3 b, Vec3 c, Rgb col) {
    Mesh m;
    m.addTri(a, b, c, col, Vec3(0, 0, -1000));
    return m;
}

DrawOpts unlitOpts() {
    DrawOpts o;
    o.unlit = true;
    return o;
}

void testProjection() {
    Renderer r(64, 64);
    r.setCamera({0, 0, 0}, {0, 0, -1}, 1.5707963f); // 90 degrees vertical, square view
    float sx = 0, sy = 0;
    CHECK(r.project({0, 0, -10}, sx, sy));
    CHECK(near(sx, 32) && near(sy, 32)); // straight ahead -> centre
    CHECK(r.project({1, 0, -1}, sx, sy));
    CHECK(near(sx, 64) && near(sy, 32)); // x = 1 at depth 1 with fov 90 -> right edge
    CHECK(r.project({0, 1, -1}, sx, sy));
    CHECK(near(sx, 32) && near(sy, 0));  // y = 1 -> top edge (screen y grows downwards)
    CHECK(!r.project({0, 0, 1}, sx, sy)); // behind the camera

    // Non-square view: the horizontal and vertical focal lengths in pixels are equal.
    Renderer w(128, 64);
    w.setCamera({0, 0, 0}, {0, 0, -1}, 1.5707963f);
    CHECK(w.project({1, 0, -1}, sx, sy));
    CHECK(near(sx, 96) && near(sy, 32)); // 32 px per unit at depth 1, centre x = 64

    // View matrix: a camera 5 units back along +z sees the origin 5 units ahead along -z.
    Renderer v(16, 16);
    v.setCamera({0, 0, 5}, {0, 0, 0}, 1.0f);
    Vec3 p = v.view().point({0, 0, 0});
    CHECK(near(p.x, 0) && near(p.y, 0) && near(p.z, -5));
}

void testClipping() {
    const float n = 0.1f;
    // One vertex in front of the near plane: the polygon becomes a quad, i.e. two triangles.
    ClipVert tri[3];
    tri[0].p = {-1, 0, -0.05f};
    tri[1].p = {1, 0, -5};
    tri[2].p = {0, 1, -5};
    for (ClipVert &v : tri) v.c = {1, 1, 1};
    ClipVert out[4];
    int m = clipPolygonNear(tri, 3, n, out);
    CHECK(m == 4);
    bool allBehind = true;
    for (int i = 0; i < m; ++i) allBehind = allBehind && out[i].p.z <= -n + 1e-4f;
    CHECK(allBehind);
    // Every resulting fan triangle must be non-degenerate and in front of the near plane.
    for (int i = 1; i + 1 < m; ++i) {
        Vec3 fn = (out[i].p - out[0].p).cross(out[i + 1].p - out[0].p);
        CHECK(fn.length() > 1e-6f);
    }

    // Fully in front: passes through unchanged.
    tri[0].p = {-1, 0, -3};
    m = clipPolygonNear(tri, 3, n, out);
    CHECK(m == 3);
    // Fully behind the near plane: nothing left.
    for (ClipVert &v : tri) v.p.z = -0.05f;
    m = clipPolygonNear(tri, 3, n, out);
    CHECK(m == 0);

    // End to end: a triangle straddling the near plane is drawn (two triangles) and counted as clipped.
    Renderer r(32, 32);
    r.setCamera({0, 0, 0}, {0, 0, -1}, 1.0f);
    Canvas cv(32, 32);
    r.beginFrame(cv, Rgb(0, 0, 0), Rgb(0, 0, 0));
    Mesh m2;
    m2.addTri({-2, -1, -0.02f}, {2, -1, -6}, {0, 2, -6}, Rgb(1, 0, 0), Vec3(0, 0, -1000));
    r.draw(m2, unlitOpts());
    CHECK(r.stats().clipped == 1);
    CHECK(r.stats().drawn >= 1);
    CHECK(r.stats().pixels > 0);
}

void testZBuffer() {
    // Far red triangle and near green triangle overlapping at the centre pixel. Result must be the
    // green one whatever the draw order.
    const Rgb red(1, 0, 0), green(0, 1, 0);
    Mesh farTri = frontTri({-30, -30, -20}, {30, -30, -20}, {0, 30, -20}, red);
    Mesh nearTri = frontTri({-3, -3, -5}, {3, -3, -5}, {0, 3, -5}, green);
    Fog noFog;
    noFog.color = Rgb(0, 0, 0);
    noFog.start = 1000;
    noFog.end = 2000;

    for (int order = 0; order < 2; ++order) {
        Renderer r(32, 32);
        r.setCamera({0, 0, 0}, {0, 0, -1}, 1.5707963f);
        r.setFog(noFog);
        Canvas cv(32, 32);
        r.beginFrame(cv, Rgb(0, 0, 0), Rgb(0, 0, 0));
        if (order == 0) {
            r.draw(farTri, unlitOpts());
            r.draw(nearTri, unlitOpts());
        } else {
            r.draw(nearTri, unlitOpts());
            r.draw(farTri, unlitOpts());
        }
        uint32_t centre = cv.pixel(16, 16);
        CHECK(((centre >> 8) & 255) == 255 && ((centre >> 16) & 255) == 0); // green
        uint32_t off = cv.pixel(16, 28); // below the near triangle (its base is at y = 25.6), inside the far one
        CHECK(((off >> 16) & 255) == 255 && ((off >> 8) & 255) == 0);      // red
    }
}

void testCulling() {
    // Same triangle with the winding reversed (normal pointing away from the eye) is culled.
    Mesh m;
    m.addTri({-1, -1, -5}, {1, -1, -5}, {0, 1, -5}, Rgb(1, 1, 1), Vec3(0, 0, 1000));
    Renderer r(16, 16);
    r.setCamera({0, 0, 0}, {0, 0, -1}, 1.0f);
    Canvas cv(16, 16);
    r.beginFrame(cv, Rgb(0, 0, 0), Rgb(0, 0, 0));
    r.draw(m, unlitOpts());
    CHECK(r.stats().culled == 1);
    CHECK(r.stats().drawn == 0);
}

void testMeshes() {
    Mesh box;
    box.addBox({0, 0, 0}, {1, 1, 1}, Rgb(1, 1, 1));
    CHECK(box.triangles() == 12);
    bool unit = true;
    for (const Vertex &v : box.vertices()) unit = unit && near(v.nrm.length(), 1.0f, 1e-4f);
    CHECK(unit);

    const shapes::Models &m = shapes::models();
    CHECK(m.crystal.triangles() == 12);
    CHECK(m.spark.triangles() == 8);
    CHECK(m.drone.triangles() == 36);
    CHECK(m.ground.triangles() == 2 * 18 * 20);
    CHECK(m.ship.triangles() == 8);
}

} // namespace

int main() {
    testProjection();
    testClipping();
    testZBuffer();
    testCulling();
    testMeshes();
    std::printf("test_3d: %d checks, %d failures\n", gChecks, gFailures);
    return gFailures ? 1 : 0;
}
