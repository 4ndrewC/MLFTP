#include "meshes.hpp"

#include <cmath>

namespace shapes {

using r3d::Mesh;
using r3d::Rgb;
using r3d::Vec3;

Mesh makeGround(float cell, int cols, int rows, float x0, float z0, Rgb a, Rgb b) {
    Mesh m;
    for (int j = 0; j < rows; ++j)
        for (int i = 0; i < cols; ++i) {
            float x = x0 + float(i) * cell, z = z0 + float(j) * cell;
            Rgb c = ((i + j) & 1) ? b : a;
            Vec3 inside(x + cell * 0.5f, -1.0f, z + cell * 0.5f);
            m.addQuad({x, 0, z}, {x + cell, 0, z}, {x + cell, 0, z + cell}, {x, 0, z + cell}, c, inside);
        }
    return m;
}

Mesh makeCrystal(float radius, float halfHeight, Rgb col) {
    Mesh m;
    const Vec3 top(0, halfHeight, 0), bottom(0, -halfHeight, 0);
    Vec3 ring[6];
    for (int k = 0; k < 6; ++k) {
        float a = float(k) * 1.04719755f; // 60 degrees
        ring[k] = {radius * std::cos(a), 0, radius * std::sin(a)};
    }
    for (int k = 0; k < 6; ++k) {
        const Vec3 &p = ring[k], &q = ring[(k + 1) % 6];
        m.addTri(top, p, q, col, Vec3{});
        m.addTri(bottom, q, p, col, Vec3{});
    }
    return m;
}

Mesh makeShip(Rgb col) {
    Mesh m;
    const Vec3 N(0, 0.15f, -1.5f);  // nose
    const Vec3 C(0, 0.6f, -0.1f);   // canopy ridge
    const Vec3 TL(-1.4f, 0, 0.7f);  // left wing tip
    const Vec3 TR(1.4f, 0, 0.7f);   // right wing tip
    const Vec3 TT(0, 0.35f, 1.0f);  // tail top
    const Vec3 B(0, -0.3f, 0);      // belly
    const Vec3 inside((N.x + C.x + TL.x + TR.x + TT.x + B.x) / 6, (N.y + C.y + TL.y + TR.y + TT.y + B.y) / 6,
                      (N.z + C.z + TL.z + TR.z + TT.z + B.z) / 6);
    const Rgb under = col * 0.55f;
    m.addTri(N, TL, C, col, inside);
    m.addTri(N, C, TR, col, inside);
    m.addTri(C, TL, TT, col, inside);
    m.addTri(C, TT, TR, col, inside);
    m.addTri(N, B, TL, under, inside);
    m.addTri(N, TR, B, under, inside);
    m.addTri(TL, B, TT, under, inside);
    m.addTri(TR, TT, B, under, inside);
    return m;
}

Mesh makeOcta(float s, Rgb col) {
    Mesh m;
    const Vec3 px(s, 0, 0), nx(-s, 0, 0), py(0, s, 0), ny(0, -s, 0), pz(0, 0, s), nz(0, 0, -s);
    const Vec3 tips[6] = {px, nx, py, ny, pz, nz};
    for (int a = 0; a < 6; ++a)
        for (int b = a + 1; b < 6; ++b) {
            if ((a / 2) == (b / 2)) continue; // opposite tips do not form a face
            for (int c = b + 1; c < 6; ++c) {
                if ((c / 2) == (a / 2) || (c / 2) == (b / 2)) continue;
                m.addTri(tips[a], tips[b], tips[c], col, Vec3{});
            }
        }
    return m;
}

Mesh makeDrone(Rgb col) {
    Mesh m;
    m.addBox(Vec3(0, 0, 0), Vec3(0.45f, 0.28f, 0.55f), col);
    m.addBox(Vec3(-0.95f, 0, 0.05f), Vec3(0.45f, 0.04f, 0.3f), col * 0.7f);
    m.addBox(Vec3(0.95f, 0, 0.05f), Vec3(0.45f, 0.04f, 0.3f), col * 0.7f);
    return m;
}

const Models &models() {
    static const Models m = [] {
        Models r;
        r.ground = makeGround(4.0f, 18, 20, -36.0f, -56.0f, Rgb::hex(0x0b1629), Rgb::hex(0x13233f));
        r.crystal = makeCrystal(1.0f, 1.6f, Rgb::hex(0x58e6ff));
        r.ship = makeShip(Rgb::hex(0x7af6ff));
        r.spark = makeOcta(0.35f, Rgb::hex(0xffd84a));
        r.shot = makeOcta(0.22f, Rgb::hex(0xff5a5a));
        r.drone = makeDrone(Rgb::hex(0xff4fd8));
        return r;
    }();
    return m;
}

} // namespace shapes
