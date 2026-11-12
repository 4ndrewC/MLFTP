#include "nature.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace nature {

using r3d::Mat4;
using r3d::Mesh;
using r3d::Rgb;
using r3d::Vec3;

namespace {

constexpr Rgb kSkyTop = Rgb::hex(0x050a1a);
constexpr Rgb kHorizon = Rgb::hex(0x22405a); // misty blue-green; also the fog colour

// ---------------------------------------------------------------- hashing (placement)

uint32_t hash3(int32_t i, int32_t j, uint32_t salt) {
    uint32_t h = uint32_t(i) * 0x8da6b343u ^ uint32_t(j) * 0xd8163841u ^ salt * 0xcb1ab31fu;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    h *= 0x27d4eb2du;
    h ^= h >> 16;
    return h;
}
float h01(int32_t i, int32_t j, uint32_t salt) { return float(hash3(i, j, salt) >> 8) / 16777216.0f; }

float smoothstep(float a, float b, float x) {
    float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3 - 2 * t);
}

// ---------------------------------------------------------------- meshes (built once)

/** Closed n-sided cone: base ring at y0 with radius r, tip at y1. */
void addCone(Mesh &m, float y0, float y1, float r, int n, Rgb col, float twist) {
    const Vec3 tip(0, y1, 0), base(0, y0, 0), inside(0, y0 + (y1 - y0) * 0.25f, 0);
    for (int k = 0; k < n; ++k) {
        float a0 = twist + float(k) * 6.2831853f / float(n), a1 = twist + float(k + 1) * 6.2831853f / float(n);
        Vec3 p(r * std::cos(a0), y0, r * std::sin(a0)), q(r * std::cos(a1), y0, r * std::sin(a1));
        // Alternate shades around the cone give the low-poly faceted look.
        Rgb c = (k & 1) ? col * 0.88f : col;
        m.addTri(tip, p, q, c, inside);
        m.addTri(base, q, p, col * 0.6f, inside);
    }
}

/** Irregular octahedron ("blob") centred on c with radius r: 8 triangles. */
void addBlob(Mesh &m, Vec3 c, float r, Rgb col, uint32_t seed) {
    auto j = [&](int k) { return 0.8f + 0.4f * float((seed >> (k * 3)) & 7) / 7.0f; };
    const Vec3 v[6] = {c + Vec3(r * j(0), 0, 0), c + Vec3(-r * j(1), 0, 0), c + Vec3(0, r * j(2), 0),
                       c + Vec3(0, -r * 0.7f * j(3), 0), c + Vec3(0, 0, r * j(4)), c + Vec3(0, 0, -r * j(5))};
    static constexpr int faces[8][3] = {{0, 2, 4}, {4, 2, 1}, {1, 2, 5}, {5, 2, 0},
                                        {0, 4, 3}, {4, 1, 3}, {1, 5, 3}, {5, 0, 3}};
    for (int f = 0; f < 8; ++f) m.addTri(v[faces[f][0]], v[faces[f][1]], v[faces[f][2]], f < 4 ? col : col * 0.75f, c);
}

Mesh makePine() {
    Mesh m;
    m.addBox(Vec3(0, 0.45f, 0), Vec3(0.16f, 0.45f, 0.16f), Rgb::hex(0x4a3020));
    addCone(m, 0.7f, 2.3f, 1.25f, 7, Rgb::hex(0x1b5233), 0.0f);
    addCone(m, 1.5f, 3.0f, 0.95f, 7, Rgb::hex(0x21613b), 0.4f);
    addCone(m, 2.3f, 3.7f, 0.62f, 7, Rgb::hex(0x287144), 0.8f);
    return m;
}

Mesh makeRoundTree() {
    Mesh m;
    m.addBox(Vec3(0, 0.75f, 0), Vec3(0.18f, 0.75f, 0.18f), Rgb::hex(0x56381f));
    addBlob(m, Vec3(0, 2.2f, 0), 1.25f, Rgb::hex(0x3c7f33), 0x5a3c91u);
    addBlob(m, Vec3(0.55f, 2.75f, 0.2f), 0.85f, Rgb::hex(0x4b9439), 0x1d77e2u);
    addBlob(m, Vec3(-0.5f, 2.6f, -0.3f), 0.8f, Rgb::hex(0x35722e), 0xc4a157u);
    return m;
}

/** Three thin leaning blades; open at the bottom (the ground hides it). 9 triangles. */
Mesh makeTuft() {
    Mesh m;
    const Rgb col = Rgb::hex(0x4f9a3e);
    for (int b = 0; b < 3; ++b) {
        float a = float(b) * 2.0943951f, lean = 0.12f;
        Vec3 c(0.12f * std::cos(a), 0, 0.12f * std::sin(a));
        Vec3 tip = c + Vec3(lean * std::cos(a), 0.42f + 0.08f * float(b), lean * std::sin(a));
        Vec3 p[3];
        for (int k = 0; k < 3; ++k) {
            float t = float(k) * 2.0943951f;
            p[k] = c + Vec3(0.045f * std::cos(t), 0, 0.045f * std::sin(t));
        }
        Rgb shade = b == 1 ? col * 1.15f : col;
        for (int k = 0; k < 3; ++k) m.addTri(tip, p[k], p[(k + 1) % 3], shade, c + Vec3(0, 0.05f, 0));
    }
    return m;
}

Mesh makeRock() {
    Mesh m;
    addBlob(m, Vec3(0, 0.15f, 0), 0.55f, Rgb::hex(0x5d6570), 0x9e3779b9u);
    return m;
}

Mesh makeFlower(Rgb col) {
    Mesh m;
    addBlob(m, Vec3(0, 0.3f, 0), 0.07f, col, 0x12345678u);
    return m;
}

Mesh makeFirefly() {
    Mesh m;
    addBlob(m, Vec3(0, 0, 0), 0.06f, Rgb::hex(0xd8ff7a), 0u);
    return m;
}

struct Kit {
    Mesh pine = makePine(), round = makeRoundTree(), tuft = makeTuft(), rock = makeRock(), firefly = makeFirefly();
    Mesh flowers[3] = {makeFlower(Rgb::hex(0xff8fc8)), makeFlower(Rgb::hex(0xf2f2ff)), makeFlower(Rgb::hex(0xb58cff))};
};
const Kit &kit() {
    static const Kit k;
    return k;
}

r3d::DrawOpts at(Vec3 p, float yaw = 0, float s = 1, bool unlit = false) {
    r3d::DrawOpts o;
    o.model = Mat4::translate(p) * Mat4::rotateY(yaw) * Mat4::scale(s);
    o.unlit = unlit;
    return o;
}

// ---------------------------------------------------------------- terrain

constexpr float kTerrainHalfX = 44.0f; // terrain spans x in [-44, 44]
constexpr float kAhead = 44.0f;        // how far ahead of the glider scenery is built
constexpr float kBehind = 12.0f;       // and behind it (the camera sits 7.5 behind)

Rgb grassColor(int i, int j, float h) {
    const Rgb low = Rgb::hex(0x1c4724), high = Rgb::hex(0x2e6a34), dry = Rgb::hex(0x4a6a2a);
    Rgb c = Rgb::lerp(low, high, std::clamp(h / 6.0f, 0.0f, 1.0f));
    float v = h01(i, j, 11);
    if (v > 0.93f) c = Rgb::lerp(c, dry, 0.5f); // occasional dry patch
    return c * (0.9f + 0.2f * h01(i, j, 12));
}

void buildTerrain(Mesh &m, float pz, float ahead, float kCell) {
    m.clear();
    const int i0 = int(std::floor(-kTerrainHalfX / kCell)), i1 = int(std::ceil(kTerrainHalfX / kCell));
    const int j0 = int(std::floor((pz - ahead) / kCell)), j1 = int(std::ceil((pz + kBehind) / kCell));
    for (int j = j0; j < j1; ++j)
        for (int i = i0; i < i1; ++i) {
            const float x0 = float(i) * kCell, x1 = x0 + kCell, z0 = float(j) * kCell, z1 = z0 + kCell;
            const Vec3 a(x0, height(x0, z0), z0), b(x1, height(x1, z0), z0), c(x1, height(x1, z1), z1),
                d(x0, height(x0, z1), z1);
            const float hm = (a.y + b.y + c.y + d.y) * 0.25f;
            const Rgb col = grassColor(i, j, hm);
            const Vec3 below((x0 + x1) * 0.5f, hm - 10.0f, (z0 + z1) * 0.5f);
            // Split along alternating diagonals so the hills do not all crease the same way.
            if ((i + j) & 1) {
                m.addTri(a, b, c, col, below);
                m.addTri(a, c, d, col * 0.94f, below);
            } else {
                m.addTri(a, b, d, col, below);
                m.addTri(b, c, d, col * 0.94f, below);
            }
        }
}

} // namespace

float height(float x, float z) {
    const float ax = std::fabs(x);
    const float ripple = 0.10f * std::sin(x * 0.55f + z * 0.31f) + 0.07f * std::sin(z * 0.83f - x * 0.37f);
    const float hills = 2.2f + 1.8f * std::sin(z * 0.071f + x * 0.13f) + 1.3f * std::sin(z * 0.029f - x * 0.061f) +
                        0.6f * std::sin(z * 0.19f + x * 0.27f);
    const float edge = smoothstep(kClearX, 30.0f, ax);
    return ripple * (1.0f - edge) + edge * edge * std::max(0.0f, hills) * 1.6f;
}

Detail detailFor(const spark::Quality &q) {
    static constexpr float kView[] = {1.0f, 0.85f, 0.7f, 0.55f};
    Detail d;
    d.view = kView[std::min<int>(q.glow, 3)];
    d.density = q.particleDensity();
    d.terrainCell = q.antialias() ? 4.0f : 8.0f;
    return d;
}

r3d::Fog fog(const Detail &d) {
    r3d::Fog f;
    f.color = kHorizon;
    f.start = 16 * d.view;
    f.end = 48 * d.view;
    return f;
}
Rgb skyTop() { return kSkyTop; }
Rgb skyHorizon() { return kHorizon; }

void drawSky(spark::Canvas &cv, float camX, float t) {
    using spark::Color;
    using spark::Vec2;
    // Canvas drawing uses logical coordinates; the pixel buffer may be scaled (render-scale lever).
    const float w = float(cv.logicalWidth()), h = float(cv.logicalHeight());
    cv.setOffset({});

    // Moon with a soft halo, fixed in the sky.
    const Vec2 moon(w * 0.8f, h * 0.13f);
    cv.glow(moon, 70, Color::hex(0x9fb8ff).fade(0.25f));
    cv.circle(moon, 17, Color::hex(0xe9eeff));
    cv.circle(moon + Vec2(6, -3), 14, Color::hex(0xc9d3f2).fade(0.35f)); // a hint of shading

    // Two mountain ranges; the nearer one shifts more when the glider steers (parallax).
    auto range = [&](float base, float amp, float freq, float shift, Color col, float seed) {
        std::vector<Vec2> pts;
        pts.push_back({-40, h * 0.62f});
        for (float x = -40; x <= w + 40; x += 24) {
            float u = (x + shift) * freq;
            float y = base - amp * (0.55f + 0.30f * std::sin(u + seed) + 0.15f * std::sin(u * 2.7f + seed * 3) +
                                    0.08f * std::sin(u * 6.1f + seed));
            pts.push_back({x, y});
        }
        pts.push_back({w + 40, h * 0.62f});
        cv.polygon(pts, col);
    };
    const Color far = Color::hex(0x1a3047), nearC = Color::hex(0x132438);
    range(h * 0.47f, h * 0.17f, 0.006f, camX * 4.0f, far, 1.3f);
    range(h * 0.50f, h * 0.11f, 0.011f, camX * 9.0f, nearC, 4.1f);
    (void)t;
}

void drawWorld(r3d::Renderer &rend, float pz, float t, const Detail &d) {
    const Kit &k = kit();
    const float ahead = kAhead * d.view;

    static Mesh terrain; // rebuilt each frame around the glider (world-anchored tiles)
    buildTerrain(terrain, pz, ahead, d.terrainCell);
    rend.draw(terrain, r3d::DrawOpts{});

    // Trees: one candidate per 6x6 world cell outside the corridor.
    constexpr float kTreeCell = 6.0f;
    const int tj0 = int(std::floor((pz - ahead) / kTreeCell)), tj1 = int(std::ceil((pz + kBehind) / kTreeCell));
    const int ti = int(std::ceil(kTerrainHalfX / kTreeCell));
    for (int j = tj0; j < tj1; ++j)
        for (int i = -ti; i < ti; ++i) {
            if (h01(i, j, 1) > 0.62f) continue;
            const float x = (float(i) + 0.15f + 0.7f * h01(i, j, 2)) * kTreeCell;
            const float z = (float(j) + 0.15f + 0.7f * h01(i, j, 3)) * kTreeCell;
            if (std::fabs(x) < kClearX + 1.0f || std::fabs(x) > kTerrainHalfX - 1.0f) continue;
            const float s = 0.85f + 0.7f * h01(i, j, 4), yaw = 6.283f * h01(i, j, 5);
            const Vec3 p(x, height(x, z) - 0.1f, z);
            rend.draw(h01(i, j, 6) < 0.62f ? k.pine : k.round, at(p, yaw, s));
        }

    // Small things only near the glider: grass tufts, flowers, rocks (2x2 cells).
    constexpr float kSmall = 2.0f;
    const int sj0 = int(std::floor((pz - 28.0f) / kSmall)), sj1 = int(std::ceil((pz + 9.0f) / kSmall));
    for (int j = sj0; j < sj1; ++j)
        for (int i = -12; i < 12; ++i) {
            const float r = h01(i, j, 7);
            if (r > 0.48f || h01(i, j, 16) >= d.density) continue;
            const float x = (float(i) + h01(i, j, 8)) * kSmall, z = (float(j) + h01(i, j, 9)) * kSmall;
            const Vec3 p(x, height(x, z) - 0.03f, z);
            const float yaw = 6.283f * h01(i, j, 10);
            if (r < 0.025f && std::fabs(x) > 3.0f) {
                rend.draw(k.rock, at(p, yaw, 0.7f + 0.8f * h01(i, j, 13)));
            } else if (r < 0.09f) {
                rend.draw(k.tuft, at(p, yaw, 0.8f));
                rend.draw(k.flowers[hash3(i, j, 14) % 3], at(p, yaw, 1.0f, true));
            } else {
                rend.draw(k.tuft, at(p, yaw, 0.8f + 0.6f * h01(i, j, 15)));
            }
        }

    // Fireflies drifting near the tree lines.
    constexpr float kFlyCell = 5.0f;
    const int fj0 = int(std::floor((pz - 30.0f) / kFlyCell)), fj1 = int(std::ceil((pz + 6.0f) / kFlyCell));
    for (int j = fj0; j < fj1; ++j)
        for (int side = -1; side <= 1; side += 2) {
            if (h01(side, j, 20) > 0.7f * d.density) continue;
            const float ph = 6.283f * h01(side, j, 21);
            const float x = float(side) * (kClearX + 2.0f + 8.0f * h01(side, j, 22)) + 0.8f * std::sin(t * 0.7f + ph);
            const float z = (float(j) + h01(side, j, 23)) * kFlyCell + 0.6f * std::cos(t * 0.5f + ph);
            const float y = height(x, z) + 1.2f + 0.9f * std::sin(t * 1.1f + ph * 2.0f);
            rend.draw(k.firefly, at(Vec3(x, y, z), 0, 1, true));
        }
}

void drawGlows(spark::Canvas &cv, const r3d::Renderer &rend, float pz, float t, const Detail &d) {
    using spark::Color;
    constexpr float kFlyCell = 5.0f;
    const int fj0 = int(std::floor((pz - 30.0f) / kFlyCell)), fj1 = int(std::ceil((pz + 6.0f) / kFlyCell));
    const spark::Blend prev = cv.blend();
    cv.setBlend(spark::Blend::Add);
    for (int j = fj0; j < fj1; ++j)
        for (int side = -1; side <= 1; side += 2) {
            if (h01(side, j, 20) > 0.7f * d.density) continue;
            const float ph = 6.283f * h01(side, j, 21);
            const float x = float(side) * (kClearX + 2.0f + 8.0f * h01(side, j, 22)) + 0.8f * std::sin(t * 0.7f + ph);
            const float z = (float(j) + h01(side, j, 23)) * kFlyCell + 0.6f * std::cos(t * 0.5f + ph);
            const float y = height(x, z) + 1.2f + 0.9f * std::sin(t * 1.1f + ph * 2.0f);
            const Vec3 v = rend.view().point(Vec3(x, y, z));
            float sx, sy;
            if (!rend.project(v, sx, sy)) continue;
            sx /= cv.pixelScale(); // renderer pixels -> canvas logical coordinates
            sy /= cv.pixelScale();
            const float depth = -v.z;
            const float pulse = 0.6f + 0.4f * std::sin(t * 3.0f + ph * 5.0f);
            const float r = std::clamp(90.0f / depth, 3.0f, 14.0f);
            cv.glow({sx, sy}, r, Color::hex(0xc8ff6a).fade(0.35f * pulse * std::clamp(1.0f - depth / 40.0f, 0.0f, 1.0f)));
        }
    cv.setBlend(prev);
}

} // namespace nature
