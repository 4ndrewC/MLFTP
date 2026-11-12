// Math utilities: 2D vectors, angles, easing, deterministic RNG, colors.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace spark {

constexpr float PI = 3.14159265358979f;
constexpr float TAU = 6.28318530717959f;

struct Vec2 {
    float x = 0, y = 0;

    constexpr Vec2() = default;
    constexpr Vec2(float x_, float y_) : x(x_), y(y_) {}

    static Vec2 fromAngle(float a, float len = 1) { return {std::cos(a) * len, std::sin(a) * len}; }

    constexpr Vec2 operator+(Vec2 o) const { return {x + o.x, y + o.y}; }
    constexpr Vec2 operator-(Vec2 o) const { return {x - o.x, y - o.y}; }
    constexpr Vec2 operator*(float s) const { return {x * s, y * s}; }
    constexpr Vec2 operator/(float s) const { return {x / s, y / s}; }
    constexpr Vec2 operator-() const { return {-x, -y}; }
    constexpr Vec2 &operator+=(Vec2 o) { x += o.x; y += o.y; return *this; }
    constexpr Vec2 &operator-=(Vec2 o) { x -= o.x; y -= o.y; return *this; }
    constexpr Vec2 &operator*=(float s) { x *= s; y *= s; return *this; }
    constexpr bool operator==(const Vec2 &) const = default;

    constexpr float dot(Vec2 o) const { return x * o.x + y * o.y; }
    constexpr float lengthSq() const { return x * x + y * y; }
    float length() const { return std::sqrt(lengthSq()); }
    float angle() const { return std::atan2(y, x); }
    Vec2 normalized() const {
        float l = length();
        return l > 0 ? *this / l : Vec2{};
    }
    Vec2 rotated(float r) const {
        float c = std::cos(r), s = std::sin(r);
        return {x * c - y * s, x * s + y * c};
    }
    /** Shorten to at most `max` length. */
    Vec2 limited(float max) const {
        float l2 = lengthSq();
        return l2 > max * max ? *this * (max / std::sqrt(l2)) : *this;
    }
};

constexpr Vec2 operator*(float s, Vec2 v) { return v * s; }
inline float distance(Vec2 a, Vec2 b) { return (b - a).length(); }
constexpr float distanceSq(Vec2 a, Vec2 b) { return (b - a).lengthSq(); }

constexpr float lerp(float a, float b, float t) { return a + (b - a) * t; }
inline float approach(float v, float target, float delta) {
    return v < target ? std::min(v + delta, target) : std::max(v - delta, target);
}
/** Shortest signed difference from angle a to angle b, in (-PI, PI]. */
inline float angleDiff(float a, float b) {
    float d = std::fmod(b - a, TAU);
    if (d > PI) d -= TAU;
    if (d <= -PI) d += TAU;
    return d;
}

namespace ease {
constexpr float outQuad(float t) { return t * (2 - t); }
constexpr float inQuad(float t) { return t * t; }
constexpr float outCubic(float t) { float u = 1 - t; return 1 - u * u * u; }
constexpr float outBack(float t) {
    constexpr float c = 1.70158f;
    float u = t - 1;
    return 1 + (c + 1) * u * u * u + c * u * u;
}
} // namespace ease

/** Deterministic PRNG (splitmix64). Same seed => same game. */
class Rng {
public:
    explicit Rng(uint64_t seed = 1) : s_(seed) {}
    void seed(uint64_t s) { s_ = s; }

    uint64_t next() {
        uint64_t z = (s_ += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    float uniform() { return float(next() >> 40) / 16777216.0f; } // [0, 1)
    float range(float lo, float hi) { return lo + (hi - lo) * uniform(); }
    int range(int lo, int hi) { return lo + int(next() % uint64_t(hi - lo + 1)); } // inclusive
    bool chance(float p) { return uniform() < p; }
    float angle() { return uniform() * TAU; }
    template <class T, size_t N> const T &pick(const T (&arr)[N]) { return arr[next() % N]; }

private:
    uint64_t s_;
};

/** 32-bit ARGB color. */
struct Color {
    uint32_t argb = 0xff000000u;

    constexpr Color() = default;
    constexpr explicit Color(uint32_t v) : argb(v) {}
    constexpr Color(int r, int g, int b, int a = 255)
        : argb(uint32_t(a) << 24 | uint32_t(r) << 16 | uint32_t(g) << 8 | uint32_t(b)) {}
    /** From 0xRRGGBB. */
    static constexpr Color hex(uint32_t rgb, int a = 255) { return Color(uint32_t(a) << 24 | (rgb & 0xffffff)); }

    constexpr int r() const { return (argb >> 16) & 255; }
    constexpr int g() const { return (argb >> 8) & 255; }
    constexpr int b() const { return argb & 255; }
    constexpr int a() const { return argb >> 24; }

    /** Copy with alpha multiplied by f (0..1). */
    Color fade(float f) const {
        int na = int(std::clamp(f, 0.0f, 1.0f) * float(a()) + 0.5f);
        return Color((argb & 0xffffffu) | uint32_t(na) << 24);
    }
    Color withAlpha(int na) const { return Color((argb & 0xffffffu) | uint32_t(std::clamp(na, 0, 255)) << 24); }

    static Color lerp(Color x, Color y, float t) {
        t = std::clamp(t, 0.0f, 1.0f);
        auto mix = [t](int p, int q) { return int(float(p) + float(q - p) * t + 0.5f); };
        return Color(mix(x.r(), y.r()), mix(x.g(), y.g()), mix(x.b(), y.b()), mix(x.a(), y.a()));
    }
};

} // namespace spark
