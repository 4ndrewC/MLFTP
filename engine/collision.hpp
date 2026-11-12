// Collision detection: uniform spatial hash for the broad phase, circle
// overlap for the narrow phase, with layer/mask filtering.
//
// `layer` is a bit flag for what a collider IS; `mask` is what it wants to
// HIT. A pair (a, b) is reported if (a.mask & b.layer) || (b.mask & a.layer).
#pragma once

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include "ecs.hpp"
#include "math.hpp"

namespace spark {

struct Collider {
    Entity entity;
    Vec2 pos;
    float radius = 0;
    uint32_t layer = 0;
    uint32_t mask = 0;
};

inline bool overlaps(Vec2 a, float ra, Vec2 b, float rb) {
    float r = ra + rb;
    return distanceSq(a, b) < r * r;
}

class SpatialHash {
public:
    explicit SpatialHash(float cellSize = 64) : cell_(cellSize) {}

    void clear();
    void insert(int item, Vec2 p, float r);
    /** Items in cells overlapping the circle (may include false positives, may repeat). */
    void query(Vec2 p, float r, std::vector<int> &out) const;

    /**
     * Reports every overlapping, mask-compatible pair exactly once as
     * fn(hitter, target), where hitter.mask matches target.layer.
     * Returns the number of pairs.
     */
    size_t findPairs(const std::vector<Collider> &items, const std::function<void(const Collider &, const Collider &)> &fn);

private:
    static uint64_t key(int cx, int cy) { return uint64_t(uint32_t(cx)) << 32 | uint32_t(cy); }
    struct Range { int x0, y0, x1, y1; };
    Range cells(Vec2 p, float r) const;

    float cell_;
    std::unordered_map<uint64_t, std::vector<int>> grid_;
};

} // namespace spark
