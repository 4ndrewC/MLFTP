#include "collision.hpp"

namespace spark {

SpatialHash::Range SpatialHash::cells(Vec2 p, float r) const {
    return {int(std::floor((p.x - r) / cell_)), int(std::floor((p.y - r) / cell_)),
            int(std::floor((p.x + r) / cell_)), int(std::floor((p.y + r) / cell_))};
}

void SpatialHash::clear() {
    // Keep the vectors' capacity: we refill the hash every frame.
    for (auto &[k, v] : grid_) v.clear();
}

void SpatialHash::insert(int item, Vec2 p, float r) {
    Range c = cells(p, r);
    for (int cx = c.x0; cx <= c.x1; ++cx)
        for (int cy = c.y0; cy <= c.y1; ++cy) grid_[key(cx, cy)].push_back(item);
}

void SpatialHash::query(Vec2 p, float r, std::vector<int> &out) const {
    Range c = cells(p, r);
    for (int cx = c.x0; cx <= c.x1; ++cx)
        for (int cy = c.y0; cy <= c.y1; ++cy)
            if (auto it = grid_.find(key(cx, cy)); it != grid_.end())
                out.insert(out.end(), it->second.begin(), it->second.end());
}

size_t SpatialHash::findPairs(const std::vector<Collider> &items,
                              const std::function<void(const Collider &, const Collider &)> &fn) {
    clear();
    for (int i = 0; i < int(items.size()); ++i) insert(i, items[i].pos, items[i].radius);

    // seen[j] == i means j was already tested against i (it can sit in several cells).
    std::vector<int> seen(items.size(), -1);
    std::vector<int> near;
    size_t pairs = 0;
    for (int i = 0; i < int(items.size()); ++i) {
        const Collider &a = items[i];
        if (!a.mask) continue; // only "hitters" search; pairs are found from the hitter's side
        near.clear();
        query(a.pos, a.radius, near);
        for (int j : near) {
            if (j == i || seen[j] == i) continue;
            seen[j] = i;
            const Collider &b = items[j];
            if (!(a.mask & b.layer)) continue;
            // If both want to hit each other, only the lower index reports the pair.
            if ((b.mask & a.layer) && j < i) continue;
            if (!overlaps(a.pos, a.radius, b.pos, b.radius)) continue;
            ++pairs;
            fn(a, b);
        }
    }
    return pairs;
}

} // namespace spark
