#include "particles.hpp"

namespace spark {

void Particles::emit(Vec2 at, Vec2 vel, float life, float size, Color c, float drag, bool streak) {
    if (list_.size() >= max_) return;
    list_.push_back({at, vel, life, life, size, drag, c, streak});
    streaks_ += streak;
    ++counters_.spawned;
}

void Particles::burst(Vec2 at, const Burst &b) {
    int n = b.count;
    if (burstScale_ != 1) {
        float f = float(b.count) * burstScale_;
        n = int(f);
        if (rng_.uniform() < f - float(n)) ++n;
    }
    ++counters_.bursts;
    for (int i = 0; i < n; ++i) {
        float a = b.angle + (rng_.uniform() - 0.5f) * b.spread;
        float s = b.speed * rng_.range(0.3f, 1.0f);
        Color c = (b.color2.a() && rng_.chance(0.5f)) ? b.color2 : b.color;
        emit(at, Vec2::fromAngle(a, s), b.life * rng_.range(0.5f, 1.0f), b.size * rng_.range(0.6f, 1.2f), c,
             b.drag, b.streak);
    }
}

void Particles::update(float dt) {
    expiring_ = 0;
    for (size_t i = list_.size(); i-- > 0;) {
        P &p = list_[i];
        p.life -= dt;
        if (p.life <= 0) {
            streaks_ -= p.streak;
            p = list_.back(); // swap-remove keeps this O(n)
            list_.pop_back();
            continue;
        }
        expiring_ += p.life <= dt;
        p.vel *= std::max(0.0f, 1 - p.drag * dt);
        p.pos += p.vel * dt;
    }
}

void Particles::draw(Canvas &cv) const {
    Blend prev = cv.blend();
    cv.setBlend(Blend::Add);
    for (const P &p : list_) {
        float t = p.life / p.maxLife;
        Color c = p.color.fade(t);
        float s = p.size * t;
        if (p.streak) cv.line(p.pos, p.pos - p.vel * 0.035f, std::max(1.0f, s), c);
        else cv.rect(p.pos.x - s / 2, p.pos.y - s / 2, std::max(1.0f, s), std::max(1.0f, s), c);
    }
    cv.setBlend(prev);
}

} // namespace spark
