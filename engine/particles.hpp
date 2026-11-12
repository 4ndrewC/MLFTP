// Particle pool. Kept outside the ECS on purpose: particles are numerous,
// short-lived and purely visual, so a flat array is far cheaper.
#pragma once

#include <vector>

#include "canvas.hpp"
#include "math.hpp"

namespace spark {

struct Burst {
    int count = 16;
    float speed = 200;     // max speed; each particle gets 30-100%
    float angle = 0;       // center direction
    float spread = TAU;    // radians around `angle`
    float life = 0.6f;     // seconds; each gets 50-100%
    float size = 3;
    float drag = 3;        // velocity damping per second
    Color color{255, 255, 255};
    Color color2{0u};      // if alpha != 0, particles pick between color and color2
    bool streak = false;   // draw as motion lines instead of squares
};

class Particles {
public:
    explicit Particles(size_t max = 4000, uint64_t seed = 7) : max_(max), rng_(seed) { list_.reserve(max); }

    void burst(Vec2 at, const Burst &b);
    void emit(Vec2 at, Vec2 vel, float life, float size, Color c, float drag = 2, bool streak = false);
    void update(float dt);
    void draw(Canvas &cv) const;
    void clear() {
        list_.clear();
        streaks_ = 0;
    }
    size_t size() const { return list_.size(); }

    /**
     * Multiply every burst's particle count by `s` (quality density times any
     * stress multiplier). Fractions are rounded stochastically with the
     * particle RNG, which is separate from gameplay RNG, so gameplay is
     * unaffected. s = 1 (the default) leaves bursts exactly as authored.
     */
    void setBurstScale(float s) { burstScale_ = s; }
    float burstScale() const { return burstScale_; }
    void setMax(size_t max) {
        max_ = max;
        list_.reserve(max);
    }
    size_t max() const { return max_; }

    // Instrumentation for frame-time research.
    size_t streaks() const { return streaks_; }        // alive particles drawn as lines
    size_t expiringNext() const { return expiring_; }  // alive particles that die within one more update
    struct Counters { uint32_t spawned = 0, bursts = 0; };
    Counters takeCounters() { Counters c = counters_; counters_ = {}; return c; }

private:
    struct P {
        Vec2 pos, vel;
        float life, maxLife, size, drag;
        Color color;
        bool streak;
    };
    size_t max_;
    Rng rng_;
    std::vector<P> list_;
    float burstScale_ = 1;
    size_t streaks_ = 0, expiring_ = 0;
    Counters counters_;
};

} // namespace spark
