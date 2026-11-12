// Autopilot for headless runs (`--bot`). It only touches Input, so it exercises the same
// code paths a player does. Each frame it scores a grid of glider positions (x, y) against the
// hazards ahead and steers towards the cheapest one. Sparks lower the cost; pylons, drones and
// shots raise it. Boost is held while the lane ahead is clear.
#include <algorithm>
#include <cmath>

#include "drift.hpp"

namespace drift {

namespace {
void hold(Input &in, Key k, bool down) {
    in.setKey(k, false); // re-press every step so `pressed` edges fire too
    if (down) in.setKey(k, true);
}
} // namespace

void botStep(Engine &e, long frame, BotReport &report) {
    Input &in = e.input();
    Scene *top = e.top();

    for (Key k : {Key::W, Key::A, Key::S, Key::D, Key::Space, Key::Enter}) hold(in, k, false);

    if (dynamic_cast<TitleScene *>(top)) {
        hold(in, Key::Enter, frame % 30 == 0);
        return;
    }
    if (dynamic_cast<GameOverScene *>(top)) {
        if (!report.onGameOver) ++report.deaths;
        report.onGameOver = true;
        hold(in, Key::Enter, frame % 60 == 0);
        return;
    }
    report.onGameOver = false;
    auto *play = dynamic_cast<PlayScene *>(top);
    if (!play) return; // paused: the bot does not unpause

    const PlayState &s = play->state();
    report.bestScore = std::max(report.bestScore, s.score);
    report.bestWave = std::max(report.bestWave, s.wave);

    float bestCost = 1e30f, bx = s.pos.x, by = s.pos.y;
    for (int ix = 0; ix <= 22; ++ix) {
        const float cx = -kHalfWidth + float(ix);
        for (int iy = 0; iy <= 7; ++iy) {
            const float cy = 1.0f + 0.75f * float(iy);
            float cost = 0.6f * std::fabs(cx - s.pos.x) + 0.8f * std::fabs(cy - s.pos.y); // prefer short moves
            for (const Pylon &q : s.pylons) {
                const float ahead = s.pos.z - q.pos.z;
                if (ahead > -1.5f && ahead < 22.0f && std::fabs(cx - q.pos.x) < 1.9f && std::fabs(cy - q.pos.y) < 2.2f)
                    cost += 100.0f;
            }
            for (const Drone &d : s.drones) {
                const float ahead = s.pos.z - d.pos.z;
                if (ahead > -1.5f && ahead < 16.0f && std::fabs(cx - d.pos.x) < 1.6f && std::fabs(cy - d.pos.y) < 1.6f)
                    cost += 150.0f;
            }
            for (const Shot &sh : s.shots) {
                const float ahead = s.pos.z - sh.pos.z;
                if (ahead > -1.0f && ahead < 10.0f && std::fabs(cx - sh.pos.x) < 1.2f && std::fabs(cy - sh.pos.y) < 1.2f)
                    cost += 120.0f;
            }
            for (const Spark &sp : s.sparks) {
                const float ahead = s.pos.z - sp.pos.z;
                if (ahead > 0.0f && ahead < 22.0f && std::fabs(cx - sp.pos.x) < 1.6f && std::fabs(cy - sp.pos.y) < 1.6f)
                    cost -= 6.0f / (1.0f + 0.2f * ahead);
            }
            if (cost < bestCost) {
                bestCost = cost;
                bx = cx;
                by = cy;
            }
        }
    }

    if (bx < s.pos.x - 0.3f) hold(in, Key::A, true);
    if (bx > s.pos.x + 0.3f) hold(in, Key::D, true);
    if (by > s.pos.y + 0.3f) hold(in, Key::W, true);
    if (by < s.pos.y - 0.3f) hold(in, Key::S, true);

    // Boost when the lane ahead is clear and there is energy to spare.
    bool clear = s.energy > 0.5f;
    for (const Pylon &q : s.pylons)
        if (clear && s.pos.z - q.pos.z > 0 && s.pos.z - q.pos.z < 30 && std::fabs(q.pos.x - s.pos.x) < 3) clear = false;
    for (const Drone &d : s.drones)
        if (clear && s.pos.z - d.pos.z > 0 && s.pos.z - d.pos.z < 30 && std::fabs(d.pos.x - s.pos.x) < 3) clear = false;
    if (clear) hold(in, Key::Space, true);
}

} // namespace drift
