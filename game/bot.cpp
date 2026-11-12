// Autopilot for headless smoke tests. It only touches Input (never the game
// state directly), so it exercises the same code paths a human player does.
#include <cstdio>

#include "play.hpp"
#include "scenes.hpp"

namespace warden {

namespace {
void hold(Input &in, Key k, bool down) {
    in.setKey(k, false); // re-press every step so `pressed` edges fire too
    if (down) in.setKey(k, true);
}
} // namespace

void botStep(Engine &e, long frame, BotReport &report) {
    Input &in = e.input();
    Scene *top = e.top();

    for (Key k : {Key::W, Key::A, Key::S, Key::D, Key::MouseLeft, Key::Space, Key::Enter}) hold(in, k, false);

    if (dynamic_cast<TitleScene *>(top)) {
        hold(in, Key::Enter, frame % 30 == 0);
        return;
    }
    bool over = dynamic_cast<GameOverScene *>(top) != nullptr;
    if (over && !report.onGameOver) {
        ++report.deaths;
        std::printf("bot: died at frame %ld\n", frame);
    }
    report.onGameOver = over;
    if (over) {
        hold(in, Key::Enter, frame % 60 == 0);
        return;
    }
    auto *play = dynamic_cast<PlayScene *>(top);
    if (!play) return;

    World &w = play->world();
    report.peakEntities = std::max(report.peakEntities, w.size());
    report.bestScore = std::max(report.bestScore, play->stats().score);
    report.bestWave = std::max(report.bestWave, play->stats().wave);
    Transform *me = w.get<Transform>(play->player());
    if (!me) return;

    // Find the nearest threat, and steer away from everything close.
    Vec2 flee, nearest;
    float nearestD = 1e9f;
    w.each<Enemy, Transform>([&](Entity, Enemy &, Transform &tr) {
        Vec2 d = me->pos - tr.pos;
        float len = d.length();
        if (len < nearestD) nearestD = len, nearest = tr.pos;
        if (len < 170 && len > 0.01f) flee += d / len * (170 - len);
    });
    w.each<EnemyShot, Transform>([&](Entity, EnemyShot &, Transform &tr) {
        Vec2 d = me->pos - tr.pos;
        float len = d.length();
        if (len < 90 && len > 0.01f) flee += d / len * (90 - len) * 2;
    });
    // Drift toward the nearest spark when it's safe, and keep off the walls.
    if (flee.lengthSq() < 1) {
        float best = 1e9f;
        Vec2 goal{kWidth / 2.0f, kHeight / 2.0f};
        w.each<Spark, Transform>([&](Entity, Spark &, Transform &tr) {
            float d = distance(me->pos, tr.pos);
            if (d < best) best = d, goal = tr.pos;
        });
        flee = (goal - me->pos) * 0.5f;
    }
    Vec2 center{kWidth / 2.0f, kHeight / 2.0f};
    flee += (center - me->pos) * 0.15f;

    if (flee.x < -5) hold(in, Key::A, true);
    if (flee.x > 5) hold(in, Key::D, true);
    if (flee.y < -5) hold(in, Key::W, true);
    if (flee.y > 5) hold(in, Key::S, true);
    if (nearestD < 55) hold(in, Key::Space, true); // dash out of trouble

    if (nearestD < 1e8f) {
        in.setMouse(nearest);
        hold(in, Key::MouseLeft, true);
    }
}

} // namespace warden
