// Spark Drift scenes: title, flight (play), pause, game over.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

#include "drift.hpp"
#include "nature.hpp"

namespace drift {

namespace {

const r3d::Rgb kSkyTop = nature::skyTop();
const r3d::Rgb kHorizon = nature::skyHorizon();

uint64_t gBaseSeed = 0, gRuns = 0;
long gBest = 0;
DrawTotals gTotals;

float dist2(r3d::Vec3 a, r3d::Vec3 b) { return (a - b).dot(a - b); }

r3d::DrawOpts opts(const r3d::Mat4 &model, bool unlit = false) {
    r3d::DrawOpts o;
    o.model = model;
    o.unlit = unlit;
    return o;
}

/** Title-style text: crisp glyphs over a few additive offset copies. */
void bigText(Canvas &cv, float x, float y, const char *s, int scale, Color c) {
    Blend prev = cv.blend();
    cv.setBlend(Blend::Add);
    Color halo = c.fade(0.18f);
    const float o = float(std::max(1, scale / 2));
    for (Vec2 d : {Vec2{-o, 0}, Vec2{o, 0}, Vec2{0, -o}, Vec2{0, o}})
        cv.text(x + d.x, y + d.y, s, scale, halo, Align::Center);
    cv.setBlend(prev);
    cv.text(x, y, s, scale, c, Align::Center);
}

const Sound kPickup{Wave::Square, 880.0f, 500.0f, 0.06f, 0.12f};
const Sound kHit{Wave::Noise, 240.0f, -200.0f, 0.35f, 0.35f};
const Sound kShot{Wave::Saw, 420.0f, -260.0f, 0.12f, 0.06f};

void setupFog(r3d::Renderer &rend, const nature::Detail &d) { rend.setFog(nature::fog(d)); }

} // namespace

DrawTotals &drawTotals() { return gTotals; }

void setBaseSeed(uint64_t seed) { gBaseSeed = seed; }

uint64_t nextRunSeed() {
    ++gRuns;
    if (gBaseSeed) return gBaseSeed + gRuns * 7919;
    return uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
}

// ---------------------------------------------------------------- title

TitleScene::TitleScene(Engine &e) : Scene(e), rend_(kWidth, kHeight) {}

void TitleScene::update(float dt) {
    t_ += dt;
    Input &in = engine.input();
    if (in.pressed(ActConfirm)) {
        engine.audio().melody({{392, 0.07f}, {523, 0.07f}, {659, 0.07f}, {784, 0.16f}}, Wave::Square, 0.08f);
        engine.switchTo<PlayScene>(nextRunSeed());
    } else if (in.pressed(ActBack)) {
        engine.quit();
    }
    if (in.pressed(ActMute)) engine.audio().setMuted(!engine.audio().muted());
}

void TitleScene::draw(Canvas &cv) {
    const float pz = -t_ * 14.0f; // the title is a slow flight through the same world
    const shapes::Models &m = shapes::models();
    const float camX = std::sin(t_ * 0.3f) * 1.5f;
    rend_.setCamera({camX, 2.0f, pz + 7.5f}, {0, 1.0f, pz - 12.0f}, kFovY);
    const nature::Detail detail = nature::detailFor(engine.quality());
    setupFog(rend_, detail);
    rend_.beginFrame(cv, kSkyTop, kHorizon);
    nature::drawSky(cv, camX, t_);
    nature::drawWorld(rend_, pz, t_, detail);
    rend_.draw(m.ship, opts(r3d::Mat4::translate({0, 1.2f + std::sin(t_ * 2) * 0.15f, pz}) *
                            r3d::Mat4::rotateY(std::sin(t_ * 0.7f) * 0.25f)));
    for (int i = 0; i < 8; ++i) {
        float z = pz - 14.0f - float(i) * 7.0f;
        float x = std::sin(float(i) * 1.7f) * 7.0f;
        float y = 2.8f + std::sin(t_ * 0.9f + float(i)) * 0.6f;
        rend_.draw(m.crystal, opts(r3d::Mat4::translate({x, y, z}) * r3d::Mat4::rotateY(t_ * 0.8f + float(i))));
    }
    nature::drawGlows(cv, rend_, pz, t_, detail);
    gTotals.frames++;
    gTotals.triangles += rend_.stats().drawn;

    cv.setOffset({});
    bigText(cv, kWidth / 2.0f, 110 + std::sin(t_ * 2) * 4, "SPARK DRIFT", 8, pal::player);
    cv.text(kWidth / 2.0f, 190, "FLY THE CRYSTAL FIELD. COLLECT THE SPARKS.", 2, pal::dim, Align::Center);

    const char *controls[][2] = {
        {"WASD / ARROWS", "STEER"},
        {"SPACE", "BOOST"},
        {"P / ESC", "PAUSE"},
        {"M", "MUTE"},
    };
    float y = 300;
    for (auto &row : controls) {
        cv.text(kWidth / 2.0f - 20, y, row[0], 2, pal::player.fade(0.9f), Align::Right);
        cv.text(kWidth / 2.0f + 10, y, row[1], 2, pal::text.fade(0.85f));
        y += 24;
    }
    if (int(t_ * 2) % 2 == 0) bigText(cv, kWidth / 2.0f, 420, "PRESS ENTER OR CLICK TO START", 2, pal::spark);
    cv.text(kWidth / 2.0f, kHeight - 30, "BUILT ON SPARK ENGINE  -  SOFTWARE 3D RENDERER", 1, pal::dim.fade(0.6f), Align::Center);
}

// ---------------------------------------------------------------- flight

PlayScene::PlayScene(Engine &e, uint64_t seed) : Scene(e), rng_(seed), rend_(kWidth, kHeight) {
    s_.pos = r3d::Vec3(0, 3.0f, 0);
}

void PlayScene::spawnPylon() {
    Pylon q;
    q.pos = r3d::Vec3(rng_.range(-kHalfWidth, kHalfWidth), rng_.range(2.0f, 6.0f),
                      s_.pos.z - rng_.range(95.0f, 115.0f));
    q.spin = rng_.angle();
    s_.pylons.push_back(q);
}

void PlayScene::spawnSparkLine() {
    const float x = rng_.range(-9.0f, 9.0f), y = rng_.range(1.2f, 6.0f);
    const float z0 = s_.pos.z - rng_.range(70.0f, 90.0f);
    for (int i = 0; i < 6; ++i) s_.sparks.push_back({r3d::Vec3(x, y, z0 - 2.4f * float(i)), 0.0f});
}

void PlayScene::spawnDrone() {
    Drone d;
    d.baseX = rng_.range(-9.0f, 9.0f);
    d.pos = r3d::Vec3(d.baseX, rng_.range(2.0f, 6.0f), s_.pos.z - rng_.range(80.0f, 95.0f));
    d.phase = rng_.angle();
    d.fireT = 1.2f;
    s_.drones.push_back(d);
}

bool PlayScene::hitPlayer() {
    s_.shields -= 1;
    s_.invul = 1.5f;
    s_.combo = 0;
    s_.comboT = 0;
    flashT_ = 0.35f;
    engine.audio().play(kHit);
    if (s_.shields > 0) return false;
    RunStats st;
    st.score = s_.score;
    st.wave = s_.wave;
    st.pickups = s_.pickups;
    st.bestChain = s_.bestChain;
    st.time = s_.elapsed;
    st.distance = s_.distance;
    engine.switchTo<GameOverScene>(st);
    return true;
}

void PlayScene::update(float dt) {
    Input &in = engine.input();
    if (in.pressed(ActMute)) engine.audio().setMuted(!engine.audio().muted());
    if (in.pressed(ActPause)) {
        engine.push<PauseScene>();
        return;
    }

    PlayState &p = s_;
    p.elapsed += dt;
    const int wave = 1 + int(p.elapsed / kWaveSeconds);
    if (wave != p.wave) {
        p.wave = wave;
        p.score += 100 * (wave - 1);
        bannerT_ = 2.0f;
        engine.audio().melody({{523, 0.08f}, {659, 0.08f}, {784, 0.16f}}, Wave::Square, 0.07f);
    }
    if (p.invul > 0) p.invul -= dt;
    flashT_ = std::max(0.0f, flashT_ - dt);
    bannerT_ = std::max(0.0f, bannerT_ - dt);
    if (p.comboT > 0) {
        p.comboT -= dt;
        if (p.comboT <= 0) p.combo = 0;
    }

    // Flight: lateral and vertical steering, forward speed grows with the wave, boost drains energy.
    const float ax = in.axis(ActLeft, ActRight), ay = in.axis(ActDown, ActUp);
    p.velX = approach(p.velX, ax * 9.0f, 40.0f * dt);
    p.velY = approach(p.velY, ay * 6.0f, 30.0f * dt);
    p.pos.x = std::clamp(p.pos.x + p.velX * dt, -kHalfWidth, kHalfWidth);
    p.pos.y = std::clamp(p.pos.y + p.velY * dt, kMinY, kMaxY);
    const bool boosting = in.down(ActBoost) && p.energy > 0.0f;
    p.energy = std::clamp(p.energy + (boosting ? -0.35f : 0.18f) * dt, 0.0f, 1.0f);
    p.forward = std::min(30.0f, 12.0f + 1.5f * float(wave - 1)) + (boosting ? 9.0f : 0.0f);
    p.pos.z -= p.forward * dt;
    p.distance += p.forward * dt;
    bank_ += (-p.velX * 0.04f - bank_) * std::min(1.0f, dt * 8.0f);

    // Spawning. Density rises with the wave.
    pylonT_ -= dt;
    if (pylonT_ <= 0) {
        spawnPylon();
        pylonT_ = std::max(0.22f, 0.9f - 0.09f * float(wave - 1)) * rng_.range(0.7f, 1.3f);
    }
    sparkT_ -= dt;
    if (sparkT_ <= 0) {
        spawnSparkLine();
        sparkT_ = rng_.range(1.8f, 2.6f);
    }
    if (wave >= 3) {
        droneT_ -= dt;
        if (droneT_ <= 0) {
            spawnDrone();
            droneT_ = std::max(3.0f, 7.0f - 0.5f * float(wave)) * rng_.range(0.8f, 1.2f);
        }
    }

    // Motion and despawn (anything more than 15 units behind the glider).
    const float zBehind = p.pos.z + 15.0f;
    for (Pylon &q : p.pylons) q.spin += dt * 0.9f;
    std::erase_if(p.pylons, [&](const Pylon &q) { return q.pos.z > zBehind; });
    for (Spark &sp : p.sparks) sp.spin += dt * 3.0f;
    for (Drone &d : p.drones) {
        d.pos.z += 2.0f * dt;
        d.pos.x = d.baseX + std::sin(p.elapsed * 1.2f + d.phase) * 2.2f;
        d.fireT -= dt;
        const bool inRange = p.pos.z - d.pos.z < 80.0f;
        if (d.fireT <= 0 && inRange) {
            const r3d::Vec3 to = p.pos - d.pos;
            p.shots.push_back({d.pos, to.normalized() * 8.0f, 8.0f});
            d.fireT = 2.4f;
            engine.audio().play(kShot);
        }
    }
    std::erase_if(p.drones, [&](const Drone &d) { return d.pos.z > zBehind; });
    for (Shot &s : p.shots) {
        s.pos = s.pos + s.vel * dt;
        s.life -= dt;
    }
    std::erase_if(p.shots, [](const Shot &s) { return s.life <= 0; });

    // Pickups: always collectable.
    std::erase_if(p.sparks, [&](const Spark &sp) {
        if (dist2(sp.pos, p.pos) > 1.44f) return false; // 1.2 units
        p.combo = std::min(p.combo + 1, 99);
        p.bestChain = std::max(p.bestChain, p.combo);
        p.comboT = 2.5f;
        p.score += 10 * std::min(p.combo, 8);
        p.pickups++;
        engine.audio().play(kPickup);
        return true;
    });
    std::erase_if(p.sparks, [&](const Spark &sp) { return sp.pos.z > zBehind; });

    // Hazards. Pylons are columns, so a pylon contact is a cylinder test. Shots are
    // consumed on contact even while invulnerable; only vulnerable contacts cost a shield.
    const bool vulnerable = p.invul <= 0;
    bool hit = false;
    if (vulnerable) {
        for (const Pylon &q : p.pylons)
            if (std::fabs(p.pos.y - q.pos.y) < 2.0f && std::hypot(p.pos.x - q.pos.x, p.pos.z - q.pos.z) < 1.5f) {
                hit = true;
                break;
            }
        std::erase_if(p.drones, [&](const Drone &d) {
            if (dist2(d.pos, p.pos) > 1.44f) return false;
            hit = true;   // ramming a drone costs a shield too
            return true;  // ...and destroys it
        });
    }
    std::erase_if(p.shots, [&](const Shot &s) {
        if (dist2(s.pos, p.pos) > 0.49f) return false;
        if (vulnerable) hit = true;
        return true;
    });
    if (hit && hitPlayer()) return; // game over (the scene has already been switched)
}

void PlayScene::draw(Canvas &cv) {
    const PlayState &p = s_;
    const shapes::Models &m = shapes::models();
    const float jitter = flashT_ > 0 ? std::sin(float(engine.time()) * 60.0f) * 0.15f * flashT_ : 0.0f;
    const float camX = p.pos.x * 0.7f + jitter;
    rend_.setCamera({camX, p.pos.y + 2.0f, p.pos.z + 7.5f}, {p.pos.x * 0.9f, p.pos.y - 0.3f, p.pos.z - 12.0f}, kFovY);
    const nature::Detail detail = nature::detailFor(engine.quality());
    setupFog(rend_, detail);
    rend_.beginFrame(cv, kSkyTop, kHorizon);
    nature::drawSky(cv, camX, p.elapsed);
    nature::drawWorld(rend_, p.pos.z, p.elapsed, detail);
    for (const Pylon &q : p.pylons) rend_.draw(m.crystal, opts(r3d::Mat4::translate(q.pos) * r3d::Mat4::rotateY(q.spin)));
    for (const Spark &sp : p.sparks) {
        r3d::Vec3 at = sp.pos + r3d::Vec3(0, std::sin(sp.spin) * 0.15f, 0);
        rend_.draw(m.spark, opts(r3d::Mat4::translate(at) * r3d::Mat4::rotateY(sp.spin), true));
    }
    for (const Drone &d : p.drones) rend_.draw(m.drone, opts(r3d::Mat4::translate(d.pos)));
    for (const Shot &s : p.shots) rend_.draw(m.shot, opts(r3d::Mat4::translate(s.pos), true));
    if (p.invul <= 0 || int(p.invul * 12) % 2 == 0)
        rend_.draw(m.ship, opts(r3d::Mat4::translate(p.pos) * r3d::Mat4::rotateZ(bank_)));
    nature::drawGlows(cv, rend_, p.pos.z, p.elapsed, detail);
    gTotals.frames++;
    gTotals.triangles += rend_.stats().drawn;

    // HUD on top of the 3D view.
    cv.setOffset({});
    cv.textf(16, 12, 2, pal::text, Align::Left, "SCORE %ld", p.score);
    cv.textf(16, 34, 1, pal::dim, Align::Left, "DISTANCE %dM", int(p.distance));
    cv.textf(kWidth - 16.0f, 12, 2, pal::text, Align::Right, "WAVE %d", p.wave);
    for (int i = 0; i < 3; ++i)
        cv.rect(float(kWidth - 16 - 20 * (3 - i)), 36, 16, 8, i < p.shields ? pal::player : pal::dim.fade(0.25f));
    cv.rect(16, kHeight - 22.0f, 160, 8, pal::dim.fade(0.3f));
    cv.rect(16, kHeight - 22.0f, 160 * p.energy, 8, p.energy > 0.25f ? pal::spark : pal::danger);
    cv.text(16, kHeight - 38.0f, "BOOST", 1, pal::dim, Align::Left);
    if (p.combo > 1) cv.textf(kWidth / 2.0f, 12, 2, pal::spark, Align::Center, "CHAIN x%d", std::min(p.combo, 8));
    if (bannerT_ > 0) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "WAVE %d", p.wave);
        bigText(cv, kWidth / 2.0f, kHeight * 0.28f, buf, 6, pal::text.fade(std::min(1.0f, bannerT_ * 2)));
    }
    if (flashT_ > 0) cv.overlay(Color(255, 40, 60, int(90 * flashT_ / 0.35f)));
    cv.text(kWidth / 2.0f, kHeight - 16.0f, "SPACE BOOST   P PAUSE", 1, pal::dim.fade(0.6f), Align::Center);
}

// ---------------------------------------------------------------- pause

void PauseScene::update(float dt) {
    t_ += dt;
    Input &in = engine.input();
    if (in.pressed(ActPause) || in.pressed(ActConfirm)) engine.pop();
    else if (in.pressed(Key::Q)) engine.switchTo<TitleScene>();
    if (in.pressed(ActMute)) engine.audio().setMuted(!engine.audio().muted());
}

void PauseScene::draw(Canvas &cv) {
    cv.setOffset({});
    cv.overlay(Color(0, 0, 0, 150));
    bigText(cv, kWidth / 2.0f, 200, "PAUSED", 7, pal::text);
    cv.text(kWidth / 2.0f, 290, "ESC / P / ENTER  RESUME", 2, pal::dim, Align::Center);
    cv.text(kWidth / 2.0f, 318, "Q  QUIT TO TITLE", 2, pal::dim, Align::Center);
    cv.text(kWidth / 2.0f, 346, engine.audio().muted() ? "M  UNMUTE" : "M  MUTE", 2, pal::dim, Align::Center);
}

// ---------------------------------------------------------------- game over

GameOverScene::GameOverScene(Engine &e, RunStats stats)
    : Scene(e), stats_(stats), best_(std::max(gBest, stats.score)), newBest_(stats.score > 0 && stats.score >= best_) {
    gBest = best_;
}

void GameOverScene::update(float dt) {
    t_ += dt;
    if (t_ < 0.6f) return; // don't let a held key skip the screen
    Input &in = engine.input();
    if (in.pressed(ActConfirm)) engine.switchTo<PlayScene>(nextRunSeed());
    else if (in.pressed(ActBack)) engine.switchTo<TitleScene>();
}

void GameOverScene::draw(Canvas &cv) {
    cv.setOffset({});
    const float a = std::min(1.0f, t_ * 2);
    cv.overlay(Color(4, 6, 14, int(170 * a)));
    const float drop = (1 - ease::outBack(std::min(1.0f, t_ * 2))) * -40;
    bigText(cv, kWidth / 2.0f, 120 + drop, "GAME OVER", 8, pal::danger.fade(a));

    char buf[96];
    std::snprintf(buf, sizeof buf, "%ld", stats_.score);
    bigText(cv, kWidth / 2.0f, 210, buf, 6, pal::text.fade(a));
    if (newBest_ && int(t_ * 3) % 2 == 0) cv.text(kWidth / 2.0f, 262, "NEW BEST!", 2, pal::spark, Align::Center);
    else if (!newBest_) cv.textf(kWidth / 2.0f, 262, 2, pal::dim, Align::Center, "BEST  %ld", best_);

    const int secs = int(stats_.time);
    std::snprintf(buf, sizeof buf, "WAVE %d   SPARKS %d   CHAIN %d   TIME %d:%02d", stats_.wave, stats_.pickups,
                  stats_.bestChain, secs / 60, secs % 60);
    cv.text(kWidth / 2.0f, 310, buf, 2, pal::text.fade(0.8f * a), Align::Center);

    if (t_ > 0.6f && int(t_ * 2) % 2 == 0)
        cv.text(kWidth / 2.0f, 380, "ENTER / CLICK  RETRY      ESC  TITLE", 2, pal::spark, Align::Center);
}

} // namespace drift
