#include "scenes.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace warden {

// ---------------------------------------------------------------- persistence

namespace {
std::string bestPath() {
    const char *home = std::getenv("HOME");
    return std::string(home ? home : ".") + "/.spark_warden_best";
}
uint64_t gBaseSeed = 0;
uint64_t gRuns = 0;
} // namespace

long loadBest() {
    long v = 0;
    if (FILE *f = std::fopen(bestPath().c_str(), "r")) {
        if (std::fscanf(f, "%ld", &v) != 1) v = 0;
        std::fclose(f);
    }
    return v;
}

void saveBest(long score) {
    if (FILE *f = std::fopen(bestPath().c_str(), "w")) {
        std::fprintf(f, "%ld\n", score);
        std::fclose(f);
    }
}

void setBaseSeed(uint64_t s) { gBaseSeed = s; }

Stress &stress() {
    static Stress s;
    return s;
}

uint64_t nextRunSeed(Engine &e) {
    ++gRuns;
    if (gBaseSeed) return gBaseSeed + gRuns * 7919;
    (void)e;
    return uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
}

// ---------------------------------------------------------------- shared visuals

void drawBackground(Canvas &cv, float time, Vec2 focus) {
    cv.clear(pal::bg);
    constexpr float kCell = 48;
    for (float x = kArenaPad; x <= kWidth - kArenaPad; x += kCell) cv.rect(x, kArenaPad, 1, kHeight - 2 * kArenaPad, pal::grid);
    for (float y = kArenaPad; y <= kHeight - kArenaPad; y += kCell) cv.rect(kArenaPad, y, kWidth - 2 * kArenaPad, 1, pal::grid);
    cv.glow(focus, 260, pal::border.fade(0.05f)); // soft light around the player

    // Drifting dust: positions derived from the index, so no state is needed.
    for (int i = 0; i < 70; ++i) {
        uint32_t h = uint32_t(i) * 2654435761u;
        float sx = float(h % 997) / 997.0f, sy = float((h >> 10) % 991) / 991.0f;
        float speed = 4 + float(h % 13);
        float x = std::fmod(sx * kWidth + time * speed, float(kWidth));
        float y = std::fmod(sy * kHeight + time * speed * 0.3f, float(kHeight));
        cv.rect(x, y, 1, 1, pal::dim.fade(0.25f + float(h % 5) * 0.08f));
    }
}

void drawTitleText(Canvas &cv, float x, float y, const char *s, int scale, Color c) {
    // Cheap bloom: a few offset additive copies under the crisp text.
    Blend prev = cv.blend();
    cv.setBlend(Blend::Add);
    Color halo = c.fade(0.18f);
    const float o = float(std::max(1, scale / 2));
    for (Vec2 d : {Vec2{-o, 0}, Vec2{o, 0}, Vec2{0, -o}, Vec2{0, o}})
        cv.text(x + d.x, y + d.y, s, scale, halo, Align::Center);
    cv.setBlend(prev);
    cv.text(x, y, s, scale, c, Align::Center);
}

// ---------------------------------------------------------------- title

void TitleScene::update(float dt) {
    t_ += dt;
    Input &in = engine.input();
    if (in.pressed(Confirm)) {
        engine.audio().melody({{392, 0.07f}, {523, 0.07f}, {659, 0.07f}, {784, 0.16f}}, Wave::Square, 0.08f);
        engine.switchTo(makePlay(engine, nextRunSeed(engine)));
    } else if (in.pressed(Back)) {
        engine.quit();
    }
    if (in.pressed(Mute)) engine.audio().setMuted(!engine.audio().muted());
}

void TitleScene::draw(Canvas &cv) {
    Vec2 center{kWidth / 2.0f, kHeight / 2.0f};
    drawBackground(cv, t_, center + Vec2{std::cos(t_ * 0.4f) * 200, std::sin(t_ * 0.6f) * 80});

    // Enemy shapes orbiting the title, as a preview of what's coming.
    const Color cols[] = {pal::drifter, pal::dart, pal::splitter, pal::spinner, pal::mini};
    const Vec2 shape[] = {{1.2f, 0}, {0, 1.2f}, {-1.2f, 0}, {0, -1.2f}};
    for (int i = 0; i < 10; ++i) {
        float a = t_ * 0.35f + float(i) * TAU / 10;
        Vec2 p = center + Vec2{std::cos(a) * 380, std::sin(a) * 190};
        Color c = cols[i % 5];
        cv.glow(p, 34, c.fade(0.25f));
        cv.shape(shape, p, t_ * 1.5f + float(i), 12, c.fade(0.2f), c.fade(0.7f), 2);
    }

    float bob = std::sin(t_ * 2) * 4;
    drawTitleText(cv, center.x, 130 + bob, "SPARK WARDEN", 9, pal::player);
    cv.text(center.x, 205, "HOLD THE ARENA. COLLECT THE SPARKS.", 2, pal::dim, Align::Center);

    const char *controls[][2] = {
        {"WASD", "MOVE"},
        {"MOUSE / ARROWS", "AIM + FIRE"},
        {"SPACE / SHIFT", "DASH (BRIEFLY INVULNERABLE)"},
        {"P / ESC", "PAUSE"},
        {"M", "MUTE"},
    };
    float y = 250;
    for (auto &row : controls) {
        cv.text(center.x - 20, y, row[0], 2, pal::player.fade(0.9f), Align::Right);
        cv.text(center.x + 10, y, row[1], 2, pal::text.fade(0.85f));
        y += 24;
    }

    if (int(t_ * 2) % 2 == 0) drawTitleText(cv, center.x, 408, "PRESS ENTER OR CLICK TO START", 2, pal::spark);
    if (best_ > 0) cv.textf(center.x, 450, 2, pal::dim, Align::Center, "BEST  %ld", best_);
    cv.text(center.x, kHeight - 30, "BUILT ON SPARK ENGINE  -  C++ SOFTWARE RENDERER", 1, pal::dim.fade(0.6f), Align::Center);
}

// ---------------------------------------------------------------- pause

void PauseScene::update(float dt) {
    t_ += dt;
    Input &in = engine.input();
    if (in.pressed(Pause) || in.pressed(Confirm)) engine.pop();
    else if (in.pressed(Key::Q)) engine.switchTo<TitleScene>();
    if (in.pressed(Mute)) engine.audio().setMuted(!engine.audio().muted());
}

void PauseScene::draw(Canvas &cv) {
    cv.setOffset({});
    cv.overlay(Color(0, 0, 0, 150));
    drawTitleText(cv, kWidth / 2.0f, 200, "PAUSED", 7, pal::text);
    cv.text(kWidth / 2.0f, 290, "ESC / P / ENTER  RESUME", 2, pal::dim, Align::Center);
    cv.text(kWidth / 2.0f, 318, "Q  QUIT TO TITLE", 2, pal::dim, Align::Center);
    cv.text(kWidth / 2.0f, 346, engine.audio().muted() ? "M  UNMUTE" : "M  MUTE", 2, pal::dim, Align::Center);
}

// ---------------------------------------------------------------- game over

GameOverScene::GameOverScene(Engine &e, RunStats stats)
    : Scene(e), stats_(stats), best_(std::max(loadBest(), stats.score)), newBest_(stats.score > 0 && stats.score >= best_) {}

void GameOverScene::update(float dt) {
    t_ += dt;
    if (t_ < 0.6f) return; // don't let a held fire button skip the screen
    Input &in = engine.input();
    if (in.pressed(Confirm)) engine.switchTo(makePlay(engine, nextRunSeed(engine)));
    else if (in.pressed(Back)) engine.switchTo<TitleScene>();
}

void GameOverScene::draw(Canvas &cv) {
    cv.setOffset({});
    float a = std::min(1.0f, t_ * 2);
    cv.overlay(Color(4, 6, 14, int(170 * a)));
    float drop = (1 - ease::outBack(std::min(1.0f, t_ * 2))) * -40;
    drawTitleText(cv, kWidth / 2.0f, 120 + drop, "GAME OVER", 8, pal::danger.fade(a));

    char buf[64];
    std::snprintf(buf, sizeof buf, "%ld", stats_.score);
    drawTitleText(cv, kWidth / 2.0f, 210, buf, 6, pal::text.fade(a));
    if (newBest_ && int(t_ * 3) % 2 == 0) cv.text(kWidth / 2.0f, 262, "NEW BEST!", 2, pal::spark, Align::Center);
    else if (!newBest_) cv.textf(kWidth / 2.0f, 262, 2, pal::dim, Align::Center, "BEST  %ld", best_);

    int secs = int(stats_.time);
    std::snprintf(buf, sizeof buf, "WAVE %d   KILLS %d   CHAIN %d   TIME %d:%02d", stats_.wave, stats_.kills,
                  stats_.bestChain, secs / 60, secs % 60);
    cv.text(kWidth / 2.0f, 310, buf, 2, pal::text.fade(0.8f * a), Align::Center);

    if (t_ > 0.6f && int(t_ * 2) % 2 == 0)
        cv.text(kWidth / 2.0f, 380, "ENTER / CLICK  RETRY      ESC  TITLE", 2, pal::spark, Align::Center);
}

std::unique_ptr<Scene> makeTitle(Engine &e) { return std::make_unique<TitleScene>(e); }

} // namespace warden
