// Spark Drift: pilot a glider down an endless corridor of crystal pylons,
// collecting sparks and dodging drones. Rendered by r3d, UI by the engine.
#pragma once

#include <cstdint>
#include <vector>

#include "meshes.hpp"
#include "r3d.hpp"
#include "spark.hpp"

namespace drift {

using namespace spark;

constexpr int kWidth = 960;
constexpr int kHeight = 540;
constexpr float kWaveSeconds = 15.0f; // a new wave every 15 s of flight
constexpr float kHalfWidth = 11.0f;   // lateral limit of the glider
constexpr float kMinY = 0.8f, kMaxY = 6.5f;
constexpr float kFovY = 1.0471976f;  // 60 degrees

// Input actions (bound in main.cpp).
enum Action { ActUp, ActDown, ActLeft, ActRight, ActBoost, ActPause, ActConfirm, ActBack, ActMute };

// Palette: neon on deep navy, as the rest of the Spark games.
namespace pal {
constexpr Color text = Color::hex(0xe8f6ff);
constexpr Color dim = Color::hex(0x6d86a8);
constexpr Color player = Color::hex(0x5cf2ff);
constexpr Color spark = Color::hex(0xffd84a);
constexpr Color danger = Color::hex(0xff3b4f);
} // namespace pal

// Flight world. Forward is -z: the glider starts at the origin, pylons and drones
// are spawned at negative z ahead of it, and objects behind it are removed.
struct Pylon {
    r3d::Vec3 pos;
    float spin = 0;
};
struct Spark {
    r3d::Vec3 pos;
    float spin = 0;
};
struct Drone {
    r3d::Vec3 pos;
    float baseX = 0, phase = 0, fireT = 0;
};
struct Shot {
    r3d::Vec3 pos, vel;
    float life = 0;
};

/** Result of a finished run, shown on the game-over screen. */
struct RunStats {
    long score = 0;
    int wave = 1;
    int pickups = 0;
    int bestChain = 0;
    float time = 0;
    float distance = 0;
};

/** Everything about the current run. The autopilot reads this; it never writes it. */
struct PlayState {
    r3d::Vec3 pos;
    float velX = 0, velY = 0, forward = 0, energy = 1, invul = 0, elapsed = 0, distance = 0;
    float comboT = 0;
    int shields = 3, wave = 1, combo = 0, bestChain = 0, pickups = 0;
    long score = 0;
    std::vector<Pylon> pylons;
    std::vector<Spark> sparks;
    std::vector<Drone> drones;
    std::vector<Shot> shots;
};

/** Render-time totals, for the headless summary (triangles drawn per 3D frame). */
struct DrawTotals {
    long frames = 0;
    long triangles = 0;
};
DrawTotals &drawTotals();

/** Run seed for the next run: base seed plus a run counter (deterministic when --seed is set). */
void setBaseSeed(uint64_t seed);
uint64_t nextRunSeed();

class TitleScene : public Scene {
public:
    explicit TitleScene(Engine &e);
    void update(float dt) override;
    void draw(Canvas &cv) override;

private:
    float t_ = 0;
    r3d::Renderer rend_;
};

class PlayScene : public Scene {
public:
    PlayScene(Engine &e, uint64_t seed);
    void update(float dt) override;
    void draw(Canvas &cv) override;
    const PlayState &state() const { return s_; }

private:
    bool hitPlayer();
    void spawnPylon();
    void spawnSparkLine();
    void spawnDrone();

    PlayState s_;
    Rng rng_;
    float pylonT_ = 0.5f, sparkT_ = 1.0f, droneT_ = 0;
    float bank_ = 0, bannerT_ = 0, flashT_ = 0;
    r3d::Renderer rend_;
};

class PauseScene : public Scene {
public:
    explicit PauseScene(Engine &e) : Scene(e) {}
    void update(float dt) override;
    void draw(Canvas &cv) override;

private:
    float t_ = 0;
};

class GameOverScene : public Scene {
public:
    GameOverScene(Engine &e, RunStats stats);
    void update(float dt) override;
    void draw(Canvas &cv) override;

private:
    RunStats stats_;
    long best_;
    bool newBest_;
    float t_ = 0;
};

/** Headless autopilot for `--bot`. It only touches Input, like a player would. */
struct BotReport {
    int deaths = 0;
    bool onGameOver = false;
    long bestScore = 0;
    int bestWave = 0;
};
void botStep(Engine &e, long frame, BotReport &report);

} // namespace drift
