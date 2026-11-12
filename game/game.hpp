// Spark Warden — shared game declarations.
#pragma once

#include <string>

#include "spark.hpp"

namespace warden {

using namespace spark;

constexpr int kWidth = 960;
constexpr int kHeight = 540;
constexpr float kArenaPad = 24; // playable area is the screen minus this margin

// Input actions (bound in main.cpp).
enum Action { MoveUp, MoveDown, MoveLeft, MoveRight, AimUp, AimDown, AimLeft, AimRight, Fire, Dash, Pause, Confirm, Back, Mute };

// Palette: neon on deep navy.
namespace pal {
constexpr Color bg = Color::hex(0x060912);
constexpr Color grid = Color::hex(0x101a2e);
constexpr Color border = Color::hex(0x2be0ff);
constexpr Color player = Color::hex(0x5cf2ff);
constexpr Color bullet = Color::hex(0xd8fbff);
constexpr Color spark = Color::hex(0xffd84a);
constexpr Color drifter = Color::hex(0xff4fd8);
constexpr Color dart = Color::hex(0xff8a3d);
constexpr Color splitter = Color::hex(0x6bff7a);
constexpr Color mini = Color::hex(0xb8ff9c);
constexpr Color spinner = Color::hex(0xfff06a);
constexpr Color eshot = Color::hex(0xff5a5a);
constexpr Color danger = Color::hex(0xff3b4f);
constexpr Color text = Color::hex(0xe8f6ff);
constexpr Color dim = Color::hex(0x6d86a8);
} // namespace pal

/** Result of a finished run, shown on the game-over screen. */
struct RunStats {
    long score = 0;
    int wave = 0;
    int kills = 0;
    int bestChain = 0;
    float time = 0;
};

/** Stress knobs for frame-time data collection (research/frametime). Defaults = normal game. */
struct Stress {
    float spawnMult = 1; // enemies per wave and spawn rate multiplier
};
Stress &stress();

// Persistent best score ($HOME/.spark_warden_best). Disabled in headless runs.
long loadBest();
void saveBest(long score);

// Shared visuals.
void drawBackground(Canvas &cv, float time, Vec2 focus);
void drawTitleText(Canvas &cv, float x, float y, const char *s, int scale, Color c);

// Scenes.
class TitleScene;
class PlayScene;
class PauseScene;
class GameOverScene;

std::unique_ptr<Scene> makeTitle(Engine &e);
std::unique_ptr<Scene> makePlay(Engine &e, uint64_t seed);

// Headless autopilot used by tests (`--bot`). Drives input like a player would.
struct BotReport {
    int deaths = 0;
    bool onGameOver = false;
    long bestScore = 0;
    int bestWave = 0;
    size_t peakEntities = 0;
};
void botStep(Engine &e, long frame, BotReport &report);

} // namespace warden
