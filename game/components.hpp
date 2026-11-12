// ECS components for Spark Warden. Plain data only; behavior lives in the
// systems in play.cpp.
#pragma once

#include <string>

#include "spark.hpp"

namespace warden {

using spark::Color;
using spark::Vec2;

// Collision layers (what something IS) — masks say what it wants to HIT.
enum Layer : uint32_t {
    L_PLAYER = 1 << 0,
    L_ENEMY = 1 << 1,
    L_BULLET = 1 << 2,
    L_SPARK = 1 << 3,
    L_ESHOT = 1 << 4,
};

struct Transform {
    Vec2 pos;
    float angle = 0;
};

struct Velocity {
    Vec2 v;
    float drag = 0; // fraction of speed lost per second
};

struct Body {
    float radius = 8;
    uint32_t layer = 0;
    uint32_t mask = 0;
};

struct Player {
    int hp = 3;
    float invuln = 0;
    float fireCd = 0;
    float dashCd = 0;
    float dashT = 0;
    Vec2 dashDir;
    float aim = 0;
};

enum class EnemyKind { Drifter, Dart, Splitter, Mini, Spinner };

struct Enemy {
    EnemyKind kind = EnemyKind::Drifter;
    int hp = 1;
    int score = 10;
    float flash = 0;  // white flash after a hit
    float timer = 0;  // behavior timer
    int state = 0;    // behavior state machine
    float spin = 0;   // visual rotation speed
    float phase = 0;  // per-enemy offset for wobble/orbit
};

struct Bullet {
    float life = 0.8f;
    int damage = 1;
};

struct EnemyShot {
    float life = 4;
};

struct Spark {
    float life = 8;
};

/** Spawn telegraph: an enemy appears here when the timer runs out. */
struct Warning {
    float t = 0.9f;
    float total = 0.9f;
    EnemyKind kind = EnemyKind::Drifter;
};

/** Expanding ring effect (shockwaves, spawns). */
struct Ring {
    float r0 = 0, r1 = 100;
    float life = 0.4f, total = 0.4f;
    float thick = 3;
    Color color;
};

/** Floating score text. */
struct Popup {
    std::string text;
    float life = 0.8f;
    Color color;
};

} // namespace warden
