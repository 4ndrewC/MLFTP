#include "play.hpp"
#include "scenes.hpp"

#include <cstdio>
#include <optional>

namespace warden {

namespace {

// Local-space shapes (unit size; scaled by each entity's radius).
const Vec2 kShipShape[] = {{1.5f, 0}, {-0.7f, 0.95f}, {-0.3f, 0.35f}, {-0.6f, 0}, {-0.3f, -0.35f}, {-0.7f, -0.95f}};
const Vec2 kDiamond[] = {{1.2f, 0}, {0, 1.2f}, {-1.2f, 0}, {0, -1.2f}};
const Vec2 kArrow[] = {{1.5f, 0}, {-0.9f, 0.9f}, {-0.4f, 0}, {-0.9f, -0.9f}};
const Vec2 kHex[] = {{1, 0}, {0.5f, 0.866f}, {-0.5f, 0.866f}, {-1, 0}, {-0.5f, -0.866f}, {0.5f, -0.866f}};
const Vec2 kTri[] = {{1.3f, 0}, {-0.8f, 1}, {-0.8f, -1}};
const Vec2 kStar[] = {{1.3f, 0},  {0.4f, 0.4f},  {0, 1.3f},  {-0.4f, 0.4f},
                      {-1.3f, 0}, {-0.4f, -0.4f}, {0, -1.3f}, {0.4f, -0.4f}};

struct EnemyDef {
    float radius;
    int hp;
    int score;
    int sparks;
    Color color;
};

EnemyDef defOf(EnemyKind k) {
    switch (k) {
    case EnemyKind::Drifter: return {13, 1, 10, 1, pal::drifter};
    case EnemyKind::Dart: return {12, 2, 25, 2, pal::dart};
    case EnemyKind::Splitter: return {22, 5, 40, 2, pal::splitter};
    case EnemyKind::Mini: return {8, 1, 5, 0, pal::mini};
    case EnemyKind::Spinner: return {15, 4, 60, 3, pal::spinner};
    }
    return {10, 1, 10, 1, pal::drifter};
}

Vec2 clampToArena(Vec2 p, float r) {
    return {std::clamp(p.x, PlayScene::kMinX + r, PlayScene::kMaxX - r),
            std::clamp(p.y, PlayScene::kMinY + r, PlayScene::kMaxY - r)};
}

} // namespace

PlayScene::PlayScene(Engine &e, uint64_t seed) : Scene(e), fx_(5000, seed ^ 0xF00D), rng_(seed) {
    if (e.config().burstMult > 1) fx_.setMax(size_t(5000 * e.config().burstMult));
}

int PlayScene::multiplier() const { return std::min(8, 1 + chain_ / 8); }

void PlayScene::enter() {
    spawnPlayer();
    startWave(1);
}

// ---------------------------------------------------------------- spawning

void PlayScene::spawnPlayer() {
    player_ = world_.create();
    world_.add<Transform>(player_, {{kWidth / 2.0f, kHeight / 2.0f}, -PI / 2});
    world_.add<Velocity>(player_, {});
    world_.add<Body>(player_, {11, L_PLAYER, L_SPARK});
    world_.add<Player>(player_, {});
    spawnRing({kWidth / 2.0f, kHeight / 2.0f}, 0, 90, 0.5f, pal::player);
}

Entity PlayScene::spawnEnemy(EnemyKind kind, Vec2 pos) {
    EnemyDef d = defOf(kind);
    Entity e = world_.create();
    world_.add<Transform>(e, {pos, rng_.angle()});
    world_.add<Velocity>(e, {{}, 0});
    world_.add<Body>(e, {d.radius, L_ENEMY, L_PLAYER | L_ENEMY});
    Enemy en;
    en.kind = kind;
    en.hp = d.hp;
    en.score = d.score;
    en.spin = rng_.range(-2.0f, 2.0f);
    en.phase = rng_.angle();
    world_.add<Enemy>(e, en);
    return e;
}

void PlayScene::spawnWarning(EnemyKind kind, Vec2 pos) {
    Entity e = world_.create();
    world_.add<Transform>(e, {pos, 0});
    world_.add<Warning>(e, {0.9f, 0.9f, kind});
}

void PlayScene::spawnBullet(Vec2 pos, float angle) {
    Entity b = world_.create();
    world_.add<Transform>(b, {pos, angle});
    world_.add<Velocity>(b, {Vec2::fromAngle(angle, 780), 0});
    world_.add<Body>(b, {4, L_BULLET, L_ENEMY});
    world_.add<Bullet>(b, {});
}

void PlayScene::spawnShot(Vec2 pos, Vec2 vel) {
    Entity s = world_.create();
    world_.add<Transform>(s, {pos, 0});
    world_.add<Velocity>(s, {vel, 0});
    world_.add<Body>(s, {6, L_ESHOT, L_PLAYER});
    world_.add<EnemyShot>(s, {});
}

void PlayScene::spawnSpark(Vec2 pos, Vec2 vel) {
    Entity s = world_.create();
    world_.add<Transform>(s, {pos, rng_.angle()});
    world_.add<Velocity>(s, {vel, 3.5f});
    world_.add<Body>(s, {7, L_SPARK, 0});
    world_.add<Spark>(s, {});
}

void PlayScene::spawnRing(Vec2 pos, float r0, float r1, float life, Color c, float thick) {
    Entity r = world_.create();
    world_.add<Transform>(r, {pos, 0});
    world_.add<Ring>(r, {r0, r1, life, life, thick, c});
}

void PlayScene::spawnPopup(Vec2 pos, std::string text, Color c) {
    Entity p = world_.create();
    world_.add<Transform>(p, {pos, 0});
    world_.add<Popup>(p, {std::move(text), 0.8f, c});
}

Vec2 PlayScene::randomSpawnPoint() {
    // Never spawn on top of the player.
    for (int tries = 0; tries < 30; ++tries) {
        Vec2 p{rng_.range(kMinX + 40, kMaxX - 40), rng_.range(kMinY + 40, kMaxY - 40)};
        if (distance(p, lastPlayerPos_) > 230) return p;
    }
    return {lastPlayerPos_.x < kWidth / 2.0f ? kMaxX - 60 : kMinX + 60, rng_.range(kMinY + 40, kMaxY - 40)};
}

EnemyKind PlayScene::pickEnemyKind() {
    int w = stats_.wave;
    float weights[] = {5, w >= 2 ? 3.0f : 0, w >= 3 ? 2.0f : 0, 0, w >= 4 ? 1.0f + float(w) / 5 : 0};
    float total = 0;
    for (float x : weights) total += x;
    float r = rng_.range(0.0f, total);
    for (int i = 0; i < 5; ++i) {
        if (r < weights[i]) return EnemyKind(i);
        r -= weights[i];
    }
    return EnemyKind::Drifter;
}

// ---------------------------------------------------------------- update

void PlayScene::update(float dt) {
    time_ += dt;
    stats_.time = deathTimer_ < 0 ? time_ : stats_.time;
    Input &in = engine.input();

    if (in.pressed(Pause) && deathTimer_ < 0) {
        engine.push<PauseScene>();
        return;
    }
    if (in.pressed(Mute)) engine.audio().setMuted(!engine.audio().muted());

    fx_.setBurstScale(engine.particleScale());
    Profiler &prof = engine.prof();
    playerSystem(dt);
    enemySystem(dt);
    motionSystem(dt);
    boundsSystem();
    warningSystem(dt);
    sparkSystem(dt);
    {
        ProfScope ps(prof, Prof::Collision);
        collisionSystem();
    }
    lifetimeSystem(dt);
    waveSystem(dt);
    {
        ProfScope ps(prof, Prof::ParticlesUpd);
        fx_.update(dt);
    }
    {
        ProfScope ps(prof, Prof::Flush);
        world_.flush();
    }

    hurtFlash_ = std::max(0.0f, hurtFlash_ - dt);
    bannerT_ = std::max(0.0f, bannerT_ - dt);

    if (deathTimer_ >= 0) {
        deathTimer_ += dt;
        if (deathTimer_ > 1.6f && engine.top() == this) {
            if (!engine.headless() && stats_.score > loadBest()) saveBest(stats_.score);
            engine.push<GameOverScene>(stats_);
            deathTimer_ = -100; // pushed once; stays "dead"
        }
    }
}

void PlayScene::playerSystem(float dt) {
    Player *pl = world_.get<Player>(player_);
    if (!pl) return;
    Transform &tr = *world_.get<Transform>(player_);
    Velocity &vel = *world_.get<Velocity>(player_);
    Input &in = engine.input();
    Audio &au = engine.audio();

    pl->invuln = std::max(0.0f, pl->invuln - dt);
    pl->fireCd -= dt;
    pl->dashCd -= dt;

    Vec2 move = Vec2{in.axis(MoveLeft, MoveRight), in.axis(MoveUp, MoveDown)}.normalized();
    Vec2 aimKeys{in.axis(AimLeft, AimRight), in.axis(AimUp, AimDown)};
    bool keyAim = aimKeys.lengthSq() > 0;
    Vec2 toMouse = in.mouse() - tr.pos;
    if (keyAim) pl->aim = aimKeys.angle();
    else if (toMouse.lengthSq() > 4) pl->aim = toMouse.angle();

    if (pl->dashT > 0) {
        pl->dashT -= dt;
        vel.v = pl->dashDir * 900;
        fx_.emit(tr.pos, -pl->dashDir * 60, 0.3f, 6, pal::player.fade(0.6f), 4);
    } else {
        // Snappy but not instant: blend toward the target velocity.
        Vec2 target = move * 290;
        vel.v += (target - vel.v) * std::min(1.0f, 16 * dt);
        if (in.pressed(Dash) && pl->dashCd <= 0) {
            pl->dashDir = move.lengthSq() > 0 ? move : Vec2::fromAngle(pl->aim);
            pl->dashT = 0.12f;
            pl->dashCd = 0.75f;
            pl->invuln = std::max(pl->invuln, 0.22f);
            spawnRing(tr.pos, 6, 40, 0.25f, pal::player, 2);
            au.play({.wave = Wave::Noise, .freq = 3000, .slide = -2500, .dur = 0.18f, .vol = 0.18f});
        }
    }
    tr.angle = pl->aim;

    bool firing = in.down(Fire) || keyAim;
    if (firing && pl->fireCd <= 0) {
        pl->fireCd = 0.095f;
        Vec2 dir = Vec2::fromAngle(pl->aim);
        spawnBullet(tr.pos + dir * 17, pl->aim + rng_.range(-0.035f, 0.035f));
        fx_.burst(tr.pos + dir * 18, {.count = 3, .speed = 160, .angle = pl->aim, .spread = 0.8f, .life = 0.12f,
                                      .size = 3, .color = pal::bullet});
        vel.v -= dir * 25; // a little recoil
        au.play({.wave = Wave::Square, .freq = 880, .slide = -500, .dur = 0.06f, .vol = 0.06f, .duty = 0.25f});
    }
    lastPlayerPos_ = tr.pos;
}

void PlayScene::enemySystem(float dt) {
    Vec2 target = lastPlayerPos_;
    bool hasTarget = world_.alive(player_);
    float speedBoost = 1.0f + float(std::min(stats_.wave, 12)) * 0.04f;
    Audio &au = engine.audio();

    world_.each<Enemy, Transform, Velocity>([&](Entity, Enemy &en, Transform &tr, Velocity &vel) {
        en.flash = std::max(0.0f, en.flash - dt);
        en.timer += dt;
        Vec2 to = hasTarget ? target - tr.pos : Vec2::fromAngle(en.phase + time_, 1);
        Vec2 dir = to.normalized();
        auto steer = [&](Vec2 desired, float rate) { vel.v += (desired - vel.v) * std::min(1.0f, rate * dt); };

        switch (en.kind) {
        case EnemyKind::Drifter:
        case EnemyKind::Mini:
        case EnemyKind::Splitter: {
            float speed = en.kind == EnemyKind::Mini ? 165.0f : en.kind == EnemyKind::Splitter ? 55.0f : 95.0f;
            // Weave side to side so groups don't move as one blob.
            Vec2 side{-dir.y, dir.x};
            float weave = std::sin(time_ * 3 + en.phase) * 0.45f;
            steer((dir + side * weave).normalized() * speed * speedBoost, en.kind == EnemyKind::Mini ? 5.0f : 2.5f);
            tr.angle += en.spin * dt;
            break;
        }
        case EnemyKind::Dart: {
            // stalk -> wind up (telegraphed) -> charge in a straight line
            if (en.state == 0) {
                tr.angle += angleDiff(tr.angle, dir.angle()) * std::min(1.0f, 6 * dt);
                steer(dir * 60, 2);
                if (en.timer > 1.1f + std::fmod(en.phase, 0.6f)) en.state = 1, en.timer = 0;
            } else if (en.state == 1) {
                if (en.timer < 0.3f) tr.angle += angleDiff(tr.angle, dir.angle()) * std::min(1.0f, 10 * dt);
                vel.v *= 0.9f;
                if (en.timer > 0.45f) {
                    en.state = 2, en.timer = 0;
                    vel.v = Vec2::fromAngle(tr.angle, 620 * speedBoost);
                    au.play({.wave = Wave::Saw, .freq = 200, .slide = 500, .dur = 0.15f, .vol = 0.06f});
                }
            } else {
                vel.v *= std::max(0.0f, 1 - 1.2f * dt);
                if (en.timer > 0.6f) en.state = 0, en.timer = 0;
            }
            break;
        }
        case EnemyKind::Spinner: {
            // Orbit the player at a distance and lob slow shots.
            float orbitAngle = en.phase + time_ * 0.55f;
            Vec2 spot = clampToArena(target + Vec2::fromAngle(orbitAngle, 210), 30);
            steer((spot - tr.pos).limited(150), 3);
            tr.angle += 3.5f * dt;
            if (en.timer > 2.4f && hasTarget) {
                en.timer = 0;
                for (int i = -1; i <= 1; ++i)
                    spawnShot(tr.pos, Vec2::fromAngle(dir.angle() + float(i) * 0.25f, 170));
                fx_.burst(tr.pos, {.count = 8, .speed = 120, .life = 0.3f, .size = 3, .color = pal::eshot});
                au.play({.wave = Wave::Triangle, .freq = 520, .slide = -300, .dur = 0.15f, .vol = 0.1f});
            }
            break;
        }
        }
    });
}

void PlayScene::motionSystem(float dt) {
    world_.each<Transform, Velocity>([&](Entity, Transform &tr, Velocity &vel) {
        tr.pos += vel.v * dt;
        if (vel.drag > 0) vel.v *= std::max(0.0f, 1 - vel.drag * dt);
    });
}

void PlayScene::boundsSystem() {
    world_.each<Transform, Velocity, Body>([&](Entity e, Transform &tr, Velocity &vel, Body &b) {
        Vec2 c = clampToArena(tr.pos, b.radius);
        if (c == tr.pos) return;
        if (b.layer == L_BULLET || b.layer == L_ESHOT) {
            fx_.burst(c, {.count = 4, .speed = 120, .life = 0.2f, .size = 2,
                          .color = b.layer == L_BULLET ? pal::bullet : pal::eshot});
            world_.destroy(e);
            return;
        }
        if (c.x != tr.pos.x) vel.v.x = -vel.v.x * 0.6f; // bounce
        if (c.y != tr.pos.y) vel.v.y = -vel.v.y * 0.6f;
        tr.pos = c;
    });
}

void PlayScene::lifetimeSystem(float dt) {
    world_.each<Bullet>([&](Entity e, Bullet &b) {
        if ((b.life -= dt) <= 0) world_.destroy(e);
    });
    world_.each<EnemyShot>([&](Entity e, EnemyShot &s) {
        if ((s.life -= dt) <= 0) world_.destroy(e);
    });
    world_.each<Ring>([&](Entity e, Ring &r) {
        if ((r.life -= dt) <= 0) world_.destroy(e);
    });
    world_.each<Popup, Transform>([&](Entity e, Popup &p, Transform &tr) {
        tr.pos.y -= 40 * dt;
        if ((p.life -= dt) <= 0) world_.destroy(e);
    });
}

void PlayScene::warningSystem(float dt) {
    world_.each<Warning, Transform>([&](Entity e, Warning &w, Transform &tr) {
        if ((w.t -= dt) > 0) return;
        world_.destroy(e);
        spawnEnemy(w.kind, tr.pos);
        Color c = defOf(w.kind).color;
        fx_.burst(tr.pos, {.count = 14, .speed = 220, .life = 0.4f, .size = 3, .color = c});
        spawnRing(tr.pos, 4, 50, 0.3f, c, 2);
        engine.audio().play({.wave = Wave::Sine, .freq = 300, .slide = 400, .dur = 0.12f, .vol = 0.08f});
    });
}

void PlayScene::sparkSystem(float dt) {
    bool hasTarget = world_.alive(player_);
    world_.each<Spark, Transform, Velocity>([&](Entity e, Spark &s, Transform &tr, Velocity &vel) {
        tr.angle += 4 * dt;
        if ((s.life -= dt) <= 0) {
            world_.destroy(e);
            return;
        }
        if (!hasTarget) return;
        Vec2 to = lastPlayerPos_ - tr.pos;
        float d = to.length();
        if (d < 150 && d > 0.01f) vel.v += to / d * (2600 * (1 - d / 150) + 300) * dt; // magnet
    });
}

void PlayScene::collisionSystem() {
    colliders_.clear();
    world_.each<Transform, Body>([&](Entity e, Transform &tr, Body &b) {
        colliders_.push_back({e, tr.pos, b.radius, b.layer, b.mask});
    });

    colliderCount_ = uint32_t(colliders_.size());
    collPairs_ += uint32_t(hash_.findPairs(colliders_, [&](const Collider &a, const Collider &b) {
        // Either side may already have been destroyed by an earlier pair this frame.
        if (!world_.alive(a.entity) || !world_.alive(b.entity)) return;

        if (a.layer == L_BULLET && b.layer == L_ENEMY) {
            Vec2 dir = world_.get<Velocity>(a.entity)->v.normalized();
            int damage = world_.get<Bullet>(a.entity)->damage;
            world_.destroy(a.entity); // after this, get() on the bullet returns nullptr
            hitEnemy(b.entity, damage, dir);
        } else if (a.layer == L_ENEMY && b.layer == L_ENEMY) {
            // Soft separation so enemies don't stack into one blob.
            Transform *ta = world_.get<Transform>(a.entity), *tb = world_.get<Transform>(b.entity);
            Vec2 d = tb->pos - ta->pos;
            float len = d.length(), overlap = a.radius + b.radius - len;
            Vec2 n = len > 0.001f ? d / len : Vec2{1, 0};
            ta->pos -= n * (overlap * 0.25f);
            tb->pos += n * (overlap * 0.25f);
        } else if (a.layer == L_ENEMY && b.layer == L_PLAYER) {
            hurtPlayer();
        } else if (a.layer == L_ESHOT && b.layer == L_PLAYER) {
            Player *pl = world_.get<Player>(b.entity);
            if (pl && pl->invuln <= 0) {
                world_.destroy(a.entity);
                hurtPlayer();
            }
        } else if (a.layer == L_PLAYER && b.layer == L_SPARK) {
            collectSpark(b.entity);
        }
    }));
}

void PlayScene::waveSystem(float dt) {
    switch (phase_) {
    case Phase::Intermission:
        if ((phaseTimer_ -= dt) <= 0) {
            phase_ = Phase::Spawning;
            int w = stats_.wave;
            toSpawn_ = 4 + w * 2 + w * w / 6;
            if (stress().spawnMult != 1) toSpawn_ = std::max(1, int(float(toSpawn_) * stress().spawnMult + 0.5f));
            spawnCd_ = 0;
        }
        break;
    case Phase::Spawning:
        if ((spawnCd_ -= dt) <= 0) {
            EnemyKind k = pickEnemyKind();
            Vec2 p = randomSpawnPoint();
            spawnWarning(k, p);
            --toSpawn_;
            // Drifters sometimes arrive as a small pack.
            if (k == EnemyKind::Drifter && toSpawn_ > 2 && rng_.chance(0.35f)) {
                for (int i = 0; i < 2; ++i) spawnWarning(k, clampToArena(p + Vec2::fromAngle(rng_.angle(), 40), 20));
                toSpawn_ -= 2;
            }
            spawnCd_ = std::max(0.22f, 0.85f - float(stats_.wave) * 0.05f) * rng_.range(0.6f, 1.3f);
            if (stress().spawnMult != 1) spawnCd_ /= stress().spawnMult;
            if (toSpawn_ <= 0) phase_ = Phase::Fighting;
        }
        break;
    case Phase::Fighting:
        if (world_.count<Enemy>() == 0 && world_.count<Warning>() == 0 && deathTimer_ < 0) {
            long bonus = 100L * stats_.wave;
            stats_.score += bonus;
            spawnPopup({kWidth / 2.0f, kHeight / 2.0f + 40}, "+" + std::to_string(bonus) + " CLEAR BONUS", pal::spark);
            if (Player *pl = world_.get<Player>(player_); pl && pl->hp < 3) {
                ++pl->hp;
                spawnPopup(lastPlayerPos_ + Vec2{0, -24}, "+1 HP", pal::player);
            }
            startWave(stats_.wave + 1);
        }
        break;
    }
}

void PlayScene::startWave(int n) {
    stats_.wave = n;
    phase_ = Phase::Intermission;
    phaseTimer_ = n == 1 ? 1.2f : 1.8f;
    bannerT_ = phaseTimer_;
    banner_ = "WAVE " + std::to_string(n);
    engine.audio().melody({{523, 0.08f}, {659, 0.08f}, {784, 0.14f}}, Wave::Triangle, 0.12f);
}

// ---------------------------------------------------------------- events

void PlayScene::hitEnemy(Entity e, int damage, Vec2 knock) {
    Enemy *en = world_.get<Enemy>(e);
    if (!en) return;
    en->hp -= damage;
    en->flash = 0.08f;
    Transform &tr = *world_.get<Transform>(e);
    world_.get<Velocity>(e)->v += knock * (en->kind == EnemyKind::Splitter ? 40.0f : 120.0f);
    fx_.burst(tr.pos, {.count = 5, .speed = 180, .angle = knock.angle() + PI, .spread = 1.4f, .life = 0.25f,
                       .size = 3, .color = defOf(en->kind).color, .color2 = pal::bullet});
    if (en->hp <= 0) killEnemy(e);
    else engine.audio().play({.wave = Wave::Square, .freq = 300, .slide = -120, .dur = 0.05f, .vol = 0.06f});
}

void PlayScene::killEnemy(Entity e) {
    Enemy en = *world_.get<Enemy>(e);
    Vec2 pos = world_.get<Transform>(e)->pos;
    EnemyDef d = defOf(en.kind);
    world_.destroy(e);

    long gained = long(en.score) * multiplier();
    stats_.score += gained;
    ++stats_.kills;
    spawnPopup(pos, std::to_string(gained), d.color);

    bool big = en.kind == EnemyKind::Splitter || en.kind == EnemyKind::Spinner;
    fx_.burst(pos, {.count = big ? 46 : 24, .speed = big ? 380.0f : 300.0f, .life = 0.7f, .size = 4, .drag = 2.5f,
                    .color = d.color, .color2 = Color(255, 255, 255), .streak = true});
    spawnRing(pos, d.radius, d.radius * 4, 0.35f, d.color, 3);
    engine.shake(big ? 0.35f : 0.15f);
    if (big) engine.hitstop(0.04f);
    engine.audio().play({.wave = Wave::Noise, .freq = big ? 900.0f : 1800.0f, .slide = -600,
                         .dur = big ? 0.4f : 0.22f, .vol = big ? 0.3f : 0.18f});

    int sparks = d.sparks + (en.kind == EnemyKind::Mini && rng_.chance(0.4f) ? 1 : 0);
    for (int i = 0; i < sparks; ++i) spawnSpark(pos, Vec2::fromAngle(rng_.angle(), rng_.range(80.0f, 220.0f)));

    if (en.kind == EnemyKind::Splitter) {
        for (int i = 0; i < 3; ++i) {
            Entity m = spawnEnemy(EnemyKind::Mini, pos + Vec2::fromAngle(float(i) * TAU / 3, 14));
            world_.get<Velocity>(m)->v = Vec2::fromAngle(float(i) * TAU / 3 + 0.5f, 260);
        }
    }
}

void PlayScene::hurtPlayer() {
    Player *pl = world_.get<Player>(player_);
    if (!pl || pl->invuln > 0) return;
    Vec2 pos = world_.get<Transform>(player_)->pos;
    --pl->hp;
    pl->invuln = 1.6f;
    chain_ = 0;
    hurtFlash_ = 0.3f;
    engine.shake(0.7f);
    engine.hitstop(0.12f);
    Audio &au = engine.audio();
    au.play({.wave = Wave::Noise, .freq = 600, .slide = -500, .dur = 0.5f, .vol = 0.35f});
    au.play({.wave = Wave::Square, .freq = 220, .slide = -170, .dur = 0.35f, .vol = 0.15f});

    // Mercy shockwave: shove enemies away and clear enemy shots nearby.
    constexpr float kRadius = 170;
    spawnRing(pos, 10, kRadius, 0.45f, pal::danger, 4);
    fx_.burst(pos, {.count = 40, .speed = 420, .life = 0.6f, .size = 4, .color = pal::danger, .color2 = pal::player,
                    .streak = true});
    world_.each<Enemy, Transform, Velocity>([&](Entity, Enemy &, Transform &tr, Velocity &vel) {
        Vec2 d = tr.pos - pos;
        float len = d.length();
        if (len < kRadius && len > 0.01f) vel.v += d / len * 520 * (1 - len / kRadius);
    });
    world_.each<EnemyShot, Transform>([&](Entity s, EnemyShot &, Transform &tr) {
        if (distance(tr.pos, pos) < kRadius) world_.destroy(s);
    });

    if (pl->hp <= 0) {
        fx_.burst(pos, {.count = 120, .speed = 520, .life = 1.2f, .size = 5, .drag = 1.5f, .color = pal::player,
                        .color2 = Color(255, 255, 255), .streak = true});
        spawnRing(pos, 10, 400, 1.0f, pal::player, 5);
        world_.destroy(player_);
        deathTimer_ = 0;
        engine.shake(1.0f);
        au.melody({{392, 0.15f}, {330, 0.15f}, {262, 0.15f}, {196, 0.4f}}, Wave::Triangle, 0.18f);
    }
}

void PlayScene::collectSpark(Entity s) {
    Vec2 pos = world_.get<Transform>(s)->pos;
    world_.destroy(s);
    int before = multiplier();
    ++chain_;
    stats_.bestChain = std::max(stats_.bestChain, chain_);
    stats_.score += 5L * multiplier();
    fx_.burst(pos, {.count = 6, .speed = 120, .life = 0.25f, .size = 3, .color = pal::spark});
    float pitch = 660 + float(chain_ % 8) * 70;
    engine.audio().play({.wave = Wave::Sine, .freq = pitch, .slide = 200, .dur = 0.08f, .vol = 0.1f});
    if (multiplier() > before) {
        spawnPopup(pos + Vec2{0, -18}, "x" + std::to_string(multiplier()), pal::spark);
        spawnRing(lastPlayerPos_, 12, 60, 0.3f, pal::spark, 2);
        engine.audio().melody({{880, 0.06f}, {1175, 0.1f}}, Wave::Square, 0.07f);
    }
}

// ---------------------------------------------------------------- drawing

void PlayScene::drawEnemy(Canvas &cv, const Enemy &en, const Transform &tr, float radius) {
    Color c = defOf(en.kind).color;
    if (en.kind == EnemyKind::Dart && en.state == 1 && int(en.timer * 20) % 2 == 0) c = pal::danger; // wind-up blink
    Color line = en.flash > 0 ? Color(255, 255, 255) : c;
    Color fill = (en.flash > 0 ? Color(255, 255, 255) : c).fade(0.25f);
    cv.glow(tr.pos, radius * 2.6f, c.fade(0.35f));

    std::span<const Vec2> shape;
    switch (en.kind) {
    case EnemyKind::Drifter: shape = kDiamond; break;
    case EnemyKind::Dart: shape = kArrow; break;
    case EnemyKind::Splitter: shape = kHex; break;
    case EnemyKind::Mini: shape = kTri; break;
    case EnemyKind::Spinner: shape = kStar; break;
    }
    cv.shape(shape, tr.pos, tr.angle, radius, fill, line, 2);
    if (en.kind == EnemyKind::Splitter) cv.shape(kHex, tr.pos, -tr.angle * 1.5f, radius * 0.5f, Color(0u), line, 1.5f);
    if (en.kind == EnemyKind::Spinner) cv.circle(tr.pos, 3, line);
}

void PlayScene::draw(Canvas &cv) {
    Profiler &prof = engine.prof();
    std::optional<ProfScope> phase;
    phase.emplace(prof, Prof::RBackground);
    drawBackground(cv, time_, lastPlayerPos_);
    phase.emplace(prof, Prof::RWorld);

    // Arena border, brighter while hurt.
    Color border = Color::lerp(pal::border, pal::danger, hurtFlash_ / 0.3f);
    cv.rectLine(kMinX - 3, kMinY - 3, kMaxX - kMinX + 6, kMaxY - kMinY + 6, 2, border.fade(0.8f));
    cv.rectLine(kMinX - 7, kMinY - 7, kMaxX - kMinX + 14, kMaxY - kMinY + 14, 1, border.fade(0.25f));

    world_.each<Ring, Transform>([&](Entity, Ring &r, Transform &tr) {
        float t = 1 - r.life / r.total;
        cv.ring(tr.pos, lerp(r.r0, r.r1, ease::outCubic(t)), r.thick * (1 - t) + 0.5f, r.color.fade(1 - t));
    });

    world_.each<Warning, Transform>([&](Entity, Warning &w, Transform &tr) {
        float t = 1 - w.t / w.total; // 0 -> 1
        Color c = defOf(w.kind).color;
        bool blink = int(w.t * 16) % 2 == 0;
        cv.ring(tr.pos, lerp(46, defOf(w.kind).radius, t), 2, c.fade(blink ? 0.9f : 0.4f));
        cv.glow(tr.pos, 30 * t, c.fade(0.5f));
    });

    world_.each<Spark, Transform>([&](Entity, Spark &s, Transform &tr) {
        if (s.life < 2 && int(s.life * 10) % 2 == 0) return; // blink before vanishing
        cv.glow(tr.pos, 16, pal::spark.fade(0.5f));
        cv.shape(kDiamond, tr.pos, tr.angle, 4.5f, pal::spark, Color(255, 255, 255), 1);
    });

    world_.each<EnemyShot, Transform>([&](Entity, EnemyShot &, Transform &tr) {
        cv.glow(tr.pos, 18, pal::eshot.fade(0.6f));
        cv.circle(tr.pos, 5, pal::eshot);
        cv.circle(tr.pos, 2.5f, Color(255, 230, 230));
    });

    world_.each<Enemy, Transform, Body>([&](Entity, Enemy &en, Transform &tr, Body &b) { drawEnemy(cv, en, tr, b.radius); });

    world_.each<Bullet, Transform>([&](Entity, Bullet &, Transform &tr) {
        Vec2 tail = tr.pos - Vec2::fromAngle(tr.angle, 14);
        cv.setBlend(Blend::Add);
        cv.line(tail, tr.pos, 5, pal::player.fade(0.35f));
        cv.line(tail, tr.pos, 2.5f, pal::bullet);
        cv.setBlend(Blend::Alpha);
    });

    if (Player *pl = world_.get<Player>(player_)) {
        Transform &tr = *world_.get<Transform>(player_);
        bool visible = pl->invuln <= 0 || pl->dashT > 0 || int(pl->invuln * 14) % 2 == 0;
        cv.glow(tr.pos, 46, pal::player.fade(0.35f));
        if (visible) cv.shape(kShipShape, tr.pos, tr.angle, 11, pal::player.fade(0.35f), pal::player, 2);
        if (pl->dashCd > 0) { // dash recharge bar
            float t = 1 - pl->dashCd / 0.75f;
            cv.rect(tr.pos.x - 12, tr.pos.y + 20, 24, 3, pal::dim.fade(0.4f));
            cv.rect(tr.pos.x - 12, tr.pos.y + 20, 24 * t, 3, pal::player);
        }
    }

    phase.emplace(prof, Prof::RParticles);
    fx_.draw(cv);

    phase.emplace(prof, Prof::RHud);
    world_.each<Popup, Transform>([&](Entity, Popup &p, Transform &tr) {
        cv.text(tr.pos.x, tr.pos.y, p.text, 2, p.color.fade(std::min(1.0f, p.life * 3)), Align::Center);
    });

    drawHud(cv);
}

void PlayScene::drawHud(Canvas &cv) {
    cv.setOffset({}); // HUD doesn't shake

    char buf[64];
    cv.text(34, 34, "SCORE", 1, pal::dim);
    std::snprintf(buf, sizeof buf, "%ld", stats_.score);
    cv.text(34, 46, buf, 3, pal::text);

    // Multiplier with progress toward the next level.
    int m = multiplier();
    std::snprintf(buf, sizeof buf, "x%d", m);
    cv.text(34, 76, buf, 2, m > 1 ? pal::spark : pal::dim);
    float prog = m >= 8 ? 1.0f : float(chain_ % 8) / 8.0f;
    cv.rect(64, 80, 70, 5, pal::grid);
    cv.rect(64, 80, 70 * prog, 5, pal::spark.fade(0.9f));

    std::snprintf(buf, sizeof buf, "WAVE %d", stats_.wave);
    cv.text(kWidth - 34, 34, buf, 2, pal::text, Align::Right);
    int hp = 0;
    if (Player *pl = world_.get<Player>(player_)) hp = pl->hp;
    for (int i = 0; i < std::max(3, hp); ++i) {
        Vec2 at{kWidth - 44.0f - float(i) * 26, 66};
        bool full = i < hp;
        cv.shape(kShipShape, at, -PI / 2, 8, full ? pal::player.fade(0.5f) : Color(0u),
                 full ? pal::player : pal::dim.fade(0.5f), 1.5f);
    }

    if (bannerT_ > 0) {
        float a = std::min(1.0f, bannerT_ * 2);
        float rise = (1 - ease::outBack(std::min(1.0f, (1.8f - bannerT_) * 3))) * 20;
        drawTitleText(cv, kWidth / 2.0f, kHeight / 2.0f - 40 + rise, banner_.c_str(), 5, pal::text.fade(a));
    }

    // Crosshair (the OS cursor is hidden).
    if (world_.alive(player_)) {
        Vec2 mp = engine.input().mouse();
        cv.ring(mp, 7, 1.5f, pal::player.fade(0.8f));
        cv.line(mp + Vec2{-12, 0}, mp + Vec2{-5, 0}, 1.5f, pal::player.fade(0.8f));
        cv.line(mp + Vec2{12, 0}, mp + Vec2{5, 0}, 1.5f, pal::player.fade(0.8f));
        cv.line(mp + Vec2{0, -12}, mp + Vec2{0, -5}, 1.5f, pal::player.fade(0.8f));
        cv.line(mp + Vec2{0, 12}, mp + Vec2{0, 5}, 1.5f, pal::player.fade(0.8f));
    }

    if (hurtFlash_ > 0) cv.overlay(pal::danger.fade(hurtFlash_ * 0.6f));
}

// ---------------------------------------------------------------- frame-time research

void PlayScene::planWork(FrameRecord &r) const {
    auto u16 = [](size_t v) { return uint16_t(std::min<size_t>(v, 65535)); };
    World &w = const_cast<World &>(world_); // each() is non-const; nothing is modified here
    r.scene = 2;
    r.entities = u16(w.size());
    uint16_t kinds[5] = {};
    w.each<Enemy>([&](Entity, Enemy &en) { ++kinds[int(en.kind)]; });
    r.enemies = u16(size_t(kinds[0]) + kinds[1] + kinds[2] + kinds[3] + kinds[4]);
    r.enemy_k0 = kinds[0];
    r.enemy_k1 = kinds[1];
    r.enemy_k2 = kinds[2];
    r.enemy_k3 = kinds[3];
    r.enemy_k4 = kinds[4];
    r.projectiles = u16(w.count<Bullet>());
    r.enemy_projectiles = u16(w.count<EnemyShot>());
    r.pickups = u16(w.count<Spark>());
    r.telegraphs = u16(w.count<Warning>());
    r.effects = u16(w.count<Ring>());
    r.popups = u16(w.count<Popup>());

    // Known-ahead work. Updates only run on steps not frozen by hit-stop, and only
    // when this scene is on top (a pause/game-over overlay freezes it).
    const float dt = float(engine.config().step);
    const bool live = engine.top() == this;
    const int active = live ? r.n_steps - r.n_frozen : 0;
    const float horizon = float(active) * dt;
    size_t dueNow = 0, due6 = 0, due30 = 0;
    w.each<Warning>([&](Entity, Warning &wr) {
        dueNow += active > 0 && wr.t <= horizon;
        due6 += wr.t <= 6 * dt;
        due30 += wr.t <= 30 * dt;
    });
    size_t firing = 0;
    if (active > 0 && world_.alive(player_))
        w.each<Enemy>([&](Entity, Enemy &en) { firing += en.kind == EnemyKind::Spinner && en.timer + horizon > 2.4f; });
    r.tele_due_now = u16(dueNow);
    r.tele_due_6 = u16(due6);
    r.tele_due_30 = u16(due30);
    // Spawn bursts are 14 particles, spinner volleys 8, before quality density.
    r.burst_due = u16(size_t(float(dueNow * 14 + firing * 8) * engine.config().burstMult + 0.5f));
    r.particles_alive = uint32_t(fx_.size());
    r.particles_streak = u16(fx_.streaks());
    r.particles_expiring = u16(fx_.expiringNext());
    r.wave = u16(size_t(std::max(0, stats_.wave)));
    r.wave_to_spawn = u16(size_t(phase_ == Phase::Fighting ? 0 : std::max(0, toSpawn_)));
}

void PlayScene::reportWork(FrameRecord &r) {
    Particles::Counters c = fx_.takeCounters();
    r.particles_spawned = uint16_t(std::min<uint32_t>(c.spawned, 65535));
    r.bursts = uint16_t(std::min<uint32_t>(c.bursts, 65535));
    r.colliders = uint16_t(std::min<uint32_t>(colliderCount_, 65535));
    r.coll_pairs = uint16_t(std::min<uint32_t>(collPairs_, 65535));
    collPairs_ = 0;
}

std::unique_ptr<Scene> makePlay(Engine &e, uint64_t seed) { return std::make_unique<PlayScene>(e, seed); }

} // namespace warden
