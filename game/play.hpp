// The gameplay scene: an ECS world plus the systems that drive it.
#pragma once

#include <vector>

#include "components.hpp"
#include "game.hpp"

namespace warden {

class PlayScene : public Scene {
public:
    PlayScene(Engine &e, uint64_t seed);

    void enter() override;
    void update(float dt) override;
    void draw(Canvas &cv) override;
    void planWork(FrameRecord &r) const override;
    void reportWork(FrameRecord &r) override;

    // Read access for the bot and tests.
    World &world() { return world_; }
    Entity player() const { return player_; }
    bool playerDead() const { return deathTimer_ >= 0; }
    const RunStats &stats() const { return stats_; }
    int multiplier() const;
    static constexpr float kMinX = kArenaPad, kMinY = kArenaPad;
    static constexpr float kMaxX = kWidth - kArenaPad, kMaxY = kHeight - kArenaPad;

private:
    // Systems, run in this order each update.
    void playerSystem(float dt);
    void enemySystem(float dt);
    void motionSystem(float dt);
    void boundsSystem();
    void lifetimeSystem(float dt);
    void warningSystem(float dt);
    void sparkSystem(float dt);
    void collisionSystem();
    void waveSystem(float dt);

    // Spawning.
    void spawnPlayer();
    Entity spawnEnemy(EnemyKind kind, Vec2 pos);
    void spawnWarning(EnemyKind kind, Vec2 pos);
    void spawnBullet(Vec2 pos, float angle);
    void spawnShot(Vec2 pos, Vec2 vel);
    void spawnSpark(Vec2 pos, Vec2 vel);
    void spawnRing(Vec2 pos, float r0, float r1, float life, Color c, float thick = 3);
    void spawnPopup(Vec2 pos, std::string text, Color c);
    Vec2 randomSpawnPoint();
    EnemyKind pickEnemyKind();

    // Events.
    void hitEnemy(Entity e, int damage, Vec2 knock);
    void killEnemy(Entity e);
    void hurtPlayer();
    void collectSpark(Entity s);
    void startWave(int n);

    // Drawing.
    void drawEnemy(Canvas &cv, const Enemy &en, const Transform &tr, float radius);
    void drawHud(Canvas &cv);

    World world_;
    Particles fx_;
    Rng rng_;
    SpatialHash hash_{48};
    std::vector<Collider> colliders_;
    Entity player_;
    RunStats stats_;
    int chain_ = 0;

    enum class Phase { Intermission, Spawning, Fighting } phase_ = Phase::Intermission;
    float phaseTimer_ = 0;
    int toSpawn_ = 0;
    float spawnCd_ = 0;
    float bannerT_ = 0;
    std::string banner_;

    float time_ = 0;
    float deathTimer_ = -1;
    uint32_t collPairs_ = 0, colliderCount_ = 0; // research instrumentation
    float hurtFlash_ = 0;
    Vec2 lastPlayerPos_{kWidth / 2.0f, kHeight / 2.0f};
};

} // namespace warden
