// Menu scenes: title, pause overlay, game over overlay.
#pragma once

#include "game.hpp"

namespace warden {

class TitleScene : public Scene {
public:
    using Scene::Scene;
    void update(float dt) override;
    void draw(Canvas &cv) override;
    void planWork(FrameRecord &r) const override { r.scene = 1; }

private:
    float t_ = 0;
    long best_ = loadBest();
};

/** Pushed over PlayScene; the game stays visible (frozen) underneath. */
class PauseScene : public Scene {
public:
    using Scene::Scene;
    void update(float dt) override;
    void draw(Canvas &cv) override;
    void planWork(FrameRecord &r) const override { r.scene = 3; }

private:
    float t_ = 0;
};

/** Pushed over PlayScene after the player dies. */
class GameOverScene : public Scene {
public:
    GameOverScene(Engine &e, RunStats stats);
    void update(float dt) override;
    void draw(Canvas &cv) override;
    void planWork(FrameRecord &r) const override { r.scene = 3; }

private:
    RunStats stats_;
    long best_;
    bool newBest_;
    float t_ = 0;
};

/** Seed for the next run: fixed sequence in headless mode, time-based otherwise. */
uint64_t nextRunSeed(Engine &e);
void setBaseSeed(uint64_t s);

} // namespace warden
