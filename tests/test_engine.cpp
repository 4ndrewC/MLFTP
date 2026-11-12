// Engine unit tests + a deterministic game simulation test.
// Tiny self-contained harness: no test framework dependency.
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "harness.hpp"
#include "../game/play.hpp"
#include "../game/scenes.hpp"
#include "spark.hpp"

using namespace spark;

namespace {

struct Pos { float x = 0, y = 0; };
struct Vel { float x = 0, y = 0; };
struct Tag {};

} // namespace

// ---------------------------------------------------------------- ECS

TEST(ecs_add_get_query) {
    World w;
    Entity a = w.create(), b = w.create(), c = w.create();
    w.add<Pos>(a, {1, 2});
    w.add<Vel>(a, {3, 4});
    w.add<Pos>(b, {5, 6});
    w.add<Vel>(c, {});
    CHECK(w.get<Pos>(a)->x == 1);
    CHECK(w.get<Vel>(b) == nullptr);
    int n = 0;
    w.each<Pos, Vel>([&](Entity e, Pos &p, Vel &v) {
        ++n;
        CHECK(e == a);
        p.x += v.x;
    });
    CHECK(n == 1);
    CHECK(w.get<Pos>(a)->x == 4);
    w.remove<Vel>(a);
    n = 0;
    w.each<Pos, Vel>([&](Entity, Pos &, Vel &) { ++n; });
    CHECK(n == 0);
    CHECK(w.count<Pos>() == 2);
}

TEST(ecs_deferred_destroy_and_generations) {
    World w;
    Entity a = w.create();
    w.add<Pos>(a);
    w.each<Pos>([&](Entity e, Pos &) { w.destroy(e); });
    CHECK(!w.alive(a));         // dying entities report dead immediately...
    CHECK(w.size() == 1);       // ...but are only removed at flush
    CHECK(w.get<Pos>(a) == nullptr);
    w.flush();
    CHECK(w.size() == 0);
    CHECK(w.count<Pos>() == 0);

    Entity b = w.create(); // reuses a's slot
    CHECK(b.index == a.index);
    CHECK(b.gen != a.gen);
    CHECK(!w.alive(a)); // stale handle stays dead
    CHECK(w.alive(b));
}

TEST(ecs_dying_entities_skipped_in_each) {
    World w;
    Entity a = w.create(), b = w.create();
    w.add<Pos>(a);
    w.add<Pos>(b);
    w.destroy(a);
    int n = 0;
    w.each<Pos>([&](Entity e, Pos &) { ++n; CHECK(e == b); });
    CHECK(n == 1);
}

TEST(ecs_references_stable_while_spawning_in_each) {
    World w;
    for (int i = 0; i < 10; ++i) w.add<Pos>(w.create(), {float(i), 0});
    int visited = 0;
    w.each<Pos>([&](Entity, Pos &p) {
        ++visited;
        float before = p.x;
        for (int k = 0; k < 200; ++k) w.add<Pos>(w.create(), {-1, -1}); // forces growth
        CHECK(p.x == before); // reference still valid
    });
    CHECK(visited == 10); // entities created during the loop aren't visited
    CHECK(w.count<Pos>() == 10 + 10 * 200);
}

TEST(ecs_remove_during_each_visits_everyone) {
    World w;
    for (int i = 0; i < 20; ++i) w.add<Pos>(w.create(), {float(i), 0});
    int visited = 0;
    w.each<Pos>([&](Entity e, Pos &) {
        ++visited;
        w.remove<Pos>(e); // swap-remove inside iteration
    });
    CHECK(visited == 20);
    CHECK(w.count<Pos>() == 0);
}

TEST(ecs_first_and_clear) {
    World w;
    CHECK(!w.first<Tag>());
    Entity a = w.create();
    w.add<Tag>(a);
    CHECK(w.first<Tag>() == a);
    w.clear();
    CHECK(w.size() == 0);
    CHECK(!w.first<Tag>());
}

// ---------------------------------------------------------------- collision

TEST(collision_masks_and_pairs) {
    World w;
    Entity p = w.create(), e = w.create(), b = w.create(), far = w.create();
    std::vector<Collider> items = {
        {p, {100, 100}, 10, 1, 2},  // player hits enemies
        {e, {115, 100}, 10, 2, 0},  // enemy hits nothing
        {b, {120, 100}, 3, 4, 2},   // bullet hits enemies (overlaps player too, but masks don't match)
        {far, {500, 500}, 10, 2, 0},
    };
    SpatialHash h(32);
    std::vector<std::pair<Entity, Entity>> pairs;
    size_t n = h.findPairs(items, [&](const Collider &a, const Collider &c) { pairs.emplace_back(a.entity, c.entity); });
    CHECK(n == 2);
    CHECK(pairs.size() == 2);
    for (auto &[hitter, target] : pairs) {
        CHECK(target == e);
        CHECK(hitter == p || hitter == b);
    }
}

TEST(collision_big_and_mutual_reported_once) {
    World w;
    Entity a = w.create(), b = w.create();
    // Huge colliders spanning many cells, both wanting to hit each other.
    std::vector<Collider> items = {{a, {0, 0}, 100, 1, 1}, {b, {50, 50}, 100, 1, 1}};
    SpatialHash h(16);
    int n = 0;
    h.findPairs(items, [&](const Collider &, const Collider &) { ++n; });
    CHECK(n == 1);
}

TEST(collision_touching_is_not_overlap) {
    CHECK(!overlaps({0, 0}, 5, {10, 0}, 5));
    CHECK(overlaps({0, 0}, 5, {9.9f, 0}, 5));
}

TEST(collision_negative_coordinates) {
    World w;
    Entity a = w.create(), b = w.create();
    std::vector<Collider> items = {{a, {-100, -100}, 5, 1, 2}, {b, {-104, -100}, 5, 2, 0}};
    SpatialHash h(64);
    CHECK(h.findPairs(items, [](const Collider &, const Collider &) {}) == 1);
}

// ---------------------------------------------------------------- input

TEST(input_actions_and_edges) {
    enum { Fire, Left, Right };
    Input in;
    in.bind(Fire, {Key::Space, Key::MouseLeft});
    in.bind(Left, {Key::A});
    in.bind(Right, {Key::D});
    in.setKey(Key::MouseLeft, true);
    CHECK(in.pressed(Fire));
    CHECK(in.down(Fire));
    in.endStep();
    CHECK(!in.pressed(Fire));
    CHECK(in.down(Fire));
    in.setKey(Key::MouseLeft, false);
    CHECK(in.released(Fire));
    in.setKey(Key::D, true);
    CHECK(in.axis(Left, Right) == 1);
    in.setKey(Key::A, true);
    CHECK(in.axis(Left, Right) == 0);
    in.releaseAll();
    CHECK(!in.down(Left));
    CHECK(in.released(Left));
}

// ---------------------------------------------------------------- canvas

TEST(canvas_circle_coverage_and_clipping) {
    Canvas cv(64, 64);
    cv.clear(Color(0, 0, 0));
    cv.circle({32, 32}, 10, Color(255, 0, 0));
    CHECK(cv.pixel(32, 32) == Color(255, 0, 0).argb);   // center: full
    CHECK(cv.pixel(32, 50) == Color(0, 0, 0).argb);     // outside: untouched
    uint32_t edge = cv.pixel(38, 39);                    // on the edge (dist ~9.9): partial
    int red = int((edge >> 16) & 255);
    CHECK(red > 0 && red < 255);
    // Off-screen and huge shapes must not crash or write out of bounds.
    cv.circle({-500, -500}, 20, Color(255, 255, 255));
    cv.circle({32, 32}, 5000, Color(0, 255, 0));
    cv.line({-1000, -1000}, {1000, 1000}, 3, Color(0, 0, 255));
    cv.rect(60, 60, 100, 100, Color(1, 2, 3));
    cv.rect(10, 10, -5, -5, Color(1, 2, 3)); // negative size: draws nothing
    CHECK(cv.pixel(0, 0) != 0);
}

TEST(canvas_additive_blend_saturates) {
    Canvas cv(4, 4);
    cv.clear(Color(200, 100, 0));
    cv.setBlend(Blend::Add);
    cv.rect(0, 0, 4, 4, Color(100, 100, 100));
    Color c(cv.pixel(1, 1));
    CHECK(c.r() == 255);
    CHECK(c.g() == 200);
    CHECK(c.b() == 100);
}

TEST(canvas_polygon_fill_and_offset) {
    Canvas cv(40, 40);
    cv.clear(Color(0, 0, 0));
    const Vec2 sq[] = {{10, 10}, {20, 10}, {20, 20}, {10, 20}};
    cv.setOffset({5, 0});
    cv.polygon(sq, Color(0, 255, 0));
    CHECK(Color(cv.pixel(16, 15)).g() == 255); // shifted right by 5
    CHECK(Color(cv.pixel(12, 15)).g() == 0);
}

TEST(canvas_text) {
    CHECK(Canvas::textWidth("", 2) == 0);
    CHECK(Canvas::textWidth("AB", 1) == 11);
    CHECK(Canvas::textWidth("AB", 3) == 33);
    Canvas cv(64, 16);
    cv.clear(Color(0, 0, 0));
    cv.text(0, 0, "I", 1, Color(255, 255, 255));
    CHECK(Color(cv.pixel(2, 3)).r() == 255); // the I's stem
    CHECK(Color(cv.pixel(0, 3)).r() == 0);
}

// ---------------------------------------------------------------- particles / audio / math

TEST(particles_lifecycle_and_cap) {
    Particles ps(100);
    ps.burst({0, 0}, {.count = 50, .life = 0.2f});
    CHECK(ps.size() == 50);
    for (int i = 0; i < 20; ++i) ps.update(1 / 60.0f);
    CHECK(ps.size() == 0);
    ps.burst({0, 0}, {.count = 500});
    CHECK(ps.size() == 100);
}

TEST(audio_mix_silence_without_voices) {
    Audio a; // not started: play() is ignored, mix() yields silence
    a.play({});
    int16_t buf[128];
    a.mix(buf, 128);
    bool silent = true;
    for (int16_t s : buf) silent &= (s == 0);
    CHECK(silent);
}

TEST(math_basics) {
    Rng r1(42), r2(42);
    CHECK(r1.next() == r2.next());
    for (int i = 0; i < 1000; ++i) {
        int v = r1.range(1, 3);
        CHECK(v >= 1 && v <= 3);
        float f = r1.uniform();
        CHECK(f >= 0 && f < 1);
    }
    CHECK(std::fabs(angleDiff(0.1f, TAU - 0.1f) + 0.2f) < 1e-4f);
    CHECK(Vec2(0, 0).normalized() == Vec2(0, 0));
    CHECK(std::fabs(Vec2(3, 4).normalized().length() - 1) < 1e-6f);
    CHECK(Vec2(30, 40).limited(5).length() < 5.001f);
    CHECK(Color::lerp(Color(0, 0, 0), Color(255, 255, 255), 0.5f).r() == 128);
}

// ---------------------------------------------------------------- engine

namespace {
struct CountingScene : Scene {
    int *updates;
    std::vector<std::string> *log;
    const char *name;
    CountingScene(Engine &e, int *u, std::vector<std::string> *l, const char *n) : Scene(e), updates(u), log(l), name(n) {}
    void enter() override { log->push_back(std::string("enter ") + name); }
    void exit() override { log->push_back(std::string("exit ") + name); }
    void update(float) override {
        ++*updates;
        if (*updates == 3) engine.push<CountingScene>(updates, log, "overlay"); // deferred
        if (*updates == 5) engine.pop();
        if (*updates == 7) engine.hitstop(0.05f);
    }
};
} // namespace

TEST(engine_scene_stack_hitstop_and_max_frames) {
    Config cfg;
    cfg.headless = true;
    cfg.maxFrames = 20;
    cfg.width = 32;
    cfg.height = 32;
    // Declared before the engine: ~Engine calls exit() on remaining scenes, which logs.
    int updates = 0;
    std::vector<std::string> log;
    Engine e(cfg);
    CHECK(e.run(std::make_unique<CountingScene>(e, &updates, &log, "base")) == 0);
    CHECK(e.frame() == 20);
    // 3 hitstop frames (0.05s at 60Hz) skip their updates.
    CHECK(updates == 17);
    CHECK(log.size() >= 3);
    CHECK(log[0] == "enter base");
    CHECK(log[1] == "enter overlay");
    CHECK(log[2] == "exit overlay");
}

// ---------------------------------------------------------------- game

namespace {
warden::RunStats simulate(uint64_t seed, long frames) {
    Config cfg;
    cfg.headless = true;
    cfg.maxFrames = frames;
    cfg.width = warden::kWidth;
    cfg.height = warden::kHeight;
    warden::BotReport report;
    Engine *ep = nullptr;
    cfg.onStep = [&](long f) { warden::botStep(*ep, f, report); };
    Engine e(cfg);
    ep = &e;
    e.input().bind(warden::Fire, {Key::MouseLeft});
    e.input().bind(warden::Dash, {Key::Space});
    e.input().bind(warden::MoveUp, {Key::W});
    e.input().bind(warden::MoveDown, {Key::S});
    e.input().bind(warden::MoveLeft, {Key::A});
    e.input().bind(warden::MoveRight, {Key::D});
    e.input().bind(warden::Confirm, {Key::Enter});
    warden::setBaseSeed(seed);
    e.run(warden::makePlay(e, seed));
    auto *play = dynamic_cast<warden::PlayScene *>(e.top());
    return play ? play->stats() : warden::RunStats{};
}
} // namespace

TEST(game_simulation_is_deterministic) {
    auto a = simulate(11, 1800);
    auto b = simulate(11, 1800);
    CHECK(a.score > 0);
    CHECK(a.kills > 0);
    CHECK(a.wave >= 2);
    CHECK(a.score == b.score);
    CHECK(a.kills == b.kills);
    CHECK(a.wave == b.wave);
}

int main() {
    using namespace testing;
    for (auto &[name, fn] : registry()) {
        int before = gFailures;
        fn();
        std::printf("%s %s\n", gFailures == before ? "ok  " : "FAIL", name);
    }
    std::printf("\n%d checks, %d failures\n", gChecks, gFailures);
    return gFailures ? 1 : 0;
}
