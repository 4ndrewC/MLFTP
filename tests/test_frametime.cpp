// Tests for the frame-time research code: frame logger, .spkm loader,
// quality controllers, render scale / glow quality, feature bookkeeping.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

#include "harness.hpp"
#include "spark.hpp"

using namespace spark;
using namespace spark::ft;

namespace {

std::string tempPath(const char *name) {
    const char *dir = std::getenv("TMPDIR");
    return std::string(dir ? dir : "/tmp") + "/spark_test_" + std::to_string(getpid()) + "_" + name;
}

// ---------------------------------------------------------------- .spkm writer (tests only)

void put32(std::vector<unsigned char> &b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(uint8_t(v >> (8 * i)));
}
void putF(std::vector<unsigned char> &b, float f) {
    uint32_t v;
    std::memcpy(&v, &f, 4);
    put32(b, v);
}
void chunk(std::vector<unsigned char> &out, const char *tag, const std::vector<unsigned char> &payload) {
    out.insert(out.end(), tag, tag + 4);
    put32(out, uint32_t(payload.size()));
    out.insert(out.end(), payload.begin(), payload.end());
    while (out.size() % 4) out.push_back(0);
}
std::vector<unsigned char> text(const std::string &s) { return {s.begin(), s.end()}; }

struct SpkmOpts {
    uint32_t version = kSpkmVersion;
    std::string outs = "t_p95_ms identity -\n";
    std::string hash; // empty = the engine's own hash
    std::string featureSet = kFeatureSet;
    bool renameFeature = false;
};

/** A linear model: t_p95_ms = 2 * n_steps + 1.5 * naive_ms + 0.5 (identity normalisation). */
std::vector<unsigned char> linearSpkm(const SpkmOpts &o = {}) {
    char hash[32];
    std::snprintf(hash, sizeof hash, "%016llx", static_cast<unsigned long long>(featureHash()));
    std::string info = "family=linear\nfeature_set=" + o.featureSet + "\nfeature_hash=" + (o.hash.empty() ? hash : o.hash) +
                       "\nn_features=" + std::to_string(kNumFeatures) + "\nhistory_len=0\nhistory_dim=" +
                       std::to_string(kHistoryDim) + "\nbudget_ms=16.667\n";
    std::string feat;
    for (int i = 0; i < kNumFeatures; ++i) feat += std::string(o.renameFeature && i == 3 ? "bogus" : kFeatureNames[i]) + "\n";
    for (const char *n : kHistoryExtraNames) feat += std::string(n) + "\n";
    std::vector<unsigned char> lin;
    put32(lin, 1);
    put32(lin, uint32_t(kNumFeatures));
    for (int i = 0; i < kNumFeatures; ++i) putF(lin, i == 0 ? 2.0f : i == 37 ? 1.5f : 0.0f);
    putF(lin, 0.5f);
    std::vector<unsigned char> body;
    chunk(body, "INFO", text(info));
    chunk(body, "FEAT", text(feat));
    chunk(body, "OUTS", text(o.outs));
    chunk(body, "LINR", lin);
    std::vector<unsigned char> file = {'S', 'P', 'K', 'M'};
    put32(file, o.version);
    put32(file, 4);
    put32(file, crc32(body.data(), body.size()));
    file.insert(file.end(), body.begin(), body.end());
    return file;
}

} // namespace

// ---------------------------------------------------------------- frame log

TEST(framelog_round_trip) {
    std::string path = tempPath("log.sfl"), err;
    {
        TableWriter w;
        CHECK(w.open(path, "SPKFLOG", kFrameLogVersion, sizeof(FrameRecord), frameLogHeaderJson("{\"m\":1}", "{\"s\":2}"),
                     err));
        for (uint32_t i = 0; i < 5000; ++i) { // crosses the 4096-row buffer flush
            FrameRecord r{};
            r.frame = i;
            r.t_total_us = float(i) * 0.5f;
            r.n_steps = uint8_t(i % 3);
            r.enemy_k4 = uint16_t(i * 7);
            r.px_glow_req = i * 1000u;
            w.write(&r);
        }
        CHECK(w.rows() == 5000);
    }
    TableData d;
    CHECK(readTable(path, d, err));
    CHECK(d.magic == "SPKFLOG");
    CHECK(d.version == kFrameLogVersion);
    CHECK(d.recordSize == sizeof(FrameRecord));
    CHECK(d.rowCount == 5000);
    CHECK(d.json.find("\"machine\":{\"m\":1}") != std::string::npos);
    CHECK(d.json.find("\"session\":{\"s\":2}") != std::string::npos);
    for (const FieldDesc &f : frameFields()) CHECK(d.json.find(std::string("\"name\":\"") + f.name + "\"") != std::string::npos);
    bool same = true;
    for (uint32_t i = 0; i < 5000; i += 37) {
        FrameRecord r;
        std::memcpy(&r, &d.rows[size_t(i) * sizeof r], sizeof r);
        same = same && r.frame == i && r.t_total_us == float(i) * 0.5f && r.n_steps == i % 3 &&
               r.enemy_k4 == uint16_t(i * 7) && r.px_glow_req == i * 1000u;
    }
    CHECK(same);
    // A crash mid-write leaves a partial row: readers ignore it.
    FILE *f = std::fopen(path.c_str(), "ab");
    std::fwrite("partial", 1, 7, f);
    std::fclose(f);
    CHECK(readTable(path, d, err) && d.rowCount == 5000);
    std::remove(path.c_str());
}

TEST(framelog_field_table_matches_struct) {
    size_t covered = 0, prevEnd = 0;
    bool ordered = true;
    for (const FieldDesc &f : frameFields()) {
        ordered = ordered && f.offset == prevEnd; // packed, no gaps
        prevEnd = f.offset + f.size;
        covered += f.size;
        CHECK(std::strlen(f.desc) > 0);
        bool knownWhen = false;
        for (const char *w : {"id", "before", "quality", "system", "after", "pred"}) knownWhen |= !std::strcmp(f.when, w);
        CHECK(knownWhen);
    }
    CHECK(ordered);
    CHECK(covered == sizeof(FrameRecord));
}

namespace {
struct BusyScene : Scene {
    int planned = 0, reported = 0;
    using Scene::Scene;
    void update(float) override {
        if (engine.frame() == 20) engine.hitstop(0.05f);
    }
    void draw(Canvas &cv) override {
        cv.clear(Color(10, 10, 10));
        cv.glow({16, 16}, 10, Color(255, 0, 0));
        cv.rect(2, 2, 4, 4, Color(255, 255, 255));
    }
    void planWork(FrameRecord &r) const override { r.scene = 2; }
    void reportWork(FrameRecord &r) override {
        ++reported;
        r.bursts = 7;
    }
};
} // namespace

TEST(engine_logs_every_frame_in_emulate_mode) {
    std::string path = tempPath("engine.sfl"), err;
    Config cfg;
    cfg.headless = true;
    cfg.maxFrames = 300;
    cfg.width = 32;
    cfg.height = 32;
    cfg.loop = LoopMode::Emulate;
    cfg.logPath = path;
    cfg.fpFile = "none";
    cfg.quality = {1, 0, 2, 1};
    long steps = 0;
    {
        Engine e(cfg);
        CHECK(e.run(std::make_unique<BusyScene>(e)) == 0);
        steps = e.frame();
    }
    TableData d;
    CHECK(readTable(path, d, err));
    CHECK(d.json.find("\"loop\":\"emulate\"") != std::string::npos);
    CHECK(d.version == 2 && d.json.find("\"fingerprint\":null") != std::string::npos);
    long stepSum = 0, frozen = 0;
    bool ok = true;
    for (size_t i = 0; i < d.rowCount; ++i) {
        FrameRecord r;
        std::memcpy(&r, &d.rows[i * sizeof r], sizeof r);
        stepSum += r.n_steps;
        frozen += r.n_frozen;
        ok = ok && r.frame == i && r.scene == 2 && r.bursts == 7 && r.t_total_us > 0 && r.q_scale == 1 && r.q_glow == 2 &&
             r.q_aa == 1 && (r.flags & 16) == 0 &&
             r.render_scale == Quality::kScale[1] && r.px_clear == uint32_t(Canvas::scaledSize(32, 0.8f) * Canvas::scaledSize(32, 0.8f)) &&
             r.calls_glow == 1 && r.px_glow_req > r.px_glow;
    }
    CHECK(ok);
    CHECK(d.rowCount > 100);
    CHECK(stepSum == steps); // every step is accounted to exactly one frame
    CHECK(frozen == 3);      // 0.05 s of hit-stop at 60 Hz
    std::remove(path.c_str());
}

// ---------------------------------------------------------------- .spkm loader

TEST(predictor_loads_good_model_and_predicts) {
    std::string err;
    auto m = Model::loadBytes(linearSpkm(), err);
    CHECK(m != nullptr);
    if (!m) return;
    CHECK(m->family() == "linear");
    CHECK(m->numOutputs() == 1 && m->outputIndex("t_p95_ms") == 0);
    CHECK(std::fabs(m->budgetMs() - 16.667) < 1e-9);
    float x[kNumFeatures] = {}, out[1];
    x[0] = 2;   // n_steps
    x[37] = 4;  // naive_ms
    HistoryRing h;
    m->beginFrame(h);
    m->predict(x, out);
    CHECK(std::fabs(out[0] - (2 * 2 + 1.5f * 4 + 0.5f)) < 1e-5f);
}

TEST(predictor_mulexp_and_add_outputs) {
    // LIT_REVIEW_HW options A/B: y = t_ref_ms * exp(raw) + fp_noise_p99_ms
    SpkmOpts o;
    o.outs = "t_p95_ms mulexp t_ref_ms fp_noise_p99_ms\n";
    std::string err;
    auto m = Model::loadBytes(linearSpkm(o), err);
    CHECK(m != nullptr);
    if (!m) return;
    float x[kNumFeatures] = {}, out[1];
    x[0] = 0.25f;  // n_steps (raw = 2 * 0.25 + 0.5 = 1)
    x[51] = 3.0f;  // t_ref_ms
    x[52] = 0.5f;  // fp_noise_p99_ms
    HistoryRing h;
    m->beginFrame(h);
    m->predict(x, out);
    CHECK(std::fabs(out[0] - (3.0f * std::exp(1.0f) + 0.5f)) < 1e-4f);
    o.outs = "t_p95_ms mulexp - -\n"; // mulexp needs a base feature
    CHECK(Model::loadBytes(linearSpkm(o), err) == nullptr);
    o.outs = "t_p95_ms identity - no_such_feature\n";
    CHECK(Model::loadBytes(linearSpkm(o), err) == nullptr);
}

TEST(predictor_rejects_bad_files) {
    std::string err;
    SpkmOpts o;
    o.version = kSpkmVersion + 1;
    CHECK(Model::loadBytes(linearSpkm(o), err) == nullptr && err.find("version") != std::string::npos);
    o.version = 1; // phase-1 models
    CHECK(Model::loadBytes(linearSpkm(o), err) == nullptr && err.find("spark-ft-v1") != std::string::npos);
    o.version = kSpkmVersion;
    o = {};
    o.hash = "0123456789abcdef";
    CHECK(Model::loadBytes(linearSpkm(o), err) == nullptr && err.find("hash") != std::string::npos);
    o = {};
    o.featureSet = "spark-ft-v1";
    CHECK(Model::loadBytes(linearSpkm(o), err) == nullptr && err.find("feature set") != std::string::npos);
    o = {};
    o.renameFeature = true;
    CHECK(Model::loadBytes(linearSpkm(o), err) == nullptr && err.find("bogus") != std::string::npos);
    auto good = linearSpkm();
    for (size_t cut : {size_t(0), size_t(10), good.size() / 2, good.size() - 1}) {
        std::vector<unsigned char> t(good.begin(), good.begin() + long(cut));
        CHECK(Model::loadBytes(t, err) == nullptr);
    }
    auto flipped = good;
    flipped[flipped.size() - 3] ^= 0x40;
    CHECK(Model::loadBytes(flipped, err) == nullptr && err.find("CRC") != std::string::npos);
    std::vector<unsigned char> notModel(good.size(), 'x');
    CHECK(Model::loadBytes(notModel, err) == nullptr && err.find("magic") != std::string::npos);
    CHECK(Model::load("/nonexistent/model.spkm", err) == nullptr);
}

TEST(engine_keeps_running_when_model_is_rejected) {
    std::string path = tempPath("bad.spkm");
    SpkmOpts o;
    o.hash = "0123456789abcdef";
    auto bytes = linearSpkm(o);
    FILE *f = std::fopen(path.c_str(), "wb");
    std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    Config cfg;
    cfg.headless = true;
    cfg.maxFrames = 30;
    cfg.width = cfg.height = 32;
    cfg.modelPath = path;
    cfg.fpFile = "none";
    cfg.controller = CtlMode::Model;
    Engine e(cfg);
    CHECK(e.run(std::make_unique<BusyScene>(e)) == 0);
    CHECK(e.frame() == 30);
    CHECK(e.model() == nullptr);
    CHECK(e.modelError().find("hash") != std::string::npos);
    std::remove(path.c_str());
}

// ---------------------------------------------------------------- controllers

namespace {
// Predicted cost per ladder level (best quality first).
float levelCost(const Quality &q, const float *costs) {
    int l = QualityLadder::find(q);
    return costs[l < 0 ? 0 : l];
}
} // namespace

TEST(controller_picks_cheaper_level_when_over_budget) {
    Controller c;
    CtlParams p;
    p.budgetMs = 16.667;
    p.margin = 0.1; // target 15 ms
    c.configure(CtlMode::Model, p);
    const float costs[] = {22, 19, 16, 14, 11, 8, 6, 4, 3};
    Quality cur = QualityLadder::kLevels[0];
    auto d = c.decide(cur, 0, p.budgetMs, [&](const Quality &q) { return levelCost(q, costs); });
    CHECK(QualityLadder::find(d.q) == 3); // best level predicted <= 15 ms
    CHECK(d.action == 1);
    CHECK(d.evals == 4);
    CHECK(d.value == 14);
    // Everything over budget: the cheapest level.
    const float hopeless[] = {40, 39, 38, 37, 36, 35, 34, 33, 32};
    d = c.decide(cur, 0, p.budgetMs, [&](const Quality &q) { return levelCost(q, hopeless); });
    CHECK(QualityLadder::find(d.q) == QualityLadder::kCount - 1);
    // Fits at the current level: keep it.
    const float easy[] = {12, 10, 9, 8, 7, 6, 5, 4, 3};
    d = c.decide(cur, 0, p.budgetMs, [&](const Quality &q) { return levelCost(q, easy); });
    CHECK(d.q == cur && d.action == 0);
}

TEST(controller_hysteresis_on_the_way_up) {
    Controller c;
    CtlParams p;
    p.budgetMs = 16.667;
    p.margin = 0.1;   // target 15
    p.upFrac = 0.8;   // raise only if the better level is predicted <= 12
    p.upHold = 30;
    p.upCooldown = 60;
    c.configure(CtlMode::Model, p);
    Quality cur = QualityLadder::kLevels[3];
    float costs[] = {30, 25, 13, 10, 8, 6, 4, 3, 2}; // level 2 fits (13 <= 15) but not with headroom (> 12)
    auto run = [&](int frames) {
        int changes = 0;
        for (int i = 0; i < frames; ++i) {
            auto d = c.decide(cur, 0, p.budgetMs, [&](const Quality &q) { return levelCost(q, costs); });
            if (!(d.q == cur)) ++changes;
            cur = d.q;
        }
        return changes;
    };
    CHECK(run(300) == 0); // inside the hysteresis band: never moves up
    costs[2] = 11;        // now comfortably fits
    CHECK(run(29) == 0);  // ... but only after upHold frames
    CHECK(run(1) == 1);
    CHECK(QualityLadder::find(cur) == 2);
    // A drop blocks raising again for upCooldown frames even if there is headroom.
    costs[2] = 20;
    CHECK(run(1) == 1 && QualityLadder::find(cur) == 3);
    costs[2] = 5;
    CHECK(run(59) == 0);
    CHECK(run(31) == 1 && QualityLadder::find(cur) == 2);
}

TEST(controller_does_not_oscillate_on_noisy_predictions) {
    Controller c;
    CtlParams p;
    c.configure(CtlMode::Model, p);
    Rng rng(3);
    Quality cur;
    int changes = 0;
    // Level 1 hovers right at the target with +-8% noise; better levels are clearly over.
    const double target = p.budgetMs * (1 - p.margin);
    for (int i = 0; i < 3000; ++i) {
        auto d = c.decide(cur, 0, p.budgetMs, [&](const Quality &q) {
            int l = QualityLadder::find(q);
            double base = target * (1.0 + 0.25 * (1 - l));
            return float(base * (1 + rng.range(-0.08f, 0.08f)));
        });
        changes += !(d.q == cur);
        cur = d.q;
    }
    CHECK(changes < 60); // naive "pick whatever fits now" would flip every few frames
}

TEST(reactive_controller_steps_and_panics) {
    Controller c;
    CtlParams p;
    p.downEvery = 6;
    p.panicFrames = 3;
    c.configure(CtlMode::Reactive, p);
    Quality cur = QualityLadder::kLevels[0];
    // Slightly over target but under budget: one step down, rate limited.
    int drops = 0;
    for (int i = 0; i < 12; ++i) {
        auto d = c.decide(cur, 15.5, p.budgetMs, {});
        drops += d.action == 1;
        cur = d.q;
    }
    CHECK(drops == 2);
    CHECK(QualityLadder::find(cur) == 2);
    // Three consecutive misses: panic drop of two levels at once.
    c.configure(CtlMode::Reactive, p);
    cur = QualityLadder::kLevels[0];
    Quality before = cur;
    for (int i = 0; i < 3; ++i) cur = c.decide(cur, 40, p.budgetMs, {}).q;
    CHECK(QualityLadder::find(cur) >= 2 && !(cur == before));
    // Plenty of headroom for long enough: back up one level.
    c.configure(CtlMode::Reactive, p);
    cur = QualityLadder::kLevels[4];
    int ups = 0;
    for (int i = 0; i < 61; ++i) {
        auto d = c.decide(cur, 3.0, p.budgetMs, {});
        ups += d.action == 2;
        cur = d.q;
    }
    CHECK(ups == 1 && QualityLadder::find(cur) == 3);
}

TEST(random_controller_explores) {
    Controller c;
    CtlParams p;
    p.randMinFrames = 10;
    p.randMaxFrames = 20;
    p.seed = 9;
    c.configure(CtlMode::Random, p);
    Quality cur;
    int changes = 0, offLadder = 0;
    for (int i = 0; i < 2000; ++i) {
        auto d = c.decide(cur, 1, p.budgetMs, {});
        if (!(d.q == cur)) {
            ++changes;
            offLadder += QualityLadder::find(d.q) < 0;
        }
        cur = d.q;
    }
    CHECK(changes > 60 && changes < 200);
    CHECK(offLadder > 10);
}

TEST(quality_parse_and_ladder) {
    Quality q;
    CHECK(parseQuality("1,2,3", q) && q.scale == 1 && q.particles == 2 && q.glow == 3 && q.aa == 0);
    CHECK(parseQuality("1,2,3,1", q) && q.aa == 1 && !q.antialias());
    CHECK(parseQuality("L8", q) && q == QualityLadder::kLevels[8]);
    CHECK(!parseQuality("4,0,0", q));
    CHECK(!parseQuality("0,0,0,2", q));
    CHECK(!parseQuality("L9", q));
    CHECK(!parseQuality("x", q));
    CHECK(QualityLadder::find(Quality{}) == 0);
    for (int i = 1; i < QualityLadder::kCount; ++i) { // every step lowers at least one lever and raises none
        const Quality &a = QualityLadder::kLevels[i - 1], &b = QualityLadder::kLevels[i];
        CHECK(b.scale >= a.scale && b.particles >= a.particles && b.glow >= a.glow && b.aa >= a.aa && !(a == b));
    }
}

// ---------------------------------------------------------------- render levers

TEST(canvas_pixel_scale_and_glow_quality) {
    Canvas cv(40, 20);
    cv.setPixelScale(2);
    CHECK(cv.width() == 80 && cv.height() == 40);
    cv.clear(Color(0, 0, 0));
    cv.rect(10, 5, 4, 3, Color(255, 0, 0));
    CHECK(cv.stats().px[size_t(Prim::Rect)] == 8u * 6u); // 4x3 logical -> 8x6 pixels
    CHECK(cv.pixel(20, 10) == Color(255, 0, 0).argb && cv.pixel(27, 15) == Color(255, 0, 0).argb);
    CHECK(cv.pixel(28, 10) != Color(255, 0, 0).argb);
    cv.resetStats();
    cv.clear(Color(0, 0, 0));
    cv.setGlowScale(0);
    cv.glow({20, 10}, 6, Color(255, 255, 255));
    CHECK(cv.pixel(40, 20) == Color(0, 0, 0).argb); // glow off draws nothing...
    CHECK(cv.stats().glowReqPx > 0 && cv.stats().px[size_t(Prim::Glow)] == 0); // ...but the request is counted
    CHECK(cv.stats().calls[size_t(Prim::Glow)] == 1);
    cv.setGlowScale(0.5f);
    cv.glow({20, 10}, 6, Color(255, 255, 255));
    CHECK(cv.pixel(40, 20) != Color(0, 0, 0).argb);
    CHECK(cv.stats().px[size_t(Prim::Glow)] < cv.stats().glowReqPx / 2);
    // Shrinking reuses the buffer; overlay covers the whole (smaller) image.
    cv.setPixelScale(0.5f);
    CHECK(cv.width() == 20 && cv.height() == 10);
    cv.clear(Color(0, 0, 0));
    cv.overlay(Color(0, 0, 255));
    CHECK(cv.pixel(0, 0) == Color(0, 0, 255).argb && cv.pixel(19, 9) == Color(0, 0, 255).argb);
}

TEST(canvas_aliased_rasterisation) {
    // AA off: same shapes as aliased spans; stats count the same AA bounding boxes.
    Canvas a(64, 64), b(64, 64);
    b.setAntialias(false);
    for (Canvas *cv : {&a, &b}) {
        cv->clear(Color(0, 0, 0));
        cv->circle({20, 20}, 8, Color(255, 0, 0));
        cv->ring({40, 40}, 10, 3, Color(0, 255, 0));
        cv->line({5, 60}, {60, 5}, 3, Color(0, 0, 255));
    }
    for (Prim p : {Prim::Circle, Prim::Ring, Prim::Line}) CHECK(a.stats().px[size_t(p)] == b.stats().px[size_t(p)]);
    CHECK(b.pixel(20, 20) == Color(255, 0, 0).argb);                // circle centre
    CHECK(b.pixel(20, 13) == Color(255, 0, 0).argb);                // just inside the edge
    CHECK(b.pixel(20, 10) != Color(255, 0, 0).argb);                // outside
    CHECK((b.pixel(40, 30) >> 8 & 255) == 255);                     // on the ring
    CHECK(b.pixel(40, 40) == Color(0, 0, 0).argb);                  // ring hole
    CHECK((b.pixel(32, 32) & 255) == 255);                          // on the line
    CHECK(b.pixel(10, 10) == Color(0, 0, 0).argb);
    int partial = 0; // aliased output has no partial-coverage (blended) red pixels
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 64; ++x) {
            int r = int(b.pixel(x, y) >> 16 & 255);
            partial += r != 0 && r != 255;
        }
    CHECK(partial == 0);
}

TEST(fingerprint_json_round_trip) {
    std::string j = std::string("{\"format\":\"") + fp::kFormat +
                    "\",\"key\":\"host=x|core=P\",\"state\":{},\"fp\":{\"step_ms_base\":0.001,\"step_ms_per_entity\":2e-4,"
                    "\"step_ms_normal\":0.0012,\"raster_calib\":0.9,\"cliff_ratio\":1.05,\"ns_px_clear\":0.36,\"ns_px_rect\":3.8,"
                    "\"ns_px_poly\":6,\"ns_px_glow\":4.5,\"ns_px_circle\":5,\"ns_px_circle_aliased\":3,\"ns_px_ring\":3,"
                    "\"ns_px_ring_aliased\":1,\"ns_px_line\":5,\"ns_px_line_aliased\":1.6,\"frame_px_s1\":518400,"
                    "\"frame_px_s32\":5308416,\"frame_ms_s1\":1.4,\"lat_dram_ns\":111,\"noise_excess_p99_us\":3}}";
    fp::Fingerprint f;
    std::string err;
    CHECK(fp::parse(j, f, err));
    CHECK(f.valid && f.key == "host=x|core=P" && f.get("step_ms_per_entity") == 2e-4 && f.get("frame_px_s32") == 5308416);
    FpCoef c = FpCoef::from(f);
    CHECK(c.valid && c.nsLine[1] == 1.6 && c.noiseP99Ms == 0.003);
    // t_ref grows with resolution and falls with AA off / glow off.
    FeatureState st;
    FrameRecord done{};
    done.render_scale = 1;
    done.px_line = 100000;
    done.px_glow_req = 200000;
    done.px_clear = 518400;
    st.update(done);
    FrameRecord next{};
    next.n_steps = 1;
    next.entities = 10;
    FrameContext ctx{960, 540, 1.0f, c};
    float hi[kNumFeatures], lo[kNumFeatures], big[kNumFeatures];
    st.compute(next, Quality{}, ctx, hi);
    st.compute(next, Quality{0, 0, 3, 1}, ctx, lo);
    ctx.outScale = 2;
    st.compute(next, Quality{}, ctx, big);
    CHECK(hi[51] > lo[51] && big[51] > 3.9f * hi[50] && hi[49] > 0 && hi[47] == 1 && lo[47] == 0);
    CHECK(!fp::parse("{\"format\":\"other\"}", f, err));
}

TEST(particles_burst_scale) {
    Particles a(100000, 1), b(100000, 1);
    b.setBurstScale(0.25f);
    Burst burst;
    burst.count = 40;
    burst.streak = true;
    for (int i = 0; i < 50; ++i) {
        a.burst({0, 0}, burst);
        b.burst({0, 0}, burst);
    }
    CHECK(a.size() == 2000);
    CHECK(b.size() == 500); // 40 * 0.25 is exact: no stochastic rounding needed
    CHECK(a.streaks() == 2000);
    auto c = a.takeCounters();
    CHECK(c.bursts == 50 && c.spawned == 2000);
    CHECK(a.takeCounters().spawned == 0);
    for (int i = 0; i < 100; ++i) a.update(0.05f); // all particles die
    CHECK(a.size() == 0 && a.streaks() == 0);
}

// ---------------------------------------------------------------- features

TEST(feature_state_history_and_counterfactual_quality) {
    FeatureState st;
    FrameContext ctx{100, 50, 2.0f, {}};
    FrameRecord r{};
    r.n_steps = 2;
    r.n_frozen = 1;
    r.t_total_us = 4000;
    r.t_steps_us = 1000;
    r.t_render_us = 3000;
    r.render_scale = 2.0f;
    r.px_clear = 200 * 100;
    r.px_rect = 4000;
    r.px_glow_req = 8000;
    r.q_glow = 0;
    st.update(r);
    float x[kNumFeatures];
    FrameRecord next{};
    next.n_steps = 1;
    st.compute(next, Quality{}, ctx, x);
    CHECK(x[38] == 4.0f);               // t_prev_ms
    CHECK(x[44] == 2.0f);               // steps_prev
    CHECK(x[31] == 0.02f);              // 200x100 px
    CHECK(std::fabs(x[34] - 0.004f) < 1e-7f); // raster px carried over at the same scale
    float xLow[kNumFeatures];
    st.compute(next, Quality{3, 0, 3}, ctx, xLow); // half resolution, glow off
    CHECK(std::fabs(xLow[34] - 0.001f) < 1e-7f);   // a quarter of the pixels
    CHECK(xLow[35] == 0.0f && x[35] > 0.0f);
    CHECK(xLow[37] < x[37]);                       // naive estimate falls with quality
    CHECK(x[1] == 1.0f);                           // n_steps_active of `next` (no hit-stop)
}
