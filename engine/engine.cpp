#include "engine.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <thread>

#include "window.hpp"

namespace spark {

using Clock = std::chrono::steady_clock;

Engine::Engine(Config cfg) : cfg_(std::move(cfg)), canvas_(cfg_.width, cfg_.height), showStats_(cfg_.showStats) {}

Engine::~Engine() {
    // Exit scenes top-down before the subsystems they use go away.
    while (!scenes_.empty()) {
        scenes_.back()->exit();
        scenes_.pop_back();
    }
    audio_.stop();
}

void Engine::switchTo(std::unique_ptr<Scene> s) { changes_.push_back({Change::Switch, std::move(s)}); }
void Engine::push(std::unique_ptr<Scene> s) { changes_.push_back({Change::Push, std::move(s)}); }
void Engine::pop() { changes_.push_back({Change::Pop, nullptr}); }

void Engine::applySceneChanges() {
    // Changes may queue further changes (e.g. enter() pushing a scene).
    while (!changes_.empty()) {
        auto batch = std::move(changes_);
        changes_.clear();
        for (auto &c : batch) {
            if (c.kind == Change::Switch || c.kind == Change::Pop) {
                bool all = c.kind == Change::Switch;
                do {
                    if (scenes_.empty()) break;
                    scenes_.back()->exit();
                    scenes_.pop_back();
                } while (all);
            }
            if (c.scene) {
                scenes_.push_back(std::move(c.scene));
                scenes_.back()->enter();
            }
        }
    }
}

void Engine::step() {
    const float dt = float(cfg_.step);
    if (cfg_.onStep) {
        ProfScope ps(prof_, Prof::OnStep);
        cfg_.onStep(frame_);
    }

    // Engine-level hotkeys.
    if (input_.pressed(Key::F3)) showStats_ = !showStats_;
    if (input_.pressed(Key::F2)) {
        char name[64];
        std::snprintf(name, sizeof name, "screenshot-%ld.ppm", frame_);
        if (canvas_.savePPM(name)) std::printf("saved %s\n", name);
    }

    if (hitstop_ > 0) {
        hitstop_ -= dt; // frozen: skip the scene update but keep time flowing
    } else if (Scene *s = top()) {
        ProfScope ps(prof_, Prof::Update);
        s->update(dt);
    }
    input_.endStep();
    applySceneChanges();

    // Screen shake: offset grows with trauma squared, which feels punchier.
    trauma_ = std::max(0.0f, trauma_ - dt * 1.5f);
    float mag = trauma_ * trauma_ * 16.0f;
    shakeOffset_ = {rng_.range(-1.0f, 1.0f) * mag, rng_.range(-1.0f, 1.0f) * mag};

    ++frame_;
    time_ += dt;
    if (cfg_.maxFrames > 0 && frame_ >= cfg_.maxFrames) quit_ = true;
    if (scenes_.empty()) quit_ = true;
}

void Engine::render() {
    for (auto &s : scenes_) {
        canvas_.setBlend(Blend::Alpha);
        canvas_.setOffset(shakeOffset_);
        s->draw(canvas_);
    }
    canvas_.setOffset({});
    canvas_.setBlend(Blend::Alpha);
    if (showStats_) drawStats();
}

void Engine::drawStats() {
    char buf[96];
    const float right = float(cfg_.width) - 8, x0 = float(cfg_.width) - 246;
    canvas_.rect(x0, 4, 242, 98, Color(0, 0, 0, 170));
    std::snprintf(buf, sizeof buf, "%d FPS  %.2f MS", fps_, frameMs_);
    canvas_.text(right, 8, buf, 1, Color(120, 255, 140), Align::Right);

    const float budget = float(cfg_.ctl.budgetMs);
    const bool havePred = model_ && (rec_.flags & 4);
    if (havePred) std::snprintf(buf, sizeof buf, "PRED %.2f  ACT %.2f MS", double(rec_.pred_ctl), lastTotalMs_);
    else std::snprintf(buf, sizeof buf, "PRED -  ACT %.2f MS", lastTotalMs_);
    canvas_.text(right, 20, buf, 1, Color(255, 230, 120), Align::Right);
    int lvl = QualityLadder::find(quality_);
    std::snprintf(buf, sizeof buf, "Q %s%d S%.2f P%.2f G%.2f AA%d %s", lvl >= 0 ? "L" : "C", lvl >= 0 ? lvl : 0,
                  double(fctx_.outScale * quality_.renderScale()), double(quality_.particleDensity()),
                  double(quality_.glowScale()), quality_.antialias() ? 1 : 0, ft::ctlName(ctl_.mode()));
    canvas_.text(right, 32, buf, 1, Color(160, 210, 255), Align::Right);
    if (model_) std::snprintf(buf, sizeof buf, "%s  %.1f US", model_->family().c_str(), double(rec_.t_pred_us));
    else std::snprintf(buf, sizeof buf, "%s", modelError_.empty() ? "PREDICTOR OFF" : "MODEL REJECTED");
    canvas_.text(right, 44, buf, 1, modelError_.empty() ? Color(200, 200, 200) : Color(255, 90, 90), Align::Right);

    // Actual (green) vs predicted (yellow) frame time; red line = budget. Scale 0..2x budget.
    const float gx = x0 + 6, gy = 98, gh = 40, gw = 230;
    canvas_.rect(gx, gy - gh * 0.5f, gw, 1, Color(255, 70, 70, 200));
    for (int i = 0; i < kGraph; ++i) {
        int k = (graphPos_ + i) % kGraph;
        float px = gx + float(i) * gw / float(kGraph);
        float a = std::min(1.0f, graphAct_[k] / (2 * budget)) * gh;
        canvas_.rect(px, gy - a, 1.5f, std::max(1.0f, a), Color(120, 255, 140, 200));
        if (graphPred_[k] > 0) {
            float p = std::min(1.0f, graphPred_[k] / (2 * budget)) * gh;
            canvas_.rect(px, gy - p, 1.5f, 1.5f, Color(255, 230, 120));
        }
    }
}

void Engine::setQuality(const Quality &q) {
    float s = fctx_.outScale * q.renderScale();
    if (s != canvas_.pixelScale()) canvas_.setPixelScale(s);
    canvas_.setGlowScale(q.glowScale());
    canvas_.setAntialias(q.antialias());
    quality_ = q;
}

namespace {
std::string utcNow() {
    std::time_t t = std::time(nullptr);
    char b[32];
    std::strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    return b;
}

/** Resolve --model: a .spkm file, or a checkpoint folder (manifest.json engine_file, else model.spkm). */
std::string resolveModelPath(const std::string &p) {
    struct stat st{};
    if (stat(p.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) return p;
    std::string file = "model.spkm";
    std::ifstream mf(p + "/manifest.json");
    if (mf) {
        std::stringstream ss;
        ss << mf.rdbuf();
        std::string m = ss.str();
        auto k = m.find("\"engine_file\"");
        if (k != std::string::npos) {
            auto q1 = m.find('"', m.find(':', k) + 1), q2 = q1 == std::string::npos ? q1 : m.find('"', q1 + 1);
            if (q2 != std::string::npos) file = m.substr(q1 + 1, q2 - q1 - 1);
        }
    }
    return p + "/" + file;
}

const char *loopName(LoopMode m) { return m == LoopMode::Fast ? "fast" : m == LoopMode::Emulate ? "emulate" : "paced"; }
} // namespace

void Engine::setupResearch() {
    fctx_ = {cfg_.width, cfg_.height, cfg_.outScale, {}};
    // Co-runners first: a fingerprint measured now must see the same "virtual hardware".
    if (!cfg_.corunner.empty()) {
        std::string err;
        if (!corunner_.start(cfg_.corunner, err)) std::fprintf(stderr, "spark: co-runner disabled: %s\n", err.c_str());
    }
    if (!cfg_.logPath.empty() || !cfg_.featuresPath.empty() || !cfg_.modelPath.empty()) {
        fp_ = fp::resolve(cfg_.fpFile, cfg_.fpCacheDir, cfg_, cfg_.fpHooks, cfg_.corunner, cfg_.fpAllowRun);
        fctx_.fp = ft::FpCoef::from(fp_);
    }
    emuRng_.seed(cfg_.researchSeed * 0x2545F4914F6CDD1Dull + 3);
    if (cfg_.outScale != 1) canvas_.setPixelScale(cfg_.outScale); // allocate the largest buffer up front
    if (cfg_.outScale != 1 || !(cfg_.quality == Quality{})) setQuality(cfg_.quality);

    if (!cfg_.modelPath.empty()) {
        std::string path = resolveModelPath(cfg_.modelPath);
        model_ = ft::Model::load(path, modelError_);
        if (!model_) {
            std::fprintf(stderr, "spark: MODEL REJECTED (%s): %s\nspark: continuing without a predictor\n", path.c_str(),
                         modelError_.c_str());
        } else {
            outP50_ = model_->outputIndex("t_p50_ms");
            outP95_ = model_->outputIndex("t_p95_ms");
            outP99_ = model_->outputIndex("t_p99_ms");
            outPmiss_ = model_->outputIndex("p_miss");
            outCtl_ = model_->outputIndex(cfg_.ctlOutput);
            for (const char *n : {"t_p99_ms", "t_p95_ms", "t_p90_ms", "t_mean_ms", "t_p50_ms"})
                if (outCtl_ < 0) outCtl_ = model_->outputIndex(n);
            if (outCtl_ < 0) {
                modelError_ = "model has no frame-time output the controller can use";
                std::fprintf(stderr, "spark: MODEL REJECTED (%s): %s\n", path.c_str(), modelError_.c_str());
                model_.reset();
            } else {
                std::fprintf(stderr, "spark: loaded %s model %s (%zu params, controller output %s)\n",
                             model_->family().c_str(), path.c_str(), model_->numParameters(),
                             model_->outputs()[size_t(outCtl_)].name.c_str());
                if (std::fabs(model_->budgetMs() - cfg_.ctl.budgetMs) > 0.01 * cfg_.ctl.budgetMs)
                    std::fprintf(stderr, "spark: note: model was trained for a %.3f ms budget, engine uses %.3f ms\n",
                                 model_->budgetMs(), cfg_.ctl.budgetMs);
            }
        }
    }
    ft::CtlMode mode = cfg_.controller;
    if (mode == ft::CtlMode::Model && !model_) {
        std::fprintf(stderr, "spark: --ctl model needs a valid --model; using fixed quality\n");
        mode = ft::CtlMode::Fixed;
    }
    ft::CtlParams cp = cfg_.ctl;
    cp.seed = cfg_.researchSeed;
    ctl_.configure(mode, cp);

    needFeatures_ = model_ || !cfg_.featuresPath.empty();
    instrumented_ = needFeatures_ || !cfg_.logPath.empty() || mode != ft::CtlMode::Fixed ||
                    cfg_.loop == LoopMode::Emulate;
    prof_.enabled = instrumented_;

    if (cfg_.contendThreads > 0)
        contention_.start(cfg_.contendThreads, cfg_.contendBusyMs, cfg_.contendIdleMs, cfg_.researchSeed);
    onAC_ = powerSource() != "battery";

    if (!cfg_.logPath.empty() || !cfg_.featuresPath.empty()) {
        char hash[32];
        std::snprintf(hash, sizeof hash, "%016llx", static_cast<unsigned long long>(ft::featureHash()));
        std::string sess = "{\"start_utc\":\"" + utcNow() + "\",\"windowed\":" + (cfg_.headless ? "false" : "true") +
                           ",\"loop\":\"" + (cfg_.headless ? loopName(cfg_.loop) : "windowed") +
                           "\",\"step_s\":" + std::to_string(cfg_.step) + ",\"logical_w\":" +
                           std::to_string(cfg_.width) + ",\"logical_h\":" + std::to_string(cfg_.height) +
                           ",\"window_scale\":" + std::to_string(cfg_.windowScale) +
                           ",\"out_scale\":" + std::to_string(cfg_.outScale) + ",\"burst_mult\":" +
                           std::to_string(cfg_.burstMult) + ",\"overshoot_us\":" + std::to_string(cfg_.overshootUs) +
                           ",\"contend_threads\":" + std::to_string(cfg_.contendThreads) + ",\"contend_busy_ms\":" +
                           std::to_string(cfg_.contendBusyMs) + ",\"contend_idle_ms\":" +
                           std::to_string(cfg_.contendIdleMs) + ",\"controller\":\"" + ft::ctlName(mode) +
                           "\",\"budget_ms\":" + std::to_string(cfg_.ctl.budgetMs) + ",\"quality_start\":[" +
                           std::to_string(cfg_.quality.scale) + "," + std::to_string(cfg_.quality.particles) + "," +
                           std::to_string(cfg_.quality.glow) + "],\"research_seed\":" +
                           std::to_string(cfg_.researchSeed) + ",\"model\":\"" +
                           jsonEscape(model_ ? cfg_.modelPath : "") + "\",\"model_family\":\"" +
                           (model_ ? model_->family() : "") + "\",\"feature_set\":\"" + ft::kFeatureSet +
                           "\",\"feature_hash\":\"" + hash + "\",\"corunner\":\"" + jsonEscape(cfg_.corunner) +
                           "\",\"fingerprint_key\":\"" + jsonEscape(fp_.key) + "\"" +
                           (cfg_.sessionJson.empty() ? "" : "," + cfg_.sessionJson) + "}";
        std::string machine = machineInfoJson(), err;
        // The fingerprint record is embedded in the machine block (null if none was available).
        machine.pop_back();
        machine += ",\"fingerprint\":" + (fp_.valid ? fp_.json : std::string("null")) + "}";
        if (!cfg_.logPath.empty() &&
            !log_.open(cfg_.logPath, "SPKFLOG", kFrameLogVersion, sizeof(FrameRecord),
                       frameLogHeaderJson(machine, sess), err))
            std::fprintf(stderr, "spark: frame log disabled: %s\n", err.c_str());
        if (!cfg_.featuresPath.empty()) {
            std::string names;
            for (int i = 0; i < ft::kNumFeatures; ++i) names += std::string(i ? "," : "") + "\"" + ft::kFeatureNames[i] + "\"";
            std::string j = "{\"format\":\"spark-features\",\"feature_set\":\"" + std::string(ft::kFeatureSet) +
                            "\",\"feature_hash\":\"" + hash + "\",\"features\":[" + names + "],\"session\":" + sess + "}";
            if (!featLog_.open(cfg_.featuresPath, "SPKFEAT", 1, sizeof x_, j, err))
                std::fprintf(stderr, "spark: feature dump disabled: %s\n", err.c_str());
        }
    }
}

void Engine::beginFrame(int steps, double acc) {
    auto tb = Clock::now();
    sys_.sample();
    rec_ = FrameRecord{};
    rec_.frame = iter_;
    rec_.step0 = uint32_t(frame_);
    rec_.n_steps = uint8_t(std::min(steps, 255));
    int frozen = 0;
    for (float h = hitstop_; frozen < steps && h > 0; ++frozen) h -= float(cfg_.step);
    rec_.n_frozen = uint8_t(std::min(frozen, 255));
    rec_.acc_ms = float(acc * 1000.0);
    rec_.hitstop_ms = hitstop_ * 1000.0f;
    rec_.trauma = trauma_;
    rec_.flags = uint8_t((cfg_.headless ? 0 : 1) | (fp_.valid ? 16 : 0));
    if (powerPoll_++ % 120 == 0) onAC_ = powerSource() != "battery"; // ~every 2 s
    if (onAC_) rec_.flags |= 8;
    for (auto &s : scenes_) s->planWork(rec_);
    rec_.cpu = uint8_t(std::min(sys_.cpu, 255));
    rec_.cpu_mhz = uint16_t(std::clamp(sys_.mhz, 0, 65535));
    rec_.ctxsw = uint16_t(std::clamp(sys_.ctxsw, 0L, 65535L));
    rec_.minflt = uint16_t(std::clamp(sys_.minflt, 0L, 65535L));
    rec_.contend = contention_.busy();
    auto ta = Clock::now();
    preLogUs_ = std::chrono::duration<double, std::micro>(ta - tb).count();

    // Controller, then the features/prediction for the quality actually used.
    ft::CtlDecision d;
    d.q = quality_;
    bool encoded = false, cached = false;
    Quality lastEval;
    if (ctl_.mode() != ft::CtlMode::Fixed) {
        double budget = cfg_.ctl.budgetMs - (cfg_.headless ? 0.0 : presentEma_);
        ft::Controller::Predict fn;
        if (ctl_.mode() == ft::CtlMode::Model && model_) {
            model_->beginFrame(hist_);
            encoded = true;
            fn = [this, &lastEval, &cached](const Quality &q) {
                feat_.compute(rec_, q, fctx_, xc_);
                model_->predict(xc_, out_);
                lastEval = q;
                cached = true;
                return out_[outCtl_];
            };
        }
        d = ctl_.decide(quality_, lastTotalMs_, budget, fn);
        if (!(d.q == quality_)) {
            setQuality(d.q);
            rec_.flags |= 2;
        }
    }
    rec_.pred_p50 = rec_.pred_p95 = rec_.pred_p99 = rec_.pred_pmiss = rec_.pred_ctl = -1;
    if (needFeatures_) feat_.compute(rec_, quality_, fctx_, x_);
    if (model_) {
        if (!encoded) model_->beginFrame(hist_);
        if (!cached || !(lastEval == quality_)) model_->predict(x_, out_); // else out_ already holds it
        if (outP50_ >= 0) rec_.pred_p50 = out_[outP50_];
        if (outP95_ >= 0) rec_.pred_p95 = out_[outP95_];
        if (outP99_ >= 0) rec_.pred_p99 = out_[outP99_];
        if (outPmiss_ >= 0) rec_.pred_pmiss = out_[outPmiss_];
        rec_.pred_ctl = out_[outCtl_];
        rec_.flags |= 4;
    }
    rec_.render_scale = fctx_.outScale * quality_.renderScale();
    int lvl = QualityLadder::find(quality_);
    rec_.q_level = uint8_t(lvl < 0 ? 255 : lvl);
    rec_.q_scale = quality_.scale;
    rec_.q_particles = quality_.particles;
    rec_.q_glow = quality_.glow;
    rec_.q_aa = quality_.aa;
    rec_.ctl_mode = uint8_t(ctl_.mode());
    rec_.ctl_evals = uint8_t(std::min(d.evals, 255));
    rec_.ctl_action = uint8_t(d.action);
    // Predictor cost of this frame: the model's end-of-frame update for the previous
    // frame (RLS update, streaming GRU step) plus everything above.
    rec_.t_pred_us = float(pendingPredUs_) + std::chrono::duration<float, std::micro>(Clock::now() - ta).count();
    pendingPredUs_ = 0;
}

void Engine::endFrame(double totalUs, double stepsUs, double renderUs, double presentUs, double sleepReqUs,
                      double sleepUs) {
    auto te = Clock::now();
    rec_.t_total_us = float(totalUs);
    rec_.t_steps_us = float(stepsUs);
    rec_.t_render_us = float(renderUs);
    rec_.t_present_us = float(presentUs);
    rec_.t_sleep_req_us = float(sleepReqUs);
    rec_.t_sleep_us = float(sleepUs);
    const float *p = prof_.us;
    rec_.t_onstep_us = p[size_t(Prof::OnStep)];
    rec_.t_update_us = p[size_t(Prof::Update)];
    rec_.t_collision_us = p[size_t(Prof::Collision)];
    rec_.t_particles_us = p[size_t(Prof::ParticlesUpd)];
    rec_.t_flush_us = p[size_t(Prof::Flush)];
    rec_.t_r_bg_us = p[size_t(Prof::RBackground)];
    rec_.t_r_world_us = p[size_t(Prof::RWorld)];
    rec_.t_r_particles_us = p[size_t(Prof::RParticles)];
    rec_.t_r_hud_us = p[size_t(Prof::RHud)];
    rec_.t_log_us = float(lastLogUs_);
    const CanvasStats &cs = canvas_.stats();
    auto px = [&](Prim k) { return uint32_t(std::min<uint64_t>(cs.px[size_t(k)], 0xffffffffu)); };
    auto calls = [&](Prim k) { return uint16_t(std::min<uint32_t>(cs.calls[size_t(k)], 65535u)); };
    rec_.px_clear = px(Prim::Clear);
    rec_.px_rect = px(Prim::Rect);
    rec_.px_circle = px(Prim::Circle);
    rec_.px_ring = px(Prim::Ring);
    rec_.px_glow = px(Prim::Glow);
    rec_.px_line = px(Prim::Line);
    rec_.px_poly = px(Prim::Poly);
    rec_.px_glow_req = uint32_t(std::min<uint64_t>(cs.glowReqPx, 0xffffffffu));
    rec_.calls_rect = calls(Prim::Rect);
    rec_.calls_circle = calls(Prim::Circle);
    rec_.calls_ring = calls(Prim::Ring);
    rec_.calls_glow = calls(Prim::Glow);
    rec_.calls_line = calls(Prim::Line);
    rec_.calls_poly = calls(Prim::Poly);
    rec_.text_chars = uint16_t(std::min<uint32_t>(cs.textChars, 65535u));
    for (auto &s : scenes_) s->reportWork(rec_);

    if (log_.isOpen()) log_.write(&rec_);
    // Derived from the logged float fields so offline code reproduces them bit for bit.
    const float tMs = rec_.t_total_us / 1000.0f;
    if (needFeatures_) {
        float h[ft::kHistoryDim];
        std::copy(x_, x_ + ft::kNumFeatures, h);
        h[ft::kNumFeatures] = tMs;
        h[ft::kNumFeatures + 1] = rec_.t_steps_us / 1000.0f;
        h[ft::kNumFeatures + 2] = rec_.t_render_us / 1000.0f;
        hist_.push(h);
        if (model_) {
            auto tm = Clock::now();
            model_->endFrame(x_, h, tMs);
            pendingPredUs_ = std::chrono::duration<double, std::micro>(Clock::now() - tm).count();
        }
        feat_.update(rec_);
        if (featLog_.isOpen()) featLog_.write(x_);
    }
    graphAct_[graphPos_] = tMs;
    graphPred_[graphPos_] = rec_.pred_ctl;
    graphPos_ = (graphPos_ + 1) % kGraph;
    lastTotalMs_ = tMs;
    if (!cfg_.headless) presentEma_ = presentEma_ == 0 ? presentUs / 1000.0 : 0.9 * presentEma_ + 0.1 * presentUs / 1000.0;
    prof_.reset();
    canvas_.resetStats();
    ++iter_;
    lastLogUs_ = preLogUs_ + std::chrono::duration<double, std::micro>(Clock::now() - te).count() - pendingPredUs_;
}

int Engine::run(std::unique_ptr<Scene> first) {
    if (!cfg_.headless) {
        window_ = std::make_unique<Window>();
        if (!window_->open(cfg_.title.c_str(), cfg_.width, cfg_.height, cfg_.windowScale, cfg_.showCursor))
            return 1;
        if (cfg_.audio && !audio_.start()) std::fprintf(stderr, "spark: no audio player found; sound disabled\n");
    }
    setupResearch();
    switchTo(std::move(first));
    applySceneChanges();

    // Headless Fast: one step + one render per iteration, as fast as possible.
    // Windowed (and headless Paced): fixed-timestep accumulator on the real clock,
    // sleeping off the rest of each frame. Headless Emulate: the same accumulator
    // on a virtual clock (see LoopMode).
    const bool windowed = !cfg_.headless;
    const bool fast = cfg_.headless && cfg_.loop == LoopMode::Fast;
    const bool paced = windowed || cfg_.loop == LoopMode::Paced;
    const auto stepDur = std::chrono::duration<double>(cfg_.step);
    using us = std::chrono::duration<double, std::micro>;
    auto last = Clock::now();
    const auto startTime = last;
    auto fpsTimer = last;
    int fpsFrames = 0;
    double acc = 0, virtualElapsed = 0;
    while (!quit_) {
        if (windowed && !window_->poll(input_)) break;
        auto now = Clock::now();
        int steps = 1;
        if (!fast) {
            double elapsed = std::min(0.25, paced ? std::chrono::duration<double>(now - last).count() : virtualElapsed);
            last = now;
            // Fixed timestep: same dt every update regardless of frame rate.
            acc += elapsed;
            steps = 0;
            for (double a = acc; a >= cfg_.step; a -= cfg_.step) ++steps;
        }
        if (instrumented_) beginFrame(steps, acc);

        auto t0 = Clock::now();
        for (int i = 0; i < steps && !quit_; ++i) {
            step();
            if (!fast) acc -= cfg_.step;
        }
        auto t1 = Clock::now();
        render();
        auto t2 = Clock::now();
        frameMs_ = std::chrono::duration<double, std::milli>(t2 - (windowed ? now : t0)).count();

        double presentUs = 0, sleepReqUs = 0, sleepUs = 0;
        if (windowed) {
            window_->present(canvas_);
            presentUs = us(Clock::now() - t2).count();
        }
        if (paced) {
            ++fpsFrames;
            if (now - fpsTimer >= std::chrono::milliseconds(500)) {
                fps_ = int(fpsFrames / std::chrono::duration<double>(now - fpsTimer).count() + 0.5);
                fpsFrames = 0;
                fpsTimer = now;
            }
            // No vsync with plain X11, so sleep off the rest of the frame.
            auto spent = Clock::now() - now;
            if (spent < stepDur) {
                auto req = stepDur - spent;
                auto ts = Clock::now();
                std::this_thread::sleep_for(req);
                sleepReqUs = us(req).count();
                sleepUs = us(Clock::now() - ts).count();
            }
        }
        double overshootUs = 0;
        if (!paced && !fast) {
            // Emulate: the loop period a windowed run would have had.
            double spent = std::chrono::duration<double>(Clock::now() - now).count();
            // Overshoot model fitted to windowed runs on the dev laptop: a floor
            // (timer slack + wake-up) plus an exponential tail; mean = overshootUs.
            if (spent < cfg_.step)
                overshootUs = cfg_.overshootUs * (0.8 - 0.2 * std::log(1.0 - double(emuRng_.uniform())));
            virtualElapsed = std::max(spent, cfg_.step) + overshootUs * 1e-6;
            sleepReqUs = overshootUs;
        }
        if (instrumented_) {
            endFrame(us(t2 - t0).count(), us(t1 - t0).count(), us(t2 - t1).count(), presentUs, sleepReqUs, sleepUs);
        } else if (showStats_) { // keep the overlay's actual-time graph alive in plain play
            lastTotalMs_ = std::chrono::duration<double, std::milli>(t2 - t0).count();
            graphAct_[graphPos_] = float(lastTotalMs_);
            graphPred_[graphPos_] = -1;
            graphPos_ = (graphPos_ + 1) % kGraph;
        }
        if (cfg_.maxSeconds > 0 && std::chrono::duration<double>(Clock::now() - startTime).count() >= cfg_.maxSeconds)
            quit_ = true;
    }
    log_.close();
    featLog_.close();
    contention_.stop();
    corunner_.stop();
    return 0;
}

} // namespace spark
