#include "features.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include "canvas.hpp"

namespace spark::ft {

const char *const kFeatureNames[kNumFeatures] = {
    "n_steps", "n_steps_active", "acc_ms", "hitstop_ms", "trauma",                                   // 0-4
    "scene_menu", "scene_play", "scene_overlay",                                                     // 5-7
    "entities", "enemies", "enemy_k0", "enemy_k1", "enemy_k2", "enemy_k3", "enemy_k4",              // 8-14
    "projectiles", "enemy_projectiles", "pickups", "telegraphs", "effects", "popups",               // 15-20
    "tele_due_now", "tele_due_6", "tele_due_30", "burst_due",                                        // 21-24
    "particles_alive", "particles_streak", "particles_expiring", "wave", "wave_to_spawn",           // 25-29
    "q_render_scale", "q_render_mpx", "q_particle_density", "q_glow_scale",                         // 30-33
    "est_raster_mpx", "est_glow_mpx", "est_particles", "naive_ms",                                  // 34-37
    "t_prev_ms", "t_prev2_ms", "t_ema_ms", "t_max8_ms", "t_steps_prev_ms", "t_render_prev_ms",     // 38-43
    "steps_prev", "cpu_ghz", "ctxsw_prev",                                                           // 44-46
    "q_aa", "est_aa_mpx", "t_ref_steps_ms", "t_ref_render_ms", "t_ref_ms",                           // 47-51
    "fp_noise_p99_ms", "fp_noise_p999_ms", "fp_sleep_p99_ms",                                        // 52-54
    "fp_log_k_step", "fp_log_k_rpx", "fp_log_cliff", "fp_log_k_glow", "fp_log_k_part", "fp_log_ghz", // 55-60
    "fp_log_lat_l3", "fp_log_lat_dram", "fp_log_bw_dram",                                            // 61-63
};

const char *const kHistoryExtraNames[kHistoryExtra] = {"h_t_total_ms", "h_t_steps_ms", "h_t_render_ms"};

uint64_t featureHash() {
    std::string s = kFeatureSet;
    for (const char *n : kFeatureNames) s += std::string("\n") + n;
    for (const char *n : kHistoryExtraNames) s += std::string("\n") + n;
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

FpCoef FpCoef::from(const fp::Fingerprint &f) {
    FpCoef c;
    if (!f.valid) return c;
    c.valid = true;
    c.stepBase = f.get("step_ms_base");
    c.stepSlope = f.get("step_ms_per_entity");
    c.stepNormal = f.get("step_ms_normal");
    c.nsClear = f.get("ns_px_clear");
    c.nsRect = f.get("ns_px_rect");
    c.nsPoly = f.get("ns_px_poly");
    c.nsGlow = f.get("ns_px_glow");
    c.nsCircle[0] = f.get("ns_px_circle");
    c.nsCircle[1] = f.get("ns_px_circle_aliased");
    c.nsRing[0] = f.get("ns_px_ring");
    c.nsRing[1] = f.get("ns_px_ring_aliased");
    c.nsLine[0] = f.get("ns_px_line");
    c.nsLine[1] = f.get("ns_px_line_aliased");
    c.calib = f.get("raster_calib", 1);
    c.cliff = f.get("cliff_ratio", 1);
    c.px1 = std::max(1.0, f.get("frame_px_s1", 1));
    c.px32 = std::max(c.px1 + 1, f.get("frame_px_s32", c.px1 + 1));
    c.noiseP99Ms = f.get("noise_excess_p99_us") / 1000.0;
    c.noiseP999Ms = f.get("noise_excess_p999_us") / 1000.0;
    c.sleepP99Ms = f.get("sleep_over_p99_us") / 1000.0;
    auto lg = [](double v) { return v > 0 ? std::log10(v) : 0.0; };
    c.logKStep = lg(c.stepNormal);
    c.logKRpx = lg(f.get("frame_ms_s1") * 1e6 / c.px1);
    c.logCliff = lg(c.cliff);
    c.logKGlow = lg(c.nsGlow);
    c.logKPart = lg(f.get("part_update_ns"));
    c.logGhz = lg(f.get("ghz_start"));
    c.logLatL3 = lg(f.get("lat_l3_ns"));
    c.logLatDram = lg(f.get("lat_dram_ns"));
    c.logBwDram = lg(f.get("bw_dram_gbs"));
    return c;
}

void FeatureState::compute(const FrameRecord &r, const Quality &q, const FrameContext &ctx, float *x) const {
    const float s = effectiveScale(ctx, q);
    const double s2 = double(s) * double(s);
    const double g = double(q.glowScale());
    const double density = double(q.particleDensity());
    const double renderMpx =
        double(Canvas::scaledSize(ctx.logicalW, s)) * double(Canvas::scaledSize(ctx.logicalH, s)) / 1e6;
    const double estRaster = rasterLpx * s2 / 1e6;
    const double estGlow = glowReqLpx * s2 * g * g / 1e6;

    x[0] = float(r.n_steps);
    x[1] = float(r.n_steps - r.n_frozen);
    x[2] = r.acc_ms;
    x[3] = r.hitstop_ms;
    x[4] = r.trauma;
    x[5] = r.scene == 1;
    x[6] = r.scene == 2;
    x[7] = r.scene == 3;
    const uint16_t counts[] = {r.entities,  r.enemies,     r.enemy_k0,          r.enemy_k1, r.enemy_k2,
                               r.enemy_k3,  r.enemy_k4,    r.projectiles,       r.enemy_projectiles,
                               r.pickups,   r.telegraphs,  r.effects,           r.popups,   r.tele_due_now,
                               r.tele_due_6, r.tele_due_30, r.burst_due};
    for (size_t i = 0; i < sizeof counts / sizeof counts[0]; ++i) x[8 + i] = float(counts[i]); // 8-24
    x[25] = float(r.particles_alive);
    x[26] = float(r.particles_streak);
    x[27] = float(r.particles_expiring);
    x[28] = float(r.wave);
    x[29] = float(r.wave_to_spawn);
    x[30] = s;
    x[31] = float(renderMpx);
    x[32] = float(density);
    x[33] = float(g);
    x[34] = float(estRaster);
    x[35] = float(estGlow);
    x[36] = float(double(r.particles_alive) + double(r.burst_due) * density);
    x[37] = float(double(r.n_steps) * emaStep + emaRpw * (renderMpx + estRaster + estGlow));
    x[38] = float(tPrev);
    x[39] = float(tPrev2);
    x[40] = float(tEma);
    double mx = 0;
    for (int i = 0; i < n8; ++i) mx = std::max(mx, last8[i]);
    x[41] = float(mx);
    x[42] = float(tStepsPrev);
    x[43] = float(tRenderPrev);
    x[44] = float(stepsPrev);
    x[45] = float(double(r.cpu_mhz) / 1000.0);
    x[46] = float(r.ctxsw);

    // v2: anti-aliasing lever and fingerprint-based reference times.
    const int aa = q.aa ? 1 : 0;
    x[47] = aa ? 0.0f : 1.0f;
    x[48] = float((lpxCircle + lpxRing + lpxLine) * s2 / 1e6);
    const FpCoef &f = ctx.fp;
    double tSteps = 0, tRender = 0;
    if (f.valid) {
        double perStep = std::max(f.stepBase + f.stepSlope * double(r.entities), 0.25 * f.stepNormal);
        tSteps = double(r.n_steps) * perStep;
        const double bufPx = renderMpx * 1e6;
        double t = std::clamp((bufPx - f.px1) / (f.px32 - f.px1), 0.0, 1.0);
        double cliff = 1 + (f.cliff - 1) * t;
        double ns = f.nsClear * bufPx + s2 * (f.nsRect * lpxRect + f.nsPoly * lpxPoly + f.nsCircle[aa] * lpxCircle +
                                              f.nsRing[aa] * lpxRing + f.nsLine[aa] * lpxLine + f.nsGlow * glowReqLpx * g * g);
        tRender = f.calib * cliff * ns / 1e6;
    }
    x[49] = float(tSteps);
    x[50] = float(tRender);
    x[51] = float(tSteps + tRender);
    x[52] = float(f.noiseP99Ms);
    x[53] = float(f.noiseP999Ms);
    x[54] = float(f.sleepP99Ms);
    x[55] = float(f.logKStep);
    x[56] = float(f.logKRpx);
    x[57] = float(f.logCliff);
    x[58] = float(f.logKGlow);
    x[59] = float(f.logKPart);
    x[60] = float(f.logGhz);
    x[61] = float(f.logLatL3);
    x[62] = float(f.logLatDram);
    x[63] = float(f.logBwDram);
}

void FeatureState::update(const FrameRecord &d) {
    const double t = double(d.t_total_us) / 1000.0;
    const double tSteps = double(d.t_steps_us) / 1000.0, tRender = double(d.t_render_us) / 1000.0;
    tPrev2 = tPrev;
    tPrev = t;
    tEma = tEmaInit ? 0.8 * tEma + 0.2 * t : t;
    tEmaInit = true;
    last8[i8] = t;
    i8 = (i8 + 1) % 8;
    n8 = std::min(8, n8 + 1);
    tStepsPrev = tSteps;
    tRenderPrev = tRender;
    stepsPrev = d.n_steps;

    const double s = double(d.render_scale), s2 = s * s;
    const double rasterPx = double(d.px_rect) + double(d.px_circle) + double(d.px_ring) + double(d.px_line) +
                            double(d.px_poly);
    rasterLpx = rasterPx / s2;
    lpxRect = double(d.px_rect) / s2;
    lpxCircle = double(d.px_circle) / s2;
    lpxRing = double(d.px_ring) / s2;
    lpxLine = double(d.px_line) / s2;
    lpxPoly = double(d.px_poly) / s2;
    glowReqLpx = double(d.px_glow_req) / s2;
    if (d.n_steps >= 1) {
        double v = tSteps / double(d.n_steps);
        emaStep = emaStepInit ? 0.9 * emaStep + 0.1 * v : v;
        emaStepInit = true;
    }
    const double g = double(Quality::kGlow[std::min<int>(d.q_glow, Quality::kGlowLevels - 1)]);
    const double work = (double(d.px_clear) + rasterPx + double(d.px_glow_req) * g * g) / 1e6;
    if (work > 0) {
        double v = tRender / work;
        emaRpw = emaRpwInit ? 0.9 * emaRpw + 0.1 * v : v;
        emaRpwInit = true;
    }
}

void HistoryRing::push(const float *v) {
    std::memcpy(buf_[head_], v, sizeof buf_[0]);
    head_ = (head_ + 1) % kMaxHistory;
    n_ = std::min(kMaxHistory, n_ + 1);
}

} // namespace spark::ft
