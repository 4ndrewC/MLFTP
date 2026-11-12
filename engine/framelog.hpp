// Per-frame records for frame-time research (off unless --log is given).
//
// One fixed-size, packed, little-endian FrameRecord per loop iteration. The
// field list below is the single source of truth: it generates the struct and
// the field table written into each log's JSON header, so a reader never has
// to hard-code offsets. research/frametime/DATA_FORMAT.md documents the file.
//
// "when" says when a field's value is known:
//   id      bookkeeping
//   before  planned work, known before the frame runs (usable as a feature)
//   quality quality levels in effect for the frame (chosen before it runs)
//   system  machine state sampled before the frame runs
//   after   outcome, only known once the frame has run (targets, measured work)
//   pred    the predictor's outputs/decisions made before the frame ran
#pragma once

#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace spark {

// X(type, name, unit, when, description). Ordered by size so the struct has no padding.
#define SPARK_FRAME_FIELDS(X)                                                                                         \
    X(uint32_t, frame, "", "id", "loop iteration (frame) index within the session, from 0")                         \
    X(uint32_t, step0, "", "id", "engine fixed-step counter before this frame's steps")                             \
    X(float, t_total_us, "us", "after", "TARGET: sim steps + render CPU time (excludes predictor, present, sleep)")   \
    X(float, t_steps_us, "us", "after", "all fixed sim steps of this frame (incl. onStep hook / bot)")              \
    X(float, t_render_us, "us", "after", "render(): all scenes + overlay")                                          \
    X(float, t_present_us, "us", "after", "Window::present (0 when headless)")                                      \
    X(float, t_sleep_req_us, "us", "after", "sleep requested (windowed/paced); emulate mode: emulated overshoot")   \
    X(float, t_sleep_us, "us", "after", "sleep actually taken (windowed/paced)")                                    \
    X(float, t_onstep_us, "us", "after", "Config::onStep hook (the bot) summed over steps")                         \
    X(float, t_update_us, "us", "after", "top scene update() summed over steps (incl. collision/particles/flush)")   \
    X(float, t_collision_us, "us", "after", "collision system (spatial hash + responses), summed over steps")        \
    X(float, t_particles_us, "us", "after", "particle update, summed over steps")                                   \
    X(float, t_flush_us, "us", "after", "ECS deferred-destroy flush, summed over steps")                            \
    X(float, t_r_bg_us, "us", "after", "render: background (clear, grid, big glow, dust)")                          \
    X(float, t_r_world_us, "us", "after", "render: world entities (rings, telegraphs, pickups, shots, enemies...)") \
    X(float, t_r_particles_us, "us", "after", "render: particles")                                                  \
    X(float, t_r_hud_us, "us", "after", "render: popups + HUD + full-screen flashes")                               \
    X(float, t_pred_us, "us", "after", "predictor + controller for this frame (runs before the steps)")            \
    X(float, t_log_us, "us", "after", "logging/feature bookkeeping of the PREVIOUS frame (incl. buffer flush)")     \
    X(float, acc_ms, "ms", "before", "fixed-step accumulator before this frame's steps")                           \
    X(float, hitstop_ms, "ms", "before", "remaining hit-stop (frozen updates)")                                     \
    X(float, trauma, "", "before", "screen-shake trauma 0..1")                                                       \
    X(float, render_scale, "px/unit", "quality", "effective pixels per logical unit (stress out-scale x quality)")  \
    X(float, pred_p50, "ms", "pred", "model output t_p50_ms for the chosen quality (-1 = n/a)")                      \
    X(float, pred_p95, "ms", "pred", "model output t_p95_ms for the chosen quality (-1 = n/a)")                      \
    X(float, pred_p99, "ms", "pred", "model output t_p99_ms for the chosen quality (-1 = n/a)")                      \
    X(float, pred_pmiss, "prob", "pred", "model output p_miss for the chosen quality (-1 = n/a)")                    \
    X(float, pred_ctl, "ms", "pred", "value the controller compared to the budget (-1 = n/a)")                      \
    X(uint32_t, particles_alive, "", "before", "live particles")                                                     \
    X(uint32_t, px_clear, "px", "after", "pixels cleared (= render buffer size)")                                    \
    X(uint32_t, px_rect, "px", "after", "pixels covered by rects (incl. text glyphs, square particles)")            \
    X(uint32_t, px_circle, "px", "after", "filled-circle bounding-box pixels (AA-path box, also when AA is off)")   \
    X(uint32_t, px_ring, "px", "after", "ring bounding-box pixels (AA-path box, also when AA is off)")              \
    X(uint32_t, px_glow, "px", "after", "bounding-box pixels iterated by glows at the current glow quality")       \
    X(uint32_t, px_line, "px", "after", "line bounding-box pixels (AA-path box, also when AA is off; incl. streaks)") \
    X(uint32_t, px_poly, "px", "after", "polygon span pixels")                                                       \
    X(uint32_t, px_glow_req, "px", "after", "glow bounding-box pixels requested at FULL glow quality")              \
    X(uint16_t, entities, "", "before", "live ECS entities")                                                         \
    X(uint16_t, enemies, "", "before", "live enemies")                                                               \
    X(uint16_t, enemy_k0, "", "before", "enemies of kind 0 (Drifter)")                                              \
    X(uint16_t, enemy_k1, "", "before", "enemies of kind 1 (Dart)")                                                 \
    X(uint16_t, enemy_k2, "", "before", "enemies of kind 2 (Splitter)")                                             \
    X(uint16_t, enemy_k3, "", "before", "enemies of kind 3 (Mini)")                                                 \
    X(uint16_t, enemy_k4, "", "before", "enemies of kind 4 (Spinner)")                                              \
    X(uint16_t, projectiles, "", "before", "player bullets")                                                        \
    X(uint16_t, enemy_projectiles, "", "before", "enemy shots")                                                     \
    X(uint16_t, pickups, "", "before", "sparks (pickups)")                                                          \
    X(uint16_t, telegraphs, "", "before", "spawn warnings alive")                                                   \
    X(uint16_t, effects, "", "before", "ring effects alive")                                                        \
    X(uint16_t, popups, "", "before", "score popups alive")                                                         \
    X(uint16_t, tele_due_now, "", "before", "warnings that turn into enemies (+burst) during this frame's steps")  \
    X(uint16_t, tele_due_6, "", "before", "warnings expiring within 6 steps")                                       \
    X(uint16_t, tele_due_30, "", "before", "warnings expiring within 30 steps")                                     \
    X(uint16_t, burst_due, "particles", "before", "particles of bursts known to fire this frame, before quality density") \
    X(uint16_t, particles_streak, "", "before", "live particles drawn as lines")                                    \
    X(uint16_t, particles_expiring, "", "before", "live particles that die in the next update")                     \
    X(uint16_t, wave, "", "before", "wave number")                                                                   \
    X(uint16_t, wave_to_spawn, "", "before", "enemies still to be announced in this wave")                          \
    X(uint16_t, calls_rect, "", "after", "rect calls")                                                               \
    X(uint16_t, calls_circle, "", "after", "circle calls")                                                           \
    X(uint16_t, calls_ring, "", "after", "ring calls")                                                               \
    X(uint16_t, calls_glow, "", "after", "glow calls (incl. ones skipped by glow quality)")                         \
    X(uint16_t, calls_line, "", "after", "line calls")                                                               \
    X(uint16_t, calls_poly, "", "after", "polygon calls")                                                            \
    X(uint16_t, text_chars, "", "after", "characters of text drawn")                                                 \
    X(uint16_t, particles_spawned, "", "after", "particles spawned during this frame's steps")                      \
    X(uint16_t, bursts, "", "after", "particle bursts fired during this frame's steps")                             \
    X(uint16_t, colliders, "", "after", "colliders tested (last step of the frame)")                                \
    X(uint16_t, coll_pairs, "", "after", "overlapping pairs found, summed over steps")                              \
    X(uint16_t, cpu_mhz, "MHz", "system", "current core frequency (sampled every 8 frames)")                        \
    X(uint16_t, ctxsw, "", "system", "involuntary context switches of the main thread during the previous frame")  \
    X(uint16_t, minflt, "", "system", "minor page faults of the main thread during the previous frame")            \
    X(uint8_t, n_steps, "", "before", "fixed sim steps this frame will run (0, 1, 2, ...)")                          \
    X(uint8_t, n_frozen, "", "before", "of those, steps frozen by hit-stop (scene update skipped)")                  \
    X(uint8_t, scene, "", "before", "top scene: 0 other, 1 menu/title, 2 gameplay, 3 overlay (pause/game over)")    \
    X(uint8_t, flags, "", "id", "bit0 windowed, bit1 quality changed, bit2 prediction valid, bit3 on AC power, bit4 fingerprint loaded") \
    X(uint8_t, q_level, "", "quality", "quality ladder index (0 = best), 255 = off-ladder combination")              \
    X(uint8_t, q_scale, "", "quality", "render-scale level (0..3)")                                                  \
    X(uint8_t, q_particles, "", "quality", "particle-density level (0..3)")                                          \
    X(uint8_t, q_glow, "", "quality", "glow level (0..3)")                                                            \
    X(uint8_t, cpu, "", "system", "CPU the main thread ran on at frame start")                                       \
    X(uint8_t, contend, "", "system", "stress: 1 if an artificial contention thread was busy at frame start")       \
    X(uint8_t, ctl_mode, "", "pred", "controller: 0 fixed, 1 reactive, 2 model, 3 random")                          \
    X(uint8_t, ctl_evals, "", "pred", "model evaluations the controller made this frame")                           \
    X(uint8_t, ctl_action, "", "pred", "0 keep, 1 lowered quality, 2 raised quality")                               \
    X(uint8_t, q_aa, "", "quality", "anti-aliasing level: 0 on, 1 off (aliased spans for circles, rings, lines)")

struct FrameRecord {
#define SPARK_X(type, name, unit, when, desc) type name;
    SPARK_FRAME_FIELDS(SPARK_X)
#undef SPARK_X
};
static_assert(sizeof(FrameRecord) == 232, "FrameRecord layout changed: bump kFrameLogVersion and DATA_FORMAT.md");

// v1: phase-1 dataset (reserved0 instead of q_aa; no fingerprint). v2: q_aa, flags bits 3-4,
// header machine.fingerprint / session.fingerprint_key / session.corunner.
constexpr uint32_t kFrameLogVersion = 2;

struct FieldDesc {
    const char *name, *type, *unit, *when, *desc;
    size_t offset, size;
};
std::span<const FieldDesc> frameFields();

/** JSON string escaping for the log headers. */
std::string jsonEscape(const std::string &s);

/**
 * Binary table file: 24-byte fixed header, a JSON header, then fixed-size rows.
 *   0  char[8] magic        "SPKFLOG\0" (frame log) or "SPKFEAT\0" (feature dump)
 *   8  u32 format_version
 *  12  u32 record_size      bytes per row
 *  16  u32 header_bytes     JSON length
 *  20  u32 data_offset      first row; 24 + header_bytes rounded up to 8
 * Rows follow until EOF. A trailing partial row (crash) is ignored by readers.
 * Rows are buffered and written in blocks, off the timed part of the frame.
 */
class TableWriter {
public:
    TableWriter() = default;
    ~TableWriter() { close(); }
    TableWriter(const TableWriter &) = delete;
    TableWriter &operator=(const TableWriter &) = delete;

    bool open(const std::string &path, const char magic[8], uint32_t version, uint32_t recordSize,
              const std::string &json, std::string &err);
    void write(const void *row);
    void flush();
    void close();
    bool isOpen() const { return f_ != nullptr; }
    uint64_t rows() const { return rows_; }

private:
    FILE *f_ = nullptr;
    uint32_t recordSize_ = 0;
    std::vector<unsigned char> buf_;
    size_t used_ = 0;
    uint64_t rows_ = 0;
};

struct TableData {
    std::string magic, json;
    uint32_t version = 0, recordSize = 0;
    std::vector<unsigned char> rows; // rowCount * recordSize bytes
    size_t rowCount = 0;
};
/** Read a whole table file (tests/tools). */
bool readTable(const std::string &path, TableData &out, std::string &err);

/** The JSON header of a frame log: fields + machine + session. */
std::string frameLogHeaderJson(const std::string &machineJson, const std::string &sessionJson);

} // namespace spark
