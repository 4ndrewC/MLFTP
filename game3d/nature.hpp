// Ambient nature scenery for Spark Drift: grassy terrain with hills on both sides of
// the flight corridor, pine and round trees, grass tufts, flowers, rocks, fireflies,
// a moon and parallax mountain silhouettes.
//
// Everything is placed from a hash of world-grid coordinates, never from an RNG, so
// the scenery stays put while the glider flies past it, is identical on every run,
// and does not touch the gameplay RNG. It is decoration only: trees stay outside
// the corridor (|x| > kClearX) and grass sits below the glider's lowest altitude,
// so nothing here takes part in collisions.
#pragma once

#include "canvas.hpp"
#include "profile.hpp"
#include "r3d.hpp"

namespace nature {

/** Half width of the flight corridor kept clear of trees and rocks. */
constexpr float kClearX = 13.5f;

/**
 * How much scenery to draw. The engine's four quality levers (scale, particles, glow, aa)
 * are reused in Spark Drift as:
 *   scale     -> internal resolution (handled by the engine's canvas; nothing to do here)
 *   glow      -> view distance: fog, terrain and trees reach 100 / 85 / 70 / 55 %
 *   particles -> ground detail: grass, flowers, rocks, fireflies at 100 / 60 / 35 / 15 % density
 *   aa        -> terrain detail: 4-unit tiles when on, 8-unit tiles when off
 */
struct Detail {
    float view = 1;        // view-distance factor
    float density = 1;     // ground-detail density
    float terrainCell = 4; // terrain tile size (world units)
};
Detail detailFor(const spark::Quality &q);

/** Terrain height at world (x, z): low ripples inside the corridor, rolling hills outside it. */
float height(float x, float z);

/** Sky details drawn straight after Renderer::beginFrame and before any 3D: moon, mountains. */
void drawSky(spark::Canvas &cv, float camX, float t);

/**
 * All 3D scenery around world depth `pz` (the glider's z; the camera sits behind it):
 * terrain, trees, rocks, grass, flowers and fireflies. Call once per frame, after drawSky.
 */
void drawWorld(r3d::Renderer &rend, float pz, float t, const Detail &d = {});

/** Firefly halos (2D, additive). Call after all 3D drawing, before the HUD. */
void drawGlows(spark::Canvas &cv, const r3d::Renderer &rend, float pz, float t, const Detail &d = {});

/** Fog for the nature scenes (a bit longer than the original corridor fog). */
r3d::Fog fog(const Detail &d = {});

/** Sky colours. */
r3d::Rgb skyTop();
r3d::Rgb skyHorizon();

} // namespace nature
