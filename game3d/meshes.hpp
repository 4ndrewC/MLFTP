// Procedurally generated meshes for Spark Drift. No asset files.
#pragma once

#include "r3d.hpp"

namespace shapes {

/** Scrolling ground: a checkerboard of `cols` x `rows` tiles of side `cell` in local space, from (x0, z0). */
r3d::Mesh makeGround(float cell, int cols, int rows, float x0, float z0, r3d::Rgb a, r3d::Rgb b);

/** Hexagonal bipyramid "crystal" pylon: equator radius `radius`, tips at +-halfHeight on y. 12 triangles. */
r3d::Mesh makeCrystal(float radius, float halfHeight, r3d::Rgb col);

/** Glider seen from behind: nose at -z, tail at +z (it flies towards -z). About 3 units long. */
r3d::Mesh makeShip(r3d::Rgb col);

/** Octahedron (spark, shot): tips at +-size on each axis. 8 triangles. */
r3d::Mesh makeOcta(float size, r3d::Rgb col);

/** Drone: a box body with two thin wings. */
r3d::Mesh makeDrone(r3d::Rgb col);

/** The meshes the game uses, built once. */
struct Models {
    r3d::Mesh ground, crystal, ship, spark, shot, drone;
};
const Models &models();

} // namespace shapes
