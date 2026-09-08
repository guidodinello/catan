// Axial/cube -> pixel layout math for the hex board.
//
// engine/board.py's Cube = (q, r, s) with q + r + s === 0 (a plain 3-tuple,
// see engine/board.py's `Cube` alias and `DIRECTIONS`). This module renders
// them as **flat-top** hexagons -- a flat edge on top, matching the real
// Catan board tiles (and every popular web implementation) -- using the
// standard axial-to-pixel formula for that orientation (redblobgames.com/
// grids/hexagons/#hex-to-pixel, "flat" layout):
//   x = size * (3/2 * q)
//   y = size * (sqrt(3)/2 * q + sqrt(3) * r)
// `s` never enters the formula (it's redundant, q + r + s = 0), matching
// engine/board.py itself only ever indexing hexes by the full 3-tuple for
// dict-key convenience, never by an independent `s`.
//
// A hex's 6 corners are never computed from a separate trig formula here.
// Every corner is also a *vertex* shared by up to 3 hexes (engine/board.py's
// `GEOMETRY.vertex_hexes`), and for a regular hex grid embedding, a shared
// vertex sits exactly at the centroid of its (up to 3) adjacent hex centers
// -- each such center is at distance `size` (the hex's circumradius) from
// the vertex, and points equidistant from a common point that also form an
// equilateral triangle (guaranteed by hex-grid 3-fold symmetry) have that
// common point as their centroid. So `vertexToPixel` is just an average of
// `hexToPixel` over the vertex's hexes, and a hex polygon's outline is just
// its own 6 vertices' pixels, in the order the server already sends them --
// no independent corner-angle formula to keep in sync with the direction
// vectors.

import type { Geometry, GeometryEdge, GeometryVertex } from "./api";

export interface PixelPoint {
  x: number;
  y: number;
}

const SQRT3 = Math.sqrt(3);

export function hexToPixel(
  hex: readonly [number, number, number],
  size: number,
): PixelPoint {
  const [q, r] = hex;
  return {
    x: size * (1.5 * q),
    y: size * ((SQRT3 / 2) * q + SQRT3 * r),
  };
}

function average(points: PixelPoint[]): PixelPoint {
  const n = points.length;
  return {
    x: points.reduce((sum, p) => sum + p.x, 0) / n,
    y: points.reduce((sum, p) => sum + p.y, 0) / n,
  };
}

export function vertexToPixel(vertex: GeometryVertex, size: number): PixelPoint {
  return average(vertex.hexes.map((h) => hexToPixel(h, size)));
}

/** Every vertex's pixel position, indexed by `vertex_id`. */
export function vertexPixels(geometry: Geometry, size: number): PixelPoint[] {
  const byId = new Array<PixelPoint>(geometry.vertices.length);
  for (const vertex of geometry.vertices) {
    byId[vertex.vertex_id] = vertexToPixel(vertex, size);
  }
  return byId;
}

/** An edge's two endpoints, as pixel points, looked up from `vertexPixels`. */
export function edgeEndpoints(
  edge: GeometryEdge,
  pixelsByVertexId: PixelPoint[],
): [PixelPoint, PixelPoint] {
  const [v1, v2] = edge.vertices;
  return [pixelsByVertexId[v1], pixelsByVertexId[v2]];
}

/** A hex's 6 corners, as pixel points in polygon-drawing order. */
export function hexCorners(
  vertexIds: readonly number[],
  pixelsByVertexId: PixelPoint[],
): PixelPoint[] {
  return vertexIds.map((id) => pixelsByVertexId[id]);
}

export function pointsAttribute(points: PixelPoint[]): string {
  return points.map((p) => `${p.x},${p.y}`).join(" ");
}
