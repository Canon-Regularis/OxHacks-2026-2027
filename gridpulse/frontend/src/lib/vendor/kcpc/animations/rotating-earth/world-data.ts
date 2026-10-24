import type { FeatureCollection, Polygon, MultiPolygon } from "geojson";
import worldData from "./world.json";

export const defaultWorldData = worldData as unknown as FeatureCollection<
  Polygon | MultiPolygon
>;

export default defaultWorldData;
