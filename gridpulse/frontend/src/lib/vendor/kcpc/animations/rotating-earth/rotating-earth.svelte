<script lang="ts">
import { cn } from "$lib/vendor/kcpc/utils";
import * as d3 from "d3";
import type {
  Feature,
  FeatureCollection,
  MultiPolygon,
  Polygon,
  Position,
} from "geojson";
import { onMount } from "svelte";
import defaultWorldData from "./world-data.js";

type LngLat = [number, number];

const pointInPolygon = (point: LngLat, polygon: LngLat[]): boolean => {
  const [x, y] = point;
  let inside = false;

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const current = polygon[i];
    const previous = polygon[j];
    if (!current || !previous) continue;

    const [xi, yi] = current;
    const [xj, yj] = previous;

    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }

  return inside;
};

const toLngLatRing = (ring?: Position[]): LngLat[] => {
  if (!Array.isArray(ring)) return [];
  return ring
    .map((coord) =>
      Array.isArray(coord) && coord.length >= 2
        ? ([coord[0], coord[1]] as LngLat)
        : null,
    )
    .filter((coord): coord is LngLat => coord !== null);
};

const pointInFeature = (
  point: LngLat,
  feature: Feature<Polygon | MultiPolygon>,
): boolean => {
  const geometry = feature.geometry;

  if (geometry.type === "Polygon") {
    const coordinates = geometry.coordinates;
    if (!coordinates.length) return false;
    const [outerRaw, ...holesRaw] = coordinates;
    const outerRing = toLngLatRing(outerRaw);
    if (!outerRing.length || !pointInPolygon(point, outerRing)) return false;

    for (const hole of holesRaw) {
      const holeRing = toLngLatRing(hole);
      if (holeRing.length && pointInPolygon(point, holeRing)) {
        return false;
      }
    }
    return true;
  }

  if (geometry.type === "MultiPolygon") {
    for (const polygon of geometry.coordinates) {
      if (!polygon.length) continue;
      const [outerRaw, ...holesRaw] = polygon;
      const outerRing = toLngLatRing(outerRaw);
      if (!outerRing.length || !pointInPolygon(point, outerRing)) continue;

      let inHole = false;
      for (const hole of holesRaw) {
        const holeRing = toLngLatRing(hole);
        if (holeRing.length && pointInPolygon(point, holeRing)) {
          inHole = true;
          break;
        }
      }
      if (!inHole) return true;
    }
    return false;
  }

  return false;
};

const generateDotsInPolygon = (
  feature: Feature<Polygon | MultiPolygon>,
  dotSpacing = 16,
): LngLat[] => {
  const dots: LngLat[] = [];
  const bounds = d3.geoBounds(feature);
  const [[minLng, minLat], [maxLng, maxLat]] = bounds;

  const stepSize = Math.max(0.5, dotSpacing * 0.08);

  for (let lng = minLng; lng <= maxLng; lng += stepSize) {
    for (let lat = minLat; lat <= maxLat; lat += stepSize) {
      const point: [number, number] = [lng, lat];
      if (pointInFeature(point, feature)) {
        dots.push(point);
      }
    }
  }
  return dots;
};

let {
  width = 800,
  height = 600,
  scale,
  class: className = "",
  rotationSpeed = 0.3,
  autoRotate = true,
  halftoneDotDensity = 16,
  landFillColor = "#00000000",
  halftoneDotColor = "#999999",
  landOutlineColor = "#ffffff",
  graticuleColor = "#ffffff",
  coastColor = "#ffffff",
  scrollable = true,
  data = defaultWorldData,
}: {
  width?: number;
  height?: number;
  scale?: number;
  class?: string;
  rotationSpeed?: number;
  autoRotate?: boolean;
  halftoneDotDensity?: number;
  landFillColor?: string;
  halftoneDotColor?: string;
  landOutlineColor?: string;
  graticuleColor?: string;
  coastColor?: string;
  scrollable?: boolean;
  data?: FeatureCollection<Polygon | MultiPolygon>;
} = $props();

let canvasRef: HTMLCanvasElement | null = $state(null);

interface DotData {
  lng: number;
  lat: number;
  visible: boolean;
}

let allDots = $state<DotData[]>([]);

$effect(() => {
  if (!data?.features) return;
  const dots: DotData[] = [];
  for (const feature of data.features) {
    const featureDots = generateDotsInPolygon(feature, halftoneDotDensity);
    for (const [lng, lat] of featureDots) {
      dots.push({ lng, lat, visible: true });
    }
  }
  allDots = dots;
});

onMount(() => {
  if (!canvasRef) return;
  const canvas = canvasRef;
  const context = canvas.getContext("2d");
  if (!context) return;

  let containerWidth = Math.min(width, window.innerWidth - 60);
  let containerHeight = Math.min(height, window.innerHeight - 80);
  let baseRadius = scale ?? Math.min(containerWidth, containerHeight) / 2.2;

  const dpr = window.devicePixelRatio || 1;
  canvas.width = containerWidth * dpr;
  canvas.height = containerHeight * dpr;
  canvas.style.width = `${containerWidth}px`;
  canvas.style.height = `${containerHeight}px`;
  context.scale(dpr, dpr);

  const projection = d3
    .geoOrthographic()
    .scale(baseRadius)
    .translate([containerWidth / 2, containerHeight / 2])
    .clipAngle(90);

  const path = d3.geoPath().projection(projection).context(context);

  const render = () => {
    context.clearRect(0, 0, containerWidth, containerHeight);

    const currentScale = projection.scale();
    const scaleFactor = currentScale / baseRadius;

    // Ocean / sphere outline
    context.beginPath();
    context.arc(
      containerWidth / 2,
      containerHeight / 2,
      currentScale,
      0,
      2 * Math.PI,
    );
    context.fillStyle = landFillColor;
    context.fill();
    context.strokeStyle = coastColor;
    context.lineWidth = 1.75 * scaleFactor;
    context.stroke();

    if (data) {
      // Graticule
      const graticule = d3.geoGraticule();
      context.beginPath();
      path(graticule());
      context.strokeStyle = graticuleColor;
      context.lineWidth = 0.875 * scaleFactor;
      context.globalAlpha = 0.25;
      context.stroke();
      context.globalAlpha = 1;

      // Land outlines
      context.beginPath();
      for (const feature of data.features) {
        path(feature);
      }
      context.strokeStyle = landOutlineColor;
      context.lineWidth = 0.875 * scaleFactor;
      context.stroke();

      // Halftone dots
      for (const dot of allDots) {
        const projected = projection([dot.lng, dot.lat] as LngLat);
        if (!projected) continue;

        const [px, py] = projected;
        if (
          px >= 0 &&
          px <= containerWidth &&
          py >= 0 &&
          py <= containerHeight
        ) {
          context.beginPath();
          context.arc(px, py, 1.05 * scaleFactor, 0, 2 * Math.PI);
          context.fillStyle = halftoneDotColor;
          context.fill();
        }
      }
    }
  };

  // Sync scale if scale prop changes
  const updateScaleEffect = () => {
    if (scale !== undefined) {
      baseRadius = scale;
      projection.scale(scale);
      render();
    }
  };

  const rotation: [number, number, number] = [0, 0, 0];
  let isDragging = false;

  const rotate = () => {
    if (autoRotate && !isDragging) {
      rotation[0] += rotationSpeed;
      projection.rotate(rotation);
      render();
    }
  };

  const rotationTimer = d3.timer(rotate);

  const handleMouseDown = (event: MouseEvent) => {
    isDragging = true;
    const startX = event.clientX;
    const startY = event.clientY;
    const startRotation: [number, number, number] = [
      rotation[0],
      rotation[1],
      rotation[2],
    ];

    const handleMouseMove = (moveEvent: MouseEvent) => {
      const sensitivity = 0.5;
      const dx = moveEvent.clientX - startX;
      const dy = moveEvent.clientY - startY;

      rotation[0] = startRotation[0] + dx * sensitivity;
      rotation[1] = startRotation[1] - dy * sensitivity;
      rotation[1] = Math.max(-90, Math.min(90, rotation[1]));

      projection.rotate(rotation);
      render();
    };

    const handleMouseUp = () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      isDragging = false;
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  };

  const handleWheel = (event: WheelEvent) => {
    if (!scrollable) return;
    event.preventDefault();
    const scaleFactor = event.deltaY > 0 ? 0.9 : 1.1;
    const newRadius = Math.max(
      baseRadius * 0.5,
      Math.min(baseRadius * 3, projection.scale() * scaleFactor),
    );
    projection.scale(newRadius);
    render();
  };

  canvas.addEventListener("mousedown", handleMouseDown);
  canvas.addEventListener("wheel", handleWheel);

  render();

  return () => {
    rotationTimer.stop();
    canvas.removeEventListener("mousedown", handleMouseDown);
    canvas.removeEventListener("wheel", handleWheel);
  };
});
</script>

<canvas
  bind:this={canvasRef}
  class={cn("w-full h-auto cursor-grab active:cursor-grabbing", className)}
  style="max-width: 100%;"
></canvas>
