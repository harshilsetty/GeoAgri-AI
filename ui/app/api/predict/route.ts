import { writeFile, unlink } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type PythonResult = {
  id?: number;
  type?: string;
  ok: boolean;
  data?: unknown;
  error?: string;
  traceback?: string;
};

type ImageKey = "original" | "detection" | "segmentation" | "combined" | "heatmap";

type ImageBundle = Record<ImageKey, string>;

type BatchPoint = {
  lat: number;
  lon: number;
};

type StaticSnapshotRequest = {
  lat: number;
  lon: number;
  zoom?: number;
};

const IMAGE_TTL_MS = 10 * 60 * 1000;
const IMAGE_KEYS: ImageKey[] = ["original", "detection", "segmentation", "combined", "heatmap"];
const AUTO_CAPTURE_MAX_RADIUS_METERS = 300;
const AUTO_CAPTURE_IMAGE_SIZE_PX = 500;
const AUTO_CAPTURE_ZOOM = 18;
const REMOTE_PREDICT_URL = (process.env.PREDICT_BACKEND_URL || process.env.REMOTE_PREDICT_URL || "")
  .trim()
  .replace(/\/+$/, "");

const workspaceRoot = path.resolve(process.cwd(), "..");
const workerScript = path.join(process.cwd(), "server", "predict_worker.py");

let worker: ChildProcessWithoutNullStreams | null = null;
let nextId = 1;
let lineBuffer = "";

function normalizeApiKey(raw: string): string {
  const trimmed = raw.trim().replace(/^['\"]|['\"]$/g, "");
  return trimmed.replace(/^Bearer\s+/i, "").replace(/\s+/g, "");
}

function parseProbabilityOrThrow(value: unknown, context: string): number {
  const probability = Number(value);
  if (!Number.isFinite(probability)) {
    throw new Error(`Prediction response contained a non-finite probability in ${context}`);
  }
  if (probability < 0 || probability > 1) {
    throw new Error(`Prediction response probability out of range [0,1] in ${context}: ${probability}`);
  }
  return probability;
}

const pending = new Map<
  number,
  {
    resolve: (value: PythonResult) => void;
    reject: (error: Error) => void;
    timeout: NodeJS.Timeout;
  }
>();

const imageCache = new Map<
  string,
  {
    createdAt: number;
    images: ImageBundle;
  }
>();

function cleanupImageCache() {
  const now = Date.now();
  for (const [jobId, entry] of imageCache.entries()) {
    if (now - entry.createdAt > IMAGE_TTL_MS) {
      imageCache.delete(jobId);
    }
  }
}

function toImageBundle(value: unknown): ImageBundle {
  const src = (value || {}) as Partial<ImageBundle>;
  return {
    original: typeof src.original === "string" ? src.original : "",
    detection: typeof src.detection === "string" ? src.detection : "",
    segmentation: typeof src.segmentation === "string" ? src.segmentation : "",
    combined: typeof src.combined === "string" ? src.combined : "",
    heatmap: typeof src.heatmap === "string" ? src.heatmap : "",
  };
}

function rejectAllPending(message: string) {
  for (const [id, entry] of pending.entries()) {
    clearTimeout(entry.timeout);
    entry.reject(new Error(`${message} (request ${id})`));
    pending.delete(id);
  }
}

function shutdownWorker() {
  if (worker && !worker.killed) {
    worker.kill();
  }
  worker = null;
  lineBuffer = "";
}

function shouldProxyToRemote() {
  return REMOTE_PREDICT_URL.length > 0;
}

function shouldRequireRemoteBackendInServerless() {
  return process.env.VERCEL === "1" && !shouldProxyToRemote();
}

function buildRemotePredictUrl(nextUrl: URL): URL {
  const remoteUrl = new URL(REMOTE_PREDICT_URL);
  remoteUrl.search = "";

  nextUrl.searchParams.forEach((value, key) => {
    remoteUrl.searchParams.append(key, value);
  });

  return remoteUrl;
}

async function proxyRequestToRemote(request: NextRequest): Promise<NextResponse> {
  const target = buildRemotePredictUrl(request.nextUrl);
  const headers = new Headers();

  const contentType = request.headers.get("content-type");
  if (contentType) {
    headers.set("content-type", contentType);
  }

  const accept = request.headers.get("accept");
  if (accept) {
    headers.set("accept", accept);
  }

  const auth = request.headers.get("authorization");
  if (auth) {
    headers.set("authorization", auth);
  }

  const body = request.method === "GET" || request.method === "HEAD" ? undefined : await request.arrayBuffer();

  const upstream = await fetch(target, {
    method: request.method,
    headers,
    body,
    cache: "no-store",
  });

  const responseHeaders = new Headers();
  const responseType = upstream.headers.get("content-type");
  if (responseType) {
    responseHeaders.set("content-type", responseType);
  }

  const responseCache = upstream.headers.get("cache-control");
  if (responseCache) {
    responseHeaders.set("cache-control", responseCache);
  }

  const payload = await upstream.arrayBuffer();
  return new NextResponse(payload, { status: upstream.status, headers: responseHeaders });
}

function ensureWorker() {
  if (worker && !worker.killed) {
    return worker;
  }

  const pythonExe = process.env.PYTHON_EXECUTABLE || "python";
  try {
    worker = spawn(pythonExe, [workerScript], { cwd: workspaceRoot });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    throw new Error(
      `Failed to start Python worker (${message}). Set PREDICT_BACKEND_URL in production to use an external Python inference API.`
    );
  }
  lineBuffer = "";

  worker.stdout.on("data", (chunk) => {
    lineBuffer += chunk.toString();
    const lines = lineBuffer.split(/\r?\n/);
    lineBuffer = lines.pop() || "";

    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line || !line.startsWith("{")) {
        continue;
      }

      try {
        const parsed = JSON.parse(line) as PythonResult;
        if (parsed.type === "ready") {
          continue;
        }

        if (typeof parsed.id !== "number") {
          continue;
        }

        const job = pending.get(parsed.id);
        if (!job) {
          continue;
        }

        clearTimeout(job.timeout);
        pending.delete(parsed.id);
        job.resolve(parsed);
      } catch {
        // Ignore malformed worker lines.
      }
    }
  });

  worker.stderr.on("data", () => {
    // stderr is intentionally ignored here; failures are returned in worker JSON responses.
  });

  worker.on("exit", () => {
    worker = null;
    rejectAllPending("Python worker exited");
  });

  worker.on("error", (err) => {
    rejectAllPending(
      `Python worker error: ${err.message}. Set PREDICT_BACKEND_URL in production to use an external Python inference API.`
    );
  });

  return worker;
}

function runWorkerJob(payload: Omit<Record<string, unknown>, "id">): Promise<PythonResult> {
  const proc = ensureWorker();
  const id = nextId++;

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      pending.delete(id);
      reject(new Error("Prediction timed out"));
    }, 240000);

    pending.set(id, { resolve, reject, timeout });
    proc.stdin.write(`${JSON.stringify({ id, ...payload })}\n`);
  });
}

async function downloadStaticSnapshot({ lat, lon, zoom = 14 }: StaticSnapshotRequest): Promise<Uint8Array> {
  const clampedZoom = AUTO_CAPTURE_ZOOM;
  const imageSize = AUTO_CAPTURE_IMAGE_SIZE_PX;
  const center = `${lat.toFixed(6)},${lon.toFixed(6)}`;
  const marker = `${center},red-pushpin`;

  const degreesPerPixel = 360 / (2 ** clampedZoom * 256);
  const halfWidthPixels = imageSize / 2;
  const halfHeightPixels = imageSize / 2;
  const deltaLon = degreesPerPixel * halfWidthPixels;
  const deltaLat = degreesPerPixel * halfHeightPixels;
  const minLon = lon - deltaLon;
  const maxLon = lon + deltaLon;
  const minLat = lat - deltaLat;
  const maxLat = lat + deltaLat;

  const candidateUrls = [
    `https://staticmap.openstreetmap.de/staticmap.php?center=${encodeURIComponent(center)}&zoom=${clampedZoom}&size=${imageSize}x${imageSize}&markers=${encodeURIComponent(marker)}`,
    `https://maps.wikimedia.org/img/osm-intl,${clampedZoom},${lat.toFixed(6)},${lon.toFixed(6)},${imageSize}x${imageSize}.png`,
    `https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox=${minLon.toFixed(6)},${minLat.toFixed(6)},${maxLon.toFixed(6)},${maxLat.toFixed(6)}&bboxSR=4326&imageSR=4326&size=${imageSize},${imageSize}&format=png32&transparent=false&f=image`,
  ];

  let lastError = "";

  for (const url of candidateUrls) {
    let timeout: ReturnType<typeof setTimeout> | null = null;
    try {
      const controller = new AbortController();
      timeout = setTimeout(() => controller.abort(), 12000);

      const response = await fetch(url, {
        cache: "no-store",
        signal: controller.signal,
        headers: {
          "User-Agent": "GeoAI-UI/1.0",
          Accept: "image/png,image/jpeg,image/*;q=0.8,*/*;q=0.5",
        },
      });

      if (!response.ok) {
        lastError = `HTTP ${response.status} from ${new URL(url).hostname}`;
        continue;
      }

      const contentType = String(response.headers.get("content-type") || "").toLowerCase();
      if (!contentType.includes("image")) {
        lastError = `non-image response (${contentType || "unknown content-type"})`;
        continue;
      }

      const buffer = await response.arrayBuffer();
      if (buffer.byteLength < 1024) {
        lastError = `response image too small from ${new URL(url).hostname}`;
        continue;
      }

      return new Uint8Array(buffer);
    } catch (err) {
      lastError = err instanceof Error ? err.message : String(err);
    } finally {
      if (timeout) {
        clearTimeout(timeout);
      }
    }
  }

  throw new Error(`Failed to capture map snapshot (<=${AUTO_CAPTURE_MAX_RADIUS_METERS}m mode). ${lastError || "All providers failed."}`);
}

export async function GET(request: NextRequest) {
  cleanupImageCache();

  const action = request.nextUrl.searchParams.get("action");
  const isLocalGetAction = action === "snapshot" || action === "image";

  if (!isLocalGetAction && shouldRequireRemoteBackendInServerless()) {
    return NextResponse.json(
      {
        error: "Prediction backend not configured",
        details: "Set PREDICT_BACKEND_URL (or REMOTE_PREDICT_URL) to your deployed Python inference endpoint.",
      },
      { status: 503 },
    );
  }

  if (!isLocalGetAction && shouldProxyToRemote()) {
    try {
      return await proxyRequestToRemote(request);
    } catch (err) {
      return NextResponse.json(
        {
          error: "Remote prediction service request failed",
          details: err instanceof Error ? err.message : String(err),
        },
        { status: 502 },
      );
    }
  }

  if (action === "snapshot") {
    try {
      const lat = Number(request.nextUrl.searchParams.get("lat"));
      const lon = Number(request.nextUrl.searchParams.get("lon"));
      const zoom = Number(request.nextUrl.searchParams.get("zoom"));

      if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
        return NextResponse.json({ error: "lat and lon are required" }, { status: 400 });
      }

      const snapshot = await downloadStaticSnapshot({
        lat,
        lon,
        zoom: Number.isFinite(zoom) ? zoom : AUTO_CAPTURE_ZOOM,
      });

      return new NextResponse(snapshot, {
        status: 200,
        headers: {
          "Content-Type": "image/png",
          "Cache-Control": "no-store, max-age=0",
        },
      });
    } catch (err) {
      return NextResponse.json(
        {
          error: "Failed to capture map snapshot",
          details: err instanceof Error ? err.message : String(err),
        },
        { status: 500 },
      );
    }
  }

  if (action !== "image") {
    return NextResponse.json({ error: "Unsupported action" }, { status: 400 });
  }

  const jobId = String(request.nextUrl.searchParams.get("jobId") || "").trim();
  const imageName = String(request.nextUrl.searchParams.get("name") || "").trim() as ImageKey;

  if (!jobId || !IMAGE_KEYS.includes(imageName)) {
    return NextResponse.json({ error: "jobId and valid image name are required" }, { status: 400 });
  }

  const cached = imageCache.get(jobId);
  if (!cached) {
    return NextResponse.json({ error: "Image job expired or not found" }, { status: 404 });
  }

  const src = cached.images[imageName] || "";

  return NextResponse.json(
    {
      name: imageName,
      src,
    },
    { status: 200 },
  );
}

export async function POST(request: NextRequest) {
  if (shouldRequireRemoteBackendInServerless()) {
    return NextResponse.json(
      {
        error: "Prediction backend not configured",
        details: "Set PREDICT_BACKEND_URL (or REMOTE_PREDICT_URL) to your deployed Python inference endpoint.",
      },
      { status: 503 },
    );
  }

  if (shouldProxyToRemote()) {
    try {
      return await proxyRequestToRemote(request);
    } catch (err) {
      return NextResponse.json(
        {
          error: "Remote prediction service request failed",
          details: err instanceof Error ? err.message : String(err),
        },
        { status: 502 },
      );
    }
  }

  cleanupImageCache();

  const action = request.nextUrl.searchParams.get("action") || "predict";

  if (action === "point") {
    try {
      const payload = (await request.json()) as {
        lat?: number;
        lon?: number;
        apiKey?: string;
        useAiInsight?: boolean;
        includeShap?: boolean;
        vegetation?: number;
        boulders?: number;
        ruins?: number;
        structures?: number;
      };

      const lat = Number(payload.lat);
      const lon = Number(payload.lon);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
        return NextResponse.json({ error: "Latitude and longitude must be valid numbers" }, { status: 400 });
      }

      const parsed = await runWorkerJob({
        action: "point",
        lat,
        lon,
        apiKey: normalizeApiKey(String(payload.apiKey || "")),
        useAiInsight: payload.useAiInsight === true,
        includeShap: payload.includeShap !== false,
        vegetation: Number.isFinite(Number(payload.vegetation)) ? Number(payload.vegetation) : undefined,
        boulders: Number.isFinite(Number(payload.boulders)) ? Number(payload.boulders) : undefined,
        ruins: Number.isFinite(Number(payload.ruins)) ? Number(payload.ruins) : undefined,
        structures: Number.isFinite(Number(payload.structures)) ? Number(payload.structures) : undefined,
      });

      if (!parsed.ok) {
        return NextResponse.json(
          {
            error: parsed.error || "Point inference failed",
            details: parsed.traceback || "Worker returned an error",
          },
          { status: 500 },
        );
      }
      const pointData = (parsed.data || {}) as Record<string, unknown>;
      const probability = parseProbabilityOrThrow(pointData.probability, "point");
      return NextResponse.json({ ...pointData, probability }, { status: 200 });
    } catch (err) {
      return NextResponse.json(
        {
          error: "Unexpected point API failure",
          details: err instanceof Error ? err.message : String(err),
        },
        { status: 500 },
      );
    }
  }

  if (action === "batch") {
    const form = await request.formData();
    const image = form.get("image");
    const pointsRaw = String(form.get("points") || "[]");
    const apiKey = normalizeApiKey(String(form.get("apiKey") || ""));
    const confidence = Number(form.get("confidence"));
    const showVegetation = String(form.get("showVegetation") || "true");
    const showRuins = String(form.get("showRuins") || "true");
    const showStructures = String(form.get("showStructures") || "true");
    const showBoulders = String(form.get("showBoulders") || "true");
    const showOthers = String(form.get("showOthers") || "true");
    const fastMode = String(form.get("fastMode") || "true");
    const useAiInsight = String(form.get("useAiInsight") || "true");
    const includeShap = String(form.get("includeShap") || "true");
    const autoCapture = String(form.get("autoCapture") || "false");
    const centerLat = Number(form.get("centerLat"));
    const centerLon = Number(form.get("centerLon"));
    const mapZoom = Number(form.get("mapZoom"));

    const hasUploadedImage = image instanceof File;
    const canAutoCapture =
      !hasUploadedImage &&
      ["1", "true", "yes", "on"].includes(autoCapture.toLowerCase()) &&
      Number.isFinite(centerLat) &&
      Number.isFinite(centerLon);

    if (!hasUploadedImage && !canAutoCapture) {
      return NextResponse.json({ error: "Image file is required (or enable autoCapture with valid centerLat/centerLon)." }, { status: 400 });
    }

    let parsedPoints: BatchPoint[] = [];
    try {
      const raw = JSON.parse(pointsRaw) as Array<{ lat?: unknown; lon?: unknown }>;
      parsedPoints = Array.isArray(raw)
        ? raw
          .map((item) => ({ lat: Number(item?.lat), lon: Number(item?.lon) }))
          .filter((item) => Number.isFinite(item.lat) && Number.isFinite(item.lon))
        : [];
    } catch {
      parsedPoints = [];
    }

    if (parsedPoints.length === 0) {
      return NextResponse.json({ error: "At least one valid point is required" }, { status: 400 });
    }

    const cappedPoints = parsedPoints.slice(0, 30);
    const tempFile = hasUploadedImage
      ? path.join(os.tmpdir(), `geo-ai-batch-${Date.now()}-${(image as File).name}`)
      : path.join(os.tmpdir(), `geo-ai-batch-${Date.now()}-autosnapshot.png`);

    try {
      if (hasUploadedImage) {
        const bytes = await (image as File).arrayBuffer();
        await writeFile(tempFile, new Uint8Array(bytes));
      } else {
        const snapshot = await downloadStaticSnapshot({
          lat: centerLat,
          lon: centerLon,
          zoom: Number.isFinite(mapZoom) ? mapZoom : 14,
        });
        await writeFile(tempFile, snapshot);
      }

      const results: Array<{
        lat: number;
        lon: number;
        probability: number;
        riskLabel: string;
        explanation?: string;
        shap?: unknown;
        insightMode?: string | null;
        insightStatus?: string | null;
      }> = [];

      for (const point of cappedPoints) {
        const parsed = await runWorkerJob({
          action: "predict",
          image: tempFile,
          lat: point.lat,
          lon: point.lon,
          apiKey,
          confidence: Number.isFinite(confidence) ? confidence : 0.25,
          classVisibility: {
            vegetation: showVegetation,
            ruins: showRuins,
            structures: showStructures,
            boulders: showBoulders,
            others: showOthers,
          },
          useAiInsight,
          includeShap,
          fastMode,
        });

        if (!parsed.ok) {
          return NextResponse.json(
            {
              error: parsed.error || "Batch inference failed",
              details: parsed.traceback || "Worker returned an error",
            },
            { status: 500 },
          );
        }

        const data = (parsed.data || {}) as {
          probability?: unknown;
          riskLabel?: unknown;
          explanation?: unknown;
          shap?: unknown;
          insightMode?: unknown;
          insightStatus?: unknown;
        };
        const probability = parseProbabilityOrThrow(data.probability, "batch");
        const riskLabel = typeof data.riskLabel === "string" ? data.riskLabel : "MODERATE";

        results.push({
          lat: point.lat,
          lon: point.lon,
          probability,
          riskLabel,
          explanation: typeof data.explanation === "string" ? data.explanation : undefined,
          shap: data.shap,
          insightMode: typeof data.insightMode === "string" ? data.insightMode : null,
          insightStatus: typeof data.insightStatus === "string" ? data.insightStatus : null,
        });
      }

      return NextResponse.json(
        {
          points: results,
          truncated: parsedPoints.length > cappedPoints.length,
        },
        { status: 200 },
      );
    } catch (err) {
      return NextResponse.json(
        {
          error: "Unexpected batch API failure",
          details: err instanceof Error ? err.message : String(err),
        },
        { status: 500 },
      );
    } finally {
      try {
        await unlink(tempFile);
      } catch {
        // Ignore temp file cleanup errors.
      }
    }
  }

  if (action === "insight") {
    try {
      const payload = (await request.json()) as {
        metrics?: unknown;
        probability?: number;
        apiKey?: string;
        useAiInsight?: boolean;
        includeShap?: boolean;
      };

      const parsed = await runWorkerJob({
        action: "insight",
        metrics: payload.metrics || {},
        probability: Number.isFinite(payload.probability) ? payload.probability : 0,
        apiKey: normalizeApiKey(String(payload.apiKey || "")),
        useAiInsight: payload.useAiInsight !== false,
        includeShap: payload.includeShap !== false,
      });

      if (!parsed.ok) {
        return NextResponse.json(
          {
            error: parsed.error || "Insight generation failed",
            details: parsed.traceback || "Worker returned an error",
          },
          { status: 500 },
        );
      }

      return NextResponse.json(parsed.data, { status: 200 });
    } catch (err) {
      return NextResponse.json(
        {
          error: "Unexpected insight API failure",
          details: err instanceof Error ? err.message : String(err),
        },
        { status: 500 },
      );
    }
  }

  if (action === "crop" || action === "agriculture") {
    try {
      const payload = (await request.json()) as {
        lat?: number;
        latitude?: number;
        lon?: number;
        longitude?: number;
        season?: string;
        apiKey?: string;
        topK?: number;
        vegetation?: number;
      };

      const lat = Number(payload.lat ?? payload.latitude ?? 15.335);
      const lon = Number(payload.lon ?? payload.longitude ?? 76.46);

      const parsed = await runWorkerJob({
        action: "crop",
        lat,
        lon,
        season: String(payload.season || "Kharif"),
        apiKey: normalizeApiKey(String(payload.apiKey || "")),
        topK: Number(payload.topK || 5),
        vegetation: Number(payload.vegetation ?? 0.35),
      });

      if (!parsed.ok) {
        return NextResponse.json(
          {
            error: parsed.error || "Crop recommendation failed",
            details: parsed.traceback || "Worker returned an error",
          },
          { status: 500 },
        );
      }

      return NextResponse.json(parsed.data, { status: 200 });
    } catch (err) {
      return NextResponse.json(
        {
          error: "Unexpected crop recommendation API failure",
          details: err instanceof Error ? err.message : String(err),
        },
        { status: 500 },
      );
    }
  }

  const form = await request.formData();
  const image = form.get("image");
  const latRaw = form.get("lat");
  const lonRaw = form.get("lon");
  const apiKey = normalizeApiKey(String(form.get("apiKey") || ""));
  const confidence = Number(form.get("confidence"));
  const showVegetation = String(form.get("showVegetation") || "true");
  const showRuins = String(form.get("showRuins") || "true");
  const showStructures = String(form.get("showStructures") || "true");
  const showBoulders = String(form.get("showBoulders") || "true");
  const showOthers = String(form.get("showOthers") || "true");
  const useAiInsight = String(form.get("useAiInsight") || "false");
  const includeShap = String(form.get("includeShap") || "false");
  const fastMode = String(form.get("fastMode") || "true");
  const autoCapture = String(form.get("autoCapture") || "false");
  const mapZoom = Number(form.get("mapZoom"));

  const hasUploadedImage = image instanceof File;

  const lat = Number(latRaw);
  const lon = Number(lonRaw);

  if (Number.isNaN(lat) || Number.isNaN(lon)) {
    return NextResponse.json({ error: "Latitude and longitude must be valid numbers" }, { status: 400 });
  }

  const canAutoCapture =
    !hasUploadedImage &&
    ["1", "true", "yes", "on"].includes(autoCapture.toLowerCase()) &&
    Number.isFinite(lat) &&
    Number.isFinite(lon);

  if (!hasUploadedImage && !canAutoCapture) {
    return NextResponse.json({ error: "Image file is required (or enable autoCapture with valid coordinates)" }, { status: 400 });
  }

  const tempFile = hasUploadedImage
    ? path.join(os.tmpdir(), `geo-ai-${Date.now()}-${(image as File).name}`)
    : path.join(os.tmpdir(), `geo-ai-${Date.now()}-autosnapshot.png`);

  try {
    if (hasUploadedImage) {
      const bytes = await (image as File).arrayBuffer();
      await writeFile(tempFile, new Uint8Array(bytes));
    } else {
      const snapshot = await downloadStaticSnapshot({
        lat,
        lon,
        zoom: Number.isFinite(mapZoom) ? mapZoom : 14,
      });
      await writeFile(tempFile, snapshot);
    }

    const parsed = await runWorkerJob({
      action: "predict",
      image: tempFile,
      lat,
      lon,
      apiKey,
      confidence: Number.isFinite(confidence) ? confidence : 0.25,
      classVisibility: {
        vegetation: showVegetation,
        ruins: showRuins,
        structures: showStructures,
        boulders: showBoulders,
        others: showOthers,
      },
      useAiInsight,
      includeShap,
      fastMode,
    });

    if (!parsed.ok) {
      return NextResponse.json(
        {
          error: parsed.error || "Inference failed",
          details: parsed.traceback || "Worker returned an error",
        },
        { status: 500 },
      );
    }

    const rawData = (parsed.data || {}) as Record<string, unknown>;
    const probability = parseProbabilityOrThrow(rawData.probability, "predict");
    const rawImages = toImageBundle(rawData.images);
    const imageJobId = `img_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;

    imageCache.set(imageJobId, { createdAt: Date.now(), images: rawImages });

    const nextData = {
      ...rawData,
      probability,
      imageJobId,
      pendingImages: IMAGE_KEYS.filter((key) => key !== "original" && Boolean(rawImages[key])),
      images: {
        original: rawImages.original,
        detection: "",
        segmentation: "",
        combined: "",
        heatmap: "",
      },
    };

    return NextResponse.json(nextData, { status: 200 });
  } catch (err) {
    return NextResponse.json(
      {
        error: "Unexpected API failure",
        details: err instanceof Error ? err.message : String(err),
      },
      { status: 500 },
    );
  } finally {
    try {
      await unlink(tempFile);
    } catch {
      // Ignore temp file cleanup errors.
    }
  }
}
