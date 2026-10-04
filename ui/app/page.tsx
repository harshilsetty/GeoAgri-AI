"use client";

import dynamic from "next/dynamic";
import { AnimatePresence, MotionConfig, motion } from "framer-motion";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import html2canvas from "html2canvas";
import jsPDF from "jspdf";
import { Bell, BrainCircuit, CheckCircle2, Crosshair, FileDown, ImageDown, Loader2, Settings, UploadCloud, X, Zap } from "lucide-react";
import HeroRiskCard from "@/components/dashboard/HeroRiskCard";
import AnalystPanel from "@/components/dashboard/AnalystPanel";
import MetricsPanel from "@/components/dashboard/MetricsPanel";
import ShapPanel from "@/components/dashboard/ShapPanel";
import ReportDocument from "@/components/dashboard/ReportDocument";
import VisualAnalysisGrid, { type VisualTile } from "@/components/dashboard/VisualAnalysisGrid";
import CropIntelligencePanel, { type CropIntelligenceData } from "@/components/dashboard/CropIntelligencePanel";

const MapPicker = dynamic(() => import("@/components/MapPicker"), { ssr: false });

type PredictionResult = {
  probability: number;
  riskLabel: "LOW" | "MODERATE" | "HIGH";
  explanation: string;
  insightMode?: string;
  insightStatus?: string | null;
  imageJobId?: string;
  pendingImages?: Array<"detection" | "segmentation" | "combined" | "heatmap">;
  metrics: {
    vegetation: number;
    slope: number;
    rainfall: number;
    elevation: number;
    soil: string;
    boulders: number;
    ruins: number;
    structures: number;
    lat: number;
    lon: number;
  };
  shap: { feature: string; value: number }[];
  images: {
    original: string;
    detection: string;
    segmentation: string;
    combined: string;
    heatmap: string;
  };
};

type NotificationItem = {
  id: string;
  message: string;
  timestamp: string;
};

type PersistedSettings = {
  confidence: number;
  showVegetation: boolean;
  showRuins: boolean;
  showStructures: boolean;
  showBoulders: boolean;
  showOthers: boolean;
  useAiInsight: boolean;
  fastMode: boolean;
};

type ReportSections = {
  summary: boolean;
  location: boolean;
  visuals: boolean;
  metrics: boolean;
  shap: boolean;
  analyst: boolean;
};

type TabId = "dashboard" | "terrain" | "agriculture" | "analytics";

type TerrainRiskPoint = {
  id: string;
  lat: number;
  lon: number;
  riskPercent: number;
  riskLabel: "LOW" | "MODERATE" | "HIGH";
  aiAnalysis: string;
  updatedAt: string;
};

type TerrainBatchPointPrediction = {
  lat: number;
  lon: number;
  probability: number;
  riskLabel: string;
  explanation?: string;
  shap?: Array<{ feature: string; value: number }>;
  insightMode?: string | null;
  insightStatus?: string | null;
};

type TerrainScanMode = "radius" | "polygon4";
const TERRAIN_POINT_REVEAL_DELAY_MS = 320;

const NAV_TABS: Array<{ id: TabId; label: string }> = [
  { id: "dashboard", label: "Dashboard" },
  { id: "terrain", label: "Terrain Map" },
  { id: "agriculture", label: "Crop Intelligence" },
  { id: "analytics", label: "Analytics" },
];

function PageContent() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const [lat, setLat] = useState("22.5726");
  const [lon, setLon] = useState("88.3639");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadPreviewUrl, setUploadPreviewUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<PredictionResult | null>(null);

  const [showVisuals, setShowVisuals] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const [notificationHistory, setNotificationHistory] = useState<NotificationItem[]>([]);
  const [hasUnreadNotification, setHasUnreadNotification] = useState(false);

  const [activeTab, setActiveTab] = useState<TabId>("dashboard");
  const [livePointSampling, setLivePointSampling] = useState(true);
  const [terrainScanMode, setTerrainScanMode] = useState<TerrainScanMode>("radius");
  const [terrainAreaRadius, setTerrainAreaRadius] = useState(300);
  const [terrainPolygonVertices, setTerrainPolygonVertices] = useState<Array<[number, number]>>([]);
  const [terrainRiskPoints, setTerrainRiskPoints] = useState<TerrainRiskPoint[]>([]);
  const [compactPointAnalysis, setCompactPointAnalysis] = useState(true);
  const [terrainBatchLoading, setTerrainBatchLoading] = useState(false);
  const [terrainBatchRevealCount, setTerrainBatchRevealCount] = useState(0);
  const [terrainBatchRevealTotal, setTerrainBatchRevealTotal] = useState(0);
  const [confidence, setConfidence] = useState(0.25);
  const [showVegetation, setShowVegetation] = useState(true);
  const [showRuins, setShowRuins] = useState(true);
  const [showStructures, setShowStructures] = useState(true);
  const [showBoulders, setShowBoulders] = useState(true);
  const [showOthers, setShowOthers] = useState(true);
  const [fastMode, setFastMode] = useState(true);
  const [useAiInsight, setUseAiInsight] = useState(false);
  const [insightLoading, setInsightLoading] = useState(false);
  const [imageStreamLoading, setImageStreamLoading] = useState(false);
  const [streamCompletePulse, setStreamCompletePulse] = useState(false);
  const [motionMode, setMotionMode] = useState<"full" | "reduced">("full");
  const [showReportPreview, setShowReportPreview] = useState(false);
  const [reportBusy, setReportBusy] = useState<"quick" | "pdf" | "png" | null>(null);
  const [reportFeedback, setReportFeedback] = useState("");
  const [reportGeneratedAt, setReportGeneratedAt] = useState<Date>(new Date());
  const [reportRegionName, setReportRegionName] = useState("");
  const [reportSections, setReportSections] = useState<ReportSections>({
    summary: true,
    location: true,
    visuals: true,
    metrics: true,
    shap: true,
    analyst: true,
  });

  const reportRef = useRef<HTMLDivElement | null>(null);
  const terrainBatchRequestRef = useRef(0);
  const terrainRevealTimeoutsRef = useRef<number[]>([]);
  const tabScrollRef = useRef<Record<TabId, number>>({
    dashboard: 0,
    terrain: 0,
    agriculture: 0,
    analytics: 0,
  });

  const [cropData, setCropData] = useState<CropIntelligenceData | null>(null);
  const [cropLoading, setCropLoading] = useState(false);
  const [cropSeason, setCropSeason] = useState("Kharif");

  const fetchCropRecommendation = async (seasonToFetch?: string) => {
    const s = seasonToFetch || cropSeason;
    setCropLoading(true);
    try {
      const resp = await fetch("/api/predict?action=crop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          lat: Number(lat) || 15.335,
          lon: Number(lon) || 76.46,
          season: s,
          vegetation: result?.metrics?.vegetation ?? 0.35,
        }),
      });
      if (resp.ok) {
        const data = await resp.json();
        setCropData(data);
      }
    } catch (e) {
      console.error("Failed to fetch crop recommendation:", e);
    } finally {
      setCropLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === "agriculture" && !cropData && !cropLoading) {
      fetchCropRecommendation();
    }
  }, [activeTab, cropData, cropLoading]);

  function updateConfidence(next: number) {
    const clamped = Math.max(0.05, Math.min(0.95, next));
    setConfidence(Number(clamped.toFixed(2)));
  }

  function getEffectiveRadiusMeters() {
    return terrainAreaRadius;
  }

  useEffect(() => {
    if (!selectedFile) {
      setUploadPreviewUrl("");
      return;
    }

    const url = URL.createObjectURL(selectedFile);
    setUploadPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [selectedFile]);

  useEffect(() => {
    try {
      const rawSettings = window.localStorage.getItem("geoai-ui-settings");
      if (rawSettings) {
        const parsed = JSON.parse(rawSettings) as Partial<PersistedSettings>;
        if (typeof parsed.confidence === "number") updateConfidence(parsed.confidence);
        if (typeof parsed.showVegetation === "boolean") setShowVegetation(parsed.showVegetation);
        if (typeof parsed.showRuins === "boolean") setShowRuins(parsed.showRuins);
        if (typeof parsed.showStructures === "boolean") setShowStructures(parsed.showStructures);
        if (typeof parsed.showBoulders === "boolean") setShowBoulders(parsed.showBoulders);
        if (typeof parsed.showOthers === "boolean") setShowOthers(parsed.showOthers);
        if (typeof parsed.fastMode === "boolean") setFastMode(parsed.fastMode);
        if (typeof parsed.useAiInsight === "boolean") setUseAiInsight(parsed.useAiInsight);
      }

      const rawNotifications = window.localStorage.getItem("geoai-ui-notifications");
      if (rawNotifications) {
        const parsed = JSON.parse(rawNotifications) as NotificationItem[];
        if (Array.isArray(parsed)) setNotificationHistory(parsed.slice(0, 8));
      }

      const rawUnread = window.localStorage.getItem("geoai-ui-notifications-unread");
      if (rawUnread === "true") setHasUnreadNotification(true);

      const rawMotionMode = window.localStorage.getItem("geoai-ui-motion-mode");
      if (rawMotionMode === "full" || rawMotionMode === "reduced") {
        setMotionMode(rawMotionMode);
      }
    } catch {
      // Ignore malformed local data.
    }
  }, []);

  useEffect(() => {
    const view = searchParams?.get("view");
    if (view === "dashboard" || view === "terrain" || view === "analytics") {
      if (view !== activeTab) {
        setActiveTab(view);
      }
    }
  }, [searchParams, activeTab]);

  useEffect(() => {
    const saved = tabScrollRef.current[activeTab] ?? 0;
    window.requestAnimationFrame(() => {
      window.scrollTo({ top: saved, behavior: "auto" });
    });
  }, [activeTab]);

  useEffect(() => {
    function trackScroll() {
      tabScrollRef.current[activeTab] = window.scrollY;
    }

    window.addEventListener("scroll", trackScroll, { passive: true });
    return () => window.removeEventListener("scroll", trackScroll);
  }, [activeTab]);

  useEffect(() => {
    const payload: PersistedSettings = {
      confidence,
      showVegetation,
      showRuins,
      showStructures,
      showBoulders,
      showOthers,
      fastMode,
      useAiInsight,
    };
    window.localStorage.setItem("geoai-ui-settings", JSON.stringify(payload));
  }, [confidence, showVegetation, showRuins, showStructures, showBoulders, showOthers, fastMode, useAiInsight]);

  useEffect(() => {
    window.localStorage.setItem("geoai-ui-notifications", JSON.stringify(notificationHistory));
  }, [notificationHistory]);

  useEffect(() => {
    window.localStorage.setItem("geoai-ui-notifications-unread", hasUnreadNotification ? "true" : "false");
  }, [hasUnreadNotification]);

  useEffect(() => {
    window.localStorage.setItem("geoai-ui-motion-mode", motionMode);
  }, [motionMode]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key !== "Escape") return;
      setShowSettings(false);
      setShowNotifications(false);
      setShowReportPreview(false);
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => {
    function onShortcut(event: KeyboardEvent) {
      const tagName = (event.target as HTMLElement | null)?.tagName;
      if (tagName === "INPUT" || tagName === "TEXTAREA") {
        return;
      }

      if (event.key === "1") setActiveTab("dashboard");
      if (event.key === "2") setActiveTab("terrain");
      if (event.key === "3") setActiveTab("analytics");
    }

    window.addEventListener("keydown", onShortcut);
    return () => window.removeEventListener("keydown", onShortcut);
  }, []);

  const riskPercent = useMemo(() => {
    if (!result) return 0;
    return Math.max(0.1, Math.min(100, Number((result.probability * 100).toFixed(1))));
  }, [result]);

  function resolveRiskLabel(riskValue: number): "LOW" | "MODERATE" | "HIGH" {
    if (riskValue >= 67) return "HIGH";
    if (riskValue >= 34) return "MODERATE";
    return "LOW";
  }

  function estimatePointRisk(nextLat: number, nextLon: number) {
    const baseProbability = result?.probability ?? 0.42;
    const latWave = (Math.sin((nextLat * Math.PI) / 180) + 1) / 2;
    const lonWave = (Math.cos((nextLon * Math.PI) / 180) + 1) / 2;

    const parsedLat = Number(lat);
    const parsedLon = Number(lon);
    const referenceLat = result?.metrics.lat ?? (Number.isFinite(parsedLat) ? parsedLat : nextLat);
    const referenceLon = result?.metrics.lon ?? (Number.isFinite(parsedLon) ? parsedLon : nextLon);
    const distance = Math.sqrt((nextLat - referenceLat) ** 2 + (nextLon - referenceLon) ** 2);
    const distanceFactor = Math.min(1, distance / 1.2);

    const blendedProbability =
      baseProbability * 0.72 +
      latWave * 0.14 +
      lonWave * 0.1 +
      distanceFactor * 0.04;

    const pointRiskPercent = Math.max(0.1, Math.min(99.9, Number((blendedProbability * 100).toFixed(1))));
    return {
      riskPercent: pointRiskPercent,
      riskLabel: resolveRiskLabel(pointRiskPercent),
    };
  }

  function buildPointAiAnalysis(nextLat: number, nextLon: number, riskPercent: number, riskLabel: "LOW" | "MODERATE" | "HIGH") {
    const parsedLat = Number(lat);
    const parsedLon = Number(lon);
    const referenceLat = result?.metrics.lat ?? (Number.isFinite(parsedLat) ? parsedLat : nextLat);
    const referenceLon = result?.metrics.lon ?? (Number.isFinite(parsedLon) ? parsedLon : nextLon);
    const distanceKm = Math.sqrt((nextLat - referenceLat) ** 2 + (nextLon - referenceLon) ** 2) * 111;

    const signal = shapItems.length > 0
      ? [...shapItems].sort((a, b) => Math.abs(b.value) - Math.abs(a.value))[0]
      : null;

    const terrainSignal = signal
      ? `${signal.feature.toLowerCase()} is the strongest explainability signal for this scan window`
      : "terrain factors around this location show a mixed pattern";

    const confidenceTone = useAiInsight ? "AI insight mode is active" : "standard model mode is active";

    if (riskLabel === "HIGH") {
      return `${confidenceTone}; this point shows HIGH risk (${riskPercent}%) and should be prioritized for inspection. ${terrainSignal}, with stronger instability closer to this coordinate.`;
    }

    if (riskLabel === "MODERATE") {
      return `${confidenceTone}; this point is MODERATE risk (${riskPercent}%), indicating watch-list conditions. ${terrainSignal}, and field validation is recommended${distanceKm > 0.8 ? " because it is spatially offset from the selected center" : " near the selected center"}.`;
    }

    return `${confidenceTone}; this point is LOW risk (${riskPercent}%), with relatively stable terrain indicators. ${terrainSignal}, and no immediate intervention is suggested.`;
  }

  function buildTerrainRiskPoint(nextLat: number, nextLon: number): TerrainRiskPoint {
    const estimate = estimatePointRisk(nextLat, nextLon);
    return {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
      lat: Number(nextLat.toFixed(6)),
      lon: Number(nextLon.toFixed(6)),
      riskPercent: estimate.riskPercent,
      riskLabel: estimate.riskLabel,
      aiAnalysis: buildPointAiAnalysis(nextLat, nextLon, estimate.riskPercent, estimate.riskLabel),
      updatedAt: new Date().toLocaleTimeString(),
    };
  }

  function buildTerrainRiskPointFromBackend(payload: TerrainBatchPointPrediction): TerrainRiskPoint {
    const riskPercent = Math.max(0.1, Math.min(99.9, Number((Number(payload.probability) * 100).toFixed(1))));
    const normalizedLabel = String(payload.riskLabel || "").toUpperCase();
    const riskLabel =
      normalizedLabel === "HIGH" || normalizedLabel === "LOW" || normalizedLabel === "MODERATE"
        ? (normalizedLabel as "LOW" | "MODERATE" | "HIGH")
        : resolveRiskLabel(riskPercent);

    const fallbackAnalysis = buildPointAiAnalysis(payload.lat, payload.lon, riskPercent, riskLabel);
    const resolvedExplanation =
      typeof payload.explanation === "string" && payload.explanation.trim().length > 0
        ? payload.explanation.trim()
        : fallbackAnalysis;

    return {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
      lat: Number(payload.lat.toFixed(6)),
      lon: Number(payload.lon.toFixed(6)),
      riskPercent,
      riskLabel,
      aiAnalysis: resolvedExplanation,
      updatedAt: new Date().toLocaleTimeString(),
    };
  }

  function invalidateTerrainBatchRequests() {
    terrainBatchRequestRef.current += 1;
    clearTerrainRevealTimers();
    setTerrainBatchLoading(false);
    setTerrainBatchRevealCount(0);
    setTerrainBatchRevealTotal(0);
  }

  function clearTerrainRevealTimers() {
    terrainRevealTimeoutsRef.current.forEach((timeoutId) => window.clearTimeout(timeoutId));
    terrainRevealTimeoutsRef.current = [];
  }

  function revealTerrainPointsSequentially(points: TerrainRiskPoint[], requestId: number) {
    clearTerrainRevealTimers();
    setTerrainRiskPoints([]);
    setTerrainBatchRevealCount(0);
    setTerrainBatchRevealTotal(points.length);

    if (points.length === 0) {
      return;
    }

    points.forEach((point, index) => {
      const timeoutId = window.setTimeout(() => {
        if (terrainBatchRequestRef.current !== requestId) {
          return;
        }

        setTerrainRiskPoints((prev) => [...prev, point]);
        setTerrainBatchRevealCount(index + 1);
      }, index * TERRAIN_POINT_REVEAL_DELAY_MS);

      terrainRevealTimeoutsRef.current.push(timeoutId);
    });
  }

  function generateRadiusSamples(centerLat: number, centerLon: number, radiusMeters: number) {
    const samples: Array<[number, number]> = [[centerLat, centerLon]];
    const angleStep = 30;
    const latMeters = 111320;
    const lonMeters = 111320 * Math.cos((centerLat * Math.PI) / 180);

    for (let angle = 0; angle < 360; angle += angleStep) {
      const radians = (angle * Math.PI) / 180;
      const dLat = (Math.sin(radians) * radiusMeters) / latMeters;
      const dLon = lonMeters === 0 ? 0 : (Math.cos(radians) * radiusMeters) / lonMeters;
      samples.push([centerLat + dLat, centerLon + dLon]);
    }

    return samples;
  }

  function isPointInsidePolygon(point: [number, number], polygon: Array<[number, number]>) {
    const x = point[1];
    const y = point[0];
    let inside = false;

    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const xi = polygon[i][1];
      const yi = polygon[i][0];
      const xj = polygon[j][1];
      const yj = polygon[j][0];

      const intersects = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / ((yj - yi) || 1e-12) + xi;
      if (intersects) {
        inside = !inside;
      }
    }

    return inside;
  }

  function generatePolygonSamples(polygon: Array<[number, number]>) {
    if (polygon.length < 3) {
      return [];
    }

    const latitudes = polygon.map((item) => item[0]);
    const longitudes = polygon.map((item) => item[1]);
    const minLat = Math.min(...latitudes);
    const maxLat = Math.max(...latitudes);
    const minLon = Math.min(...longitudes);
    const maxLon = Math.max(...longitudes);

    const rows = 5;
    const cols = 5;
    const latStep = (maxLat - minLat) / rows;
    const lonStep = (maxLon - minLon) / cols;
    const samples: Array<[number, number]> = [];

    for (let row = 0; row <= rows; row += 1) {
      for (let col = 0; col <= cols; col += 1) {
        const candidate: [number, number] = [minLat + row * latStep, minLon + col * lonStep];
        if (isPointInsidePolygon(candidate, polygon)) {
          samples.push(candidate);
        }
      }
    }

    return [...polygon, ...samples].slice(0, 30);
  }

  function computePolygonAreaSqMeters(polygon: Array<[number, number]>) {
    if (polygon.length < 3) {
      return 0;
    }

    const avgLat =
      polygon.reduce((sum, item) => sum + item[0], 0) / Math.max(1, polygon.length);
    const metersPerDegLat = 111320;
    const metersPerDegLon = 111320 * Math.cos((avgLat * Math.PI) / 180);

    let areaTwice = 0;
    for (let i = 0; i < polygon.length; i += 1) {
      const current = polygon[i];
      const next = polygon[(i + 1) % polygon.length];
      const x1 = current[1] * metersPerDegLon;
      const y1 = current[0] * metersPerDegLat;
      const x2 = next[1] * metersPerDegLon;
      const y2 = next[0] * metersPerDegLat;
      areaTwice += x1 * y2 - x2 * y1;
    }

    return Math.abs(areaTwice) * 0.5;
  }

  async function updateTerrainRiskBatch(points: Array<[number, number]>) {
    if (points.length === 0) {
      clearTerrainRevealTimers();
      setTerrainRiskPoints([]);
      setTerrainBatchRevealCount(0);
      setTerrainBatchRevealTotal(0);
      return;
    }

    const requestId = terrainBatchRequestRef.current + 1;
    terrainBatchRequestRef.current = requestId;
    setTerrainBatchLoading(true);
    clearTerrainRevealTimers();
    setTerrainRiskPoints([]);
    setTerrainBatchRevealCount(0);
    setTerrainBatchRevealTotal(points.length);

    try {
      for (let index = 0; index < points.length; index += 1) {
        if (terrainBatchRequestRef.current !== requestId) {
          return;
        }

        const [pointLat, pointLon] = points[index];
        const response = await fetch("/api/predict?action=point", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            lat: pointLat,
            lon: pointLon,
            useAiInsight: false,
            includeShap: true,
            vegetation: result?.metrics.vegetation,
            boulders: result?.metrics.boulders,
            ruins: result?.metrics.ruins,
            structures: result?.metrics.structures,
          }),
        });

        const payload = await response.json();

        if (terrainBatchRequestRef.current !== requestId) {
          return;
        }

        let nextPoint: TerrainRiskPoint;
        if (!response.ok) {
          nextPoint = buildTerrainRiskPoint(pointLat, pointLon);
          const details = typeof payload?.details === "string" ? ` ${payload.details}` : "";
          setError((payload?.error || "Point prediction failed.") + details);
        } else {
          const backendPointPayload: TerrainBatchPointPrediction = {
            lat: pointLat,
            lon: pointLon,
            probability: Number(payload?.probability ?? 0),
            riskLabel: String(payload?.riskLabel || "MODERATE"),
            explanation: typeof payload?.explanation === "string" ? payload.explanation : undefined,
            shap: Array.isArray(payload?.shap) ? payload.shap : undefined,
            insightMode: typeof payload?.insightMode === "string" ? payload.insightMode : null,
            insightStatus: typeof payload?.insightStatus === "string" ? payload.insightStatus : null,
          };
          nextPoint = buildTerrainRiskPointFromBackend(backendPointPayload);
        }

        setTerrainRiskPoints((prev) => [...prev, nextPoint]);
        setTerrainBatchRevealCount(index + 1);
      }
      if (terrainBatchRequestRef.current === requestId) {
        setError("");
      }
    } catch (err) {
      if (terrainBatchRequestRef.current === requestId) {
        setError(err instanceof Error ? err.message : "Point prediction failed.");
      }
    } finally {
      if (terrainBatchRequestRef.current === requestId) {
        setTerrainBatchLoading(false);
      }
    }
  }

  function onTerrainMapSelect(nextLat: number, nextLon: number) {
    setLat(nextLat.toFixed(6));
    setLon(nextLon.toFixed(6));

    if (!livePointSampling) {
      return;
    }

    if (terrainScanMode === "radius") {
      const effectiveRadius = getEffectiveRadiusMeters();
      setTerrainRiskPoints([]);
      const samples = generateRadiusSamples(nextLat, nextLon, effectiveRadius);
      void updateTerrainRiskBatch(samples);
      return;
    }

    setTerrainPolygonVertices((prev) => {
      const nextVertices = prev.length >= 4 ? [[nextLat, nextLon]] : [...prev, [nextLat, nextLon]];
      // Clear stale results while user is still defining the polygon vertices.
      if (nextVertices.length < 4) {
        setTerrainRiskPoints([]);
      }
      if (nextVertices.length === 4) {
        setTerrainRiskPoints([]);
        const samples = generatePolygonSamples(nextVertices as Array<[number, number]>);
        void updateTerrainRiskBatch(samples);
      }
      return nextVertices as Array<[number, number]>;
    });
  }

  const terrainOverlayPoints = useMemo(
    () => terrainRiskPoints.map((point) => [point.lat, point.lon] as [number, number]),
    [terrainRiskPoints]
  );

  const terrainPolygonAreaSqMeters = useMemo(
    () => (terrainScanMode === "polygon4" ? computePolygonAreaSqMeters(terrainPolygonVertices) : 0),
    [terrainScanMode, terrainPolygonVertices]
  );

  function formatPointAnalysis(text: string) {
    if (!compactPointAnalysis) return text;
    const normalized = text.trim();
    if (normalized.length <= 110) return normalized;
    return `${normalized.slice(0, 110).trimEnd()}...`;
  }

  const fileLabel = useMemo(() => {
    if (!selectedFile) return "No image selected";
    return selectedFile.name;
  }, [selectedFile]);

  const visualTiles = useMemo<VisualTile[]>(() => {
    if (result?.images) {
      return [
        { title: "Original Terrain", src: result.images.original },
        { title: "Erosion Detection", src: result.images.detection },
        { title: "Segmentation Map", src: result.images.segmentation },
        { title: "Heatmap Overlay", src: result.images.heatmap }
      ];
    }

    return [
      { title: "Original Terrain", src: uploadPreviewUrl || "", accent: "from-slate-500/40 via-slate-300/10 to-slate-700/50" },
      { title: "Erosion Detection", src: uploadPreviewUrl || "", accent: "from-rose-500/35 via-red-300/10 to-slate-800/60" },
      { title: "Segmentation Map", src: uploadPreviewUrl || "", accent: "from-orange-400/30 via-amber-200/10 to-slate-800/60" },
      { title: "Heatmap Overlay", src: uploadPreviewUrl || "", accent: "from-red-500/45 via-red-400/15 to-black/70" }
    ];
  }, [result, uploadPreviewUrl]);

  const metricCards = useMemo(() => {
    if (!result) {
      return [
        { label: "Slope Steepness", value: "-", tooltip: "Slope influences erosion speed. Steeper terrain increases surface runoff and soil loss." },
        { label: "Soil Type", value: "-", tooltip: "Soil composition affects retention. Loose and granular soils are generally more erosion-prone." },
        { label: "Vegetation", value: "-", tooltip: "Vegetation roots anchor soil and reduce detachment caused by rain and wind." },
        { label: "Recent Rainfall", value: "-", tooltip: "Higher rainfall intensity increases runoff pressure and can accelerate erosion." }
      ];
    }

    return [
      {
        label: "Slope Steepness",
        value: `${result.metrics.slope.toFixed(1)} deg`,
        tooltip: "Slope influences erosion speed. Steeper terrain increases surface runoff and soil loss."
      },
      {
        label: "Soil Type",
        value: result.metrics.soil,
        tooltip: "Soil composition affects retention. Loose and granular soils are generally more erosion-prone."
      },
      {
        label: "Vegetation",
        value: `${(result.metrics.vegetation * 100).toFixed(1)}%`,
        tooltip: "Vegetation roots anchor soil and reduce detachment caused by rain and wind."
      },
      {
        label: "Recent Rainfall",
        value: `${result.metrics.rainfall.toFixed(1)} mm`,
        tooltip: "Higher rainfall intensity increases runoff pressure and can accelerate erosion."
      }
    ];
  }, [result]);

  const shapItems = useMemo(() => {
    const raw = result?.shap;
    return Array.isArray(raw)
      ? raw
      .filter((item) => item && typeof item.feature === "string" && Number.isFinite(item.value))
      .map((item) => ({ feature: item.feature, value: Number(item.value) }))
      : [];
  }, [result]);

  const reportVisuals = useMemo(() => {
    return visualTiles.map((tile) => ({ title: tile.title, src: tile.src }));
  }, [visualTiles]);

  const reportMapSnapshotUrl = useMemo(() => {
    if (!result?.metrics) return "";
    const latValue = Number(result.metrics.lat.toFixed(6));
    const lonValue = Number(result.metrics.lon.toFixed(6));
    const cacheBust = reportGeneratedAt.getTime();
    return `/api/predict?action=snapshot&lat=${encodeURIComponent(String(latValue))}&lon=${encodeURIComponent(String(lonValue))}&zoom=12&cb=${cacheBust}`;
  }, [result, reportGeneratedAt]);

  function buildReportFilename(extension: "pdf" | "png" | "jpg") {
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    return `geo_ai_report_${timestamp}.${extension}`;
  }

  function buildDataFilename(suffix: string, extension: "json" | "csv") {
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    return `geo_ai_${suffix}_${timestamp}.${extension}`;
  }

  function openReportExport() {
    if (!result) {
      setReportFeedback("Run a prediction to unlock report export.");
      return;
    }
    setReportGeneratedAt(new Date());
    setReportRegionName((prev) => prev || `Lat ${result.metrics.lat.toFixed(4)}, Lon ${result.metrics.lon.toFixed(4)}`);
    setShowReportPreview(true);
    setReportFeedback("");
  }

  function exportTerrainScanData() {
    const payload = {
      exportedAt: new Date().toISOString(),
      scanMode: terrainScanMode,
      thresholdMeters: terrainScanMode === "radius" ? terrainAreaRadius : null,
      polygonVertices: terrainPolygonVertices,
      areaSqMeters: terrainScanMode === "polygon4" ? Math.round(terrainPolygonAreaSqMeters) : null,
      points: terrainRiskPoints,
      aiModeStats: {
        aimodeEnabled: useAiInsight,
        insightMode: result?.insightMode || null,
        insightStatus: result?.insightStatus || null,
        shapFactors: shapItems.length,
      },
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = buildDataFilename("terrain_scan", "json");
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function exportTerrainScanCsv() {
    if (terrainRiskPoints.length === 0) {
      return;
    }

    const escapeCsv = (value: string | number) => {
      const text = String(value ?? "");
      if (text.includes(",") || text.includes("\"") || text.includes("\n")) {
        return `"${text.replace(/\"/g, '""')}"`;
      }
      return text;
    };

    const header = [
      "index",
      "lat",
      "lon",
      "riskPercent",
      "riskLabel",
      "aiAnalysis",
      "updatedAt",
      "scanMode",
      "thresholdMeters",
      "areaSqMeters",
      "aimodeEnabled",
      "insightMode",
      "insightStatus",
      "shapFactors",
    ];

    const rows = terrainRiskPoints.map((point, index) => [
      index + 1,
      point.lat.toFixed(6),
      point.lon.toFixed(6),
      point.riskPercent,
      point.riskLabel,
      point.aiAnalysis,
      point.updatedAt,
      terrainScanMode,
      terrainScanMode === "radius" ? terrainAreaRadius : "",
      terrainScanMode === "polygon4" ? Math.round(terrainPolygonAreaSqMeters) : "",
      useAiInsight ? "true" : "false",
      result?.insightMode || "",
      result?.insightStatus || "",
      shapItems.length,
    ]);

    const csvContent = [header, ...rows]
      .map((line) => line.map((value) => escapeCsv(value)).join(","))
      .join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = buildDataFilename("terrain_scan", "csv");
    anchor.click();
    URL.revokeObjectURL(url);
  }

  async function renderReportCanvas(mode: "quick" | "standard" = "standard") {
    const element = reportRef.current;
    if (!element) {
      throw new Error("Report preview is not ready yet.");
    }

    // Wait for images in the report to load before rasterizing.
    const images = Array.from(element.querySelectorAll("img"));
    const imageWaiters = images.map(
      (img) =>
        new Promise<void>((resolve) => {
          if (img.complete) {
            resolve();
            return;
          }
          img.onload = () => resolve();
          img.onerror = () => resolve();
        })
    );

    if (mode === "quick") {
      await Promise.race([
        Promise.all(imageWaiters),
        new Promise<void>((resolve) => {
          window.setTimeout(() => resolve(), 900);
        }),
      ]);
    } else {
      await Promise.all(imageWaiters);
    }

    return html2canvas(element, {
      scale: mode === "quick" ? 1 : 2,
      useCORS: true,
      allowTaint: true,
      backgroundColor: "#f8fafc",
      windowWidth: element.scrollWidth,
      windowHeight: element.scrollHeight,
    });
  }

  async function exportReport(format: "quick" | "pdf" | "png") {
    if (!result) {
      setReportFeedback("Run a prediction before exporting a report.");
      return;
    }

    setReportBusy(format);
    setReportFeedback(format === "quick" ? "Generating instant report..." : "Generating report...");

    try {
      const canvas = await renderReportCanvas(format === "quick" ? "quick" : "standard");

      if (format === "quick") {
        const imageData = canvas.toDataURL("image/jpeg", 0.82);
        const link = document.createElement("a");
        link.href = imageData;
        link.download = buildReportFilename("jpg");
        link.click();
      } else if (format === "png") {
        const imageData = canvas.toDataURL("image/png");
        const link = document.createElement("a");
        link.href = imageData;
        link.download = buildReportFilename("png");
        link.click();
      } else {
        const imageData = canvas.toDataURL("image/png", 1);
        const pdf = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
        const pageWidth = pdf.internal.pageSize.getWidth();
        const pageHeight = pdf.internal.pageSize.getHeight();
        const imgProps = pdf.getImageProperties(imageData);
        const imgWidth = pageWidth;
        const imgHeight = (imgProps.height * imgWidth) / imgProps.width;

        let heightLeft = imgHeight;
        let position = 0;
        pdf.addImage(imageData, "PNG", 0, position, imgWidth, imgHeight, undefined, "FAST");
        heightLeft -= pageHeight;

        while (heightLeft > 0) {
          position -= pageHeight;
          pdf.addPage();
          pdf.addImage(imageData, "PNG", 0, position, imgWidth, imgHeight, undefined, "FAST");
          heightLeft -= pageHeight;
        }

        pdf.save(buildReportFilename("pdf"));
      }

      setReportFeedback("Report downloaded successfully.");
    } catch (err) {
      setReportFeedback(err instanceof Error ? err.message : "Report export failed.");
    } finally {
      setReportBusy(null);
    }
  }

  async function streamImagesSequentially(
    imageJobId: string,
    pendingImages: Array<"detection" | "segmentation" | "combined" | "heatmap">
  ) {
    if (!imageJobId || pendingImages.length === 0) {
      return;
    }

    setImageStreamLoading(true);

    try {
      for (const imageName of pendingImages) {
        const response = await fetch(
          `/api/predict?action=image&jobId=${encodeURIComponent(imageJobId)}&name=${encodeURIComponent(imageName)}`
        );

        if (!response.ok) {
          continue;
        }

        const payload = await response.json() as {
          name?: "detection" | "segmentation" | "combined" | "heatmap";
          src?: string;
        };
        if (!payload?.name || typeof payload.src !== "string") {
          continue;
        }

        const resolvedImageName = payload.name;
        const imageSrc = payload.src;

        setResult((prev) => {
          if (!prev) return prev;
          return {
            ...prev,
            images: {
              ...prev.images,
              [resolvedImageName]: imageSrc,
            },
          };
        });
      }
    } finally {
      setImageStreamLoading(false);
      setStreamCompletePulse(true);
      window.setTimeout(() => setStreamCompletePulse(false), 1800);
    }
  }

  async function requestDeferredInsights() {
    if (insightLoading) {
      return;
    }

    if (!result) {
      setUseAiInsight(false);
      setError("Run a prediction first to enable AI mode.");
      setReportFeedback("Prediction is required before AI insight can run.");
      return;
    }

    setInsightLoading(true);
    setError("");
    setReportFeedback("Generating AI Insight and SHAP...");

    try {
      const response = await fetch("/api/predict?action=insight", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          metrics: result.metrics,
          probability: result.probability,
          useAiInsight: true,
          includeShap: true,
        }),
      });

      const payload = await response.json();
      if (!response.ok) {
        const details = typeof payload?.details === "string" ? ` ${payload.details}` : "";
        throw new Error((payload?.error || "AI insight generation failed.") + details);
      }

      setResult((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          explanation: typeof payload?.explanation === "string" ? payload.explanation : prev.explanation,
          shap: Array.isArray(payload?.shap) ? payload.shap : prev.shap,
          insightMode: typeof payload?.insightMode === "string" ? payload.insightMode : prev.insightMode,
          insightStatus: typeof payload?.insightStatus === "string" ? payload.insightStatus : prev.insightStatus,
        };
      });

      setUseAiInsight(true);
      setReportFeedback("AI Insight and SHAP updated successfully.");
    } catch (err) {
      setUseAiInsight(false);
      setError(err instanceof Error ? err.message : "AI insight generation failed.");
    } finally {
      setInsightLoading(false);
    }
  }

  function onToggleAiMode(checked: boolean) {
    if (!checked) {
      setUseAiInsight(false);
      setReportFeedback("AI mode disabled.");
      return;
    }

    if (!result) {
      setUseAiInsight(false);
      setError("Run a prediction first to enable AI mode.");
      setReportFeedback("Prediction is required before AI insight can run.");
      return;
    }

    setUseAiInsight(true);
    void requestDeferredInsights();
  }

  function onAiButtonClick() {
    if (!result) {
      setUseAiInsight(false);
      setError("Run a prediction first to enable AI mode.");
      setReportFeedback("Prediction is required before AI insight can run.");
      return;
    }

    setUseAiInsight(true);
    void requestDeferredInsights();
  }

  async function runPrediction() {
    const parsedLat = Number(lat);
    const parsedLon = Number(lon);

    if (Number.isNaN(parsedLat) || Number.isNaN(parsedLon)) {
      setError("Latitude and longitude must be valid numbers.");
      return;
    }

    if (!selectedFile) {
      setError("Please upload a terrain image first.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("image", selectedFile);

      formData.append("lat", String(parsedLat));
      formData.append("lon", String(parsedLon));
      formData.append("confidence", String(confidence));
      formData.append("showVegetation", String(showVegetation));
      formData.append("showRuins", String(showRuins));
      formData.append("showStructures", String(showStructures));
      formData.append("showBoulders", String(showBoulders));
      formData.append("showOthers", String(showOthers));
      formData.append("fastMode", String(fastMode));
      formData.append("useAiInsight", "false");
      formData.append("includeShap", "false");

      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData
      });

      const payload = await response.json();
      if (!response.ok) {
        const details = typeof payload?.details === "string" ? ` ${payload.details}` : "";
        throw new Error((payload?.error || "Prediction failed.") + details);
      }

      const next = payload as PredictionResult;
      setResult(next);
  setUseAiInsight(false);

      if (next.imageJobId && Array.isArray(next.pendingImages) && next.pendingImages.length > 0) {
        streamImagesSequentially(next.imageJobId, next.pendingImages);
      }

      setNotificationHistory((prev) => [
        {
          id: `${Date.now()}`,
          message: `Prediction ready: ${Math.round(next.probability * 100)}% ${next.riskLabel} risk`,
          timestamp: new Date().toLocaleTimeString()
        },
        ...prev
      ].slice(0, 8));
      setHasUnreadNotification(true);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  function navigateTab(tab: TabId) {
    tabScrollRef.current[activeTab] = window.scrollY;
    setActiveTab(tab);

    const params = new URLSearchParams(searchParams?.toString() || "");
    params.set("view", tab);
    router.replace(`${pathname}?${params.toString()}`, { scroll: false });
  }

  return (
    <MotionConfig reducedMotion={motionMode === "reduced" ? "always" : "never"}>
      <div className="min-h-screen bg-surface text-ink">
      <header className="fixed inset-x-0 top-0 z-50 border-b border-white/10 bg-[#0f1218]/75 backdrop-blur-xl">
        <div className="mx-auto flex h-16 w-full max-w-[1500px] items-center justify-between px-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-8">
            <p className="font-[var(--font-sora)] text-lg font-semibold tracking-tight text-white">GeoAI Systems</p>
            <nav className="relative hidden items-center gap-2 rounded-full border border-white/10 bg-white/5 p-1 lg:flex">
              {NAV_TABS.map((tab) => (
                <motion.button
                  key={tab.id}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className={`relative z-10 rounded-full px-3 py-1.5 text-sm transition ${activeTab === tab.id ? "text-slate-900" : "text-dim hover:text-white"}`}
                  onClick={() => navigateTab(tab.id)}
                >
                  {activeTab === tab.id ? (
                    <motion.span
                      layoutId="nav-tab-pill"
                      className="absolute inset-0 -z-10 rounded-full bg-white"
                      transition={{ type: "spring", stiffness: 360, damping: 30 }}
                    />
                  ) : null}
                  {tab.label}
                </motion.button>
              ))}
              <motion.span
                key={activeTab}
                initial={{ opacity: 0.2 }}
                animate={{ opacity: 0.45 }}
                transition={{ duration: 0.28 }}
                className="pointer-events-none absolute inset-0 rounded-full bg-[radial-gradient(circle_at_center,rgba(56,189,248,0.30),transparent_70%)]"
              />
            </nav>
          </div>

          <div className="relative flex items-center gap-2 text-dim">
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="inline-flex items-center gap-2 rounded-full border border-white/20 bg-white px-3 py-1.5 text-xs font-semibold text-slate-900 transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-55"
              onClick={openReportExport}
              disabled={!result || loading}
            >
              <FileDown className="h-3.5 w-3.5" />
              Export Report
            </motion.button>

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="relative rounded-full p-2 transition hover:bg-white/10 hover:text-white"
              onClick={() => {
                setShowNotifications((prev) => !prev);
                setHasUnreadNotification(false);
              }}
              aria-label="Notifications"
            >
              <Bell className="h-4 w-4" />
              {hasUnreadNotification ? <span className="absolute right-1 top-1 h-2 w-2 rounded-full bg-rose-400" /> : null}
            </motion.button>

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="rounded-full p-2 transition hover:bg-white/10 hover:text-white"
              onClick={() => setShowSettings(true)}
              aria-label="Open settings"
            >
              <Settings className="h-4 w-4" />
            </motion.button>

            {showNotifications ? (
              <div className="absolute right-0 top-12 z-[66] w-[320px] rounded-xl border border-white/10 bg-[#101722]/95 p-3 shadow-lg backdrop-blur">
                <div className="mb-2 flex items-center justify-between">
                  <p className="text-xs font-semibold uppercase tracking-[0.12em] text-zinc-200">Notifications</p>
                  {notificationHistory.length > 0 ? (
                    <button className="text-[10px] text-dim transition hover:text-white" onClick={() => setNotificationHistory([])}>
                      Clear
                    </button>
                  ) : null}
                </div>

                {notificationHistory.length === 0 ? (
                  <p className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs text-dim">No notifications yet.</p>
                ) : (
                  <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
                    {notificationHistory.map((item) => (
                      <div key={item.id} className="rounded-lg border border-emerald-300/20 bg-[#13201a]/70 px-3 py-2">
                        <p className="text-xs leading-relaxed text-emerald-100">{item.message}</p>
                        <p className="mt-1 text-[10px] text-emerald-200/75">{item.timestamp}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : null}
          </div>
        </div>
        {reportFeedback ? <p className="mx-auto mt-1 w-full max-w-[1500px] px-4 text-right text-xs text-zinc-300 sm:px-6 lg:px-8">{reportFeedback}</p> : null}
      </header>

      <main className="mx-auto grid w-full max-w-[1500px] grid-cols-1 gap-6 px-4 pb-10 pt-20 sm:px-6 lg:grid-cols-[300px_minmax(0,1fr)] lg:px-8">
        <div className="panel-soft lg:hidden">
          <div className="flex items-center gap-2 rounded-2xl p-2">
            {NAV_TABS.map((tab) => (
              <button
                key={`mobile-${tab.id}`}
                className={`flex-1 rounded-xl px-3 py-2 text-xs font-semibold transition ${activeTab === tab.id ? "bg-white text-slate-900" : "text-zinc-300"}`}
                onClick={() => navigateTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        <aside className="panel h-fit p-4 sm:p-5 lg:sticky lg:top-20 lg:h-[calc(100vh-6.5rem)] lg:overflow-y-auto">
          <div className="flex h-full flex-col gap-5">
            <section>
              {activeTab !== "terrain" ? (
                <>
                  <p className="mb-3 text-[11px] font-semibold uppercase tracking-[0.18em] text-dim">Input Image</p>
                <label className="group flex cursor-pointer flex-col items-center justify-center gap-2 rounded-3xl border border-dashed border-borderSoft bg-[#171d27]/80 px-4 py-9 text-center transition hover:border-white/25 hover:bg-[#1b2230]">
                  <UploadCloud className="h-5 w-5 text-dim transition group-hover:text-ink" />
                  <span className="text-xs text-dim transition group-hover:text-ink">Upload Terrain Image</span>
                  <input
                    type="file"
                    accept="image/png,image/jpeg,image/jpg"
                    className="hidden"
                    onChange={(e) => {
                      setSelectedFile(e.target.files?.[0] ?? null);
                      setResult(null);
                      setError("");
                    }}
                  />
                </label>
                <p className="mt-2 truncate text-[11px] text-dim">{fileLabel}</p>
                </>
              ) : (
                <div className="space-y-3 rounded-2xl border border-cyan-300/20 bg-cyan-500/10 p-3 text-xs text-cyan-100">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.16em]">Terrain Point Controls</p>
                  <label className="inline-flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={livePointSampling}
                      onChange={(event) => setLivePointSampling(event.target.checked)}
                    />
                    Live point sampling
                  </label>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-[10px] uppercase tracking-[0.12em] text-cyan-100/90">Scan Mode</span>
                    <select
                      className="rounded-full border border-white/25 bg-black/30 px-2 py-1 text-[11px] text-zinc-100 outline-none"
                      value={terrainScanMode}
                      onChange={(event) => {
                        const mode = event.target.value as TerrainScanMode;
                        invalidateTerrainBatchRequests();
                        setTerrainScanMode(mode);
                        setTerrainRiskPoints([]);
                        if (mode === "radius") {
                          setTerrainPolygonVertices([]);
                        }
                      }}
                    >
                      <option value="radius">Radius Sweep</option>
                      <option value="polygon4">4-Point Polygon</option>
                    </select>
                  </div>
                  {terrainScanMode === "radius" ? (
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-[10px] uppercase tracking-[0.12em] text-cyan-100/90">Threshold</span>
                      <select
                        className="rounded-full border border-white/25 bg-black/30 px-2 py-1 text-[11px] text-zinc-100 outline-none"
                        value={terrainAreaRadius}
                        onChange={(event) => {
                          const value = Number(event.target.value);
                          invalidateTerrainBatchRequests();
                          setTerrainAreaRadius(value);
                          setTerrainRiskPoints([]);
                        }}
                      >
                        <option value={200}>200 m</option>
                        <option value={300}>300 m</option>
                        <option value={500}>500 m</option>
                        <option value={1000}>1000 m</option>
                      </select>
                    </div>
                  ) : (
                    <div className="inline-flex items-center gap-2 rounded-full border border-amber-300/25 bg-amber-500/10 px-2.5 py-1 text-[10px] uppercase tracking-[0.12em] text-amber-100">
                      Vertices {terrainPolygonVertices.length}/4
                      <button
                        type="button"
                        className="rounded-full border border-white/20 px-2 py-0.5 text-[9px] font-semibold text-zinc-100"
                        onClick={() => {
                          invalidateTerrainBatchRequests();
                          setTerrainPolygonVertices([]);
                          setTerrainRiskPoints([]);
                        }}
                      >
                        Reset
                      </button>
                    </div>
                  )}
                  <button
                    type="button"
                    className="rounded-full border border-white/25 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-zinc-100 transition hover:bg-white/10 disabled:opacity-50"
                    onClick={() => {
                      invalidateTerrainBatchRequests();
                      setTerrainRiskPoints([]);
                      setTerrainPolygonVertices([]);
                    }}
                    disabled={terrainRiskPoints.length === 0 && terrainPolygonVertices.length === 0}
                  >
                    Clear points
                  </button>
                </div>
              )}
            </section>

            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="inline-flex w-full items-center justify-center gap-2 rounded-full bg-white px-4 py-3 text-sm font-semibold text-slate-900 transition hover:brightness-105 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-60"
              onClick={runPrediction}
              disabled={loading}
            >
              <Crosshair className="h-4 w-4" />
              {loading ? "Running..." : "Run Prediction"}
            </motion.button>

            <section className="transition-all duration-300 hover:scale-[1.01]">
              <p className="mb-3 text-[11px] font-semibold uppercase tracking-[0.18em] text-dim">Map Selection</p>
              <div className="rounded-3xl border border-white/10 bg-[#141b26]/85 p-2 transition hover:border-cyan-300/25">
                {activeTab === "terrain" ? (
                  <p className="px-2 py-3 text-xs text-zinc-300">
                    Mini map paused for smoother performance. Use the large Terrain Navigator map on the right.
                  </p>
                ) : (
                  <MapPicker
                    lat={Number(lat) || 0}
                    lon={Number(lon) || 0}
                    onChange={(nextLat, nextLon) => {
                      setLat(nextLat.toFixed(6));
                      setLon(nextLon.toFixed(6));
                    }}
                  />
                )}
              </div>
            </section>

            <section>
              <p className="mb-3 text-[11px] font-semibold uppercase tracking-[0.18em] text-dim">Coordinates</p>
              <div className="grid grid-cols-1 gap-2">
                <label>
                  <span className="mb-1 block text-[10px] uppercase tracking-[0.12em] text-dim">Latitude</span>
                  <input className="input rounded-full" value={lat} onChange={(e) => setLat(e.target.value)} inputMode="decimal" />
                </label>
                <label>
                  <span className="mb-1 block text-[10px] uppercase tracking-[0.12em] text-dim">Longitude</span>
                  <input className="input rounded-full" value={lon} onChange={(e) => setLon(e.target.value)} inputMode="decimal" />
                </label>
              </div>
            </section>

            {error ? <p className="text-xs text-rose-300">{error}</p> : null}
          </div>
        </aside>

        <section className="space-y-6">
          <div className="panel-soft cinematic-card flex flex-wrap items-center justify-between gap-3 rounded-3xl p-4">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-dim">Workspace Mode</p>
              <p className="mt-1 text-sm text-zinc-100">
                {activeTab === "dashboard"
                  ? "Live risk briefing"
                  : activeTab === "terrain"
                  ? "Terrain visual lab"
                  : activeTab === "agriculture"
                  ? "Crop suitability & agro-climatic intelligence"
                  : "Explainability analytics"}
              </p>
              {streamCompletePulse ? (
                <p className="status-pulse mt-2 inline-flex items-center gap-1 rounded-full border border-emerald-300/30 bg-emerald-500/15 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-emerald-200">
                  Stream Complete
                </p>
              ) : null}
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <label className="inline-flex items-center gap-2 rounded-full border border-white/20 bg-black/30 px-3 py-2 text-xs font-semibold text-zinc-100">
                <input
                  type="checkbox"
                  checked={useAiInsight}
                  disabled={insightLoading}
                  onChange={(event) => onToggleAiMode(event.target.checked)}
                />
                <span className="inline-flex items-center gap-1">
                  <BrainCircuit className="h-3.5 w-3.5" />
                  AI Mode
                </span>
              </label>
              <button
                className="inline-flex items-center gap-2 rounded-full border border-cyan-200/40 bg-cyan-100 px-4 py-2 text-xs font-semibold text-cyan-950 transition hover:brightness-105 disabled:opacity-60"
                disabled={insightLoading}
                onClick={onAiButtonClick}
              >
                {insightLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <BrainCircuit className="h-4 w-4" />}
                {insightLoading ? "AIMODE Running..." : "Enable / Refresh AI"}
              </button>
            </div>
          </div>

          <AnimatePresence mode="wait">
            {activeTab === "dashboard" ? (
              <motion.div
                key="tab-dashboard"
                initial={{ opacity: 0, y: 22, scale: 0.97, filter: "blur(6px)" }}
                animate={{ opacity: 1, y: 0, scale: 1, filter: "blur(0px)" }}
                exit={{ opacity: 0, y: -14, scale: 0.985, filter: "blur(4px)" }}
                transition={{ duration: 0.34, ease: [0.22, 0.9, 0.3, 1] }}
                className="space-y-6"
              >
                {!result && !loading ? (
                  <motion.article
                    whileHover={{ y: -3 }}
                    className="panel-soft cinematic-card relative overflow-hidden rounded-3xl border border-cyan-300/20 bg-[linear-gradient(135deg,rgba(8,21,36,0.92),rgba(17,35,48,0.92))] p-5"
                  >
                    <div className="ambient-orbit pointer-events-none absolute -right-14 -top-14 h-44 w-44 rounded-full bg-cyan-300/15 blur-2xl" />
                    <div className="pointer-events-none absolute inset-0 bg-[repeating-linear-gradient(135deg,rgba(186,230,253,0.06)_0px,rgba(186,230,253,0.06)_1px,transparent_1px,transparent_14px)]" />
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-200">Start Here</p>
                    <p className="mt-2 text-sm text-zinc-200">Upload a terrain image, verify coordinates, and run prediction to unlock live risk cards and visual overlays.</p>
                  </motion.article>
                ) : null}

                <div className="grid grid-cols-12 gap-5 lg:gap-6">
                  <HeroRiskCard
                    loading={loading}
                    hasResult={Boolean(result)}
                    riskPercent={riskPercent}
                    riskLabel={result?.riskLabel || "MODERATE"}
                  />

                  <div className="col-span-12 space-y-5 xl:col-span-4">
                    <AnalystPanel loading={loading || insightLoading} hasResult={Boolean(result)} text={result?.explanation || ""} />

                    <article className="panel-soft card-lift p-5">
                      <p className="mb-4 text-[11px] font-semibold uppercase tracking-[0.16em] text-dim">Risk Factors</p>
                      <div className="space-y-3 text-xs">
                        <div className="flex items-center justify-between">
                          <span className="text-dim">Slope Instability</span>
                          <span className="font-semibold text-rose-300">{result ? `${result.metrics.slope.toFixed(1)} deg` : "--"}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-dim">Vegetation Density</span>
                          <span className="font-semibold text-rose-300">{result ? `${(result.metrics.vegetation * 100).toFixed(1)}%` : "--"}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-dim">Subsurface Hydrology</span>
                          <span className="font-semibold text-emerald-300">{result ? `${result.metrics.rainfall.toFixed(1)} mm` : "--"}</span>
                        </div>
                      </div>
                    </article>
                  </div>
                </div>

                <VisualAnalysisGrid tiles={visualTiles} showVisuals={showVisuals} onToggleVisuals={() => setShowVisuals((prev) => !prev)} />

                {imageStreamLoading ? (
                  <p className="-mt-3 text-xs text-zinc-400">Loading analysis images progressively...</p>
                ) : null}
              </motion.div>
            ) : null}

            {activeTab === "terrain" ? (
              <motion.div
                key="tab-terrain"
                initial={{ opacity: 0, y: 22, scale: 0.97, filter: "blur(6px)" }}
                animate={{ opacity: 1, y: 0, scale: 1, filter: "blur(0px)" }}
                exit={{ opacity: 0, y: -14, scale: 0.985, filter: "blur(4px)" }}
                transition={{ duration: 0.34, ease: [0.22, 0.9, 0.3, 1] }}
                className="space-y-6"
              >
                <div className="panel-soft rounded-3xl p-5">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-dim">Terrain Navigator</p>
                  <p className="mt-1 text-sm text-zinc-300">Select a center or polygon on map to generate point-wise erosion risk predictions.</p>
                  <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-cyan-300/20 bg-cyan-500/10 px-3 py-2 text-xs text-cyan-100">
                    <label className="inline-flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={livePointSampling}
                        onChange={(event) => setLivePointSampling(event.target.checked)}
                      />
                      Live point sampling
                    </label>
                    <button
                      type="button"
                      className="rounded-full border border-white/25 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-zinc-100 transition hover:bg-white/10 disabled:opacity-50"
                      onClick={() => {
                        invalidateTerrainBatchRequests();
                        setTerrainRiskPoints([]);
                        setTerrainPolygonVertices([]);
                      }}
                      disabled={terrainRiskPoints.length === 0 && terrainPolygonVertices.length === 0}
                    >
                      Clear points
                    </button>
                  </div>
                  <div className="mt-4 rounded-2xl border border-white/10 bg-[#141b26]/85 p-2">
                    <MapPicker
                      lat={Number(lat) || 0}
                      lon={Number(lon) || 0}
                      areaRadiusMeters={getEffectiveRadiusMeters()}
                      showRadiusArea={livePointSampling && terrainScanMode === "radius"}
                      polygonVertices={terrainScanMode === "polygon4" ? terrainPolygonVertices : []}
                      sampledPoints={terrainOverlayPoints}
                      onChange={onTerrainMapSelect}
                    />
                  </div>
                </div>

                <div className="panel-soft rounded-3xl p-5">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-dim">Point Risk Live Feed</p>
                      <p className="mt-1 text-sm text-zinc-300">Point-level risk is generated directly from selected map locations in terrain mode.</p>
                      <p className="mt-1 text-xs text-zinc-400">
                        {terrainScanMode === "radius"
                          ? `Area threshold active: ${getEffectiveRadiusMeters()} meters around selected center.`
                          : "Click 4 points to create a polygon, then area-wide prediction points are generated automatically."}
                      </p>
                      {terrainScanMode === "polygon4" ? (
                        <p className="mt-1 text-xs text-amber-200/90">
                          Area Covered: {terrainPolygonAreaSqMeters >= 1000000
                            ? `${(terrainPolygonAreaSqMeters / 1000000).toFixed(3)} km2`
                            : `${Math.round(terrainPolygonAreaSqMeters).toLocaleString()} m2`}
                        </p>
                      ) : null}
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="rounded-full border border-white/15 bg-white/5 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-zinc-200">
                        {terrainRiskPoints.length} points
                      </span>
                      <button
                        type="button"
                        className="rounded-full border border-white/25 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-zinc-100 transition hover:bg-white/10 disabled:opacity-50"
                        onClick={() => {
                          invalidateTerrainBatchRequests();
                          setTerrainRiskPoints([]);
                        }}
                        disabled={terrainRiskPoints.length === 0}
                      >
                        Remove History
                      </button>
                    </div>
                  </div>

                  {terrainBatchLoading ? (
                    <p className="mt-3 inline-flex items-center gap-2 rounded-full border border-sky-300/30 bg-sky-500/10 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.11em] text-sky-100">
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      {terrainBatchRevealTotal > 0
                        ? `Predicting point ${Math.min(terrainBatchRevealCount + 1, terrainBatchRevealTotal)} of ${terrainBatchRevealTotal}...`
                        : "Preparing points..."}
                    </p>
                  ) : null}

                  <div className="mt-3 flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      className="inline-flex items-center gap-2 rounded-full border border-cyan-200/40 bg-cyan-100 px-3 py-1.5 text-[11px] font-semibold text-cyan-950 transition hover:brightness-105 disabled:opacity-60"
                      onClick={openReportExport}
                      disabled={!result}
                    >
                      <FileDown className="h-3.5 w-3.5" />
                      Export Report
                    </button>
                    <button
                      type="button"
                      className="inline-flex items-center gap-2 rounded-full border border-white/25 bg-white/5 px-3 py-1.5 text-[11px] font-semibold text-zinc-100 transition hover:bg-white/10 disabled:opacity-60"
                      onClick={exportTerrainScanData}
                      disabled={terrainRiskPoints.length === 0}
                    >
                      <ImageDown className="h-3.5 w-3.5" />
                      Export JSON
                    </button>
                    <button
                      type="button"
                      className="inline-flex items-center gap-2 rounded-full border border-white/25 bg-white/5 px-3 py-1.5 text-[11px] font-semibold text-zinc-100 transition hover:bg-white/10 disabled:opacity-60"
                      onClick={exportTerrainScanCsv}
                      disabled={terrainRiskPoints.length === 0}
                    >
                      <ImageDown className="h-3.5 w-3.5" />
                      Export CSV
                    </button>
                    <button
                      type="button"
                      className="inline-flex items-center gap-2 rounded-full border border-white/25 bg-white/5 px-3 py-1.5 text-[11px] font-semibold text-zinc-100 transition hover:bg-white/10"
                      onClick={() => setCompactPointAnalysis((prev) => !prev)}
                    >
                      {compactPointAnalysis ? "Detailed Analysis" : "Compact Analysis"}
                    </button>
                  </div>

                  <div className="mt-3 grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
                    <div className="rounded-xl border border-white/10 bg-black/20 px-3 py-2">
                      <p className="text-[10px] uppercase tracking-[0.11em] text-dim">AIMODE</p>
                      <p className="mt-1 font-semibold text-zinc-100">{useAiInsight ? "Enabled" : "Standard"}</p>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-black/20 px-3 py-2">
                      <p className="text-[10px] uppercase tracking-[0.11em] text-dim">Insight Mode</p>
                      <p className="mt-1 font-semibold text-zinc-100">{result?.insightMode || "Pending"}</p>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-black/20 px-3 py-2">
                      <p className="text-[10px] uppercase tracking-[0.11em] text-dim">Insight Status</p>
                      <p className="mt-1 font-semibold text-zinc-100">{result?.insightStatus || "N/A"}</p>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-black/20 px-3 py-2">
                      <p className="text-[10px] uppercase tracking-[0.11em] text-dim">SHAP Factors</p>
                      <p className="mt-1 font-semibold text-zinc-100">{shapItems.length}</p>
                    </div>
                  </div>

                  {terrainRiskPoints.length === 0 ? (
                    <p className="mt-4 rounded-2xl border border-white/10 bg-black/20 px-4 py-3 text-sm text-zinc-400">
                      Turn on live point sampling and click map locations to view erosion risk instantly.
                    </p>
                  ) : (
                    <div className="mt-4 space-y-2">
                      {terrainRiskPoints.map((point, index) => {
                        const riskTone =
                          point.riskLabel === "HIGH"
                            ? "border-rose-300/30 bg-rose-500/10 text-rose-100"
                            : point.riskLabel === "MODERATE"
                              ? "border-amber-300/30 bg-amber-500/10 text-amber-100"
                              : "border-emerald-300/30 bg-emerald-500/10 text-emerald-100";

                        return (
                          <div
                            key={point.id}
                            className={`rounded-2xl border px-3 py-2 text-xs ${riskTone}`}
                          >
                            <div className="flex items-center justify-between gap-2">
                              <p className="font-semibold uppercase tracking-[0.11em]">Point {index + 1}</p>
                              <p className="font-semibold">{point.riskPercent}% {point.riskLabel}</p>
                            </div>
                            <p className="mt-1 text-[11px]">
                              Lat {point.lat.toFixed(6)}, Lon {point.lon.toFixed(6)}
                            </p>
                            <p className="mt-1 text-[11px] leading-relaxed opacity-95">{formatPointAnalysis(point.aiAnalysis)}</p>
                            <p className="mt-0.5 text-[10px] opacity-80">Updated {point.updatedAt}</p>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>

              </motion.div>
            ) : null}

            {activeTab === "agriculture" ? (
              <motion.div
                key="tab-agriculture"
                initial={{ opacity: 0, y: 22, scale: 0.97, filter: "blur(6px)" }}
                animate={{ opacity: 1, y: 0, scale: 1, filter: "blur(0px)" }}
                exit={{ opacity: 0, y: -14, scale: 0.985, filter: "blur(4px)" }}
                transition={{ duration: 0.34, ease: [0.22, 0.9, 0.3, 1] }}
                className="space-y-6"
              >
                <CropIntelligencePanel
                  data={cropData}
                  loading={cropLoading}
                  onRefreshSeason={(newSeason) => {
                    setCropSeason(newSeason);
                    fetchCropRecommendation(newSeason);
                  }}
                />
              </motion.div>
            ) : null}

            {activeTab === "analytics" ? (
              <motion.div
                key="tab-analytics"
                initial={{ opacity: 0, y: 22, scale: 0.97, filter: "blur(6px)" }}
                animate={{ opacity: 1, y: 0, scale: 1, filter: "blur(0px)" }}
                exit={{ opacity: 0, y: -14, scale: 0.985, filter: "blur(4px)" }}
                transition={{ duration: 0.34, ease: [0.22, 0.9, 0.3, 1] }}
                className="space-y-6"
              >
                {!result && !loading ? (
                  <article className="panel-soft cinematic-card relative overflow-hidden rounded-3xl border border-fuchsia-300/20 bg-[linear-gradient(120deg,rgba(32,14,39,0.9),rgba(17,23,34,0.9))] p-5">
                    <div className="ambient-orbit pointer-events-none absolute -right-16 top-0 h-44 w-44 rounded-full bg-fuchsia-300/15 blur-2xl" />
                    <div className="pointer-events-none absolute inset-0 bg-[repeating-radial-gradient(circle_at_center,rgba(244,114,182,0.12)_0px,transparent_16px,transparent_20px)] opacity-60" />
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-fuchsia-200">Analytics Locked</p>
                    <p className="mt-2 text-sm text-zinc-200">Run one prediction to open SHAP and terrain metrics. Then press AIMODE for full explainability.</p>
                  </article>
                ) : null}

                <div className="grid grid-cols-12 gap-5 lg:gap-6">
                  <MetricsPanel cards={metricCards} />
                  <ShapPanel items={shapItems} />
                </div>
                <AnalystPanel loading={loading || insightLoading} hasResult={Boolean(result)} text={result?.explanation || ""} />
              </motion.div>
            ) : null}
          </AnimatePresence>
        </section>
      </main>

      <AnimatePresence>
        {showReportPreview && result ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[68] overflow-y-auto bg-black/70 p-4 backdrop-blur-sm"
            onClick={() => setShowReportPreview(false)}
            role="dialog"
            aria-modal="true"
            aria-label="Export report"
          >
            <motion.div
              initial={{ y: 18, scale: 0.98, opacity: 0 }}
              animate={{ y: 0, scale: 1, opacity: 1 }}
              exit={{ y: 18, scale: 0.98, opacity: 0 }}
              transition={{ duration: 0.22 }}
              className="mx-auto w-full max-w-[860px]"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="mb-3 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-white/15 bg-[#0f1726]/92 p-4">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-zinc-400">Export Report</p>
                  <p className="mt-1 text-sm text-zinc-200">Preview the professional report and choose your output format.</p>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    className="inline-flex items-center gap-2 rounded-full border border-amber-200/40 bg-amber-100 px-3 py-1.5 text-xs font-semibold text-amber-900 transition hover:brightness-105 disabled:opacity-60"
                    disabled={Boolean(reportBusy)}
                    onClick={() => exportReport("quick")}
                  >
                    {reportBusy === "quick" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Zap className="h-3.5 w-3.5" />}
                    Instant JPG
                  </button>
                  <button
                    className="inline-flex items-center gap-2 rounded-full border border-sky-200/40 bg-sky-100 px-3 py-1.5 text-xs font-semibold text-sky-900 transition hover:brightness-105 disabled:opacity-60"
                    disabled={Boolean(reportBusy)}
                    onClick={() => exportReport("pdf")}
                  >
                    {reportBusy === "pdf" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <FileDown className="h-3.5 w-3.5" />}
                    Download PDF
                  </button>
                  <button
                    className="inline-flex items-center gap-2 rounded-full border border-emerald-200/40 bg-emerald-100 px-3 py-1.5 text-xs font-semibold text-emerald-900 transition hover:brightness-105 disabled:opacity-60"
                    disabled={Boolean(reportBusy)}
                    onClick={() => exportReport("png")}
                  >
                    {reportBusy === "png" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ImageDown className="h-3.5 w-3.5" />}
                    Download PNG
                  </button>
                  <button
                    className="rounded-full border border-white/20 bg-black/40 p-2 text-zinc-200 transition hover:bg-black/65 hover:text-white"
                    onClick={() => setShowReportPreview(false)}
                    aria-label="Close report preview"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              </div>

              <div className="mb-3 rounded-2xl border border-white/15 bg-[#0f1726]/92 p-4">
                <p className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-zinc-400">Included Sections</p>
                <div className="grid grid-cols-2 gap-2 text-sm text-zinc-100 md:grid-cols-3">
                  <label className="flex items-center gap-2"><input type="checkbox" checked={reportSections.summary} onChange={(event) => setReportSections((prev) => ({ ...prev, summary: event.target.checked }))} />Summary</label>
                  <label className="flex items-center gap-2"><input type="checkbox" checked={reportSections.location} onChange={(event) => setReportSections((prev) => ({ ...prev, location: event.target.checked }))} />Location</label>
                  <label className="flex items-center gap-2"><input type="checkbox" checked={reportSections.visuals} onChange={(event) => setReportSections((prev) => ({ ...prev, visuals: event.target.checked }))} />Visual Analysis</label>
                  <label className="flex items-center gap-2"><input type="checkbox" checked={reportSections.metrics} onChange={(event) => setReportSections((prev) => ({ ...prev, metrics: event.target.checked }))} />Terrain Metrics</label>
                  <label className="flex items-center gap-2"><input type="checkbox" checked={reportSections.shap} onChange={(event) => setReportSections((prev) => ({ ...prev, shap: event.target.checked }))} />SHAP Explanation</label>
                  <label className="flex items-center gap-2"><input type="checkbox" checked={reportSections.analyst} onChange={(event) => setReportSections((prev) => ({ ...prev, analyst: event.target.checked }))} />AI Analyst Notes</label>
                </div>
              </div>

              <div className="mb-3 rounded-2xl border border-white/15 bg-[#0f1726]/92 p-4">
                <p className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-zinc-400">Metadata</p>
                <label className="text-xs text-zinc-200">
                  Region Name
                  <input
                    className="mt-1 w-full rounded-lg border border-white/20 bg-black/35 px-3 py-2 text-sm text-white outline-none focus:border-sky-300/50"
                    value={reportRegionName}
                    onChange={(event) => setReportRegionName(event.target.value)}
                    placeholder="District / Region"
                  />
                </label>
              </div>

              <div className="overflow-auto rounded-2xl border border-white/15 bg-white shadow-2xl">
                <div ref={reportRef} className="mx-auto w-[794px]">
                  <ReportDocument
                    generatedAt={reportGeneratedAt}
                    riskPercent={riskPercent}
                    riskLabel={result.riskLabel}
                    explanation={result.explanation}
                    metrics={result.metrics}
                    shapItems={shapItems}
                    visuals={reportVisuals}
                    sections={reportSections}
                    regionName={reportRegionName}
                    mapSnapshotUrl={reportMapSnapshotUrl}
                    inputImageUrl={result.images.original || uploadPreviewUrl}
                    cropIntelligence={cropData}
                  />
                </div>
              </div>

              {reportBusy ? (
                <p className="mt-3 inline-flex items-center gap-2 rounded-full border border-white/20 bg-[#0f1726]/92 px-3 py-1.5 text-xs text-zinc-200">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Generating report...
                </p>
              ) : null}

              {!reportBusy && reportFeedback.includes("successfully") ? (
                <p className="mt-3 inline-flex items-center gap-2 rounded-full border border-emerald-300/30 bg-emerald-500/15 px-3 py-1.5 text-xs text-emerald-100">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Report downloaded successfully
                </p>
              ) : null}
            </motion.div>
          </motion.div>
        ) : null}

        {showSettings ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[65] flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
            onClick={() => setShowSettings(false)}
            role="dialog"
            aria-modal="true"
            aria-label="Prediction settings"
          >
            <motion.div
              initial={{ y: 14, scale: 0.98, opacity: 0 }}
              animate={{ y: 0, scale: 1, opacity: 1 }}
              exit={{ y: 14, scale: 0.98, opacity: 0 }}
              transition={{ duration: 0.22 }}
              className="panel w-full max-w-lg p-5"
              onClick={(e) => e.stopPropagation()}
            >
            <div className="mb-4 flex items-center justify-between">
              <h3 className="font-[var(--font-sora)] text-lg text-white">Prediction Settings</h3>
              <button
                className="rounded-lg border border-white/20 bg-black/40 p-1.5 text-zinc-200 transition hover:bg-black/65 hover:text-white"
                onClick={() => setShowSettings(false)}
                aria-label="Close settings"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <div className="mb-2 flex items-center justify-between text-xs text-dim">
                  <span>Animation Mode</span>
                  <span className="uppercase">{motionMode}</span>
                </div>
                <div className="grid grid-cols-2 gap-2 rounded-xl border border-white/10 bg-black/20 p-1">
                  <button
                    type="button"
                    className={`rounded-lg px-3 py-2 text-xs font-semibold transition ${motionMode === "full" ? "bg-white text-slate-900" : "text-dim hover:text-white"}`}
                    onClick={() => setMotionMode("full")}
                  >
                    Full Motion
                  </button>
                  <button
                    type="button"
                    className={`rounded-lg px-3 py-2 text-xs font-semibold transition ${motionMode === "reduced" ? "bg-white text-slate-900" : "text-dim hover:text-white"}`}
                    onClick={() => setMotionMode("reduced")}
                  >
                    Reduced Motion
                  </button>
                </div>
              </div>

              <div>
                <div className="mb-2 flex items-center justify-between text-xs text-dim">
                  <span>Detection Confidence</span>
                  <span>{confidence.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={0.05}
                  max={0.95}
                  step={0.05}
                  value={confidence}
                  onChange={(e) => updateConfidence(Number(e.target.value))}
                  onWheel={(e) => {
                    e.preventDefault();
                    const direction = e.deltaY > 0 ? -1 : 1;
                    updateConfidence(confidence + direction * 0.05);
                  }}
                  className="h-2 w-full cursor-pointer"
                />
                <p className="mt-1 text-[11px] text-dim">Tip: scroll on the bar to adjust confidence.</p>
              </div>

              <div className="grid grid-cols-2 gap-2 text-sm text-ink">
                <label className="flex items-center gap-2"><input type="checkbox" checked={showVegetation} onChange={(e) => setShowVegetation(e.target.checked)} />Vegetation</label>
                <label className="flex items-center gap-2"><input type="checkbox" checked={showRuins} onChange={(e) => setShowRuins(e.target.checked)} />Ruins</label>
                <label className="flex items-center gap-2"><input type="checkbox" checked={showStructures} onChange={(e) => setShowStructures(e.target.checked)} />Structures</label>
                <label className="flex items-center gap-2"><input type="checkbox" checked={showBoulders} onChange={(e) => setShowBoulders(e.target.checked)} />Boulders</label>
                <label className="flex items-center gap-2"><input type="checkbox" checked={showOthers} onChange={(e) => setShowOthers(e.target.checked)} />Others</label>
                <label className="flex items-center gap-2"><input type="checkbox" checked={fastMode} onChange={(e) => setFastMode(e.target.checked)} />Fast Mode</label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={useAiInsight}
                    disabled={insightLoading}
                    onChange={(e) => {
                      onToggleAiMode(e.target.checked);
                    }}
                  />
                  {insightLoading ? "AI Insight (loading...)" : "AI Insight (on demand)"}
                </label>
              </div>
            </div>
            </motion.div>
          </motion.div>
        ) : null}
      </AnimatePresence>
      </div>
    </MotionConfig>
  );
}

export default function Page() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-surface" />}>
      <PageContent />
    </Suspense>
  );
}
