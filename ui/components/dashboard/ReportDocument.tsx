"use client";

/* eslint-disable @next/next/no-img-element */

import { useMemo } from "react";
import { type CropIntelligenceData } from "./CropIntelligencePanel";

type RiskLabel = "LOW" | "MODERATE" | "HIGH";

type ReportSections = {
  summary: boolean;
  location: boolean;
  visuals: boolean;
  metrics: boolean;
  shap: boolean;
  analyst: boolean;
};

type ShapItem = {
  feature: string;
  value: number;
};

type VisualItem = {
  title: string;
  src: string;
};

type MetricsData = {
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

type ReportDocumentProps = {
  generatedAt: Date;
  riskPercent: number;
  riskLabel: RiskLabel;
  explanation: string;
  metrics: MetricsData;
  shapItems: ShapItem[];
  visuals: VisualItem[];
  sections: ReportSections;
  regionName?: string;
  mapSnapshotUrl?: string;
  inputImageUrl?: string;
  cropIntelligence?: CropIntelligenceData | null;
};

function toneByRisk(riskLabel: RiskLabel) {
  if (riskLabel === "LOW") {
    return {
      pill: "bg-emerald-100 text-emerald-700 border-emerald-300",
      marker: "bg-emerald-500",
      text: "text-emerald-700",
    };
  }
  if (riskLabel === "MODERATE") {
    return {
      pill: "bg-amber-100 text-amber-700 border-amber-300",
      marker: "bg-amber-500",
      text: "text-amber-700",
    };
  }
  return {
    pill: "bg-rose-100 text-rose-700 border-rose-300",
    marker: "bg-rose-500",
    text: "text-rose-700",
  };
}

function formatFeatureName(raw: string) {
  return raw
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase())
    .trim();
}

function inferOneLineSummary(riskLabel: RiskLabel, metrics: MetricsData) {
  const signals: string[] = [];
  if (metrics.slope >= 24) signals.push("steep slope");
  if (metrics.vegetation <= 0.3) signals.push("low vegetation cover");
  if (metrics.rainfall >= 90) signals.push("heavy rainfall pattern");

  if (signals.length === 0) {
    return `${formatFeatureName(riskLabel)} erosion risk with balanced terrain indicators.`;
  }

  const joined = signals.length === 1
    ? signals[0]
    : `${signals.slice(0, -1).join(", ")} and ${signals[signals.length - 1]}`;

  return `${formatFeatureName(riskLabel)} erosion risk due to ${joined}.`;
}

function highlightImportantTerms(text: string) {
  const pattern = /(slope|vegetation|rainfall|erosion|runoff|soil|elevation|risk)/gi;
  const parts = text.split(pattern);

  return parts.map((part, index) => {
    const match = part.match(pattern);
    if (!match) {
      return <span key={`plain-${index}`}>{part}</span>;
    }
    return (
      <strong key={`strong-${index}`} className="font-semibold text-slate-900">
        {part}
      </strong>
    );
  });
}

export default function ReportDocument({
  generatedAt,
  riskPercent,
  riskLabel,
  explanation,
  metrics,
  shapItems,
  visuals,
  sections,
  regionName,
  mapSnapshotUrl,
  inputImageUrl,
  cropIntelligence,
}: ReportDocumentProps) {
  const tone = toneByRisk(riskLabel);

  const rankedShap = useMemo(() => {
    return [...shapItems]
      .filter((item) => Number.isFinite(item.value) && item.feature)
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, 7);
  }, [shapItems]);

  const positiveDrivers = useMemo(() => {
    return rankedShap.filter((item) => item.value > 0).slice(0, 3);
  }, [rankedShap]);

  const negativeDrivers = useMemo(() => {
    return rankedShap.filter((item) => item.value < 0).slice(0, 3);
  }, [rankedShap]);

  const maxImpact = useMemo(
    () => Math.max(0.001, ...rankedShap.map((item) => Math.abs(item.value))),
    [rankedShap]
  );

  const scoredShap = useMemo(() => {
    return rankedShap.map((item, index) => {
      const relative = Math.abs(item.value) / maxImpact;
      const score = Math.max(0.1, Number((relative * 5).toFixed(1)));
      const level =
        relative >= 0.75 ? "Very Strong" :
        relative >= 0.5 ? "Strong" :
        relative >= 0.25 ? "Moderate" :
        "Mild";
      return {
        ...item,
        rank: index + 1,
        score,
        level,
      };
    });
  }, [rankedShap, maxImpact]);

  const summaryLine = useMemo(() => inferOneLineSummary(riskLabel, metrics), [metrics, riskLabel]);

  return (
    <div className="w-[794px] bg-[#f8fafc] p-10 text-slate-700">
      <header className="rounded-2xl border border-slate-200 bg-white px-7 py-6 shadow-sm">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">Geo AI System</p>
            <h1 className="mt-2 font-[var(--font-sora)] text-3xl font-semibold text-slate-900">Geo AI Terrain Risk Analysis Report</h1>
            <p className="mt-2 text-xs text-slate-500">Generated on {generatedAt.toLocaleString()}</p>
          </div>
          {inputImageUrl ? (
            <div className="w-36 rounded-xl border border-slate-200 bg-slate-50 p-2">
              <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-500">Input Glimpse</p>
              <img src={inputImageUrl} alt="Uploaded terrain glimpse" className="mt-2 h-20 w-full rounded-md border border-slate-200 object-cover" crossOrigin="anonymous" />
            </div>
          ) : (
            <div className="flex h-20 w-36 items-center justify-center rounded-xl border border-dashed border-slate-300 bg-slate-50 text-[11px] text-slate-400">
              Input image unavailable
            </div>
          )}
        </div>
      </header>

      {sections.summary ? (
        <section className="mt-6 rounded-2xl border border-slate-200 bg-white px-7 py-6 shadow-sm">
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Executive Summary</p>
          <div className="mt-4 flex items-end justify-between gap-6">
            <div>
              <p className="text-xs text-slate-500">Predicted Erosion Risk</p>
              <p className="mt-2 font-[var(--font-sora)] text-6xl font-semibold leading-none text-slate-900">{riskPercent}%</p>
            </div>
            <div className="flex flex-col items-end gap-3">
              <span className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 text-xs font-semibold uppercase tracking-[0.12em] ${tone.pill}`}>
                <span className={`h-2.5 w-2.5 rounded-full ${tone.marker}`} />
                {riskLabel} Risk
              </span>
              <p className="max-w-sm text-right text-sm text-slate-600">{summaryLine}</p>
            </div>
          </div>
        </section>
      ) : null}

      {sections.location ? (
        <section className="mt-6 rounded-2xl border border-slate-200 bg-white px-7 py-6 shadow-sm">
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Location Details</p>
          <div className="mt-4 grid grid-cols-[minmax(0,1fr)_280px] gap-5">
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                <p className="text-xs text-slate-500">Latitude</p>
                <p className="mt-1 font-semibold text-slate-900">{metrics.lat.toFixed(6)}</p>
              </div>
              <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                <p className="text-xs text-slate-500">Longitude</p>
                <p className="mt-1 font-semibold text-slate-900">{metrics.lon.toFixed(6)}</p>
              </div>
              <div className="col-span-2 rounded-xl border border-slate-200 bg-slate-50 p-3">
                <p className="text-xs text-slate-500">Region</p>
                <p className="mt-1 font-semibold text-slate-900">{regionName?.trim() || "Region not provided"}</p>
              </div>
            </div>
            <div className="rounded-xl border border-slate-200 bg-[linear-gradient(140deg,#e2e8f0,#f8fafc)] p-4">
              <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Map Snapshot</p>
              {mapSnapshotUrl ? (
                <img
                  src={mapSnapshotUrl}
                  alt="Location snapshot"
                  className="mt-3 h-28 w-full rounded-lg border border-slate-200 object-cover"
                  crossOrigin="anonymous"
                />
              ) : (
                <div className="mt-3 flex h-28 items-center justify-center rounded-lg border border-dashed border-slate-300 bg-white/75 text-xs text-slate-500">
                  Snapshot unavailable
                </div>
              )}
            </div>
          </div>
        </section>
      ) : null}

      {sections.visuals ? (
        <section className="mt-6 rounded-2xl border border-slate-200 bg-white px-7 py-6 shadow-sm">
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Visual Analysis</p>
          <div className="mt-4 grid grid-cols-2 gap-4">
            {visuals.map((item) => (
              <figure key={item.title} className="overflow-hidden rounded-xl border border-slate-200 bg-slate-50">
                {item.src ? (
                  <img src={item.src} alt={item.title} className="h-40 w-full object-cover" crossOrigin="anonymous" />
                ) : (
                  <div className="h-40 w-full bg-[linear-gradient(140deg,#dbeafe,#e2e8f0)]" />
                )}
                <figcaption className="border-t border-slate-200 px-3 py-2 text-xs font-medium text-slate-600">{item.title}</figcaption>
              </figure>
            ))}
          </div>
        </section>
      ) : null}

      {sections.metrics ? (
        <section className="mt-6 rounded-2xl border border-slate-200 bg-white px-7 py-6 shadow-sm">
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Terrain Metrics</p>
          <div className="mt-4 overflow-hidden rounded-xl border border-slate-200">
            <table className="w-full border-collapse text-sm">
              <tbody>
                <tr className="border-b border-slate-200 bg-slate-50/80">
                  <th className="px-4 py-3 text-left font-semibold text-slate-600">Slope</th>
                  <td className="px-4 py-3 font-medium text-slate-900">{metrics.slope.toFixed(1)} deg</td>
                  <th className="px-4 py-3 text-left font-semibold text-slate-600">Vegetation</th>
                  <td className="px-4 py-3 font-medium text-slate-900">{(metrics.vegetation * 100).toFixed(1)}%</td>
                </tr>
                <tr className="border-b border-slate-200 bg-white">
                  <th className="px-4 py-3 text-left font-semibold text-slate-600">Elevation</th>
                  <td className="px-4 py-3 font-medium text-slate-900">{metrics.elevation.toFixed(1)} m</td>
                  <th className="px-4 py-3 text-left font-semibold text-slate-600">Rainfall</th>
                  <td className="px-4 py-3 font-medium text-slate-900">{metrics.rainfall.toFixed(1)} mm</td>
                </tr>
                <tr className="border-b border-slate-200 bg-slate-50/80">
                  <th className="px-4 py-3 text-left font-semibold text-slate-600">Soil Type</th>
                  <td className="px-4 py-3 font-medium text-slate-900">{metrics.soil}</td>
                  <th className="px-4 py-3 text-left font-semibold text-slate-600">Structures</th>
                  <td className="px-4 py-3 font-medium text-slate-900">{metrics.structures.toFixed(3)}</td>
                </tr>
                <tr className="bg-white">
                  <th className="px-4 py-3 text-left font-semibold text-slate-600">Ruins</th>
                  <td className="px-4 py-3 font-medium text-slate-900">{metrics.ruins.toFixed(3)}</td>
                  <th className="px-4 py-3 text-left font-semibold text-slate-600">Boulders</th>
                  <td className="px-4 py-3 font-medium text-slate-900">{metrics.boulders.toFixed(3)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {sections.shap ? (
        <section className="mt-6 rounded-2xl border border-slate-200 bg-white px-7 py-6 shadow-sm">
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Model Explanation (SHAP)</p>
          <div className="mt-5 space-y-3">
            {scoredShap.length === 0 ? (
              <p className="text-sm text-slate-500">SHAP values are not available for this prediction.</p>
            ) : (
              scoredShap.map((item) => {
                const width = Math.max(12, Math.round((Math.abs(item.value) / maxImpact) * 100));
                const increasesRisk = item.value > 0;
                return (
                  <div key={item.feature} className="space-y-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-medium text-slate-700">#{item.rank} {formatFeatureName(item.feature)}</span>
                      <div className="text-right">
                        <p className={increasesRisk ? "font-semibold text-rose-600" : "font-semibold text-emerald-600"}>
                          {increasesRisk ? "Raises risk" : "Lowers risk"}
                        </p>
                        <p className="text-[10px] text-slate-500">{item.level} - Score {item.score}/5</p>
                      </div>
                    </div>
                    <div className="h-2.5 overflow-hidden rounded-full bg-slate-200">
                      <div className={`h-full ${increasesRisk ? "bg-rose-500" : "bg-emerald-500"}`} style={{ width: `${width}%` }} />
                    </div>
                  </div>
                );
              })
            )}
          </div>

          <div className="mt-5 grid grid-cols-2 gap-4 text-sm">
            <div className="rounded-xl border border-rose-200 bg-rose-50/60 p-3">
              <p className="text-xs font-semibold uppercase tracking-[0.14em] text-rose-700">Positive Contributors</p>
              <ul className="mt-2 space-y-1 text-rose-800">
                {positiveDrivers.length > 0
                  ? positiveDrivers.map((item) => <li key={item.feature}>- {formatFeatureName(item.feature)}</li>)
                  : <li>- None detected</li>}
              </ul>
            </div>
            <div className="rounded-xl border border-emerald-200 bg-emerald-50/60 p-3">
              <p className="text-xs font-semibold uppercase tracking-[0.14em] text-emerald-700">Negative Contributors</p>
              <ul className="mt-2 space-y-1 text-emerald-800">
                {negativeDrivers.length > 0
                  ? negativeDrivers.map((item) => <li key={item.feature}>- {formatFeatureName(item.feature)}</li>)
                  : <li>- None detected</li>}
              </ul>
            </div>
          </div>

          <p className="mt-4 text-sm text-slate-600">
            Vegetation reduces erosion risk, while slope increases runoff and soil displacement.
          </p>
        </section>
      ) : null}

      {sections.analyst ? (
        <section className="mt-6 rounded-2xl border border-slate-200 bg-white px-7 py-6 shadow-sm">
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">AI Analyst Explanation</p>
          <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm leading-7 text-slate-700">
            {highlightImportantTerms(explanation || "No AI explanation provided for this prediction.")}
          </div>
        </section>
      ) : null}

      {cropIntelligence ? (
        <section className="mt-6 rounded-2xl border border-emerald-200 bg-white px-7 py-6 shadow-sm">
          <div className="flex items-center justify-between border-b border-emerald-100 pb-3">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-emerald-700">
                Agricultural Intelligence (Multi-Objective Decision Support)
              </p>
              <h3 className="mt-1 text-xl font-bold text-slate-900">
                Primary Recommended Crop: {cropIntelligence.primary_recommendation.crop}
              </h3>
            </div>
            <div className="text-right">
              <span className="inline-block rounded-full bg-emerald-100 px-3 py-1 text-xs font-bold text-emerald-800 border border-emerald-300">
                Suitability: {cropIntelligence.primary_recommendation.suitability_percent}%
              </span>
              <p className="mt-0.5 text-[10px] text-slate-500">
                Model Confidence: {cropIntelligence.primary_recommendation.confidence_percent}%
              </p>
            </div>
          </div>

          {/* Model Deterministic Output Notice */}
          <div className="mt-4 rounded-lg bg-emerald-50/60 border border-emerald-200 p-3">
            <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-800 block">
              [MODEL DETERMINISTIC OUTPUT] — Agro-Climatic Profile & Multi-Objective Ranking
            </span>
            <div className="mt-2 grid grid-cols-3 gap-3 text-xs text-slate-700">
              <div>
                <span className="text-slate-500 block">Season / Zone</span>
                <span className="font-semibold">{cropIntelligence.location.season} • {cropIntelligence.location.agro_climatic_zone}</span>
              </div>
              <div>
                <span className="text-slate-500 block">Soil pH & Texture</span>
                <span className="font-semibold">{cropIntelligence.soil.soil_ph.toFixed(1)} • {cropIntelligence.soil.soil_texture_class}</span>
              </div>
              <div>
                <span className="text-slate-500 block">NPK Nutrients (kg/ha)</span>
                <span className="font-semibold">N:{cropIntelligence.soil.nitrogen.toFixed(0)} P:{cropIntelligence.soil.phosphorus.toFixed(0)} K:{cropIntelligence.soil.potassium.toFixed(0)}</span>
              </div>
              <div>
                <span className="text-slate-500 block">7d Forecast Rain</span>
                <span className="font-semibold">{cropIntelligence.weather.forecast_rainfall_7d.toFixed(1)} mm</span>
              </div>
              <div>
                <span className="text-slate-500 block">Soil Moisture</span>
                <span className="font-semibold">{cropIntelligence.weather.soil_moisture.toFixed(3)} m³/m³</span>
              </div>
              <div>
                <span className="text-slate-500 block">Erosion Risk Context</span>
                <span className="font-semibold">{cropIntelligence.terrain.erosion_risk_label} ({cropIntelligence.terrain.erosion_risk_score.toFixed(2)})</span>
              </div>
            </div>
          </div>

          {/* Alternative Crops */}
          <div className="mt-4">
            <span className="text-xs font-semibold text-slate-800 block mb-2">Suitable Alternative Crops:</span>
            <div className="grid grid-cols-4 gap-2">
              {cropIntelligence.recommendations.slice(1).map((alt, idx) => (
                <div key={alt.crop} className="rounded-lg border border-slate-200 bg-slate-50 p-2 text-center text-xs">
                  <span className="font-bold text-slate-800 block">#{idx + 2} {alt.crop}</span>
                  <span className="text-emerald-700 font-medium">{alt.suitability_percent}% suit.</span>
                </div>
              ))}
            </div>
          </div>

          {/* Natural Language Presentation Notice */}
          <div className="mt-4 rounded-lg bg-slate-50 border border-slate-200 p-3">
            <span className="text-[10px] font-bold uppercase tracking-wider text-slate-600 block">
              [AI-GENERATED NATURAL LANGUAGE EXPLANATION] — Downstream Agronomic Synthesis
            </span>
            <p className="mt-1 text-xs text-slate-700 leading-relaxed">
              {cropIntelligence.explanation.summary}
            </p>
          </div>

          {/* Data Quality & Limitations */}
          <div className="mt-3 flex items-center justify-between text-[10px] text-slate-500 border-t border-slate-100 pt-2">
            <span>Data Quality: Soil ({cropIntelligence.data_quality.soil_data_quality}), Weather ({cropIntelligence.data_quality.weather_data_quality}), Terrain ({cropIntelligence.data_quality.terrain_data_quality})</span>
            <span className="italic">Advisory only; does not guarantee agricultural yield or market success.</span>
          </div>
        </section>
      ) : null}

      <footer className="mt-6 border-t border-slate-300 pt-4 text-xs text-slate-500">
        <p>Generated by Geo AI System</p>
        <p className="mt-1">This report is AI-generated and should be used for analysis support.</p>
      </footer>
    </div>
  );
}
