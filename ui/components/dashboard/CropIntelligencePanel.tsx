"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Sprout,
  ShieldCheck,
  AlertTriangle,
  CloudRain,
  Thermometer,
  Layers,
  ArrowRight,
  RefreshCw,
  Sparkles,
  Info,
  CheckCircle2,
  Droplets,
  ChevronDown,
  ChevronUp,
  Sliders,
  Activity,
  TrendingUp,
  ShieldAlert,
} from "lucide-react";

export type CropRecommendationItem = {
  crop: string;
  suitability_score: number;
  suitability_percent: number;
  confidence: number;
  confidence_percent: number;
  positive_drivers: string[];
  limiting_factors: string[];
  expected_yield_t_ha?: number;
  expected_yield_range?: string;
  relative_yield_potential_percent?: number;
  risk?: "Low" | "Moderate" | "High" | string;
  risk_breakdown?: {
    weather_risk?: string;
    terrain_risk?: string;
    erosion_risk?: string;
    limiting_factors?: string[];
  };
};

export type CounterfactualScenario = {
  scenario: string;
  top_crop: string;
  confidence_percent: number;
  changed_from_baseline: boolean;
  description: string;
};

export type CropIntelligenceData = {
  location: {
    latitude: number;
    longitude: number;
    season: string;
    agro_climatic_zone: string;
    zone_description: string;
  };
  soil: {
    soil_ph: number;
    nitrogen: number;
    phosphorus: number;
    potassium: number;
    organic_carbon: number;
    clay: number;
    sand: number;
    silt: number;
    electrical_conductivity: number;
    soil_texture_class: string;
    soil_data_source: string;
    soil_data_quality: string;
  };
  weather: {
    temperature_mean: number;
    temperature_min: number;
    temperature_max: number;
    humidity_mean: number;
    soil_moisture: number;
    rainfall_7d: number;
    forecast_rainfall_7d: number;
    forecast_precip_probability: number;
    et0: number;
    weather_data_quality: string;
  };
  terrain: {
    elevation: number;
    slope: number;
    erosion_risk_score: number;
    erosion_risk_label: "LOW" | "MODERATE" | "HIGH";
    terrain_data_quality: string;
  };
  erosion: {
    risk_score: number;
    risk_label: string;
    advisory: string;
  };
  recommendations: CropRecommendationItem[];
  primary_recommendation: CropRecommendationItem;
  explanation: {
    summary: string;
    insight_mode: string;
    insight_status: string | null;
  };
  data_quality: {
    soil_data_quality: string;
    weather_data_quality: string;
    terrain_data_quality: string;
    overall_data_quality: string;
  };
  uncertainty?: {
    decision_margin: number;
    decision_margin_percent: number;
    ranking_stability: "High Stability" | "Moderate Stability" | "Close Decision (Multiple Crops Viable)" | string;
    recommendation_uncertainty: number;
    entropy?: number;
    normalized_entropy?: number;
    interpretation?: string;
    advisory?: string;
  };
  risk?: {
    overall_risk: "Low" | "Moderate" | "High" | string;
    weather_risk: string;
    terrain_risk: string;
    erosion_risk: string;
    advisory?: string;
  };
  drivers?: {
    model_shap_drivers?: string[];
    agronomic_compatibility?: string[];
    risk_drivers?: string[];
  };
  counterfactuals?: CounterfactualScenario[];
  model?: {
    name: string;
    version: string;
    dataset_version: string;
  };
  limitations: string[];
};


type Props = {
  data: CropIntelligenceData | null;
  loading: boolean;
  onRefreshSeason: (newSeason: string) => void;
};

const CROP_EMOJIS: Record<string, string> = {
  Rice: "🌾",
  Wheat: "🌾",
  Maize: "🌽",
  Groundnut: "🥜",
  Cotton: "🌱",
  Sorghum: "🌾",
  "Pearl Millet": "🌾",
  Chickpea: "🫘",
  Pigeonpea: "🫛",
  Moong: "🌱",
  Urad: "🫘",
  Sugarcane: "🎋",
  Soybean: "🌱",
  Mustard: "🌼",
  "Finger Millet": "🌾",
  Sesamum: "🌰",
};

export default function CropIntelligencePanel({ data, loading, onRefreshSeason }: Props) {
  const [selectedSeason, setSelectedSeason] = useState<string>("Kharif");
  const [showCounterfactuals, setShowCounterfactuals] = useState<boolean>(false);
  const [showDiagnostics, setShowDiagnostics] = useState<boolean>(false);

  const handleSeasonChange = (season: string) => {
    setSelectedSeason(season);
    onRefreshSeason(season);
  };

  if (loading) {
    return (
      <div className="panel col-span-12 p-8 md:p-12 relative overflow-hidden">
        <div className="flex flex-col items-center justify-center text-center py-16 space-y-4">
          <div className="relative">
            <Sprout className="w-14 h-14 text-emerald-400 animate-pulse" />
            <div className="absolute inset-0 bg-emerald-500/20 blur-xl rounded-full" />
          </div>
          <h3 className="font-[var(--font-sora)] text-2xl text-white font-semibold">
            Analyzing Agro-Climatic Intelligence...
          </h3>
          <p className="text-sm text-dim max-w-md">
            Querying SoilGrids, Open-Meteo weather forecasts, terrain slope, and erosion context...
          </p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="panel col-span-12 p-8 relative overflow-hidden text-center py-12">
        <Sprout className="w-10 h-10 text-emerald-400/60 mx-auto mb-3" />
        <h3 className="font-[var(--font-sora)] text-xl text-white font-medium">
          Agricultural Intelligence Ready
        </h3>
        <p className="text-sm text-dim max-w-lg mx-auto mt-2">
          Select coordinates and trigger crop recommendation to receive evidence-based suitability assessments.
        </p>
      </div>
    );
  }

  const primary = data.primary_recommendation;
  const alternatives = data.recommendations.slice(1);
  const primaryEmoji = CROP_EMOJIS[primary.crop] || "🌱";

  return (
    <div className="col-span-12 space-y-6">
      {/* Top Banner & Season Selector */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 p-5 rounded-xl border border-white/10 bg-slate-900/80 backdrop-blur-md">
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <Sprout className="w-5 h-5" />
            </span>
            <h2 className="text-xl font-semibold text-white font-[var(--font-sora)]">
              AI Agricultural Intelligence
            </h2>
            <span className="px-2 py-0.5 text-[10px] font-mono font-semibold tracking-wider uppercase rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              Model: {data.model?.version?.toUpperCase() || "V2"}
            </span>
            <span className="px-2 py-0.5 text-[10px] font-mono tracking-wider rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              Validation: Geographic + Temporal
            </span>
          </div>
          <p className="text-xs text-dim mt-1">
            Zone: <span className="text-white font-medium">{data.location.zone_description}</span> (Lat {data.location.latitude.toFixed(2)}, Lon {data.location.longitude.toFixed(2)})
          </p>

        </div>

        {/* Season Selector Buttons */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-dim mr-1">Season:</span>
          {["Kharif", "Rabi", "Summer", "Whole Year"].map((s) => (
            <button
              key={s}
              onClick={() => handleSeasonChange(s)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                selectedSeason === s
                  ? "bg-emerald-500 text-slate-950 font-semibold shadow-md shadow-emerald-500/20"
                  : "bg-white/5 text-dim hover:text-white hover:bg-white/10"
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {/* Main Grid: Primary Crop + Rationale */}
      <div className="grid grid-cols-12 gap-6">
        {/* Primary Recommended Crop Card */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          className="col-span-12 lg:col-span-6 rounded-2xl p-6 md:p-8 border border-emerald-500/30 bg-gradient-to-br from-emerald-950/40 via-slate-900/90 to-slate-950 relative overflow-hidden"
        >
          <div className="absolute top-0 right-0 w-64 h-64 bg-emerald-500/10 blur-3xl pointer-events-none rounded-full" />
          
          <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-[11px] uppercase tracking-wider font-semibold px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                Primary Candidate
              </span>
              {primary.risk && (
                <span className={`text-[11px] font-semibold px-2.5 py-0.5 rounded-full border ${
                  primary.risk === "Low"
                    ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                    : primary.risk === "Moderate"
                    ? "bg-amber-500/10 text-amber-400 border-amber-500/30"
                    : "bg-rose-500/10 text-rose-400 border-rose-500/30"
                }`}>
                  Risk: {primary.risk}
                </span>
              )}
              {data.uncertainty && (
                <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                  data.uncertainty.ranking_stability.includes("High")
                    ? "bg-emerald-500/10 text-emerald-300 border-emerald-500/20"
                    : data.uncertainty.ranking_stability.includes("Moderate")
                    ? "bg-sky-500/10 text-sky-300 border-sky-500/20"
                    : "bg-amber-500/10 text-amber-300 border-amber-500/20"
                }`}>
                  {data.uncertainty.ranking_stability}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2 text-xs text-dim">
              <span>Overall Quality:</span>
              <span className={`font-semibold ${
                data.data_quality.overall_data_quality === "HIGH" ? "text-emerald-400" : "text-amber-400"
              }`}>
                {data.data_quality.overall_data_quality}
              </span>
            </div>
          </div>

          <div className="flex items-baseline gap-4 my-2">
            <span className="text-5xl">{primaryEmoji}</span>
            <div>
              <h3 className="text-4xl md:text-5xl font-bold text-white font-[var(--font-sora)]">
                {primary.crop}
              </h3>
              <p className="text-xs text-dim mt-1">Recommended for {data.location.season} planting cycle</p>
              {primary.expected_yield_range && (
                <div className="inline-flex items-center gap-2 mt-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-xs">
                  <span className="text-emerald-400 font-semibold">🌾 Expected Yield:</span>
                  <span className="text-white font-mono font-bold">{primary.expected_yield_range}</span>
                  {primary.relative_yield_potential_percent !== undefined && (
                    <span className="text-dim text-[11px]">({primary.relative_yield_potential_percent}% potential)</span>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Scores Breakdown: Suitability vs Confidence */}
          <div className="grid grid-cols-2 gap-4 my-6 p-4 rounded-xl bg-black/30 border border-white/5">
            <div>
              <div className="flex items-center justify-between text-xs text-dim mb-1">
                <span>Suitability Score</span>
                <span className="font-semibold text-emerald-400">{primary.suitability_percent}%</span>
              </div>
              <div className="w-full h-2 rounded-full bg-white/10 overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-700"
                  style={{ width: `${primary.suitability_percent}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between text-xs text-dim mb-1">
                <span>Model Confidence</span>
                <span className="font-semibold text-sky-400">{primary.confidence_percent}%</span>
              </div>
              <div className="w-full h-2 rounded-full bg-white/10 overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-sky-500 to-indigo-400 transition-all duration-700"
                  style={{ width: `${primary.confidence_percent}%` }}
                />
              </div>
            </div>
          </div>

          {/* Key Positive Drivers (SHAP + Domain) */}
          <div className="space-y-2 mt-4">
            <h4 className="text-xs uppercase tracking-wider text-dim font-medium flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              Key Positive Factors
            </h4>
            <ul className="space-y-1.5 text-xs text-slate-200">
              {primary.positive_drivers.map((driver, i) => (
                <li key={i} className="flex items-start gap-2 bg-emerald-500/5 p-2 rounded-lg border border-emerald-500/10">
                  <span className="text-emerald-400 mt-0.5">•</span>
                  <span>{driver}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Limiting Factors */}
          {primary.limiting_factors.length > 0 && (
            <div className="space-y-2 mt-4">
              <h4 className="text-xs uppercase tracking-wider text-amber-400/90 font-medium flex items-center gap-1.5">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                Potential Limiting Factors
              </h4>
              <ul className="space-y-1.5 text-xs text-slate-300">
                {primary.limiting_factors.map((lim, i) => (
                  <li key={i} className="flex items-start gap-2 bg-amber-500/5 p-2 rounded-lg border border-amber-500/10">
                    <span className="text-amber-400 mt-0.5">•</span>
                    <span>{lim}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </motion.div>

        {/* Right Column: AI Agronomist Synthesis & Alternatives */}
        <div className="col-span-12 lg:col-span-6 space-y-6">
          {/* AI Narrative Box */}
          <div className="rounded-2xl p-6 border border-white/10 bg-slate-900/70 backdrop-blur-md">
            <div className="flex items-center gap-2 mb-3">
              <Sparkles className="w-4 h-4 text-indigo-400" />
              <h4 className="text-sm font-semibold text-white font-[var(--font-sora)]">
                Agronomic Assessment Summary
              </h4>
              <span className="ml-auto text-[10px] text-dim px-2 py-0.5 rounded bg-white/5 uppercase">
                {data.explanation.insight_mode || "Validated Model Output"}
              </span>
            </div>
            <p className="text-xs leading-relaxed text-slate-300">
              {data.explanation.summary}
            </p>
          </div>

          {/* Alternative Ranked Crops */}
          <div className="rounded-2xl p-6 border border-white/10 bg-slate-900/70 backdrop-blur-md">
            <h4 className="text-sm font-semibold text-white font-[var(--font-sora)] mb-3">
              Suitable Alternatives (Top 2–5)
            </h4>
            <div className="space-y-2.5">
              {alternatives.map((alt, idx) => (
                <div
                  key={alt.crop}
                  className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/5 hover:border-white/10 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-xs font-mono text-dim w-4">{idx + 2}.</span>
                    <span className="text-xl">{CROP_EMOJIS[alt.crop] || "🌱"}</span>
                    <div>
                      <h5 className="text-xs font-semibold text-white">{alt.crop}</h5>
                      <p className="text-[11px] text-dim">
                        Confidence: {alt.confidence_percent}%
                      </p>
                      {alt.expected_yield_range && (
                        <p className="text-[10px] font-mono text-emerald-400/90 mt-0.5">
                          Yield: {alt.expected_yield_range}
                        </p>
                      )}
                    </div>
                  </div>
                  <div className="text-right">
                    <span className="text-sm font-bold text-emerald-400 font-mono">
                      {alt.suitability_percent}%
                    </span>
                    <p className="text-[10px] text-dim uppercase">Suitability</p>
                    {alt.risk && (
                      <span className={`inline-block mt-0.5 text-[9px] px-1.5 py-0.5 rounded font-medium border ${
                        alt.risk === "Low"
                          ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                          : alt.risk === "Moderate"
                          ? "bg-amber-500/10 text-amber-400 border-amber-500/20"
                          : "bg-rose-500/10 text-rose-400 border-rose-500/20"
                      }`}>
                        {alt.risk} Risk
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* V3 Counterfactual Analysis Panel (Collapsible) */}
      {data.counterfactuals && data.counterfactuals.length > 0 && (
        <div className="rounded-2xl border border-white/10 bg-slate-900/70 backdrop-blur-md overflow-hidden">
          <button
            onClick={() => setShowCounterfactuals(!showCounterfactuals)}
            className="w-full flex items-center justify-between p-5 text-left hover:bg-white/5 transition-colors"
          >
            <div className="flex items-center gap-3">
              <span className="p-2 rounded-lg bg-sky-500/10 text-sky-400 border border-sky-500/20">
                <Sliders className="w-5 h-5" />
              </span>
              <div>
                <div className="flex items-center gap-2 flex-wrap">
                  <h4 className="text-sm md:text-base font-semibold text-white font-[var(--font-sora)]">
                    What Could Change This Recommendation? (Counterfactual Analysis)
                  </h4>
                  <span className="text-[10px] px-2 py-0.5 rounded bg-sky-500/10 text-sky-300 border border-sky-500/20 font-mono">
                    Model-Supported Perturbations
                  </span>
                </div>
                <p className="text-xs text-dim mt-0.5">
                  Simulated climate, precipitation, and soil sensitivity scenarios to identify recommendation tipping points.
                </p>
              </div>
            </div>
            <div className="p-2 rounded-lg bg-white/5 text-dim">
              {showCounterfactuals ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </div>
          </button>

          <AnimatePresence>
            {showCounterfactuals && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="border-t border-white/10 p-5 space-y-3 bg-black/20"
              >
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {data.counterfactuals.map((cf, idx) => (
                    <div
                      key={idx}
                      className={`p-4 rounded-xl border transition-all ${
                        cf.changed_from_baseline
                          ? "bg-amber-950/20 border-amber-500/30"
                          : "bg-white/5 border-white/10"
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-semibold text-white">{cf.scenario}</span>
                        <span className={`text-[10px] px-2 py-0.5 rounded font-mono font-medium border ${
                          cf.changed_from_baseline
                            ? "bg-amber-500/20 text-amber-300 border-amber-500/30"
                            : "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
                        }`}>
                          {cf.changed_from_baseline ? "Tipping Point" : "Resilient"}
                        </span>
                      </div>
                      <p className="text-[11px] text-dim mb-3">{cf.description}</p>
                      <div className="flex items-center justify-between text-xs pt-2 border-t border-white/5">
                        <span className="text-dim">New #1 Candidate:</span>
                        <div className="flex items-center gap-1.5">
                          <span className="text-base">{CROP_EMOJIS[cf.top_crop] || "🌱"}</span>
                          <span className={`font-semibold ${cf.changed_from_baseline ? "text-amber-400" : "text-white"}`}>
                            {cf.top_crop}
                          </span>
                          <span className="text-dim text-[10px] font-mono">({cf.confidence_percent}%)</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* V3 Uncertainty & Multi-Objective Risk Diagnostics (Collapsible) */}
      {data.uncertainty && (
        <div className="rounded-2xl border border-white/10 bg-slate-900/70 backdrop-blur-md overflow-hidden">
          <button
            onClick={() => setShowDiagnostics(!showDiagnostics)}
            className="w-full flex items-center justify-between p-5 text-left hover:bg-white/5 transition-colors"
          >
            <div className="flex items-center gap-3">
              <span className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <Activity className="w-5 h-5" />
              </span>
              <div>
                <div className="flex items-center gap-2 flex-wrap">
                  <h4 className="text-sm md:text-base font-semibold text-white font-[var(--font-sora)]">
                    Decision Margin & Multi-Objective Risk
                  </h4>
                  <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/20 font-mono">
                    V3 Decision Intelligence
                  </span>
                </div>
                <p className="text-xs text-dim mt-0.5">
                  Decision margin vs runner-up candidate, Shannon entropy, and environmental risk decomposition.
                </p>
              </div>
            </div>
            <div className="p-2 rounded-lg bg-white/5 text-dim">
              {showDiagnostics ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </div>
          </button>

          <AnimatePresence>
            {showDiagnostics && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="border-t border-white/10 p-5 space-y-4 bg-black/20"
              >
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Decision Margin */}
                  <div className="p-4 rounded-xl bg-white/5 border border-white/10">
                    <span className="text-xs text-dim block">Decision Margin (vs #2)</span>
                    <span className="text-2xl font-bold font-mono text-emerald-400">
                      +{data.uncertainty.decision_margin_percent}%
                    </span>
                    <p className="text-[11px] text-dim mt-1">
                      Separation margin between top candidate and runner-up.
                    </p>
                  </div>

                  {/* Ranking Stability */}
                  <div className="p-4 rounded-xl bg-white/5 border border-white/10">
                    <span className="text-xs text-dim block">Ranking Stability</span>
                    <span className="text-lg font-semibold text-white">
                      {data.uncertainty.ranking_stability}
                    </span>
                    <p className="text-[11px] text-dim mt-1">
                      {data.uncertainty.interpretation || data.uncertainty.advisory}
                    </p>
                  </div>

                  {/* Multi-Factor Risk */}
                  <div className="p-4 rounded-xl bg-white/5 border border-white/10">
                    <span className="text-xs text-dim block">Multi-Objective Risk Matrix</span>
                    <div className="mt-2 space-y-1 text-xs">
                      <div className="flex justify-between">
                        <span className="text-dim">Weather Risk:</span>
                        <span className="font-semibold text-white">{data.risk?.weather_risk || "Low"}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-dim">Terrain Risk:</span>
                        <span className="font-semibold text-white">{data.risk?.terrain_risk || "Low"}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-dim">Erosion Risk:</span>
                        <span className="font-semibold text-white">{data.risk?.erosion_risk || "Low"}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* Environmental & Soil Diagnostics Grid */}
      <div className="grid grid-cols-12 gap-6">
        {/* Soil Profile */}
        <div className="col-span-12 md:col-span-4 rounded-xl p-5 border border-white/10 bg-slate-900/60">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-amber-400" />
              <h4 className="text-xs uppercase tracking-wider text-white font-semibold">Soil Profile</h4>
            </div>
            <span className="text-[10px] text-amber-300 px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/20">
              {data.soil.soil_texture_class}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="p-2.5 rounded-lg bg-black/20 border border-white/5">
              <span className="text-dim text-[11px] block">pH Value</span>
              <span className="text-base font-bold text-white">{data.soil.soil_ph.toFixed(1)}</span>
            </div>
            <div className="p-2.5 rounded-lg bg-black/20 border border-white/5">
              <span className="text-dim text-[11px] block">Organic Carbon</span>
              <span className="text-base font-bold text-white">{data.soil.organic_carbon.toFixed(2)}%</span>
            </div>
            <div className="p-2.5 rounded-lg bg-black/20 border border-white/5">
              <span className="text-dim text-[11px] block">Nitrogen (N)</span>
              <span className="text-base font-bold text-white">{data.soil.nitrogen.toFixed(0)} kg/ha</span>
            </div>
            <div className="p-2.5 rounded-lg bg-black/20 border border-white/5">
              <span className="text-dim text-[11px] block">Potassium (K)</span>
              <span className="text-base font-bold text-white">{data.soil.potassium.toFixed(0)} kg/ha</span>
            </div>
          </div>
          <div className="mt-3 flex items-center justify-between text-[11px] text-dim">
            <span>Soil Source: {data.soil.soil_data_source}</span>
            <span className="text-emerald-400">{data.soil.soil_data_quality}</span>
          </div>
        </div>

        {/* Meteorological Parameters */}
        <div className="col-span-12 md:col-span-4 rounded-xl p-5 border border-white/10 bg-slate-900/60">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <CloudRain className="w-4 h-4 text-sky-400" />
              <h4 className="text-xs uppercase tracking-wider text-white font-semibold">Weather & Moisture</h4>
            </div>
            <span className="text-[10px] text-sky-300 px-2 py-0.5 rounded bg-sky-500/10 border border-sky-500/20">
              Open-Meteo
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="p-2.5 rounded-lg bg-black/20 border border-white/5">
              <span className="text-dim text-[11px] block">Mean Temp</span>
              <span className="text-base font-bold text-white">{data.weather.temperature_mean.toFixed(1)}°C</span>
            </div>
            <div className="p-2.5 rounded-lg bg-black/20 border border-white/5">
              <span className="text-dim text-[11px] block">Soil Moisture</span>
              <span className="text-base font-bold text-white">{data.weather.soil_moisture.toFixed(3)} m³/m³</span>
            </div>
            <div className="p-2.5 rounded-lg bg-black/20 border border-white/5">
              <span className="text-dim text-[11px] block">7d Forecast Rain</span>
              <span className="text-base font-bold text-white">{data.weather.forecast_rainfall_7d.toFixed(1)} mm</span>
            </div>
            <div className="p-2.5 rounded-lg bg-black/20 border border-white/5">
              <span className="text-dim text-[11px] block">Rain Probability</span>
              <span className="text-base font-bold text-white">{data.weather.forecast_precip_probability.toFixed(0)}%</span>
            </div>
          </div>
          <div className="mt-3 flex items-center justify-between text-[11px] text-dim">
            <span>ET0: {data.weather.et0.toFixed(2)} mm/day</span>
            <span className="text-sky-400">{data.weather.weather_data_quality}</span>
          </div>
        </div>

        {/* Terrain & Erosion Integration */}
        <div className="col-span-12 md:col-span-4 rounded-xl p-5 border border-white/10 bg-slate-900/60">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <h4 className="text-xs uppercase tracking-wider text-white font-semibold">Terrain & Erosion</h4>
            </div>
            <span className={`text-[10px] px-2 py-0.5 rounded font-semibold border ${
              data.terrain.erosion_risk_label === "LOW"
                ? "bg-emerald-500/10 text-emerald-300 border-emerald-400/30"
                : data.terrain.erosion_risk_label === "MODERATE"
                ? "bg-amber-500/10 text-amber-300 border-amber-400/30"
                : "bg-rose-500/10 text-rose-300 border-rose-400/30"
            }`}>
              {data.terrain.erosion_risk_label} RISK
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="p-2.5 rounded-lg bg-black/20 border border-white/5">
              <span className="text-dim text-[11px] block">Elevation</span>
              <span className="text-base font-bold text-white">{data.terrain.elevation.toFixed(0)} m</span>
            </div>
            <div className="p-2.5 rounded-lg bg-black/20 border border-white/5">
              <span className="text-dim text-[11px] block">Terrain Slope</span>
              <span className="text-base font-bold text-white">{data.terrain.slope.toFixed(1)}°</span>
            </div>
          </div>

          <p className="mt-3 text-[11px] text-dim leading-relaxed p-2 rounded bg-black/20 border border-white/5">
            {data.erosion.advisory}
          </p>
        </div>
      </div>
    </div>
  );
}
