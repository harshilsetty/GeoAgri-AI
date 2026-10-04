type Metric = {
  label: string;
  value: string;
};

type MetricsGridProps = {
  metrics: Metric[];
};

const ICON_BY_METRIC: Record<string, string> = {
  Vegetation: "🌿",
  Slope: "⛰️",
  Rainfall: "🌧️",
  Elevation: "📍",
  Soil: "🪨",
  Boulders: "🧱",
  Ruins: "🏛️",
  Structures: "🏗️",
  Coordinates: "🧭",
};

export default function MetricsGrid({ metrics }: MetricsGridProps) {
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
      {metrics.map((metric) => (
        <div key={metric.label} className="rounded-xl border border-borderSoft bg-slate-950/55 p-4 transition hover:-translate-y-0.5 hover:border-white/20">
          <p className="flex items-center gap-2 text-xs uppercase tracking-wide text-dim">
            <span className="text-base leading-none">{ICON_BY_METRIC[metric.label] ?? "📊"}</span>
            {metric.label}
          </p>
          <p className="mt-2 text-base font-semibold text-ink">{metric.value}</p>
        </div>
      ))}
    </div>
  );
}
