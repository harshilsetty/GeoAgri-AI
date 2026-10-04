type InsightBoxProps = {
  text: string;
  loading?: boolean;
  risk?: "LOW" | "MODERATE" | "HIGH";
  mode?: "groq" | "local" | "default";
  status?: string;
};

export default function InsightBox({ text, loading = false, risk = "MODERATE", mode = "default", status }: InsightBoxProps) {
  const riskBorder =
    risk === "LOW"
      ? "border-l-low"
      : risk === "MODERATE"
      ? "border-l-moderate"
      : "border-l-high";

  return (
    <section className={`panel-soft border-l-4 p-5 md:p-6 ${riskBorder} transition hover:shadow-soft`}>
      <div className="mb-3 flex items-center justify-between gap-3">
        <p className="text-sm font-semibold text-ink">AI Insight</p>
        {mode === "local" ? (
          <span className="rounded-full border border-amber-300/35 bg-amber-400/10 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.1em] text-amber-200">
            Local Insight Mode
          </span>
        ) : null}
      </div>
      {loading ? (
        <div className="space-y-2.5">
          <p className="text-sm text-dim">Analyzing terrain...</p>
          <div className="h-3.5 w-11/12 animate-pulseSoft rounded bg-white/15" />
          <div className="h-3.5 w-full animate-pulseSoft rounded bg-white/10" />
          <div className="h-3.5 w-9/12 animate-pulseSoft rounded bg-white/15" />
        </div>
      ) : (
        <div className="space-y-3">
          <p className="text-sm leading-relaxed text-dim md:text-[15px]">{text}</p>
          {mode === "local" && status ? (
            <p className="rounded-md border border-amber-300/25 bg-amber-300/10 px-3 py-2 text-xs text-amber-100">
              GROQ fallback reason: {status}
            </p>
          ) : null}
        </div>
      )}
    </section>
  );
}
