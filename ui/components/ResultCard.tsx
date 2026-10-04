type ResultCardProps = {
  probability: number;
};

function getRisk(probability: number) {
  if (probability < 0.3) return { label: "LOW", color: "text-low" };
  if (probability < 0.7) return { label: "MODERATE", color: "text-moderate" };
  return { label: "HIGH", color: "text-high" };
}

export default function ResultCard({ probability }: ResultCardProps) {
  const risk = getRisk(probability);
  const progress = Math.max(0, Math.min(100, Math.round(probability * 100)));

  return (
    <section className="panel relative flex flex-col items-center justify-center px-6 py-10 text-center animate-fadeUp md:px-10 md:py-12">
      <p className="text-[11px] uppercase tracking-[0.28em] text-dim">Erosion Risk</p>
      <p className={`mt-3 font-[var(--font-sora)] text-[72px] leading-none md:text-[92px] ${risk.color}`}>
        {progress}%
      </p>
      <p className={`mt-3 rounded-full border px-4 py-1 text-xs font-semibold tracking-[0.18em] ${risk.color} ${
        risk.label === "LOW"
          ? "border-low/45 bg-low/10"
          : risk.label === "MODERATE"
          ? "border-moderate/45 bg-moderate/10"
          : "border-high/45 bg-high/10"
      }`}>
        {risk.label}
      </p>

      <div className="mt-8 h-2.5 w-full max-w-xl overflow-hidden rounded-full bg-white/10">
        <div
          className={`h-full rounded-full transition-all duration-700 ${
            risk.label === "LOW"
              ? "bg-low"
              : risk.label === "MODERATE"
              ? "bg-moderate"
              : "bg-high"
          }`}
          style={{ width: `${progress}%` }}
        />
      </div>
    </section>
  );
}
