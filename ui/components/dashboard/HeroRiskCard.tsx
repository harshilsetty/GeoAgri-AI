"use client";

import { AnimatePresence, motion } from "framer-motion";
import CountUp from "@/components/ui/CountUp";
import { easePremium, transitionBase } from "@/components/ui/motion";

type HeroRiskCardProps = {
  riskPercent: number;
  riskLabel: "LOW" | "MODERATE" | "HIGH";
  loading: boolean;
  hasResult: boolean;
};

function glowClasses(riskLabel: "LOW" | "MODERATE" | "HIGH") {
  if (riskLabel === "LOW") return "from-emerald-500/18 to-emerald-500/0";
  if (riskLabel === "MODERATE") return "from-amber-500/18 to-amber-500/0";
  return "from-rose-500/22 to-rose-500/0";
}

function badgeClasses(riskLabel: "LOW" | "MODERATE" | "HIGH") {
  if (riskLabel === "LOW") return "text-emerald-300 border-emerald-400/35 bg-emerald-500/10";
  if (riskLabel === "MODERATE") return "text-amber-300 border-amber-400/35 bg-amber-500/10";
  return "text-rose-300 border-rose-400/35 bg-rose-500/10";
}

export default function HeroRiskCard({ riskPercent, riskLabel, loading, hasResult }: HeroRiskCardProps) {
  const displayPercent = Math.max(0.1, riskPercent);
  const progressWidth = loading ? 32 : hasResult ? displayPercent : 0;

  return (
    <motion.article
      initial={{ opacity: 0, scale: 0.96, y: 16 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ duration: 0.55, ease: easePremium }}
      className={`panel relative col-span-12 overflow-hidden p-8 transition-all duration-500 md:p-10 xl:col-span-8 ${
        hasResult ? "scale-100 opacity-100" : "scale-[0.985] opacity-95"
      } ${hasResult && riskLabel === "HIGH" ? "risk-pulse-high" : ""}`}
    >
      <div className={`absolute inset-0 bg-gradient-to-br ${glowClasses(riskLabel)}`} />
      <div className="relative z-10 mx-auto flex min-h-[240px] max-w-xl flex-col items-center justify-center text-center md:min-h-[280px]">
        <p className="text-[11px] uppercase tracking-[0.22em] text-dim">Erosion Risk Index</p>

        <AnimatePresence mode="wait">
          {!hasResult && !loading ? (
            <motion.div key="empty" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}>
              <h2 className="mt-4 font-[var(--font-sora)] text-2xl text-white sm:text-3xl">No result yet</h2>
              <p className="mt-3 max-w-md text-sm text-dim">Upload an image and run prediction to generate terrain risk output.</p>
            </motion.div>
          ) : null}

          {loading ? (
            <motion.div key="loading" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}>
              <h2 className="mt-3 font-[var(--font-sora)] text-5xl text-white sm:text-6xl">Analyzing terrain...</h2>
              <div className="mt-4 h-4 w-36 rounded-full bg-white/10 skeleton-shimmer" />
            </motion.div>
          ) : null}

          {hasResult ? (
            <motion.div key="result" initial={{ opacity: 0, y: 8, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -8 }}>
              <h2 className="mt-3 font-[var(--font-sora)] text-7xl font-semibold leading-none text-white sm:text-8xl">
                <CountUp value={riskPercent} duration={1} decimals={1} />
                <span className="text-4xl">%</span>
              </h2>
              <div className={`mt-4 inline-flex items-center gap-2 rounded-full border px-4 py-1.5 ${badgeClasses(riskLabel)}`}>
                <span className="h-1.5 w-1.5 rounded-full bg-current" />
                <span className="text-xs font-semibold uppercase tracking-[0.12em]">{riskLabel} RISK</span>
              </div>
            </motion.div>
          ) : null}
        </AnimatePresence>

        <div className="mt-7 h-1.5 w-full max-w-sm overflow-hidden rounded-full bg-white/10">
          <motion.div
            className="h-full bg-gradient-to-r from-emerald-400 via-amber-400 to-rose-500 transition-all duration-1000"
            initial={{ width: "0%" }}
            animate={{ width: `${progressWidth}%` }}
            transition={{ duration: 1, ease: easePremium }}
          />
        </div>
      </div>
    </motion.article>
  );
}
