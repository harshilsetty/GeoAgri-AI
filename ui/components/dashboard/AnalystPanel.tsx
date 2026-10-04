"use client";

import { motion } from "framer-motion";
import { Sparkles } from "lucide-react";
import { useMemo } from "react";
import TypingText from "@/components/ui/TypingText";
import { fadeSlideUp, transitionBase } from "@/components/ui/motion";

type AnalystPanelProps = {
  loading: boolean;
  text: string;
  hasResult: boolean;
};

const fallbackText = "I am ready to analyze terrain stability and summarize risks once prediction is complete.";

export default function AnalystPanel({ loading, text, hasResult }: AnalystPanelProps) {
  const targetText = useMemo(() => {
    if (!hasResult) return fallbackText;
    return text || fallbackText;
  }, [text, hasResult]);

  return (
    <motion.article
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, amount: 0.35 }}
      variants={fadeSlideUp}
      transition={transitionBase}
      whileHover={{ scale: 1.02 }}
      className="panel card-lift relative overflow-hidden border-l-2 border-l-rose-400 p-5"
    >
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_100%_0%,rgba(239,68,68,0.08),transparent_45%)]" />
      <div className="relative z-10">
        <div className="mb-3 inline-flex items-center gap-2">
          <span className="inline-flex h-7 w-7 items-center justify-center rounded-full border border-rose-300/30 bg-rose-500/12 text-rose-200">
            <Sparkles className="h-3.5 w-3.5" />
          </span>
          <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-zinc-300">GeoAI Analyst</p>
        </div>

        {loading ? (
          <div className="space-y-2">
            <p className="text-sm text-zinc-300 typing-caret">Analyzing terrain signals</p>
            <div className="h-3 w-full rounded bg-white/10 skeleton-shimmer" />
            <div className="h-3 w-5/6 rounded bg-white/10 skeleton-shimmer" />
            <div className="h-3 w-4/6 rounded bg-white/10 skeleton-shimmer" />
          </div>
        ) : (
          <TypingText text={targetText} delayMs={hasResult ? 700 : 120} speedMs={15} className="text-sm leading-relaxed text-zinc-300" />
        )}
      </div>
    </motion.article>
  );
}
