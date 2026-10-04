"use client";

import { motion } from "framer-motion";
import { useMemo } from "react";
import Tooltip from "@/components/ui/Tooltip";
import { easePremium, fadeSlideUp, staggerContainer, transitionBase } from "@/components/ui/motion";

type ShapItem = {
  feature: string;
  value: number;
};

type ShapPanelProps = {
  items: ShapItem[];
};

export default function ShapPanel({ items }: ShapPanelProps) {
  const safeItems = useMemo(
    () => items.filter((item) => item && typeof item.feature === "string" && Number.isFinite(item.value)),
    [items]
  );

  const normalized = useMemo(() => {
    if (!safeItems.length) return [];
    const sorted = [...safeItems].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 5);
    const maxAbs = Math.max(...sorted.map((item) => Math.abs(item.value)), 0.001);
    return sorted.map((item, index) => {
      const relative = Math.abs(item.value) / maxAbs;
      const width = Math.max(10, Math.round(relative * 100));
      const score = Math.max(0.1, Number((relative * 5).toFixed(1)));
      const level =
        relative >= 0.75 ? "Very Strong" :
        relative >= 0.5 ? "Strong" :
        relative >= 0.25 ? "Moderate" :
        "Mild";
      const direction = item.value < 0 ? "Lowers risk" : "Raises risk";
      return {
        ...item,
        rank: index + 1,
        direction,
        level,
        score,
        width,
      };
    });
  }, [safeItems]);

  return (
    <motion.section
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, amount: 0.25 }}
      variants={fadeSlideUp}
      transition={transitionBase}
      className="col-span-12 xl:col-span-7"
    >
      <h3 className="mb-4 font-[var(--font-sora)] text-lg text-white">Model Explanation</h3>
      <motion.article whileHover={{ scale: 1.02, y: -4 }} className="panel-soft rounded-3xl p-6">
        <motion.div
          initial="hidden"
          animate="visible"
          variants={staggerContainer}
          className="space-y-5"
        >
          {normalized.length === 0 ? (
            <p className="text-sm text-dim">Feature importance will appear after prediction.</p>
          ) : (
            normalized.map((factor) => (
              <motion.div key={factor.feature} variants={{ hidden: { opacity: 0, y: 10 }, visible: { opacity: 1, y: 0 } }}>
                <div className="mb-2 flex items-start justify-between gap-3 text-xs">
                  <Tooltip
                    content={
                      factor.value < 0
                        ? `${factor.feature} is reducing predicted erosion risk for this terrain.`
                        : `${factor.feature} is increasing predicted erosion risk for this terrain.`
                    }
                  >
                    <span className="text-dim">#{factor.rank} {factor.feature}</span>
                  </Tooltip>
                  <div className="text-right">
                    <p className={factor.value < 0 ? "font-semibold text-emerald-300" : "font-semibold text-rose-300"}>{factor.direction}</p>
                    <p className="text-[10px] text-dim">{factor.level} - Score {factor.score}/5</p>
                  </div>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-white/10">
                  <motion.div
                    className={`h-full ${factor.value < 0 ? "bg-emerald-400" : "bg-rose-400"}`}
                    initial={{ width: "0%" }}
                    animate={{ width: `${factor.width}%` }}
                    transition={{ duration: 0.9, ease: easePremium }}
                  />
                </div>
              </motion.div>
            ))
          )}
        </motion.div>

        <p className="mt-6 border-t border-white/10 pt-4 text-[11px] italic leading-relaxed text-dim">
          Ranked by SHAP magnitude with direction (raises or lowers risk) and an easy strength score.
        </p>
      </motion.article>
    </motion.section>
  );
}
