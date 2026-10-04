"use client";

import { motion } from "framer-motion";
import Tooltip from "@/components/ui/Tooltip";
import { fadeSlideUp, staggerContainer, transitionBase } from "@/components/ui/motion";

type MetricCard = {
  label: string;
  value: string;
  tooltip: string;
};

type MetricsPanelProps = {
  cards: MetricCard[];
};

export default function MetricsPanel({ cards }: MetricsPanelProps) {
  return (
    <motion.section
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, amount: 0.25 }}
      variants={fadeSlideUp}
      transition={transitionBase}
      className="col-span-12 xl:col-span-5"
    >
      <h3 className="mb-4 font-[var(--font-sora)] text-lg text-white">Terrain Metrics</h3>
      <motion.div
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, amount: 0.2 }}
        variants={staggerContainer}
        className="grid grid-cols-1 gap-3 sm:grid-cols-2"
      >
        {cards.map((card) => (
          <motion.article
            key={card.label}
            variants={{ hidden: { opacity: 0, y: 12 }, visible: { opacity: 1, y: 0 } }}
            whileHover={{ scale: 1.02, y: -4 }}
            className="panel-soft card-lift group relative rounded-3xl p-5"
          >
            <Tooltip content={card.tooltip}>
              <div>
                <p className="text-[10px] uppercase tracking-[0.14em] text-dim">{card.label}</p>
                <p className="mt-2 font-[var(--font-sora)] text-2xl text-white">{card.value}</p>
              </div>
            </Tooltip>
          </motion.article>
        ))}
      </motion.div>
    </motion.section>
  );
}
