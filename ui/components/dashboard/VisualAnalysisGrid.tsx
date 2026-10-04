"use client";

import Image from "next/image";
import { AnimatePresence, motion } from "framer-motion";
import { Maximize2, Minus, Plus, RotateCcw, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { fadeSlideUp, modalBackdrop, modalPanel, staggerContainer, transitionBase, transitionFast } from "@/components/ui/motion";

export type VisualTile = {
  title: string;
  src: string;
  accent?: string;
};

type VisualAnalysisGridProps = {
  tiles: VisualTile[];
  showVisuals: boolean;
  onToggleVisuals: () => void;
};

export default function VisualAnalysisGrid({ tiles, showVisuals, onToggleVisuals }: VisualAnalysisGridProps) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [touchStartX, setTouchStartX] = useState<number | null>(null);

  const expandedTile = expandedIndex !== null ? tiles[expandedIndex] ?? null : null;

  const canPan = zoom > 1;

  const transformStyle = useMemo(
    () => ({ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})` }),
    [pan.x, pan.y, zoom]
  );

  function openTile(index: number) {
    setExpandedIndex(index);
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }

  const goNext = useCallback(() => {
    setExpandedIndex((prev) => {
      if (prev === null || tiles.length === 0) return prev;
      return (prev + 1) % tiles.length;
    });
  }, [tiles.length]);

  const goPrev = useCallback(() => {
    setExpandedIndex((prev) => {
      if (prev === null || tiles.length === 0) return prev;
      return (prev - 1 + tiles.length) % tiles.length;
    });
  }, [tiles.length]);

  useEffect(() => {
    if (expandedIndex === null) return;

    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setExpandedIndex(null);
        return;
      }
      if (event.key === "ArrowRight") {
        goNext();
        return;
      }
      if (event.key === "ArrowLeft") {
        goPrev();
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [expandedIndex, goNext, goPrev]);

  return (
    <motion.section
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, amount: 0.2 }}
      variants={fadeSlideUp}
      transition={transitionBase}
      className="space-y-4"
    >
      <div className="flex items-center justify-between">
        <h3 className="inline-flex items-center gap-2 font-[var(--font-sora)] text-lg text-white">Visual Analysis Layers</h3>
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          className="text-[11px] font-semibold uppercase tracking-[0.16em] text-dim transition hover:text-white"
          onClick={onToggleVisuals}
        >
          {showVisuals ? "Hide Visuals" : "Show Visuals"}
        </motion.button>
      </div>

      <div className={`overflow-hidden transition-all duration-500 ${showVisuals ? "max-h-[1400px] opacity-100" : "max-h-0 opacity-0"}`}>
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.2 }}
          variants={staggerContainer}
          className="grid grid-cols-1 gap-3 pt-1 md:grid-cols-2 xl:grid-cols-4"
        >
          {tiles.map((tile, index) => (
            <motion.article
              key={tile.title}
              variants={{ hidden: { opacity: 0, y: 10 }, visible: { opacity: 1, y: 0 } }}
              whileHover={{ scale: 1.02, y: -4 }}
              className="group relative aspect-video overflow-hidden rounded-3xl border border-white/10 bg-slate-900/50 card-lift"
            >
              {tile.src ? (
                <Image src={tile.src} alt={tile.title} fill unoptimized className="object-cover transition duration-700 group-hover:scale-110" />
              ) : (
                <div className={`h-full w-full bg-gradient-to-br ${tile.accent || "from-slate-600/40 to-slate-900"}`} />
              )}
              <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent" />
              {tile.src ? (
                <button
                  className="absolute right-2 top-2 rounded-lg border border-white/15 bg-black/45 p-1.5 text-zinc-200 backdrop-blur transition hover:bg-black/60 hover:text-white"
                  onClick={() => openTile(index)}
                  aria-label={`Open ${tile.title} in full screen`}
                >
                  <Maximize2 className="h-3.5 w-3.5" />
                </button>
              ) : null}
              <p className="absolute bottom-3 left-3 rounded-lg bg-black/55 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-zinc-200 backdrop-blur">
                {tile.title}
              </p>
            </motion.article>
          ))}
        </motion.div>
      </div>

      <AnimatePresence>
        {expandedTile?.src ? (
          <motion.div
            className="fixed inset-0 z-[60] flex items-center justify-center bg-black/85 p-4 backdrop-blur-sm"
            onClick={() => setExpandedIndex(null)}
            role="dialog"
            aria-modal="true"
            aria-label="Full screen visual"
            initial="hidden"
            animate="visible"
            exit="exit"
            variants={modalBackdrop}
            transition={transitionFast}
          >
            <motion.div
              className="relative h-[86vh] w-full max-w-6xl overflow-hidden rounded-2xl border border-white/10 bg-black"
              onClick={(e) => e.stopPropagation()}
              initial="hidden"
              animate="visible"
              exit="exit"
              variants={modalPanel}
              transition={transitionFast}
            >
            <div className="absolute inset-x-0 top-0 z-20 flex items-center justify-between bg-gradient-to-b from-black/75 to-transparent px-4 py-3">
              <p className="text-sm font-semibold text-white">{expandedTile.title}</p>
              <div className="flex items-center gap-2">
                <button
                  className="rounded-lg border border-white/20 bg-black/40 p-1.5 text-zinc-200 transition hover:bg-black/65 hover:text-white"
                  onClick={() => setZoom((prev) => Math.max(1, Number((prev - 0.2).toFixed(2))))}
                  aria-label="Zoom out"
                >
                  <Minus className="h-4 w-4" />
                </button>
                <button
                  className="rounded-lg border border-white/20 bg-black/40 p-1.5 text-zinc-200 transition hover:bg-black/65 hover:text-white"
                  onClick={() => setZoom((prev) => Math.min(4, Number((prev + 0.2).toFixed(2))))}
                  aria-label="Zoom in"
                >
                  <Plus className="h-4 w-4" />
                </button>
                <button
                  className="rounded-lg border border-white/20 bg-black/40 p-1.5 text-zinc-200 transition hover:bg-black/65 hover:text-white"
                  onClick={() => {
                    setZoom(1);
                    setPan({ x: 0, y: 0 });
                  }}
                  aria-label="Reset view"
                >
                  <RotateCcw className="h-4 w-4" />
                </button>
                <button
                  className="rounded-lg border border-white/20 bg-black/40 p-1.5 text-zinc-200 transition hover:bg-black/65 hover:text-white"
                  onClick={() => setExpandedIndex(null)}
                  aria-label="Close full screen"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>

            <div
              className={`relative h-full w-full ${canPan ? "cursor-grab active:cursor-grabbing" : "cursor-default"}`}
              onMouseDown={(e) => {
                if (!canPan) return;
                setDragging(true);
                setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
              }}
              onMouseMove={(e) => {
                if (!dragging || !canPan) return;
                setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
              }}
              onMouseUp={() => setDragging(false)}
              onMouseLeave={() => setDragging(false)}
              onTouchStart={(e) => setTouchStartX(e.touches[0].clientX)}
              onTouchEnd={(e) => {
                if (touchStartX === null) return;
                const delta = e.changedTouches[0].clientX - touchStartX;
                if (Math.abs(delta) > 45) {
                  if (delta < 0) goNext();
                  else goPrev();
                }
                setTouchStartX(null);
              }}
            >
              <div className="h-full w-full transition-transform duration-150" style={transformStyle}>
                <Image src={expandedTile.src} alt={expandedTile.title} fill unoptimized className="object-contain" />
              </div>
            </div>
            </motion.div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </motion.section>
  );
}
