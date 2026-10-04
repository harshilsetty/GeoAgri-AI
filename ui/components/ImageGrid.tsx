"use client";

import Image from "next/image";
import { useState } from "react";

type ImageItem = {
  label: string;
  src: string;
};

type ImageGridProps = {
  items: ImageItem[];
};

export default function ImageGrid({ items }: ImageGridProps) {
  const [active, setActive] = useState<ImageItem | null>(null);

  if (items.length === 0) {
    return <p className="text-sm text-dim">No visual outputs available yet.</p>;
  }

  const orderedLabels = ["Original", "Detection", "Segmentation", "Combined"];
  const primaryItems = orderedLabels
    .map((label) => items.find((item) => item.label.toLowerCase() === label.toLowerCase()))
    .filter((item): item is ImageItem => Boolean(item));
  const heatmapItem = items.find((item) => item.label.toLowerCase() === "heatmap");

  return (
    <>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {primaryItems.map((item) => (
          <figure key={item.label} className="overflow-hidden rounded-xl border border-borderSoft bg-black/30 transition hover:-translate-y-0.5 hover:border-white/20">
            <button type="button" className="relative block aspect-[16/10] w-full" onClick={() => setActive(item)}>
              <Image src={item.src} alt={item.label} fill unoptimized className="object-cover" />
              <span className="absolute right-2 top-2 rounded-md bg-black/55 px-2 py-1 text-[11px] text-white">Fullscreen</span>
            </button>
            <figcaption className="border-t border-borderSoft px-3 py-2 text-xs text-dim">{item.label}</figcaption>
          </figure>
        ))}
      </div>

      {heatmapItem ? (
        <figure className="mt-4 overflow-hidden rounded-xl border border-borderSoft bg-black/30 transition hover:border-white/20">
          <button type="button" className="relative block aspect-[21/8] w-full" onClick={() => setActive(heatmapItem)}>
            <Image src={heatmapItem.src} alt={heatmapItem.label} fill unoptimized className="object-cover" />
            <span className="absolute right-2 top-2 rounded-md bg-black/55 px-2 py-1 text-[11px] text-white">Fullscreen</span>
          </button>
          <figcaption className="border-t border-borderSoft px-3 py-2 text-xs text-dim">Heatmap</figcaption>
        </figure>
      ) : null}

      {active ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-4" onClick={() => setActive(null)}>
          <div className="relative h-[86vh] w-full max-w-6xl" onClick={(e) => e.stopPropagation()}>
            <button
              type="button"
              className="absolute right-3 top-3 z-10 rounded-lg bg-black/65 px-3 py-1 text-xs text-white"
              onClick={() => setActive(null)}
            >
              Close
            </button>
            <Image src={active.src} alt={active.label} fill unoptimized className="object-contain" />
          </div>
        </div>
      ) : null}
    </>
  );
}
