"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";

type TooltipProps = {
  content: string;
  children: React.ReactNode;
};

export default function Tooltip({ content, children }: TooltipProps) {
  const [open, setOpen] = useState(false);

  return (
    <div
      className="relative"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      {children}
      <AnimatePresence>
        {open ? (
          <motion.div
            initial={{ opacity: 0, y: -6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.98 }}
            transition={{ duration: 0.18, ease: "easeOut" }}
            className="pointer-events-none absolute left-4 top-2 z-20 w-56 rounded-lg border border-white/10 bg-black/80 px-3 py-2 text-[11px] leading-relaxed text-zinc-200"
          >
            {content}
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
