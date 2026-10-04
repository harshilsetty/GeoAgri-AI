"use client";

import { animate } from "framer-motion";
import { useEffect, useState } from "react";

type CountUpProps = {
  value: number;
  duration?: number;
  decimals?: number;
};

export default function CountUp({ value, duration = 1, decimals = 1 }: CountUpProps) {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    const controls = animate(0, value, {
      duration,
      ease: [0.22, 1, 0.36, 1],
      onUpdate: (latest) => setDisplayValue(Number(latest.toFixed(decimals))),
    });

    return () => controls.stop();
  }, [value, duration, decimals]);

  return <>{displayValue}</>;
}
