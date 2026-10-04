import type { Variants, Transition } from "framer-motion";

export const easePremium: [number, number, number, number] = [0.22, 1, 0.36, 1];

export const transitionFast: Transition = {
  duration: 0.22,
  ease: easePremium,
};

export const transitionBase: Transition = {
  duration: 0.45,
  ease: easePremium,
};

export const fadeSlideUp: Variants = {
  hidden: { opacity: 0, y: 14 },
  visible: { opacity: 1, y: 0 },
};

export const staggerContainer: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.08,
    },
  },
};

export const modalBackdrop: Variants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1 },
  exit: { opacity: 0 },
};

export const modalPanel: Variants = {
  hidden: { opacity: 0, y: 14, scale: 0.98 },
  visible: { opacity: 1, y: 0, scale: 1 },
  exit: { opacity: 0, y: 14, scale: 0.98 },
};
