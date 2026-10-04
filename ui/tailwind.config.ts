import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        surface: "#0f1218",
        panel: "#141923",
        panelSoft: "#1a2230",
        borderSoft: "rgba(255,255,255,0.08)",
        ink: "#edf1fa",
        dim: "#a3afc2",
        low: "#23c06b",
        moderate: "#f3ba2f",
        high: "#ef5050"
      },
      boxShadow: {
        soft: "0 20px 50px rgba(0, 0, 0, 0.28)",
        card: "0 8px 24px rgba(0, 0, 0, 0.24)"
      },
      borderRadius: {
        xl2: "1.2rem"
      },
      keyframes: {
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" }
        },
        pulseSoft: {
          "0%, 100%": { opacity: "0.4" },
          "50%": { opacity: "1" }
        }
      },
      animation: {
        fadeUp: "fadeUp 420ms ease-out",
        pulseSoft: "pulseSoft 1.4s ease-in-out infinite"
      }
    }
  },
  plugins: []
};

export default config;
