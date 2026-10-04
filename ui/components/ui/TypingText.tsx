"use client";

import { useEffect, useState } from "react";

type TypingTextProps = {
  text: string;
  speedMs?: number;
  delayMs?: number;
  className?: string;
};

export default function TypingText({ text, speedMs = 15, delayMs = 600, className = "" }: TypingTextProps) {
  const [typedText, setTypedText] = useState("");

  useEffect(() => {
    let index = 0;
    let delayTimer: ReturnType<typeof setTimeout> | null = null;
    let typeTimer: ReturnType<typeof setInterval> | null = null;

    setTypedText("");

    delayTimer = setTimeout(() => {
      typeTimer = setInterval(() => {
        index += 1;
        setTypedText(text.slice(0, index));
        if (index >= text.length && typeTimer) {
          clearInterval(typeTimer);
        }
      }, speedMs);
    }, delayMs);

    return () => {
      if (delayTimer) clearTimeout(delayTimer);
      if (typeTimer) clearInterval(typeTimer);
    };
  }, [text, speedMs, delayMs]);

  return <p className={`${className} typing-caret`}>{typedText}</p>;
}
