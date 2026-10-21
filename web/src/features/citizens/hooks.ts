import { useEffect, useState, type RefObject } from "react";

export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** The element's content width in CSS pixels (`fallback` until measured, e.g. in tests). */
export function useWidth(ref: RefObject<HTMLElement | null>, fallback = 640): number {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (el === null || typeof ResizeObserver === "undefined") return;
    const update = (): void => {
      const w = el.getBoundingClientRect().width;
      setWidth((old) => (Math.abs(old - w) < 0.5 ? old : w));
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => {
      observer.disconnect();
    };
  }, [ref]);
  return width > 0 ? width : fallback;
}
