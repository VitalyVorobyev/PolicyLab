import { useLayoutEffect, useRef, useState, type RefObject } from "react";

/** Content-box width of an element, updated on resize. Falls back to `fallback` without ResizeObserver. */
export function useElementWidth<T extends HTMLElement>(fallback: number): [RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(fallback);
  useLayoutEffect(() => {
    const element = ref.current;
    if (element === null || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver((entries) => {
      const next = entries[0]?.contentRect.width;
      if (next !== undefined && next > 0) setWidth(Math.round(next));
    });
    observer.observe(element);
    return () => { observer.disconnect(); };
  }, []);
  return [ref, width];
}
