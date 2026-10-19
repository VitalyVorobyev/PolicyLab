import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type FocusEvent,
  type HTMLAttributes,
  type PointerEvent,
  type ReactNode,
} from "react";

/** Props to spread on the element that owns the tooltip. */
export type TooltipTriggerProps = Pick<
  HTMLAttributes<HTMLElement>,
  "aria-describedby" | "onPointerEnter" | "onPointerLeave" | "onPointerDown" | "onClick" | "onFocus" | "onBlur"
>;

const GAP = 8;
const MARGIN = 8;

function focusVisible(el: Element): boolean {
  try {
    return el.matches(":focus-visible");
  } catch {
    return true;
  }
}

/**
 * One accessible tooltip for the whole site. It opens on mouse hover, on keyboard focus and on
 * a tap (touch and pen), and closes on Escape, leave, blur or a tap elsewhere. The trigger gets
 * `aria-describedby`, so the text is also its accessible description. The bubble is positioned
 * with `fixed` coordinates, clamped to the viewport, so clipping ancestors never cut it off.
 */
export function Tooltip({ content, children }: { content: ReactNode; children: (props: TooltipTriggerProps) => ReactNode }) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const [style, setStyle] = useState<CSSProperties>({});
  const [anchor, setAnchor] = useState<HTMLElement | null>(null);
  const [pointer, setPointer] = useState("mouse");
  const bubbleRef = useRef<HTMLSpanElement>(null);

  const props: TooltipTriggerProps = {
    "aria-describedby": id,
    onPointerEnter: (e: PointerEvent<HTMLElement>) => {
      setAnchor(e.currentTarget);
      if (e.pointerType === "mouse") setOpen(true);
    },
    onPointerLeave: (e: PointerEvent<HTMLElement>) => {
      if (e.pointerType === "mouse") setOpen(false);
    },
    onPointerDown: (e: PointerEvent<HTMLElement>) => {
      setAnchor(e.currentTarget);
      setPointer(e.pointerType);
    },
    onClick: (e) => {
      setAnchor(e.currentTarget);
      if (pointer !== "mouse") setOpen((o) => !o);
      setPointer("mouse");
    },
    onFocus: (e: FocusEvent<HTMLElement>) => {
      setAnchor(e.currentTarget);
      if (focusVisible(e.currentTarget)) setOpen(true);
    },
    onBlur: () => { setOpen(false); },
  };

  const place = useCallback(() => {
    const trigger = anchor;
    const bubble = bubbleRef.current;
    if (trigger === null || bubble === null) return;
    const t = trigger.getBoundingClientRect();
    const b = bubble.getBoundingClientRect();
    const vw = document.documentElement.clientWidth;
    const above = t.top - b.height - GAP >= MARGIN;
    const left = Math.min(Math.max(MARGIN, t.left + t.width / 2 - b.width / 2), Math.max(MARGIN, vw - b.width - MARGIN));
    setStyle({ left, top: above ? t.top - b.height - GAP : t.bottom + GAP });
  }, [anchor]);

  useLayoutEffect(() => {
    if (open) place();
  }, [open, place]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === "Escape") setOpen(false);
    };
    const onDown = (e: Event): void => {
      if (e.target instanceof Node && !anchor?.contains(e.target)) setOpen(false);
    };
    const close = (): void => { setOpen(false); };
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onDown);
    window.addEventListener("scroll", close, true);
    window.addEventListener("resize", close);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onDown);
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("resize", close);
    };
  }, [open, anchor]);

  return (
    <>
      {children(props)}
      <span
        ref={bubbleRef}
        id={id}
        role="tooltip"
        hidden={!open}
        style={style}
        className="pointer-events-none fixed z-50 w-max max-w-[min(20rem,calc(100vw-1rem))] rounded-sm border border-canvas-axis bg-canvas-raised px-3 py-2 text-left text-[0.8125rem] font-normal leading-snug text-canvas-strong shadow-lg normal-case tracking-normal"
      >
        {content}
      </span>
    </>
  );
}

/** An inline term with a dotted underline that explains itself in a tooltip. */
export function Term({ tip, children, className = "" }: { tip: ReactNode; children: ReactNode; className?: string }) {
  return (
    <Tooltip content={tip}>
      {(props) => (
        <span
          {...props}
          tabIndex={0}
          className={`cursor-help underline decoration-current/50 decoration-dotted underline-offset-[0.25em] hover:decoration-current ${className}`}
        >
          {children}
        </span>
      )}
    </Tooltip>
  );
}
