import type { KeyboardEvent } from "react";
import { Term, Tooltip } from "../../components/Tooltip";
import type { Atlas, AtlasArm } from "../../lib/data/types";
import { fixed } from "../../lib/format";
import {
  altitudeLabel,
  axisKm,
  cellKey,
  eliteSummary,
  neighbor,
  sameCell,
  SURVIVAL_GRADIENT,
  survivalColor,
  survivalTextIsDark,
  type Cell,
  type Direction,
} from "./atlasModel";

const KEYS: Record<string, Direction> = { ArrowLeft: "left", ArrowRight: "right", ArrowUp: "up", ArrowDown: "down" };

interface Props {
  atlas: Atlas;
  arm: AtlasArm;
  /** The square whose trajectory is shown (hovered or pinned). */
  active: Cell | null;
  onHover: (cell: Cell | null) => void;
  onSelect: (cell: Cell) => void;
}

/**
 * The atlas: 10 x 10 kinds of free return (closest approach to the Moon across, trip time up).
 * Filled squares show the survival of the selected method's trajectory; empty reachable squares
 * are dashed; squares where no free return exists are left blank. Arrow keys move between filled
 * squares (one tab stop), Enter or Space selects, a mouse hover previews and a click keeps.
 */
export function AtlasGrid({ atlas, arm, active, onHover, onSelect }: Props) {
  const { d1, d2, reachable_cells } = atlas.grid;
  const reachable = new Set(reachable_cells.map(cellKey));
  const filled = new Map(arm.elites.map((e) => [cellKey(e.cell), e]));
  const filledCells = arm.elites.map((e) => e.cell);
  // Roving tab stop: the active square if filled, else the first filled one.
  const stop = active !== null && filled.has(cellKey(active)) ? active : (filledCells[0] ?? null);

  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>, from: Cell): void => {
    const direction = KEYS[e.key];
    if (direction === undefined) return;
    e.preventDefault();
    const next = neighbor(filledCells, from, direction);
    if (next === undefined) return;
    e.currentTarget.parentElement?.querySelector<HTMLElement>(`[data-cell="${cellKey(next)}"]`)?.focus();
    onSelect(next);
  };

  const rows = Array.from({ length: d2.bins }, (_, k) => d2.bins - 1 - k);
  const cols = Array.from({ length: d1.bins }, (_, i) => i);
  const xTicks = [0, 2, 4, 6, 8, 10];

  return (
    <div className="select-none">
      <div className="grid grid-cols-[1rem_1.25rem_minmax(0,1fr)] gap-x-1.5">
        <span
          className="self-center text-center text-xs text-canvas-text"
          style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
          aria-hidden="true"
        >
          Trip time (days)
        </span>
        <div className="relative" aria-hidden="true">
          {xTicks.map((k) => (
            <span key={k} className="numeric absolute right-0 -translate-y-1/2 text-[0.6875rem] text-canvas-text" style={{ top: `${100 - k * 10}%` }}>
              {fixed(d2.edges_days[k] ?? 0, 0)}
            </span>
          ))}
        </div>
        <div
          role="group"
          aria-label="Atlas of free returns: closest approach to the Moon across, trip time up. Arrow keys move between filled squares."
          className="grid aspect-square grid-cols-10 gap-[3px]"
        >
          {rows.flatMap((j) =>
            cols.map((i) => {
              const cell: Cell = [i, j];
              const key = cellKey(cell);
              const elite = filled.get(key);
              const where = `closest approach ${altitudeLabel(Math.sqrt((d1.edges_km[i] ?? 1) * (d1.edges_km[i + 1] ?? 1)))}, trip about ${fixed(((d2.edges_days[j] ?? 0) + (d2.edges_days[j + 1] ?? 0)) / 2, 1)} days`;
              if (elite !== undefined) {
                const isActive = sameCell(active, cell);
                const dark = survivalTextIsDark(elite.survival);
                return (
                  <button
                    key={key}
                    data-cell={key}
                    type="button"
                    tabIndex={sameCell(stop, cell) ? 0 : -1}
                    aria-label={eliteSummary(elite)}
                    aria-pressed={isActive}
                    onClick={() => { onSelect(cell); }}
                    onKeyDown={(e) => { onKeyDown(e, cell); }}
                    onPointerEnter={(e) => { if (e.pointerType === "mouse") onHover(cell); }}
                    onPointerLeave={(e) => { if (e.pointerType === "mouse") onHover(null); }}
                    onFocus={() => { onHover(null); }}
                    className={`numeric relative grid aspect-square place-items-center rounded-[3px] text-[0.6875rem] font-medium leading-none transition-transform ${
                      dark ? "text-ink" : "text-sheet"
                    } ${isActive ? "z-10 scale-110 outline-2 outline-offset-1 outline-accent-strong" : "hover:scale-105"}`}
                    style={{ backgroundColor: survivalColor(elite.survival) }}
                  >
                    {Math.round(elite.survival * 100)}
                  </button>
                );
              }
              const isReachable = reachable.has(key);
              return (
                <Tooltip
                  key={key}
                  content={
                    isReachable
                      ? `Free returns of this kind exist (${where}), but this method found none in this run.`
                      : `No free return of this kind exists (${where}).`
                  }
                >
                  {(props) => (
                    <div
                      {...props}
                      role="img"
                      aria-label={isReachable ? `Empty, but reachable: ${where}` : `Impossible: ${where}`}
                      className={`aspect-square rounded-[3px] ${
                        isReachable ? "border border-dashed border-canvas-text/70 bg-canvas-raised" : "bg-canvas-grid/35"
                      }`}
                    />
                  )}
                </Tooltip>
              );
            }),
          )}
        </div>
      </div>

      <div className="mt-1 grid grid-cols-[1rem_1.25rem_minmax(0,1fr)] gap-x-1.5" aria-hidden="true">
        <span />
        <span />
        <div className="relative h-8">
          {xTicks.map((k) => (
            <span key={k} className="numeric absolute -translate-x-1/2 text-[0.6875rem] text-canvas-text" style={{ left: `${k * 10}%` }}>
              {axisKm(d1.edges_km[k] ?? 0)}
            </span>
          ))}
          <span className="absolute inset-x-0 bottom-0 text-center text-xs text-canvas-text">Closest approach to the Moon (km, log scale)</span>
        </div>
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5 pl-[2.75rem] text-xs text-canvas-text">
        <span className="flex items-center gap-2">
          <span className="h-2.5 w-14 rounded-[2px]" style={{ background: SURVIVAL_GRADIENT }} aria-hidden="true" />
          <Term tip="The share of 50 launch errors the square's trajectory survives, from 0 to 100 percent. The number inside each square is the same value. Every trajectory faces the same 50 errors, and the same few errors knock out many trajectories alike, which is why some values repeat.">
            Survival 0–100 %
          </Term>
        </span>
        <span className="flex items-center gap-2">
          <span className="size-2.5 rounded-[2px] border border-dashed border-canvas-text/70" aria-hidden="true" />
          Empty
        </span>
        <span className="flex items-center gap-2">
          <span className="size-2.5 rounded-[2px] bg-canvas-grid/60" aria-hidden="true" />
          <Term tip="No free return of this kind turned up in any search, so these squares are treated as impossible and no method can fill them.">No free return</Term>
        </span>
      </div>
    </div>
  );
}
