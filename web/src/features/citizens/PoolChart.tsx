import { useRef } from "react";
import type { GameRecord } from "../../lib/data/citizens";
import { useWidth } from "./hooks";

const M = { left: 38, right: 14, top: 26 };
const GAP = 34;

/** Pool level after `t` rounds (t may be fractional: linear playback interpolation between recorded rounds). */
function poolAt(game: GameRecord, t: number): number {
  const i = Math.max(0, Math.min(game.rounds, Math.floor(t)));
  const a = game.pool[i] ?? 0;
  const b = game.pool[Math.min(game.rounds, i + 1)] ?? a;
  return a + (b - a) * (t - i);
}

function linePath(game: GameRecord, t: number, x: (r: number) => number, y: (v: number) => number): string {
  const end = Math.max(0, Math.min(game.rounds, t));
  const points: string[] = [];
  for (let i = 0; i <= Math.floor(end); i++) points.push(`${x(i).toFixed(1)},${y(game.pool[i] ?? 0).toFixed(1)}`);
  if (end > Math.floor(end)) points.push(`${x(end).toFixed(1)},${y(poolAt(game, end)).toFixed(1)}`);
  return points.length > 0 ? `M${points.join("L")}` : "";
}

function keptPerRound(game: GameRecord): number[] {
  return game.kept.map((row) => row.reduce((s, v) => s + v, 0));
}

/** A step ceiling for the kept-coins axis. */
function niceCeil(v: number): number {
  for (const step of [10, 20, 25, 50, 100, 200]) if (v <= step) return step;
  return Math.ceil(v / 100) * 100;
}

/**
 * The pool over the game (top) and the coins kept in each round (bottom), drawn up to playback
 * time `t`. A comparison game, when given, is drawn as a dashed line on the same rounds.
 */
export function PoolChart({
  game,
  ghost,
  t,
  maxPool,
  title,
  ghostLabel,
}: {
  game: GameRecord;
  ghost: GameRecord | null;
  t: number;
  maxPool: number;
  title: string;
  ghostLabel: string | null;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const width = useWidth(ref);
  const narrow = width < 520;
  const poolH = narrow ? 160 : 180;
  const barH = narrow ? 52 : 56;
  const height = M.top + poolH + GAP + barH + 22;
  const plotW = Math.max(10, width - M.left - M.right);
  const rounds = game.rounds;
  const x = (r: number): number => M.left + (r / rounds) * plotW;
  const y = (v: number): number => M.top + poolH - (v / maxPool) * poolH;
  const barTop = M.top + poolH + GAP;
  const kept = keptPerRound(game);
  const keptMax = niceCeil(Math.max(1, ...kept, ...(ghost === null ? [] : keptPerRound(ghost))));
  const yb = (v: number): number => barTop + barH - (Math.max(0, v) / keptMax) * barH;
  const current = Math.max(0, Math.min(rounds, t));
  const shownRounds = Math.ceil(current - 1e-9);
  const barW = Math.max(1.5, plotW / rounds - (narrow ? 2 : 3));
  const depleted = game.pool.findIndex((p) => p < 1);
  const path = linePath(game, current, x, y);
  const area = path === "" ? "" : `${path}L${x(current).toFixed(1)},${y(0).toFixed(1)}L${x(0).toFixed(1)},${y(0).toFixed(1)}Z`;
  const ticks = narrow ? [0, 10, 20, 30, 40] : [0, 5, 10, 15, 20, 25, 30, 35, 40];
  const summary = `${title}: the pool starts at ${Math.round(game.pool[0] ?? 0)} coins and ends at ${Math.round(game.pool[rounds] ?? 0)} after ${rounds} rounds${
    depleted >= 0 ? `; it runs dry after round ${depleted}` : ""
  }.`;

  return (
    <div ref={ref} className="w-full min-w-0 overflow-hidden">
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={summary} className="block select-none">
        {/* pool panel */}
        {[0, 50, 100, 150, 200].filter((v) => v <= maxPool).map((v) => (
          <g key={v}>
            <line x1={M.left} x2={M.left + plotW} y1={y(v)} y2={y(v)} stroke="var(--color-canvas-grid)" strokeDasharray={v === maxPool ? "3 3" : undefined} />
            <text x={M.left - 8} y={y(v) + 4} textAnchor="end" fontSize="11" fill="var(--color-canvas-text)" className="numeric">
              {v}
            </text>
          </g>
        ))}
        <text x={M.left} y={M.top - 12} fontSize="11" fill="var(--color-canvas-text)">
          Pool (coins), capped at {maxPool}
        </text>
        {area !== "" && <path d={area} fill="var(--color-canvas-strong)" opacity="0.06" />}
        {ghost !== null && (
          <path d={linePath(ghost, current, x, y)} fill="none" stroke="var(--color-canvas-text)" strokeWidth="1.5" strokeDasharray="5 4" strokeLinejoin="round" />
        )}
        <path d={path} fill="none" stroke="var(--color-canvas-strong)" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
        {depleted >= 0 && depleted <= current && (
          <g>
            <circle cx={x(depleted)} cy={y(0)} r="4" fill="var(--color-fan-fail)" />
            <text x={x(depleted) + (x(depleted) > M.left + plotW * 0.7 ? -8 : 8)} y={y(0) - 8} fontSize="11" fill="var(--color-canvas-strong)" textAnchor={x(depleted) > M.left + plotW * 0.7 ? "end" : "start"}>
              Pool exhausted
            </text>
          </g>
        )}
        <line x1={x(current)} x2={x(current)} y1={M.top} y2={barTop + barH} stroke="var(--color-accent)" strokeWidth="1" opacity="0.7" />
        <circle cx={x(current)} cy={y(poolAt(game, current))} r="4.5" fill="var(--color-accent)" stroke="var(--color-canvas)" strokeWidth="2" />

        {/* kept-coins panel */}
        <text x={M.left} y={barTop - 10} fontSize="11" fill="var(--color-canvas-text)">
          Coins kept by the players, per round
        </text>
        <line x1={M.left} x2={M.left + plotW} y1={yb(0)} y2={yb(0)} stroke="var(--color-canvas-axis)" />
        <line x1={M.left} x2={M.left + plotW} y1={yb(keptMax)} y2={yb(keptMax)} stroke="var(--color-canvas-grid)" />
        <text x={M.left - 8} y={yb(keptMax) + 4} textAnchor="end" fontSize="11" fill="var(--color-canvas-text)" className="numeric">
          {keptMax}
        </text>
        <text x={M.left - 8} y={yb(0) + 4} textAnchor="end" fontSize="11" fill="var(--color-canvas-text)" className="numeric">
          0
        </text>
        {kept.slice(0, shownRounds).map((v, r) => (
          <rect key={r} x={x(r + 0.5) - barW / 2} y={yb(v)} width={barW} height={Math.max(0, yb(0) - yb(v))} rx="1" fill="var(--color-kept)" opacity={r + 1 === shownRounds ? 1 : 0.75} />
        ))}
        {ticks.map((r) => (
          <text key={r} x={x(r)} y={barTop + barH + 16} textAnchor="middle" fontSize="11" fill="var(--color-canvas-text)" className="numeric">
            {r === 0 ? "Round 0" : r}
          </text>
        ))}
      </svg>
      {ghost !== null && ghostLabel !== null && (
        <p className="mt-1 flex flex-wrap items-center gap-x-5 gap-y-1 pl-[38px] text-xs text-canvas-text">
          <span className="inline-flex items-center gap-2">
            <svg width="22" height="6" aria-hidden="true">
              <line x1="0" x2="22" y1="3" y2="3" stroke="var(--color-canvas-strong)" strokeWidth="2" />
            </svg>
            {title}
          </span>
          <span className="inline-flex items-center gap-2">
            <svg width="22" height="6" aria-hidden="true">
              <line x1="0" x2="22" y1="3" y2="3" stroke="var(--color-canvas-text)" strokeWidth="1.5" strokeDasharray="5 4" />
            </svg>
            {ghostLabel}
          </span>
        </p>
      )}
    </div>
  );
}
