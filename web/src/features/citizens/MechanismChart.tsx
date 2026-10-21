import { useRef } from "react";
import { Term } from "../../components/Tooltip";
import type { Distilled } from "../../lib/data/citizens";
import { MODEL_DASH, spreadLabels } from "./charts";
import { useWidth } from "./hooks";
import { modelColor } from "./model";

const M = { left: 40, right: 86, top: 14, bottom: 34 };

/**
 * What each AI governor learned, distilled into the simple rule's own terms: the weight on the equal
 * split at each pool level (seed means, binned), next to the simple rule (k = 22) itself.
 */
export function MechanismChart({ distilled, maxPool }: { distilled: Distilled; maxPool: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const width = useWidth(ref);
  const height = width < 520 ? 230 : 260;
  const plotW = Math.max(10, width - M.left - M.right);
  const plotH = height - M.top - M.bottom;
  const x = (fraction: number): number => M.left + fraction * plotW;
  const y = (w: number): number => M.top + (1 - w) * plotH;
  const edges = distilled.bin_edges;

  const stepPath = (w: (number | null)[]): string => {
    let d = "";
    let open = false;
    w.forEach((v, j) => {
      const a = edges[j] ?? 0;
      const b = edges[j + 1] ?? 1;
      if (v === null) {
        open = false;
        return;
      }
      d += `${open ? "L" : "M"}${x(a).toFixed(1)},${y(v).toFixed(1)}L${x(b).toFixed(1)},${y(v).toFixed(1)}`;
      open = true;
    });
    return d;
  };
  const ref22 = distilled.reference;
  const refPath = ref22.pool_fraction.map((f, i) => `${i === 0 ? "M" : "L"}${x(f).toFixed(1)},${y(ref22.w[i] ?? 0).toFixed(1)}`).join("");

  const labels = spreadLabels(
    [
      ...distilled.planners.map((p) => {
        const model = p.policy.replace(/^Planner\[(.*)\]$/, "$1");
        const lastValue = [...p.w_mean].reverse().find((v) => v !== null) ?? 0;
        return { key: p.policy, text: `AI · ${model}`, color: modelColor(model), y: y(lastValue) };
      }),
      { key: "ref", text: "Simple rule", color: "var(--color-ink)", y: y(ref22.w[ref22.w.length - 1] ?? 1) },
    ],
    14,
    M.top + 4,
    M.top + plotH,
  );

  return (
    <figure>
      <div ref={ref} className="w-full min-w-0 overflow-hidden">
        <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Weight each AI governor puts on the equal split at each pool level, against the simple rule" className="block">
          {[0, 0.5, 1].map((v) => (
            <g key={v}>
              <line x1={M.left} x2={M.left + plotW} y1={y(v)} y2={y(v)} stroke="var(--color-rule)" />
              <text x={M.left - 8} y={y(v) + 4} textAnchor="end" fontSize="11" fill="var(--color-graphite)" className="numeric">
                {v}
              </text>
            </g>
          ))}
          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <text key={f} x={x(f)} y={M.top + plotH + 18} textAnchor="middle" fontSize="11" fill="var(--color-graphite)" className="numeric">
              {Math.round(f * maxPool)}
            </text>
          ))}
          <text x={M.left + plotW} y={height - 2} textAnchor="end" fontSize="11" fill="var(--color-graphite)">
            pool before the offers (coins)
          </text>
          <path d={refPath} fill="none" stroke="var(--color-ink)" strokeWidth="1.5" strokeDasharray="4 3" />
          {distilled.planners.map((p) => {
            const model = p.policy.replace(/^Planner\[(.*)\]$/, "$1");
            return <path key={p.policy} d={stepPath(p.w_mean)} fill="none" stroke={modelColor(model)} strokeWidth="2" strokeDasharray={MODEL_DASH[model]} strokeLinejoin="round" />;
          })}
          {labels.map((l) => (
            <text key={l.key} x={M.left + plotW + 6} y={l.ly + 4} fontSize="11.5" fill="var(--color-ink)">
              <tspan fill={l.color}>■</tspan> {l.text}
            </text>
          ))}
        </svg>
      </div>
      <figcaption className="mt-2 text-xs text-graphite">
        Vertical axis:{" "}
        <Term tip="Each governor's offers were fitted with the simple rule's own form: a weight w on the equal split and 1 − w on splitting by last round's returns, separately in ten pool-level bands (on its own training model; averaged over its training seeds). 1 = split equally, 0 = split by contribution.">
          weight on the equal split
        </Term>
        . 1 = everyone gets the same, 0 = shares follow what each player returned.
      </figcaption>
    </figure>
  );
}
