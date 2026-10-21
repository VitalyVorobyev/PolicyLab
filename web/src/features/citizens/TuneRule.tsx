import { useRef, useState } from "react";
import { Term } from "../../components/Tooltip";
import type { KTuning } from "../../lib/data/citizens";
import { MODEL_DASH, niceRange, spreadLabels } from "./charts";
import { useWidth } from "./hooks";
import { citizenInfo, coins, kLabel, MODEL_KEYS, modelColor } from "./model";

const M = { left: 48, right: 64, top: 16, bottom: 36 };
const K_TICKS = [0.01, 0.1, 1, 10, 100];

function nearestIndex(lnK: number[], target: number): number {
  let best = 0;
  lnK.forEach((v, i) => {
    if (Math.abs(v - target) < Math.abs((lnK[best] ?? v) - target)) best = i;
  });
  return best;
}

/** "Tune the rule yourself": mean surplus against k for the four citizen models, with each model's best k*. */
export function TuneRule({ tuning }: { tuning: KTuning }) {
  const lnK = tuning.ln_k;
  const models = MODEL_KEYS.filter((m) => tuning.curves[m] !== undefined);
  const publishedIndex = nearestIndex(lnK, Math.log(tuning.k_published));
  const [index, setIndex] = useState(publishedIndex);
  const ref = useRef<HTMLDivElement>(null);
  const width = useWidth(ref);
  const height = width < 520 ? 250 : 300;
  const plotW = Math.max(10, width - M.left - M.right);
  const plotH = height - M.top - M.bottom;
  const all = models.flatMap((m) => tuning.curves[m] ?? []);
  const [lo, hi] = niceRange(Math.min(...all), Math.max(...all), 200);
  const x = (v: number): number => M.left + ((v - (lnK[0] ?? 0)) / ((lnK[lnK.length - 1] ?? 1) - (lnK[0] ?? 0))) * plotW;
  const y = (v: number): number => M.top + plotH - ((v - lo) / (hi - lo)) * plotH;
  const k = Math.exp(lnK[index] ?? 0);
  const yTicks: number[] = [];
  for (let v = lo; v <= hi; v += 200) yTicks.push(v);

  const labels = spreadLabels(
    models.map((m) => {
      const curve = tuning.curves[m] ?? [];
      return { key: m, color: modelColor(m), y: y(curve[curve.length - 1] ?? lo) };
    }),
    14,
    M.top,
    M.top + plotH,
  );

  return (
    <div className="grid grid-cols-1 gap-x-10 gap-y-6 lg:grid-cols-12">
      <div className="lg:col-span-8">
        <div ref={ref} className="w-full min-w-0 overflow-hidden">
          <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Mean surplus per game against the setting k of the simple rule, for the four citizen models" className="block">
            {yTicks.map((v) => (
              <g key={v}>
                <line x1={M.left} x2={M.left + plotW} y1={y(v)} y2={y(v)} stroke="var(--color-rule)" />
                <text x={M.left - 8} y={y(v) + 4} textAnchor="end" fontSize="11" fill="var(--color-graphite)" className="numeric">
                  {v.toLocaleString("en-US")}
                </text>
              </g>
            ))}
            {K_TICKS.map((t) => (
              <text key={t} x={x(Math.log(t))} y={M.top + plotH + 18} textAnchor="middle" fontSize="11" fill="var(--color-graphite)" className="numeric">
                {t}
              </text>
            ))}
            <text x={M.left + plotW} y={height - 2} textAnchor="end" fontSize="11" fill="var(--color-graphite)">
              k (log scale)
            </text>
            <line x1={x(Math.log(tuning.k_published))} x2={x(Math.log(tuning.k_published))} y1={M.top} y2={M.top + plotH} stroke="var(--color-ink)" strokeDasharray="3 3" />
            <text x={x(Math.log(tuning.k_published)) - 5} y={M.top + plotH - 6} textAnchor="end" fontSize="11" fill="var(--color-ink)">
              published k = {tuning.k_published}
            </text>
            {models.map((m) => {
              const curve = tuning.curves[m] ?? [];
              const d = curve.map((v, i) => `${i === 0 ? "M" : "L"}${x(lnK[i] ?? 0).toFixed(1)},${y(v).toFixed(1)}`).join("");
              return <path key={m} d={d} fill="none" stroke={modelColor(m)} strokeWidth="2" strokeDasharray={MODEL_DASH[m]} strokeLinejoin="round" />;
            })}
            {models.map((m) => {
              const ks = tuning.k_star[m];
              const curve = tuning.curves[m] ?? [];
              if (ks === undefined) return null;
              const i = nearestIndex(lnK, ks.ln_k);
              return <circle key={m} cx={x(lnK[i] ?? 0)} cy={y(curve[i] ?? lo)} r="4.5" fill="var(--color-sheet)" stroke={modelColor(m)} strokeWidth="2" />;
            })}
            <line x1={x(lnK[index] ?? 0)} x2={x(lnK[index] ?? 0)} y1={M.top} y2={M.top + plotH} stroke="var(--color-accent)" strokeWidth="2" />
            {models.map((m) => (
              <circle key={m} cx={x(lnK[index] ?? 0)} cy={y(tuning.curves[m]?.[index] ?? lo)} r="3.5" fill={modelColor(m)} stroke="var(--color-sheet)" strokeWidth="1.5" />
            ))}
            {labels.map((l) => (
              <text key={l.key} x={M.left + plotW + 6} y={l.ly + 4} fontSize="11.5" fill="var(--color-ink)">
                <tspan fill={l.color}>■</tspan> {l.key}
              </text>
            ))}
          </svg>
        </div>
        <label className="mt-2 flex items-center gap-4 pl-[48px] pr-[64px]">
          <span className="sr-only">Setting k of the simple rule</span>
          <input
            type="range"
            min={0}
            max={lnK.length - 1}
            step={1}
            value={index}
            onChange={(e) => { setIndex(Number(e.target.value)); }}
            className="k-slider w-full"
            aria-valuetext={`k = ${kLabel(k)}`}
          />
        </label>
      </div>
      <div className="lg:col-span-4">
        <p className="font-serif text-2xl">
          k = <span className="numeric">{kLabel(k)}</span>
        </p>
        <p className="mt-1 text-sm text-graphite">
          {k < 1
            ? "Small k: the pool is split equally almost whatever its level — generous."
            : k < 30
              ? "Middle k: equal split while the pool is full, shares follow contributions as it drains."
              : "Large k: shares follow contributions unless the pool is nearly full — strict."}
        </p>
        <table className="mt-4 w-full text-sm">
          <thead>
            <tr className="border-b border-rule-strong text-left text-xs text-graphite">
              <th scope="col" className="py-1.5 font-medium">
                Citizens
              </th>
              <th scope="col" className="py-1.5 text-right font-medium">
                Surplus at this k
              </th>
              <th scope="col" className="py-1.5 text-right font-medium">
                <Term tip="k*: the k with the highest mean surplus for this model (open circle on the chart).">Best k*</Term>
              </th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => {
              const ks = tuning.k_star[m];
              const curve = tuning.curves[m] ?? [];
              const best = ks === undefined ? undefined : curve[nearestIndex(lnK, ks.ln_k)];
              return (
                <tr key={m} className="border-b border-rule">
                  <th scope="row" className="py-1.5 text-left font-normal">
                    <span className="inline-flex items-center gap-2">
                      <span className="inline-block size-2.5 rounded-full" style={{ background: modelColor(m) }} aria-hidden="true" />
                      {m} <span className="hidden text-graphite sm:inline xl:hidden 2xl:inline">{citizenInfo(m)?.name}</span>
                    </span>
                  </th>
                  <td className="numeric py-1.5 text-right">{coins(curve[index] ?? 0)}</td>
                  <td className="numeric py-1.5 text-right text-graphite">
                    {best === undefined || ks === undefined ? "–" : `${coins(best)} at k = ${kLabel(ks.k)}`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="mt-3 text-xs text-graphite">
          <Term tip={`Mean total surplus over ${tuning.games.toLocaleString("en-US")} simulated tuning games per model and grid point (separate games from the cross-play matrix).`}>
            Coins per game
          </Term>
          , averaged over {tuning.games.toLocaleString("en-US")} games.
        </p>
      </div>
    </div>
  );
}
