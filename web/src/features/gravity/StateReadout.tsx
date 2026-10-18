import { fixed } from "../../lib/format";
import type { Trajectory } from "../../lib/data/types";
import { usePlaybackState, type PlaybackStore } from "./playback";
import type { Units } from "./units";

interface Row {
  symbol: string;
  name: string;
  normalized: string;
  normalizedUnit: string;
  physical: string;
  physicalUnit: string;
}

/** Exact values of the current recorded sample, in normalized units and converted with the manifest's scales. */
export function StateReadout({ store, trajectory, units }: { store: PlaybackStore; trajectory: Trajectory; units: Units }) {
  const { index, playing } = usePlaybackState(store);
  const t = trajectory.t[index] ?? 0;
  const x = trajectory.x[index] ?? 0;
  const y = trajectory.y[index] ?? 0;
  const vx = trajectory.vx[index] ?? 0;
  const vy = trajectory.vy[index] ?? 0;
  const rows: Row[] = [
    { symbol: "t", name: "time", normalized: fixed(t, 6), normalizedUnit: "TU", physical: fixed(units.days(t), 4), physicalUnit: "d" },
    { symbol: "x", name: "position x", normalized: fixed(x, 7), normalizedUnit: "LU", physical: fixed(units.km(x), 1), physicalUnit: "km" },
    { symbol: "y", name: "position y", normalized: fixed(y, 7), normalizedUnit: "LU", physical: fixed(units.km(y), 1), physicalUnit: "km" },
    { symbol: "vx", name: "velocity x", normalized: fixed(vx, 7), normalizedUnit: "VU", physical: fixed(units.kmPerSecond(vx), 5), physicalUnit: "km/s" },
    { symbol: "vy", name: "velocity y", normalized: fixed(vy, 7), normalizedUnit: "VU", physical: fixed(units.kmPerSecond(vy), 5), physicalUnit: "km/s" },
  ];
  return (
    <div aria-live={playing ? "off" : "polite"} aria-atomic="true">
      <p className="flex items-baseline justify-between text-sm text-graphite">
        <span>Current sample</span>
        <span className="numeric">
          {(index + 1).toLocaleString("en-US")} of {trajectory.length.toLocaleString("en-US")}
        </span>
      </p>
      <table className="numeric mt-2 w-full border-collapse text-[0.95rem]">
        <caption className="sr-only">
          State of the spacecraft at sample {index + 1}, in normalized units and converted to physical units
        </caption>
        <thead className="sr-only">
          <tr>
            <th scope="col">Quantity</th>
            <th scope="col">Normalized value</th>
            <th scope="col">Physical value</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.symbol} className="border-t border-rule">
              <th scope="row" className="w-8 py-1.5 pr-2 text-left font-serif text-base font-medium italic">
                <span aria-hidden="true">{row.symbol}</span>
                <span className="sr-only">{row.name}</span>
              </th>
              <td className="py-1.5 pr-3 text-right whitespace-nowrap">
                {row.normalized} <span className="text-xs text-quiet">{row.normalizedUnit}</span>
              </td>
              <td className="py-1.5 text-right whitespace-nowrap">
                {row.physical} <span className="text-xs text-quiet">{row.physicalUnit}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
