import type { ReactNode } from "react";
import { fixed, humanize } from "../../lib/format";
import type { Manifest, Metric } from "../../lib/data/types";

/** A km/s difference in the most readable unit: m/s, or mm/s below 1 m/s. */
export function speedDifference(kmPerSecond: number): string {
  const metres = kmPerSecond * 1000;
  if (Math.abs(metres) >= 1) return `${fixed(metres, metres >= 10 ? 1 : 2)} m/s`;
  return `${fixed(metres * 1000, metres * 1000 >= 10 ? 1 : 2)} mm/s`;
}

/** The outcome named at the end of a delta-v margin definition ("outcome beyond the edge: <name>"), if any. */
export function beyondOutcome(metric: Metric): string | null {
  const match = /outcome beyond the edge: ([a-z_]+)/.exec(metric.definition);
  const name = match?.[1];
  if (name === undefined || name === "none") return null;
  return OUTCOME_WORDS[name] ?? humanize(name);
}

const OUTCOME_WORDS: Record<string, string> = {
  no_return_within_t_max: "no return within the time window",
  perilune_out_of_band: "perilune outside the band",
  moon_collision: "Moon collision",
  return_before_lunar_encounter: "return before the lunar encounter",
};

function sense(value: number, positive: string, negative: string): string {
  return value > 0 ? `+1, ${positive}` : value < 0 ? `−1, ${negative}` : "0, undefined";
}

interface Item {
  key: string;
  label: string;
  value: ReactNode;
  note?: ReactNode;
}

function Group({ title, caption, items }: { title: string; caption?: string; items: Item[] }) {
  if (items.length === 0) return null;
  return (
    <div className="min-w-0">
      <h4 className="font-sans text-sm font-medium text-graphite">{title}</h4>
      {caption !== undefined && <p className="mt-0.5 text-sm leading-snug text-quiet">{caption}</p>}
      <dl className="mt-3 space-y-2.5">
        {items.map((item) => (
          <div key={item.key} className="border-t border-rule pt-2">
            <dt className="text-sm text-graphite">{item.label}</dt>
            <dd className="numeric text-[1.15rem] leading-snug">{item.value}</dd>
            {item.note !== undefined && <dd className="text-sm leading-snug text-graphite">{item.note}</dd>}
          </div>
        ))}
      </dl>
    </div>
  );
}

/**
 * Key quantities of a free-return candidate, read from its manifest only. A metric the manifest
 * does not record is left out, never filled in.
 */
export function FreeReturnMetrics({ manifest }: { manifest: Manifest }) {
  const m = manifest.metrics;
  const item = (key: string, label: string, format: (metric: Metric) => ReactNode, note?: (metric: Metric) => ReactNode): Item[] => {
    const metric = m[key];
    return metric === undefined ? [] : [{ key, label, value: format(metric), note: note?.(metric) }];
  };
  const kmValue = (decimals: number) => (metric: Metric) => `${fixed(metric.value, decimals)} km`;
  const dayValue = (metric: Metric) => `day ${fixed(metric.value, 3)}`;
  const deg = (decimals: number) => (metric: Metric) => `${fixed(metric.value, decimals)}°`;

  const departure = [
    ...item("departure_delta_v_km_s", "TLI Δv", (v) => `${fixed(v.value, 5)} km/s`),
    ...item("departure_phase_deg", "Departure phase θ", deg(3), () => "From +x, the Earth–Moon line"),
    ...item("parking_altitude_km", "Parking orbit altitude", kmValue(0), () => "Circular, tangential impulse"),
  ];
  const flyby = [
    ...item("perilune_altitude_km", "Perilune altitude", kmValue(1)),
    ...item("lunar_encounter_time_days", "Perilune time", dayValue),
    ...item("lunar_passage_sense", "Lunar passage sense", (v) => sense(v.value, "counterclockwise", "clockwise"), () => "About the Moon, rotating frame"),
  ];
  const entry = [
    ...item("return_shell_crossing_time_days", "Entry-interface crossing", dayValue),
    ...item("return_shell_speed_km_s", "Entry speed", (v) => `${fixed(v.value, 4)} km/s`),
    ...item("return_flight_path_angle_deg", "Flight-path angle", deg(3), () => "Negative: descending"),
    ...item("entry_sense", "Entry sense", (v) => sense(v.value, "posigrade", "retrograde")),
    ...item("return_vacuum_perigee_altitude_km", "Vacuum perigee", kmValue(2), () => "Two-body conic, no atmosphere"),
  ];
  const margin = (key: string, label: string): Item[] =>
    item(key, label, (v) => speedDifference(v.value), (v) => {
      const beyond = beyondOutcome(v);
      return beyond === null ? undefined : `Beyond it: ${beyond}`;
    });
  const margins = [
    ...margin("delta_v_margin_below_km_s", "Δv decrease to infeasible"),
    ...margin("delta_v_margin_above_km_s", "Δv increase to infeasible"),
    ...item("perilune_margin_to_band_min_km", "Perilune above the band floor", kmValue(0)),
  ];
  // Comparisons with historical missions are shown only if the manifest records them.
  const plausibility = Object.entries(m)
    .filter(([key]) => key.includes("apollo"))
    .map(([key, metric]): Item => ({ key, label: humanize(key), value: `${fixed(metric.value, 3)} ${metric.unit}` }));

  return (
    <section aria-labelledby="key-metrics-title" className="mt-10 rounded-sm border border-rule bg-sheet px-5 py-6 sm:px-7">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <h3 id="key-metrics-title" className="text-2xl">
          Key quantities
        </h3>
        <p className="text-sm text-graphite">From the run manifest. Entry values are Earth-relative and inertial.</p>
      </div>
      <div className="mt-6 grid gap-x-8 gap-y-8 sm:grid-cols-2 xl:grid-cols-4">
        <Group title="Departure (TLI)" items={departure} />
        <Group title="Lunar flyby" items={flyby} />
        <Group title="Entry interface" items={entry} />
        <Group title="Margins" caption="Recorded with the run; not validation checks. Δv changes at fixed θ." items={margins} />
      </div>
      {plausibility.length > 0 && (
        <div className="mt-8">
          <Group title="Plausibility, not validation" items={plausibility} />
        </div>
      )}
    </section>
  );
}
