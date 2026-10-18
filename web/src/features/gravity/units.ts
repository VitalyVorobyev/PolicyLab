import type { Model } from "../../lib/data/types";

export const SECONDS_PER_DAY = 86400;

/** Conversions declared by the manifest; nothing here assumes a particular Earth–Moon distance. */
export interface Units {
  km(lengthNormalized: number): number;
  days(timeNormalized: number): number;
  kmPerSecond(velocityNormalized: number): number;
  timeUnitDays: number;
}

export function unitsFor(model: Model): Units {
  return {
    km: (v) => v * model.length_scale_km,
    days: (v) => (v * model.time_scale_seconds) / SECONDS_PER_DAY,
    kmPerSecond: (v) => v * model.velocity_scale_km_s,
    timeUnitDays: model.time_scale_seconds / SECONDS_PER_DAY,
  };
}
