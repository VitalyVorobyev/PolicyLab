import type { Atlas, ArmId } from "../../lib/data/types";
import { fixed } from "../../lib/format";
import { ARM_INFO, armOf, mean } from "./atlasModel";

/**
 * The seven methods as one radio group. Each tab carries a thin bar for its 16-run average of
 * expected survivors, so the whole comparison can be read before anything is selected.
 */
export function MethodSwitch({ atlas, value, onChange }: { atlas: Atlas; value: ArmId; onChange: (arm: ArmId) => void }) {
  const p1 = ARM_INFO.map((info) => mean(armOf(atlas, info.id), "p1"));
  const max = Math.max(...p1, 1);
  return (
    <fieldset className="m-0 min-w-0 border-0 p-0">
      <legend className="sr-only">Search method</legend>
      <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-4 lg:grid-cols-7">
        {ARM_INFO.map((info, k) => (
          <label key={info.id} className="group relative block cursor-pointer">
            <input
              type="radio"
              name="method"
              value={info.id}
              checked={value === info.id}
              onChange={() => { onChange(info.id); }}
              className="peer sr-only"
            />
            <span className="block rounded-sm border border-canvas-grid px-2.5 pt-2 pb-1.5 text-canvas-text transition-colors group-hover:border-canvas-axis peer-checked:border-accent peer-checked:bg-canvas-raised peer-checked:text-canvas-strong peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent-strong">
              <span className="block truncate text-[0.8125rem] leading-tight font-medium">{info.short}</span>
              <span className="mt-1.5 flex items-center gap-2">
                <span className="h-1 flex-1 rounded-full bg-canvas-grid" aria-hidden="true">
                  <span className="block h-full rounded-full bg-current opacity-80" style={{ width: `${((p1[k] ?? 0) / max) * 100}%` }} />
                </span>
                <span className="numeric text-[0.6875rem] leading-none">
                  <span className="sr-only">expected survivors </span>
                  {fixed(p1[k] ?? 0, 1)}
                </span>
              </span>
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}
