import { Term } from "../../components/Tooltip";
import type { Atlas } from "../../lib/data/types";
import { fixed } from "../../lib/format";
import { armOf, mean, percent } from "./atlasModel";

/** The three findings. Every number is read from the atlas file's 16-run summaries and contrasts. */
export function Findings({ atlas }: { atlas: Atlas }) {
  const [D, MS, MBH, DE, CC, T] = (["D", "MS", "MBH", "DE", "CC", "T"] as const).map((id) => armOf(atlas, id));
  if (D === undefined || MS === undefined || MBH === undefined || DE === undefined || CC === undefined || T === undefined) return null;
  const squares = atlas.grid.reachable_cells.length;
  const worstP = Math.max(...atlas.contrasts.map((c) => c.p_holm));

  return (
    <ol className="grid gap-x-10 gap-y-8 lg:grid-cols-3">
      <li>
        <p className="font-serif text-xl leading-snug">MAP-Elites builds the best catalog.</p>
        <p className="mt-2 text-graphite">
          On average it holds {fixed(mean(T, "p1"), 1)} trajectories that survive a launch error, against {fixed(mean(DE, "p1"), 1)} for differential
          evolution, {fixed(mean(CC, "p1"), 1)} for continuation and {fixed(mean(D, "p1"), 1)} for the dense sweep. It also meets{" "}
          {percent(mean(T, "p2"))} of the unseen requirements (next best baseline: continuation, {percent(mean(CC, "p2"))}). The advantage over every
          baseline held up under the{" "}
          <Term tip={`Paired sign-flip tests on the 16 runs, corrected for ten comparisons (Holm); all p ≤ ${fixed(worstP, 3)}.`}>pre-registered statistical tests</Term>.
        </p>
      </li>
      <li>
        <p className="font-serif text-xl leading-snug">It wins by covering the map, not by finding tougher orbits.</p>
        <p className="mt-2 text-graphite">
          It fills {fixed(mean(T, "elites"), 1)} of the {squares} reachable squares, the dense sweep about {fixed(mean(D, "elites"), 0)}. But each
          dense-sweep trajectory survives {percent(mean(D, "mean_s"))} of launch errors, each MAP-Elites one {percent(mean(T, "mean_s"))}. The best
          trajectory in a square tends to sit on a{" "}
          <Term tip="Each square keeps the trajectory whose entry angle is closest to −6.5°, whatever its tolerance to launch errors.">knife edge</Term>.
        </p>
      </li>
      <li>
        <p className="font-serif text-xl leading-snug">Chasing a single best answer finds very few kinds.</p>
        <p className="mt-2 text-graphite">
          Basin hopping averages {fixed(mean(MBH, "elites"), 1) === "0.5" ? "half a trajectory" : `${fixed(mean(MBH, "elites"), 1)} trajectories`}, multi-start{" "}
          {fixed(mean(MS, "elites"), 1)}: optimizers that polish one answer at a time leave most of the atlas empty.
        </p>
      </li>
    </ol>
  );
}
