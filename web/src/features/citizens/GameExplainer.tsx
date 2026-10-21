import { Term } from "../../components/Tooltip";
import type { CitizensGame } from "../../lib/data/citizens";

/** "The game in 30 seconds": four steps, every constant read from the run. */
export function GameExplainer({ game }: { game: CitizensGame }) {
  const steps = [
    {
      title: `A shared pool of ${game.max_pool} coins`,
      body: (
        <>
          Four players share one pool for {game.rounds} rounds. It starts full and can never hold more than{" "}
          <Term tip={`The pool is capped at ${game.max_pool} coins; growth above the cap is lost.`}>{game.max_pool}</Term>.
        </>
      ),
    },
    {
      title: "The governor makes offers",
      body: <>Each round the governor offers each player some coins from the pool. How it splits them is the mechanism being designed.</>,
    },
    {
      title: "Players keep or return",
      body: (
        <>
          Each player keeps part of the offer and returns the rest. Returned coins grow{" "}
          <Term
            tip={`Next pool = current pool − all offers + ${game.growth_factor} × all returns, capped at ${game.max_pool}. A pool that falls below one coin cannot fund any offer: the game is effectively over.`}
          >
            ×{game.growth_factor}
          </Term>{" "}
          and refill the pool.
        </>
      ),
    },
    {
      title: "The score is what players keep",
      body: (
        <>
          A governor is judged by the{" "}
          <Term tip="Surplus: the coins kept, summed over the four players and all rounds of the game.">total surplus</Term>. Coins can only be
          kept while the pool lasts, so generous offers now compete with a pool that keeps paying later.
        </>
      ),
    },
  ];
  return (
    <ol className="grid gap-x-8 gap-y-6 sm:grid-cols-2 lg:grid-cols-4">
      {steps.map((s, i) => (
        <li key={s.title} className="border-t border-rule-strong pt-3">
          <p className="numeric font-serif text-sm text-quiet">{i + 1}</p>
          <p className="mt-1 font-medium">{s.title}</p>
          <p className="mt-1.5 text-[0.95rem] text-graphite">{s.body}</p>
        </li>
      ))}
    </ol>
  );
}
