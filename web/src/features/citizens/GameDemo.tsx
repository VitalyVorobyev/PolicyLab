import { useCallback, useEffect, useRef, useState } from "react";
import { Term } from "../../components/Tooltip";
import type { CitizensDemo, GameRecord, HumanGame } from "../../lib/data/citizens";
import { fixed } from "../../lib/format";
import { prefersReducedMotion } from "./hooks";
import {
  CITIZENS,
  cellOf,
  citizenInfo,
  citizenOptions,
  coins,
  fallbackCitizens,
  fallbackGovernor,
  gamesFor,
  governorInfo,
  governorOptions,
  hasGames,
  PEOPLE,
  SIMPLE_RULE,
} from "./model";
import { PoolChart } from "./PoolChart";

/** Rounds per second during playback (a 40-round game takes 10 s). */
const SPEED = 4;

export interface Selection {
  governor: string;
  citizens: string;
  game: number;
}

function PlayIcon({ playing }: { playing: boolean }) {
  return (
    <svg viewBox="0 0 16 16" className="size-4" aria-hidden="true" fill="currentColor">
      {playing ? <path d="M3.5 2h3v12h-3zM9.5 2h3v12h-3z" /> : <path d="M4 2l10 6-10 6z" />}
    </svg>
  );
}

function ReplayIcon() {
  return (
    <svg viewBox="0 0 16 16" className="size-4" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.6">
      <path d="M3 8a5 5 0 1 0 1.6-3.7M3 2.5v2.8h2.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

/** A compact radio group of pills. */
function Pills({
  legend,
  name,
  options,
  value,
  onChange,
}: {
  legend: string;
  name: string;
  options: { key: string; label: string; title?: string }[];
  value: string;
  onChange: (key: string) => void;
}) {
  return (
    <fieldset className="m-0 min-w-0 border-0 p-0">
      <legend className="mb-1.5 text-xs font-medium tracking-wide text-canvas-text uppercase">{legend}</legend>
      <div className="flex flex-wrap gap-1.5">
        {options.map((o) => (
          <label key={o.key} className="group relative cursor-pointer" title={o.title}>
            <input type="radio" name={name} value={o.key} checked={value === o.key} onChange={() => { onChange(o.key); }} className="peer sr-only" />
            <span className="block rounded-sm border border-canvas-grid px-2.5 py-1.5 text-[0.8125rem] leading-tight text-canvas-text transition-colors group-hover:border-canvas-axis peer-checked:border-accent peer-checked:bg-canvas-raised peer-checked:text-canvas-strong peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent-strong">
              {o.label}
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

function PlayerRow({ index, offer, returned, kept, scale }: { index: number; offer: number; returned: number; kept: number; scale: number }) {
  const w = (v: number): string => `${Math.max(0, Math.min(100, (v / scale) * 100))}%`;
  return (
    <li className="grid grid-cols-[4.25rem_minmax(0,1fr)] items-center gap-x-3 gap-y-0.5">
      <span className="text-xs text-canvas-text">Player {index + 1}</span>
      <span className="flex h-2.5 overflow-hidden rounded-[2px] bg-canvas-grid/60" aria-hidden="true">
        <span className="block h-full bg-returned" style={{ width: w(returned) }} />
        <span className="block h-full border-l-2 border-canvas bg-kept" style={{ width: w(kept) }} />
      </span>
      <span className="col-start-2 numeric text-[0.75rem] text-canvas-text">
        offered <span className="text-canvas-strong">{fixed(offer, 1)}</span> · returned <span className="text-canvas-strong">{fixed(returned, 0)}</span> · kept{" "}
        <span className="text-canvas-strong">{fixed(kept, 1)}</span>
      </span>
    </li>
  );
}

function sourceNote(game: GameRecord, citizens: string): string {
  if (citizens === PEOPLE) {
    const h = game as HumanGame;
    return `Recorded game ${h.game_id} (${h.cohort})${h.split === "heldout" ? ", held-out data" : ""}`;
  }
  if (citizens.startsWith("B4")) return "Recorded LLM game";
  return "Simulated game";
}

/**
 * The hero: pick who governs and who the citizens are, then watch the 40 rounds. Everything shown
 * is a recorded or simulated game from the published run; only the playback is interpolated.
 */
export function GameDemo({ demo, selection, onSelect }: { demo: CitizensDemo; selection: Selection; onSelect: (s: Selection) => void }) {
  const { governor, citizens, game: gameIndex } = selection;
  const games = gamesFor(demo, governor, citizens);
  const game = games[Math.min(gameIndex, games.length - 1)];
  const rounds = game?.rounds ?? demo.game.rounds;

  const governors = governorOptions(demo).filter((g) => hasGames(demo, g.key, citizens));
  const citizensList = citizenOptions(demo);
  const [compareChoice, setCompareChoice] = useState<string | null>(SIMPLE_RULE);
  const compareOptions = governors.filter((g) => g.key !== governor);
  const compare = compareChoice !== null && compareOptions.some((g) => g.key === compareChoice) ? compareChoice : null;
  const ghostGames = compare === null ? [] : gamesFor(demo, compare, citizens);
  const ghost = ghostGames[Math.min(gameIndex, ghostGames.length - 1)] ?? null;

  const [reduced] = useState(prefersReducedMotion);
  const [t, setTState] = useState(reduced ? rounds : 0);
  const time = useRef(t);
  const setT = useCallback((v: number) => {
    time.current = v;
    setTState(v);
  }, []);
  const [playing, setPlaying] = useState(!reduced);

  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    let prev: number | null = null;
    const tick = (now: number): void => {
      const dt = prev === null ? 0 : (now - prev) / 1000;
      prev = now;
      const next = Math.min(rounds, time.current + dt * SPEED);
      setT(next);
      if (next >= rounds) {
        setPlaying(false);
        return;
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(frame);
    };
  }, [playing, rounds, setT]);

  const toggle = (): void => {
    if (playing) {
      setPlaying(false);
      return;
    }
    if (time.current >= rounds) setT(0);
    setPlaying(true);
  };

  const select = (next: Partial<Selection>): void => {
    onSelect({ ...selection, ...next });
  };

  if (game === undefined) return null;
  const round = Math.max(0, Math.min(rounds, Math.ceil(t - 1e-9)));
  const r = round - 1;
  const offers = r >= 0 ? (game.offers[r] ?? []) : [];
  const returns = r >= 0 ? (game.returns[r] ?? []) : [];
  const kept = r >= 0 ? (game.kept[r] ?? []) : [];
  const scale = Math.max(1, ...game.offers.flat());
  const surplus = r >= 0 ? (game.cumulative_surplus[r] ?? 0) : 0;
  const ghostSurplus = ghost === null || r < 0 ? 0 : (ghost.cumulative_surplus[Math.min(r, ghost.rounds - 1)] ?? 0);
  const gInfo = governorInfo(governor);
  const cInfo = citizenInfo(citizens);
  const compareInfo = compare === null ? undefined : governorInfo(compare);
  const cell = cellOf(demo, governor, citizens);
  const ghostCell = compare === null ? undefined : cellOf(demo, compare, citizens);
  const title = gInfo?.name ?? governor;

  return (
    <section aria-label="Interactive demonstration" className="on-canvas overflow-hidden rounded-sm bg-canvas text-canvas-text">
      <div className="grid gap-4 border-b border-canvas-grid p-3 sm:p-5 lg:grid-cols-2 lg:gap-8">
        <div>
          <Pills
            legend="Who governs"
            name="governor"
            value={governor}
            options={governors.map((g) => ({ key: g.key, label: g.short, title: g.name }))}
            onChange={(key) => { select({ governor: key, citizens: fallbackCitizens(demo, key, citizens) }); }}
          />
          <p className="mt-2 max-w-[60ch] text-[0.8125rem] leading-snug">
            <span className="font-medium text-canvas-strong">{gInfo?.name}.</span> {gInfo?.line}
          </p>
        </div>
        <div>
          <Pills
            legend="Who are the citizens"
            name="citizens"
            value={citizens}
            options={citizensList.map((c) => ({ key: c.key, label: c.key.startsWith("B") && !c.key.startsWith("B4") ? `${c.key} · ${c.name}` : c.name }))}
            onChange={(key) => { select({ citizens: key, governor: fallbackGovernor(demo, key, governor) }); }}
          />
          <p className="mt-2 max-w-[60ch] text-[0.8125rem] leading-snug">
            <span className="font-medium text-canvas-strong">{cInfo?.name}.</span> {cInfo?.line}
          </p>
        </div>
      </div>

      <div className="grid lg:grid-cols-[minmax(0,1fr)_21rem]">
        <div className="min-w-0 p-3 pb-2 sm:p-5 sm:pb-3">
          <PoolChart
            game={game}
            ghost={ghost}
            t={t}
            maxPool={demo.game.max_pool}
            title={title}
            ghostLabel={compareInfo === undefined ? null : `${compareInfo.name}, same ${citizens === PEOPLE ? "governor's game" : "citizens and game"}`}
          />
          <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-3">
            <button
              type="button"
              onClick={toggle}
              className="inline-flex items-center gap-2 rounded-sm border border-canvas-axis px-3 py-1.5 text-sm text-canvas-strong hover:border-accent"
            >
              {t >= rounds && !playing ? <ReplayIcon /> : <PlayIcon playing={playing} />}
              {playing ? "Pause" : t >= rounds ? "Replay" : "Play"}
            </button>
            <label className="flex min-w-[10rem] flex-1 items-center gap-3">
              <span className="sr-only">Round</span>
              <input
                type="range"
                min={0}
                max={rounds}
                step={1}
                value={round}
                onChange={(e) => {
                  setPlaying(false);
                  setT(Number(e.target.value));
                }}
                className="time-slider"
                aria-valuetext={round === 0 ? "Before round 1" : `Round ${round} of ${rounds}`}
              />
            </label>
            <div role="group" aria-label="Game" className="flex items-center gap-1">
              <span className="mr-1 text-xs text-canvas-text">Game</span>
              {games.map((_, i) => (
                <button
                  key={i}
                  type="button"
                  aria-pressed={i === gameIndex}
                  onClick={() => { select({ game: i }); }}
                  className={`numeric size-7 rounded-sm border text-[0.8125rem] ${i === gameIndex ? "border-accent bg-canvas-raised text-canvas-strong" : "border-canvas-grid hover:border-canvas-axis"}`}
                >
                  {i + 1}
                </button>
              ))}
            </div>
            <label className="flex items-center gap-2 text-xs">
              Compare with
              <select
                value={compare ?? ""}
                onChange={(e) => { setCompareChoice(e.target.value === "" ? null : e.target.value); }}
                className="rounded-sm border border-canvas-grid bg-canvas-raised px-1.5 py-1 text-[0.8125rem] text-canvas-strong"
              >
                <option value="">Nothing</option>
                {compareOptions.map((g) => (
                  <option key={g.key} value={g.key}>
                    {g.name}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </div>

        <aside aria-label="This round" className="border-t border-canvas-grid p-4 sm:p-5 lg:border-t-0 lg:border-l">
          <div className="flex items-baseline justify-between gap-3">
            <p className="font-serif text-xl text-canvas-strong" aria-live="off">
              {round === 0 ? "Before round 1" : `Round ${round}`}
              <span className="text-sm text-canvas-text"> of {rounds}</span>
            </p>
            <p className="numeric text-sm">
              <Term tip={`The shared pool before this round's offers and after the returned coins grew ×${demo.game.growth_factor} (it never exceeds ${demo.game.max_pool}).`}>Pool</Term>{" "}
              <span className="text-canvas-strong">{coins(game.pool[Math.max(0, r)] ?? 0)}</span>
              {r >= 0 && (
                <>
                  {" → "}
                  <span className="text-canvas-strong">{coins(game.pool[r + 1] ?? 0)}</span>
                </>
              )}
            </p>
          </div>
          <ul className="mt-4 space-y-3">
            {[0, 1, 2, 3].map((i) => (
              <PlayerRow key={i} index={i} offer={offers[i] ?? 0} returned={returns[i] ?? 0} kept={kept[i] ?? 0} scale={scale} />
            ))}
          </ul>
          <p className="mt-3 flex gap-4 text-xs">
            <span className="inline-flex items-center gap-1.5">
              <span className="inline-block size-2.5 rounded-[2px] bg-returned" aria-hidden="true" />
              <Term tip={`Returned coins go back to the pool and grow ×${demo.game.growth_factor}.`}>returned</Term>
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="inline-block size-2.5 rounded-[2px] bg-kept" aria-hidden="true" />
              <Term tip="Kept coins are the players' surplus: the score every governor tries to maximise, summed over players and rounds.">kept</Term>
            </span>
          </p>

          <dl className="mt-5 grid grid-cols-2 gap-x-4 gap-y-3 border-t border-canvas-grid pt-4">
            <div>
              <dt className="text-xs">Surplus so far</dt>
              <dd className="numeric mt-1 font-serif text-3xl leading-none text-canvas-strong">{coins(surplus)}</dd>
            </div>
            {ghost !== null && compareInfo !== undefined && (
              <div>
                <dt className="text-xs">{compareInfo.short}</dt>
                <dd className="numeric mt-1 font-serif text-3xl leading-none text-canvas-text">{coins(ghostSurplus)}</dd>
              </div>
            )}
          </dl>
          <p className="mt-4 text-xs leading-relaxed">
            {sourceNote(game, citizens)}: {coins(game.total_surplus)} coins in all.
            {cell !== undefined && (
              <>
                {" "}
                <Term tip={`Mean total surplus of this governor over all ${demo.crossplay.games.toLocaleString("en-US")} simulated games with these citizens${cell.seeds.length > 1 ? `, averaged over ${cell.seeds.length} training seeds` : ""}. Single games vary a lot.`}>
                  Average over {demo.crossplay.games.toLocaleString("en-US")} games
                </Term>
                : <span className="numeric text-canvas-strong">{coins(cell.total_surplus_mean)}</span>
                {ghostCell !== undefined && compareInfo !== undefined && (
                  <>
                    {" "}
                    ({compareInfo.short}: <span className="numeric">{coins(ghostCell.total_surplus_mean)}</span>)
                  </>
                )}
                .
              </>
            )}
          </p>
        </aside>
      </div>
    </section>
  );
}

export const DEFAULT_SELECTION: Selection = { governor: "Planner[B3]", citizens: "B1", game: 0 };

export function validSelection(demo: CitizensDemo, s: Selection): Selection {
  const citizens = CITIZENS.some((c) => c.key === s.citizens) ? s.citizens : "B1";
  const governor = fallbackGovernor(demo, citizens, s.governor);
  const n = gamesFor(demo, governor, citizens).length;
  return { governor, citizens, game: Math.max(0, Math.min(n - 1, s.game)) };
}
