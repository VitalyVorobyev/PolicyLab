import type { StudyStatus } from "../../lib/data/types";

/** Editorial copy for the gallery. Statuses must agree with public/research/catalog.json (tested). */
export interface StudySummary {
  study_id: string;
  title: string;
  question: string;
  status: StudyStatus;
  /** Plain-language progress, never a claim of results that do not exist. */
  progress: string;
  specPath: string;
  route?: string;
}

export const STUDIES: readonly StudySummary[] = [
  {
    study_id: "model-citizens",
    title: "Model Citizens",
    question: "Do AI governors trained against different models of people discover the same rule for sharing a common pool?",
    status: "active",
    progress:
      "Running. Four models of people were built from a published experiment and an AI governor was trained against each; every governor was then played against every model. The pre-registered verdicts are being computed. Explore the games in the demonstration.",
    specPath: "docs/studies/model-citizens/README.md",
    route: "/citizens",
  },
  {
    study_id: "gravity-atlas",
    title: "Gravity Atlas",
    question: "Which search method finds the free returns to the Moon that survive a bad launch?",
    status: "complete",
    progress:
      "Finished. Seven search methods built catalogs of Earth–Moon free returns from the same budget; the MAP-Elites archive held the most trajectories that survive launch errors. Try it in the demonstration.",
    specPath: "docs/studies/gravity-atlas/README.md",
    route: "/gravity",
  },
  {
    study_id: "last-library",
    title: "The Last Library",
    question:
      "When does a strictly bounded inherited memory, at most 256 bytes passed between otherwise fresh agents, enable cumulative improvement on unseen but related tasks?",
    status: "planned",
    progress: "Specification only. Starts after Gravity Atlas publishes its first investigation.",
    specPath: "docs/studies/last-library/README.md",
  },
  {
    study_id: "algorithm-forge",
    title: "Algorithm Forge",
    question:
      "Under a fixed evaluation budget, when does evolutionary search over very short bit-vector programs beat simple structured search at finding correct, compact implementations?",
    status: "planned",
    progress: "Specification only. No experiments have been run.",
    specPath: "docs/studies/algorithm-forge/README.md",
  },
];

export const STUDY_STATUS_LABEL: Record<StudyStatus, string> = {
  active: "Active",
  planned: "Planned",
  complete: "Complete",
  conditional: "Planned, conditional",
};
