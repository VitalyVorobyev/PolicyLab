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
    study_id: "gravity-atlas",
    title: "Gravity Atlas",
    question:
      "At equal compute, does a diverse archive of Earth–Moon free-return trajectories hold more solutions that survive unseen injection errors and changed constraints than strong conventional optimizers do?",
    status: "active",
    progress:
      "Physics engine validated against the JPL catalog; free returns found, validated and mapped. Next: a frozen protocol, then the matched-budget comparison.",
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
  {
    study_id: "constitution-machine",
    title: "Constitution Machine",
    question:
      "Do adaptive rules for governing a shared resource keep working when the behaviour of the governed population changes, compared with simple fixed rules?",
    status: "conditional",
    progress: "Specification only. Will run only if it passes a feasibility review after the other studies.",
    specPath: "docs/studies/constitution-machine/README.md",
  },
];

export const STUDY_STATUS_LABEL: Record<StudyStatus, string> = {
  active: "Active",
  planned: "Planned",
  conditional: "Planned, conditional",
};
