export const REPOSITORY_URL = "https://github.com/VitalyVorobyev/PolicyLab";

export function commitUrl(sha: string): string {
  if (!/^[0-9a-f]{40}$/.test(sha)) throw new RangeError("Expected a full commit SHA.");
  return `${REPOSITORY_URL}/commit/${sha}`;
}

/** Link to a document on the main branch, e.g. "docs/studies/gravity-atlas/README.md". */
export function docUrl(path: string): string {
  return `${REPOSITORY_URL}/blob/main/${path}`;
}
