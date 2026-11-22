/** Leading-slash deployment base prefix with no trailing slash ("" when served at "/"). */
export function basePrefix(baseUrl: string = import.meta.env.BASE_URL): string {
  const trimmed = baseUrl.replace(/\/+$/, "");
  if (trimmed === "" || trimmed === "/") return "";
  return trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
}

/** Base URL under which all published research data is served for a given Vite base URL. */
export function researchBase(baseUrl: string = import.meta.env.BASE_URL): string {
  return `${basePrefix(baseUrl)}/research`;
}

/** Base URL under which all published research data is served. */
export const RESEARCH_BASE = researchBase();

const SAFE_SEGMENT = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

/**
 * A path from a catalog or manifest is accepted only if it is a plain relative path made of
 * simple segments. This rejects absolute paths, URLs with a scheme or host, backslashes,
 * query strings, percent-encoding and any "." or ".." segment.
 */
export function isSafeRelativePath(path: string): boolean {
  if (path.length === 0 || path.length > 256) return false;
  return path.split("/").every((segment) => SAFE_SEGMENT.test(segment) && segment !== "." && segment !== "..");
}

/** Join safe relative segments onto the research base. Callers validate segments first. */
export function researchUrl(...relative: string[]): string {
  return [RESEARCH_BASE, ...relative].join("/");
}
