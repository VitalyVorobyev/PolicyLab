import { useSyncExternalStore, type AnchorHTMLAttributes, type MouseEvent } from "react";
import { basePrefix } from "./data/paths";

/** A minimal history-API router: the site has three routes and needs nothing more. */

const NAVIGATE_EVENT = "phasespace:navigate";

function subscribe(onChange: () => void): () => void {
  window.addEventListener("popstate", onChange);
  window.addEventListener(NAVIGATE_EVENT, onChange);
  return () => {
    window.removeEventListener("popstate", onChange);
    window.removeEventListener(NAVIGATE_EVENT, onChange);
  };
}

/** Pathname without a trailing slash ("/" stays "/"). */
export function normalizePath(pathname: string): string {
  return pathname.length > 1 ? pathname.replace(/\/+$/, "") || "/" : pathname;
}

/** Strip a deployment base prefix (e.g. "/PolicyLab/") so routes match "/", "/gravity", "/citizens". */
export function stripBase(pathname: string, baseUrl: string = import.meta.env.BASE_URL): string {
  const normalized = normalizePath(pathname);
  const prefix = basePrefix(baseUrl);
  if (prefix === "") return normalized;
  if (normalized === prefix) return "/";
  if (normalized.startsWith(`${prefix}/`)) return normalizePath(normalized.slice(prefix.length));
  return normalized;
}

/** Prefix an app-relative route ("/", "/citizens") with the deployment base path. */
export function withBase(to: string, baseUrl: string = import.meta.env.BASE_URL): string {
  if (!to.startsWith("/")) return to;
  const prefix = basePrefix(baseUrl);
  if (prefix === "") return to;
  return `${prefix}${to}`;
}

export function usePathname(): string {
  return useSyncExternalStore(
    subscribe,
    () => stripBase(window.location.pathname),
    () => "/",
  );
}

export function navigate(to: string): void {
  const target = withBase(to);
  if (target === window.location.pathname + window.location.hash) return;
  window.history.pushState(null, "", target);
  window.dispatchEvent(new Event(NAVIGATE_EVENT));
}

type LinkProps = Omit<AnchorHTMLAttributes<HTMLAnchorElement>, "href"> & { to: string };

/** Same-origin link that navigates without a reload; modified clicks behave like normal links. */
export function Link({ to, onClick, ...rest }: LinkProps) {
  const handleClick = (event: MouseEvent<HTMLAnchorElement>): void => {
    onClick?.(event);
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
      return;
    }
    event.preventDefault();
    navigate(to);
  };
  return <a href={withBase(to)} onClick={handleClick} {...rest} />;
}
