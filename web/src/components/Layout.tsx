import type { ReactNode } from "react";
import { REPOSITORY_URL } from "../lib/links";
import { Link } from "../lib/router";

function Wordmark() {
  return (
    <Link to="/" className="group inline-flex items-center gap-2.5 no-underline">
      <svg viewBox="0 0 28 28" width="26" height="26" aria-hidden="true" focusable="false">
        <rect width="28" height="28" rx="5" fill="var(--color-canvas)" />
        <ellipse cx="14" cy="14" rx="9.5" ry="4.6" fill="none" stroke="var(--color-canvas-strong)" strokeWidth="1.3" transform="rotate(-28 14 14)" />
        <circle cx="14" cy="14" r="2.4" fill="var(--color-canvas-strong)" />
        <circle cx="22.2" cy="9.4" r="2" fill="var(--color-accent)" />
      </svg>
      <span className="font-serif text-xl font-semibold tracking-tight">Phase Space</span>
    </Link>
  );
}

export function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-dvh flex-col">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-3 focus:z-10 focus:bg-sheet focus:px-3 focus:py-2"
      >
        Skip to content
      </a>
      <header className="border-b border-rule">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-6 px-4 py-4 sm:px-8">
          <Wordmark />
          <nav aria-label="Site" className="flex items-center gap-5 text-[0.95rem] text-graphite">
            <Link to="/gravity" className="hover:text-ink">
              Gravity Atlas
            </Link>
            <a href={REPOSITORY_URL} className="hover:text-ink" rel="noreferrer">
              Source
            </a>
          </nav>
        </div>
      </header>
      <main id="main" tabIndex={-1} className="flex-1 outline-none">
        {children}
      </main>
      <footer className="mt-24 border-t border-rule">
        <div className="mx-auto flex max-w-7xl flex-col gap-2 px-4 py-8 text-sm text-graphite sm:flex-row sm:justify-between sm:px-8">
          <p>Every trajectory on this site is computed and published with its manifest and checksum.</p>
          <p>
            <a className="prose-link" href={REPOSITORY_URL} rel="noreferrer">
              VitalyVorobyev/PolicyLab
            </a>{" "}
            on GitHub, MIT or Apache-2.0
          </p>
        </div>
      </footer>
    </div>
  );
}
