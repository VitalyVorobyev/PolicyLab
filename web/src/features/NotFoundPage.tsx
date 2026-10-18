import { Link } from "../lib/router";
import { usePageTitle } from "../lib/usePageTitle";

export function NotFoundPage() {
  usePageTitle("Page not found – Phase Space");
  return (
    <div className="mx-auto max-w-7xl px-4 py-20 sm:px-8">
      <h1 className="text-5xl">Page not found</h1>
      <p className="mt-5 max-w-prose text-lg text-graphite">
        There is nothing at this address. Study pages only exist once a study has computed data to show.
      </p>
      <p className="mt-8 flex flex-wrap gap-x-6 gap-y-2">
        <Link to="/" className="prose-link">
          Go to the studies
        </Link>
        <Link to="/gravity" className="prose-link">
          Open Gravity Atlas
        </Link>
      </p>
    </div>
  );
}
