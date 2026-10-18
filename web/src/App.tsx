import { useEffect, useRef } from "react";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { Layout } from "./components/Layout";
import { GalleryPage } from "./features/gallery/GalleryPage";
import { GravityPage } from "./features/gravity/GravityPage";
import { NotFoundPage } from "./features/NotFoundPage";
import { usePathname } from "./lib/router";

function Route({ path }: { path: string }) {
  switch (path) {
    case "/":
      return <GalleryPage />;
    case "/gravity":
      return <GravityPage />;
    default:
      return <NotFoundPage />;
  }
}

export function App() {
  const path = usePathname();
  const first = useRef(true);

  // After client-side navigation, start at the top and move focus to the new page's content.
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    window.scrollTo(0, 0);
    document.getElementById("main")?.focus({ preventScroll: true });
  }, [path]);

  return (
    <Layout>
      <ErrorBoundary key={path}>
        <Route path={path} />
      </ErrorBoundary>
    </Layout>
  );
}
