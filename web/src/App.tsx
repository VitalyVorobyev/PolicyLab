import { useEffect, useRef } from "react";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { Layout } from "./components/Layout";
import { CitizensPage } from "./features/citizens/CitizensPage";
import { CitizensReportPage } from "./features/citizens/CitizensReportPage";
import { GalleryPage } from "./features/gallery/GalleryPage";
import { GravityPage } from "./features/gravity/GravityPage";
import { GravityReportPage } from "./features/gravity/GravityReportPage";
import { LibraryPage } from "./features/library/LibraryPage";
import { LibraryReportPage } from "./features/library/LibraryReportPage";
import { MethodPage } from "./features/method/MethodPage";
import { NotFoundPage } from "./features/NotFoundPage";
import { usePathname } from "./lib/router";

function Route({ path }: { path: string }) {
  switch (path) {
    case "/":
      return <GalleryPage />;
    case "/gravity":
      return <GravityPage />;
    case "/gravity/report":
      return <GravityReportPage />;
    case "/citizens":
      return <CitizensPage />;
    case "/citizens/report":
      return <CitizensReportPage />;
    case "/library":
      return <LibraryPage />;
    case "/library/report":
      return <LibraryReportPage />;
    case "/method":
      return <MethodPage />;
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
