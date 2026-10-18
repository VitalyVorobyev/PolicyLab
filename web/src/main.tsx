import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./index.css";

const root = document.getElementById("root");
if (root === null) throw new Error("Missing #root element");

createRoot(root, {
  onUncaughtError: (error, info) => { console.error("Uncaught React error", error, info.componentStack); },
}).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
