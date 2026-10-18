import { Component, type ErrorInfo, type ReactNode } from "react";
import { ErrorState } from "./DataState";

interface Props {
  children: ReactNode;
}

interface State {
  error: unknown;
}

/** Renders a visible error instead of a blank page, and reports the error to the console. */
export class ErrorBoundary extends Component<Props, State> {
  override state: State = { error: null };

  static getDerivedStateFromError(error: unknown): State {
    return { error };
  }

  override componentDidCatch(error: unknown, info: ErrorInfo): void {
    console.error("Phase Space render error", error, info.componentStack);
  }

  override render(): ReactNode {
    if (this.state.error !== null) {
      return (
        <div className="mx-auto max-w-6xl px-4 py-12 sm:px-8">
          <ErrorState error={this.state.error} />
        </div>
      );
    }
    return this.props.children;
  }
}
