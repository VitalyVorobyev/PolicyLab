import type { SVGProps } from "react";

/** Small stroke icons drawn on a 16-unit grid. Decorative unless given a title by the caller. */
type IconProps = SVGProps<SVGSVGElement>;

function Icon({ children, ...props }: IconProps) {
  return (
    <svg
      viewBox="0 0 16 16"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...props}
    >
      {children}
    </svg>
  );
}

export const CheckCircleIcon = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="8" cy="8" r="6.25" />
    <path d="M5.2 8.2l1.9 1.9 3.7-4" />
  </Icon>
);

export const HalfCircleIcon = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="8" cy="8" r="6.25" />
    <path d="M8 1.75v12.5A6.25 6.25 0 0 1 8 1.75z" fill="currentColor" stroke="none" />
  </Icon>
);

export const DashedCircleIcon = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="8" cy="8" r="6.25" strokeDasharray="2.2 2.2" />
  </Icon>
);

export const CrossCircleIcon = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="8" cy="8" r="6.25" />
    <path d="M5.8 5.8l4.4 4.4M10.2 5.8l-4.4 4.4" />
  </Icon>
);

export const AlertIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M8 1.9l6.4 11.4H1.6z" />
    <path d="M8 6.4v3.1M8 11.4v.01" />
  </Icon>
);

export const PlayIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M5 3.2v9.6L12.6 8z" fill="currentColor" />
  </Icon>
);

export const PauseIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M5.2 3.5v9M10.8 3.5v9" strokeWidth="2.2" />
  </Icon>
);

export const RestartIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M3.5 3v4h4" />
    <path d="M3.9 7A4.6 4.6 0 1 1 4.6 11" />
  </Icon>
);

export const StepBackIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M4 3.5v9" />
    <path d="M12 3.5v9L6.5 8z" fill="currentColor" />
  </Icon>
);

export const StepForwardIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M12 3.5v9" />
    <path d="M4 3.5v9L9.5 8z" fill="currentColor" />
  </Icon>
);

export const ExternalIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M9.5 2.5h4v4M13.5 2.5L7.5 8.5" />
    <path d="M12 9.5v3.5H3V4h3.5" />
  </Icon>
);

export const DownloadIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M8 2.5v8M4.8 7.5L8 10.7l3.2-3.2" />
    <path d="M3 13.5h10" />
  </Icon>
);
