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

export const DashedCircleIcon = (p: IconProps) => (
  <Icon {...p}>
    <circle cx="8" cy="8" r="6.25" strokeDasharray="2.2 2.2" />
  </Icon>
);

export const AlertIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M8 1.9l6.4 11.4H1.6z" />
    <path d="M8 6.4v3.1M8 11.4v.01" />
  </Icon>
);

export const ExternalIcon = (p: IconProps) => (
  <Icon {...p}>
    <path d="M9.5 2.5h4v4M13.5 2.5L7.5 8.5" />
    <path d="M12 9.5v3.5H3V4h3.5" />
  </Icon>
);

