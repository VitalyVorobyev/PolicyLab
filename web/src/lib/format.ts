const MINUS = "−";

function signed(text: string): string {
  return text.startsWith("-") ? MINUS + text.slice(1) : text;
}

/** Fixed number of decimals with a typographic minus and thousands separators. */
export function fixed(value: number, decimals: number): string {
  const text = value.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
  // Avoid "−0.000" for tiny negative values that round to zero.
  return Number(text.replace(/,/g, "")) === 0 ? text.replace("-", "") : signed(text);
}

/** Compact scientific notation for tolerances and residuals, e.g. 1.6 × 10⁻¹¹. */
export function scientific(value: number, digits = 2): string {
  if (value === 0) return "0";
  const [mantissa = "", exponent = "0"] = value.toExponential(digits - 1).split("e");
  const exp = Number(exponent);
  if (exp >= -2 && exp <= 3) return signed(String(Number(value.toPrecision(digits))));
  const sup = String(exp)
    .split("")
    .map((c) => "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"["-0123456789".indexOf(c)] ?? c)
    .join("");
  return `${signed(mantissa)} × 10${sup}`;
}

/** Kilometres with a separator, e.g. 317,636 km. */
export function km(value: number, decimals = 0): string {
  return `${fixed(value, decimals)} km`;
}

export function shortSha(sha: string): string {
  return sha.slice(0, 7);
}

/** "2026-10-18T11:28:07Z" -> "18 Oct 2026, 11:28 UTC" (deterministic, independent of locale). */
export function utcDateTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const pad = (n: number): string => String(n).padStart(2, "0");
  return `${d.getUTCDate()} ${months[d.getUTCMonth()] ?? ""} ${d.getUTCFullYear()}, ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())} UTC`;
}

/** snake_case identifier -> readable phrase, e.g. "jacobi_max_abs_drift" -> "jacobi max abs drift". */
export function humanize(identifier: string): string {
  return identifier.replace(/_/g, " ");
}
