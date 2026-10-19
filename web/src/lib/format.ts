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
