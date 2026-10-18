/** Why a research artifact could not be used. Each kind maps to a distinct, actionable UI message. */
export type DataErrorKind =
  | "network" // the request did not complete
  | "http" // the server answered with a non-2xx status
  | "parse" // the bytes are not JSON
  | "schema_version" // the file declares a schema version this site does not understand
  | "invalid" // the JSON does not satisfy the data contract
  | "empty" // the file is well-formed but has nothing to show
  | "checksum" // artifact bytes do not match the manifest's SHA-256
  | "unsafe_path"; // a path in a manifest or catalog could escape the research directory

export class DataError extends Error {
  readonly kind: DataErrorKind;
  /** The URL or logical file the problem was found in. */
  readonly source: string;

  constructor(kind: DataErrorKind, source: string, message: string) {
    super(message);
    this.name = "DataError";
    this.kind = kind;
    this.source = source;
  }
}

export function isDataError(value: unknown): value is DataError {
  return value instanceof DataError;
}
