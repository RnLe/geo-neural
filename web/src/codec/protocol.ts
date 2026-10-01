// Messages between the microscope and its decoder worker. Each decode carries an id; the page drops replies
// whose id is not the latest it asked for.

export type ToDecoder =
  | { type: "init"; wasmUrl: string }
  | { type: "decode"; id: number; product: ArrayBuffer; model: ArrayBuffer | null }
  | { type: "dispose" };

export interface DecodedMessage {
  type: "decoded";
  id: number;
  rows: number;
  cols: number;
  coder: string;
  boundUnits: number;
  /** Wall time of the decode call alone, measured in the worker. */
  decodeMs: number;
  latticeSha256: string;
  /** Heights in metres, float32, row-major. */
  heights: Float32Array;
  /** Header and byte breakdown read from the file by the decoder (JSON text). */
  describe: string;
}

export type FromDecoder =
  | { type: "ready"; tableId: string }
  | DecodedMessage
  | { type: "error"; id: number | null; message: string }
  | { type: "disposed" };
