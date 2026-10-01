// Owns the WebAssembly decoder. One decode per message; timing covers the decode call only.

import init, { decode, describe, tableId } from "../wasm/gnc_wasm.js";
import type { FromDecoder, ToDecoder } from "./protocol";

interface WorkerScope {
  postMessage(message: FromDecoder, transfer?: Transferable[]): void;
  onmessage: ((event: MessageEvent<ToDecoder>) => void) | null;
  close(): void;
}
const scope = self as unknown as WorkerScope;

let ready: Promise<unknown> | null = null;

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

async function handle(msg: ToDecoder): Promise<void> {
  switch (msg.type) {
    case "init": {
      ready ??= init({ module_or_path: msg.wasmUrl });
      try {
        await ready;
        scope.postMessage({ type: "ready", tableId: tableId() });
      } catch (err) {
        ready = null;
        scope.postMessage({ type: "error", id: null, message: `could not load the WebAssembly decoder: ${message(err)}` });
      }
      return;
    }
    case "decode": {
      if (!ready) return scope.postMessage({ type: "error", id: msg.id, message: "the decoder was not initialised" });
      await ready;
      const product = new Uint8Array(msg.product);
      const model = msg.model ? new Uint8Array(msg.model) : undefined;
      let field;
      try {
        const info = describe(product);
        const started = performance.now();
        field = decode(product, model);
        const decodeMs = performance.now() - started;
        const heights = field.heights();
        scope.postMessage(
          {
            type: "decoded",
            id: msg.id,
            rows: field.rows,
            cols: field.cols,
            coder: field.coder,
            boundUnits: field.boundUnits,
            decodeMs,
            latticeSha256: field.latticeSha256(),
            heights,
            describe: info,
          },
          [heights.buffer],
        );
      } catch (err) {
        scope.postMessage({ type: "error", id: msg.id, message: message(err) });
      } finally {
        field?.free();
      }
      return;
    }
    case "dispose": {
      scope.postMessage({ type: "disposed" });
      scope.close();
      return;
    }
  }
}

// Strictly in arrival order, even across the awaits above.
let queue: Promise<void> = Promise.resolve();
scope.onmessage = (event) => {
  queue = queue.then(() => handle(event.data)).catch((err) => {
    scope.postMessage({ type: "error", id: null, message: message(err) });
  });
};
