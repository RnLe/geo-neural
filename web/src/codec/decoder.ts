// Page side of the decoder worker: start on first use, one pending decode at a time, terminate on dispose.

import type { DecodedMessage, FromDecoder, ToDecoder } from "./protocol";

export interface DecoderClient {
  /** Resolves with the decoded field, or rejects; a newer request supersedes an older one. */
  decode(product: Uint8Array, model: Uint8Array | null): Promise<DecodedMessage>;
  tableId(): string | null;
  dispose(): void;
}

export function supported(): boolean {
  return typeof Worker !== "undefined" && typeof WebAssembly === "object" && typeof WebAssembly.instantiate === "function";
}

export function createDecoder(wasmUrl: string): DecoderClient {
  let worker: Worker | null = null;
  let ready: Promise<void> | null = null;
  let table: string | null = null;
  let seq = 0;
  let disposed = false;
  const pending = new Map<number, { resolve: (m: DecodedMessage) => void; reject: (e: Error) => void }>();
  let onReady: { resolve: () => void; reject: (e: Error) => void } | null = null;

  function send(msg: ToDecoder, transfer: Transferable[] = []): void {
    worker?.postMessage(msg, transfer);
  }

  function fail(err: Error): void {
    onReady?.reject(err);
    onReady = null;
    for (const p of pending.values()) p.reject(err);
    pending.clear();
  }

  function onMessage(event: MessageEvent<FromDecoder>): void {
    const msg = event.data;
    switch (msg.type) {
      case "ready":
        table = msg.tableId;
        onReady?.resolve();
        onReady = null;
        return;
      case "decoded": {
        const p = pending.get(msg.id);
        pending.delete(msg.id);
        p?.resolve(msg);
        return;
      }
      case "error": {
        const err = new Error(msg.message);
        if (msg.id === null) return fail(err);
        const p = pending.get(msg.id);
        pending.delete(msg.id);
        p?.reject(err);
        return;
      }
      case "disposed":
        (event.target as Worker).terminate();
        return;
    }
  }

  function start(): Promise<void> {
    if (ready) return ready;
    ready = new Promise<void>((resolve, reject) => {
      onReady = { resolve, reject };
      try {
        worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
      } catch (err) {
        reject(err instanceof Error ? err : new Error(String(err)));
        return;
      }
      worker.addEventListener("message", onMessage);
      worker.addEventListener("error", (e) => fail(new Error(e.message || "the decoder worker failed to start")));
      send({ type: "init", wasmUrl });
    });
    ready.catch(() => {
      worker?.terminate();
      worker = null;
      ready = null;
    });
    return ready;
  }

  return {
    async decode(product, model) {
      if (disposed) throw new Error("the decoder was disposed");
      await start();
      const id = ++seq;
      for (const [old, p] of pending) {
        p.reject(new Error("superseded"));
        pending.delete(old);
      }
      // Copies, so the caller keeps its bytes.
      const productCopy = product.slice().buffer;
      const modelCopy = model ? model.slice().buffer : null;
      return new Promise<DecodedMessage>((resolve, reject) => {
        pending.set(id, { resolve, reject });
        send({ type: "decode", id, product: productCopy, model: modelCopy }, modelCopy ? [productCopy, modelCopy] : [productCopy]);
      });
    },
    tableId: () => table,
    dispose() {
      if (disposed) return;
      disposed = true;
      fail(new Error("the decoder was disposed"));
      if (worker) {
        const w = worker;
        w.postMessage({ type: "dispose" } satisfies ToDecoder);
        setTimeout(() => w.terminate(), 300);
      }
      worker = null;
    },
  };
}
