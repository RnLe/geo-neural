// Standalone demo page: intro, then the viewer, the chart and the lab.
// three.js loads only with the viewer chunk (dynamic import below).

import "./styles.css";
import { mountChart } from "./charts";
import { loadBundle } from "./data/bundle";
import { h } from "./data/dom";
import { createSelectionStore } from "./data/store";
import { mountLab } from "./lab";
import wasmUrl from "./wasm/landscape_wasm_bg.wasm?url";

const handles: { dispose(): void }[] = [];

function slot(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el;
}

async function main(): Promise<void> {
  const status = slot("gn-load-status");
  let bundle;
  try {
    bundle = await loadBundle(new URL("bundle/", document.baseURI).href);
  } catch (err) {
    status.textContent = `The data bundle could not be loaded: ${err instanceof Error ? err.message : String(err)}`;
    return;
  }
  status.textContent = "";
  status.hidden = true;

  const m = bundle.manifest;
  const store = createSelectionStore({ candidateId: m.candidates[0].id, overlay: "elevation" });

  const viewerSlot = slot("gn-viewer");
  const viewerPending = import("./viewer").then(({ mountViewer }) => {
    handles.push(mountViewer(viewerSlot, { bundle, store }));
  });
  handles.push(mountChart(slot("gn-chart"), { bundle, store }));
  handles.push(mountLab(slot("gn-lab"), { bundle, wasmUrl }));

  const attribution = slot("gn-attribution");
  attribution.replaceChildren(
    ...m.attribution.map((line) => h("li", {}, line)),
    h("li", {}, `Bundle ${m.schema}, written ${m.createdUtc.slice(0, 10)}. Reference grid SHA-256 ${m.region.referenceSha256.slice(0, 16)}...`),
  );

  try {
    await viewerPending;
  } catch (err) {
    viewerSlot.append(h("p", { class: "gn-message" }, `The viewer could not start: ${err instanceof Error ? err.message : String(err)}`));
  }
}

void main();

if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    for (const handle of handles.splice(0)) handle.dispose();
  });
}
