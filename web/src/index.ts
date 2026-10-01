// Public entry: framework-free mount functions and the bundle loader.
// mountViewer pulls in three.js; import it lazily where page weight matters.

export { loadBundle } from "./data/bundle";
export type { Bundle, Candidate, CandidateField, ChartRow, LabData, LoadOptions, Manifest } from "./data/bundle";
export { createSelectionStore } from "./data/store";
export type { Overlay, Selection, SelectionStore } from "./data/store";
export { mountViewer } from "./viewer";
export type { ViewerHandle, ViewerOptions } from "./viewer";
export { mountChart } from "./charts";
export type { ChartHandle, ChartOptions } from "./charts";
export { mountLab } from "./lab";
export type { LabHandle, LabOptions } from "./lab";
export { loadCodecBundle, mountBytes, mountMicroscope } from "./codec";
export type { BytesHandle, BytesOptions, CodecBundle, CodecManifest, MicroscopeHandle, MicroscopeOptions } from "./codec";
