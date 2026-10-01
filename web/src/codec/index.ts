// Codec views: the compression microscope and the byte accounting against corpus size.

export { checkManifest, errorFromCode, loadCodecBundle, productsAt } from "./bundle";
export type { CodecBundle, CodecManifest, CodecProduct, CodecRegion, CoderId } from "./bundle";
export { breakEven, corpusSummary, perField, standaloneBreakdown } from "./accounting";
export { mountMicroscope } from "./microscope";
export type { MicroscopeHandle, MicroscopeOptions } from "./microscope";
export { mountBytes } from "./bytes";
export type { BytesHandle, BytesOptions } from "./bytes";
