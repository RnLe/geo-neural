// The terrain mesh: one static indexed grid; heights, overlays and shading
// come from textures, so a candidate switch only re-uploads two textures.

import * as THREE from "three";
import { buildLut, DIVERGING_STOPS, ELEVATION_STOPS, hexToRgb, NEUTRAL_MAP, STREAM_COLORS } from "../data/palette";
import type { LegendEntry } from "../data/bundle";
import type { Overlay } from "../data/store";

export const OVERLAY_INDEX: Record<Overlay, number> = { elevation: 0, error: 1, geology: 2, streams: 3 };

/** Stream filter bits for the shader. */
export const STREAM_BOTH = 1;
export const STREAM_LOST = 2;
export const STREAM_SPURIOUS = 4;

const vertexShader = /* glsl */ `
precision highp float;
precision highp int;
uniform sampler2D uRef;
uniform sampler2D uCand;
uniform int uSide;
uniform int uShowRef;
uniform float uSpacing;
uniform float uExag;
uniform float uH0;
out float vH;
out float vErr;
out vec3 vN;
out vec2 vCell;

float shown(ivec2 p) {
  p = clamp(p, ivec2(0), ivec2(uSide - 1));
  return uShowRef == 1 ? texelFetch(uRef, p, 0).r : texelFetch(uCand, p, 0).r;
}

void main() {
  vCell = uv * float(uSide - 1);
  ivec2 p = ivec2(vCell + 0.5);
  float r = texelFetch(uRef, p, 0).r;
  float c = texelFetch(uCand, p, 0).r;
  float h = uShowRef == 1 ? r : c;
  vH = h;
  vErr = c - r;
  // x grows with column (east), z grows with row (south).
  float dx = (shown(p + ivec2(1, 0)) - shown(p - ivec2(1, 0))) / (2.0 * uSpacing);
  float dz = (shown(p + ivec2(0, 1)) - shown(p - ivec2(0, 1))) / (2.0 * uSpacing);
  vN = vec3(-dx * uExag, 1.0, -dz * uExag);
  vec3 pos = vec3(position.x, (h - uH0) * uExag, position.z);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}
`;

const fragmentShader = /* glsl */ `
precision highp float;
precision highp int;
uniform sampler2D uMask;
uniform sampler2D uElevLut;
uniform sampler2D uErrLut;
uniform sampler2D uGeoPal;
uniform int uSide;
uniform int uOverlay;
uniform int uStreamFilter;
uniform float uHMin;
uniform float uHMax;
uniform float uErrRange;
uniform float uPatternPx;
uniform vec3 uSun;
uniform vec3 uNeutral;
uniform vec3 uColBoth;
uniform vec3 uColLost;
uniform vec3 uColSpur;
in float vH;
in float vErr;
in vec3 vN;
in vec2 vCell;
out vec4 fragColor;

vec3 lut(sampler2D t, float x) {
  return texture(t, vec2(clamp(x, 0.0, 1.0) * (255.0 / 256.0) + 0.5 / 256.0, 0.5)).rgb;
}

void main() {
  ivec2 p = clamp(ivec2(vCell + 0.5), ivec2(0), ivec2(uSide - 1));
  vec4 m = texelFetch(uMask, p, 0);
  vec3 col = uNeutral;
  if (uOverlay == 0) {
    col = lut(uElevLut, (vH - uHMin) / (uHMax - uHMin));
  } else if (uOverlay == 1) {
    col = lut(uErrLut, 0.5 + 0.5 * clamp(vErr / uErrRange, -1.0, 1.0));
  } else if (uOverlay == 2) {
    int cls = int(m.b * 255.0 + 0.5);
    vec4 g = texelFetch(uGeoPal, ivec2(cls, 0), 0);
    col = mix(uNeutral, g.rgb, g.a);
  } else {
    bool r = m.r > 0.5;
    bool c = m.g > 0.5;
    vec2 f = floor(gl_FragCoord.xy / uPatternPx);
    if (r && c && (uStreamFilter & 1) != 0) {
      col = uColBoth;
    } else if (r && !c && (uStreamFilter & 2) != 0) {
      // Lost: diagonal stripes, two on, one off.
      if (mod(f.x + f.y, 3.0) < 2.0) col = uColLost;
    } else if (!r && c && (uStreamFilter & 4) != 0) {
      // Spurious: checkerboard.
      if (mod(f.x + f.y, 2.0) < 1.0) col = uColSpur;
    }
  }
  // Restrained hillshade, 1 on flat ground.
  vec3 n = normalize(vN);
  float shade = clamp(1.0 + 1.4 * (dot(n, uSun) - uSun.y), 0.62, 1.12);
  fragColor = vec4(col * shade, 1.0);
}
`;

function srgb(hex: string): THREE.Vector3 {
  const [r, g, b] = hexToRgb(hex);
  return new THREE.Vector3(r / 255, g / 255, b / 255);
}

function lutTexture(stops: readonly string[]): THREE.DataTexture {
  const tex = new THREE.DataTexture(buildLut(stops), 256, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}

function heightTexture(data: Float32Array, side: number): THREE.DataTexture {
  const tex = new THREE.DataTexture(data, side, side, THREE.RedFormat, THREE.FloatType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

export interface TerrainOptions {
  side: number;
  spacingM: number;
  h0: number;
  hMin: number;
  hMax: number;
  errRangeM: number;
  reference: Float32Array;
  referenceStreams: Uint8Array;
  geology: Uint8Array;
  legend: LegendEntry[];
}

export class Terrain {
  readonly mesh: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial>;
  readonly half: number;
  private readonly candData: Float32Array;
  private readonly maskData: Uint8Array;
  private readonly textures: THREE.Texture[] = [];
  private readonly candTex: THREE.DataTexture;
  private readonly maskTex: THREE.DataTexture;

  constructor(o: TerrainOptions) {
    const side = o.side;
    this.half = ((side - 1) * o.spacingM) / 2;
    const n = side * side;

    const positions = new Float32Array(n * 3);
    const uvs = new Float32Array(n * 2);
    for (let r = 0; r < side; r++) {
      for (let c = 0; c < side; c++) {
        const i = r * side + c;
        positions[3 * i] = c * o.spacingM - this.half;
        positions[3 * i + 2] = r * o.spacingM - this.half;
        uvs[2 * i] = c / (side - 1);
        uvs[2 * i + 1] = r / (side - 1);
      }
    }
    const index = new Uint32Array((side - 1) * (side - 1) * 6);
    let k = 0;
    for (let r = 0; r < side - 1; r++) {
      for (let c = 0; c < side - 1; c++) {
        const a = r * side + c;
        const b = a + 1;
        const d = a + side;
        const e = d + 1;
        index[k++] = a;
        index[k++] = d;
        index[k++] = b;
        index[k++] = b;
        index[k++] = d;
        index[k++] = e;
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
    geometry.setIndex(new THREE.BufferAttribute(index, 1));

    const refTex = heightTexture(o.reference, side);
    this.candData = new Float32Array(o.reference);
    this.candTex = heightTexture(this.candData, side);

    this.maskData = new Uint8Array(n * 4);
    for (let i = 0; i < n; i++) {
      this.maskData[4 * i] = o.referenceStreams[i] ? 255 : 0;
      this.maskData[4 * i + 2] = o.geology[i];
      this.maskData[4 * i + 3] = 255;
    }
    this.maskTex = new THREE.DataTexture(this.maskData, side, side, THREE.RGBAFormat, THREE.UnsignedByteType);
    this.maskTex.magFilter = THREE.NearestFilter;
    this.maskTex.minFilter = THREE.NearestFilter;
    this.maskTex.needsUpdate = true;

    const pal = new Uint8Array(256 * 4);
    for (const entry of o.legend) {
      if (entry.code < 0 || entry.code > 255) continue;
      pal.set(hexToRgb(entry.color), entry.code * 4);
    }
    const palTex = new THREE.DataTexture(pal, 256, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
    palTex.needsUpdate = true;

    const elevLut = lutTexture(ELEVATION_STOPS);
    const errLut = lutTexture(DIVERGING_STOPS);
    this.textures.push(refTex, this.candTex, this.maskTex, palTex, elevLut, errLut);

    // Sun from the north-west, 45 degrees up; north is -z.
    const alt = Math.PI / 4;
    const sun = new THREE.Vector3(-Math.cos(alt) * Math.SQRT1_2, Math.sin(alt), -Math.cos(alt) * Math.SQRT1_2);

    const material = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader,
      fragmentShader,
      uniforms: {
        uRef: { value: refTex },
        uCand: { value: this.candTex },
        uMask: { value: this.maskTex },
        uElevLut: { value: elevLut },
        uErrLut: { value: errLut },
        uGeoPal: { value: palTex },
        uSide: { value: side },
        uShowRef: { value: 0 },
        uSpacing: { value: o.spacingM },
        uExag: { value: 1 },
        uH0: { value: o.h0 },
        uOverlay: { value: 0 },
        uStreamFilter: { value: STREAM_BOTH | STREAM_LOST | STREAM_SPURIOUS },
        uHMin: { value: o.hMin },
        uHMax: { value: o.hMax },
        uErrRange: { value: o.errRangeM },
        uPatternPx: { value: 2 },
        uSun: { value: sun },
        uNeutral: { value: srgb(NEUTRAL_MAP) },
        uColBoth: { value: srgb(STREAM_COLORS.both) },
        uColLost: { value: srgb(STREAM_COLORS.lost) },
        uColSpur: { value: srgb(STREAM_COLORS.spurious) },
      },
    });
    this.mesh = new THREE.Mesh(geometry, material);
    // Heights are applied in the shader, so the CPU bounds are flat.
    this.mesh.frustumCulled = false;
  }

  private get u() {
    return this.mesh.material.uniforms;
  }

  setCandidate(height: Float32Array, streams: Uint8Array): void {
    this.candData.set(height);
    this.candTex.needsUpdate = true;
    for (let i = 0; i < streams.length; i++) this.maskData[4 * i + 1] = streams[i] ? 255 : 0;
    this.maskTex.needsUpdate = true;
  }

  setOverlay(overlay: Overlay): void {
    this.u.uOverlay.value = OVERLAY_INDEX[overlay];
  }

  setExaggeration(v: number): void {
    this.u.uExag.value = v;
  }

  setShowReference(show: boolean): void {
    this.u.uShowRef.value = show ? 1 : 0;
  }

  setStreamFilter(bits: number): void {
    this.u.uStreamFilter.value = bits;
  }

  setPixelRatio(ratio: number): void {
    this.u.uPatternPx.value = Math.max(1, Math.round(2 * ratio));
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.mesh.material.dispose();
    for (const t of this.textures) t.dispose();
  }
}
