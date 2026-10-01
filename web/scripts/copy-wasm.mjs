// Copies the wasm-pack output of native/landscape-wasm and native/gnc into src/wasm/ (gitignored).
// Runs wasm-pack first when a package has not been built yet.
import { spawnSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(web, "src/wasm");
// The same flags as the npm scripts: no machine paths in the binaries; the decoder uses 128-bit SIMD.
const remap = `--remap-path-prefix=${homedir()}=~`;
const packages = [
  { crate: "../native/landscape-wasm", name: "landscape_wasm", args: [], rustflags: remap },
  { crate: "../native/gnc", name: "gnc_wasm", args: ["--out-name", "gnc_wasm"], rustflags: `${remap} -C target-feature=+simd128` },
];

mkdirSync(out, { recursive: true });
let copied = 0;
for (const p of packages) {
  const crate = resolve(web, p.crate);
  const pkg = join(crate, "pkg");
  const files = [`${p.name}.js`, `${p.name}.d.ts`, `${p.name}_bg.wasm`, `${p.name}_bg.wasm.d.ts`];
  if (!files.every((f) => existsSync(join(pkg, f)))) {
    if (!existsSync(crate)) {
      if (files.every((f) => existsSync(join(out, f)))) continue;
      console.error(`no wasm package at ${pkg} and no crate at ${crate}`);
      process.exit(1);
    }
    console.log(`building ${p.name} with wasm-pack`);
    const run = spawnSync("wasm-pack", ["build", crate, "--target", "web", "--release", ...p.args], {
      stdio: "inherit",
      env: { ...process.env, RUSTFLAGS: p.rustflags },
    });
    if (run.status !== 0) {
      console.error("wasm-pack failed; install it or run `npm run wasm` once");
      process.exit(run.status ?? 1);
    }
  }
  for (const f of files) {
    const from = join(pkg, f);
    const to = join(out, f);
    if (existsSync(to) && statSync(to).mtimeMs >= statSync(from).mtimeMs && statSync(to).size === statSync(from).size) continue;
    copyFileSync(from, to);
    copied += 1;
  }
}
if (copied) console.log(`copied ${copied} wasm file(s) to src/wasm/`);
