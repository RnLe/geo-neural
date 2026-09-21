// Copies the wasm-pack output of native/landscape-wasm into src/wasm/ (gitignored).
// Runs wasm-pack first when the package has not been built yet.
import { spawnSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const crate = resolve(web, "../native/landscape-wasm");
const pkg = join(crate, "pkg");
const out = join(web, "src/wasm");
const files = ["landscape_wasm.js", "landscape_wasm.d.ts", "landscape_wasm_bg.wasm", "landscape_wasm_bg.wasm.d.ts"];

if (!files.every((f) => existsSync(join(pkg, f)))) {
  if (!existsSync(crate)) {
    if (files.every((f) => existsSync(join(out, f)))) process.exit(0);
    console.error(`no wasm package at ${pkg} and no crate at ${crate}`);
    process.exit(1);
  }
  console.log("building the wasm package with wasm-pack");
  const run = spawnSync("wasm-pack", ["build", crate, "--target", "web", "--release"], { stdio: "inherit" });
  if (run.status !== 0) {
    console.error("wasm-pack failed; install it or run `npm run wasm` once");
    process.exit(run.status ?? 1);
  }
}

mkdirSync(out, { recursive: true });
let copied = 0;
for (const f of files) {
  const from = join(pkg, f);
  const to = join(out, f);
  if (existsSync(to) && statSync(to).mtimeMs >= statSync(from).mtimeMs && statSync(to).size === statSync(from).size) continue;
  copyFileSync(from, to);
  copied += 1;
}
if (copied) console.log(`copied ${copied} wasm file(s) to src/wasm/`);
