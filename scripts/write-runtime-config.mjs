import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(__dirname, "..");
const outputPath = resolve(projectRoot, "web", "runtime-config.js");
const apiBaseUrl = (process.env.API_BASE_URL || "").trim().replace(/\/+$/, "");

mkdirSync(dirname(outputPath), { recursive: true });
writeFileSync(
  outputPath,
  `window.__APP_CONFIG__ = {\n  apiBaseUrl: ${JSON.stringify(apiBaseUrl)}\n};\n`,
  "utf8"
);

console.log(`runtime-config.js written with apiBaseUrl=${apiBaseUrl || "(same-origin)"}`);
