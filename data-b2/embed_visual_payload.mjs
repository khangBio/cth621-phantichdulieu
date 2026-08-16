import fs from "node:fs/promises";

const templatePath = new URL("./best-model-comparison.html", import.meta.url);
const payloadPath = new URL(
  "file:///C:/Users/Khangtn/.codex/visualizations/2026/06/27/019f07e4-c17e-7a93-8fad-05ca106b14c6/best-model-comparison-data.json",
);
const destinationPath = new URL(
  "file:///C:/Users/Khangtn/.codex/visualizations/2026/06/27/019f07e4-c17e-7a93-8fad-05ca106b14c6/best-model-comparison.html",
);

const template = await fs.readFile(templatePath, "utf8");
const payload = await fs.readFile(payloadPath, "utf8");
if (!template.includes("__PAYLOAD__")) {
  throw new Error("Payload placeholder not found.");
}
await fs.writeFile(destinationPath, template.replace("__PAYLOAD__", payload), "utf8");
