import fs from "node:fs";
import { Resvg } from "@resvg/resvg-js";
import { PNG } from "pngjs";
import pngToIco from "png-to-ico";

const svg = fs.readFileSync("build/logo.svg", "utf8");
const sizes = [16, 24, 32, 48, 64, 128, 256];
const pngs = [];

for (const size of sizes) {
  const rendered = new Resvg(svg, {
    fitTo: { mode: "width", value: size }
  }).render();

  const png = new PNG({ width: size, height: size });
  png.data = Buffer.from(rendered.pixels);
  pngs.push(PNG.sync.write(png));
}

fs.writeFileSync("build/icon.ico", await pngToIco(pngs));
console.log("Created build/icon.ico with multiple resolutions.");
