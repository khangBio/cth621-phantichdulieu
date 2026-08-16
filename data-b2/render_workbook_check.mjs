import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const dir = "outputs/thong_ke_aapl";
const input = await FileBlob.load(`${dir}/thong_ke_mo_ta_aapl.xlsx`);
const workbook = await SpreadsheetFile.importXlsx(input);
for (const [sheetName, fileName, range] of [
  ["Tong quan", "preview_tong_quan.png", "A1:H17"],
  ["Thong ke dinh luong", "preview_thong_ke.png", "A1:Z8"],
  ["Muc tieu close", "preview_muc_tieu.png", "A1:N6"],
  ["Bang tan suat", "preview_tan_suat.png", "A1:J68"],
]) {
  const image = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(`${dir}/${fileName}`, new Uint8Array(await image.arrayBuffer()));
  console.log(`rendered: ${sheetName}`);
}
