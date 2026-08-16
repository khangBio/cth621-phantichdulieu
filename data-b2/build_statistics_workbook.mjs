import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "outputs/thong_ke_aapl";
const payload = JSON.parse(await fs.readFile(`${outputDir}/workbook_data.json`, "utf8"));
const workbook = Workbook.create();
console.log("stage: workbook-created");

const navy = "#17365D";
const blue = "#D9EAF7";
const pale = "#F4F7FA";
const gold = "#F4B183";
const green = "#E2F0D9";
const white = "#FFFFFF";
const bodyFont = { name: "Aptos", size: 10, color: "#1F2937" };

function title(sheet, range, value) {
  range.merge();
  range.values = [[value]];
  range.format = {
    fill: navy,
    font: { name: "Aptos Display", size: 16, bold: true, color: white },
    verticalAlignment: "center",
    horizontalAlignment: "left",
  };
  range.format.rowHeight = 30;
}

function header(range) {
  range.format = {
    fill: blue,
    font: { ...bodyFont, bold: true, color: navy },
    wrapText: true,
    verticalAlignment: "center",
    horizontalAlignment: "center",
    borders: { preset: "outside", style: "thin", color: "#9FBAD0" },
  };
  range.format.rowHeight = 32;
}

function styleBody(range) {
  range.format.font = bodyFont;
  range.format.borders = {
    insideHorizontal: { style: "thin", color: "#E5E7EB" },
    bottom: { style: "thin", color: "#C9D2DC" },
  };
}

const overview = workbook.worksheets.add("Tong quan");
overview.showGridLines = false;
title(overview, overview.getRange("A1:H1"), "THỐNG KÊ MÔ TẢ VÀ BẢNG TẦN SUẤT – AAPL");
overview.getRange("A3:B8").values = [
  ["Nội dung", "Giá trị"],
  ["Mã cổ phiếu", payload.metadata.stock_code],
  ["Biến mục tiêu", `${payload.metadata.target} (giá đóng cửa)`],
  ["Số quan sát", payload.metadata.rows],
  ["Khoảng thời gian", `${payload.metadata.date_min} – ${payload.metadata.date_max}`],
  ["Biến định lượng", payload.metadata.quantitative_columns.join(", ")],
];
header(overview.getRange("A3:B3"));
styleBody(overview.getRange("A4:B8"));
overview.getRange("B6").format.numberFormat = "#,##0";
overview.getRange("A10:H10").merge();
overview.getRange("A10").values = [["NHẬN XÉT CHÍNH"]];
overview.getRange("A10:H10").format = { fill: gold, font: { ...bodyFont, bold: true, color: navy } };
const reportLines = payload.report
  .split("\n")
  .filter((line) => line.startsWith("- **") || line.startsWith("Giá đóng cửa trung bình"))
  .map((line) => [line.replaceAll("**", "").replaceAll("`", "")]);
overview.getRange(`A11:H${10 + reportLines.length}`).merge(true);
overview.getRange(`A11:A${10 + reportLines.length}`).values = reportLines;
overview.getRange(`A11:H${10 + reportLines.length}`).format = {
  font: bodyFont,
  fill: pale,
  wrapText: true,
  verticalAlignment: "top",
  borders: { preset: "outside", style: "thin", color: "#D1D5DB" },
};
overview.getRange(`A11:H${10 + reportLines.length}`).format.rowHeight = 46;
overview.getRange("A3:A8").format.columnWidth = 23;
overview.getRange("B3:B8").format.columnWidth = 48;
overview.getRange("A10:H20").format.columnWidth = 15;
overview.freezePanes.freezeRows(1);
console.log("stage: overview-ready");

const stats = workbook.worksheets.add("Thong ke dinh luong");
stats.showGridLines = false;
title(stats, stats.getRange("A1:Z1"), "THỐNG KÊ CÁC BIẾN ĐỊNH LƯỢNG");
console.log("stage: stats-title");
const statHeaders = payload.summary_columns;
const statRows = payload.summary_rows.map((row) => statHeaders.map((column) => row[column]));
stats.getRangeByIndexes(2, 0, 1, statHeaders.length).values = [statHeaders];
console.log("stage: stats-headers");
stats.getRangeByIndexes(3, 0, statRows.length, statHeaders.length).values = statRows;
console.log("stage: stats-values");
header(stats.getRangeByIndexes(2, 0, 1, statHeaders.length));
console.log("stage: stats-header-format");
styleBody(stats.getRangeByIndexes(3, 0, statRows.length, statHeaders.length));
console.log("stage: stats-body-format");
stats.getRangeByIndexes(3, 1, statRows.length, 3).format.numberFormat = "#,##0";
console.log("stage: stats-numfmt1");
stats.getRangeByIndexes(3, 6, statRows.length, statHeaders.length - 6).format.numberFormat = "#,##0.00";
console.log("stage: stats-numfmt2");
stats.getRangeByIndexes(3, 0, statRows.length, 1).format.fill = green;
console.log("stage: stats-fill");
stats.getRangeByIndexes(2, 0, statRows.length + 1, statHeaders.length).format.columnWidth = 15;
stats.getRange("A:A").format.columnWidth = 14;
stats.getRange("F:F").format.columnWidth = 34;
stats.freezePanes.freezeRows(3);
stats.freezePanes.freezeColumns(1);
console.log("stage: stats-ready");

const target = workbook.worksheets.add("Muc tieu close");
target.showGridLines = false;
title(target, target.getRange("A1:N1"), "THỐNG KÊ MÔ TẢ BIẾN MỤC TIÊU: CLOSE");
const targetHeaders = payload.target_columns;
const targetRows = payload.target_rows.map((row) => targetHeaders.map((column) => row[column]));
target.getRangeByIndexes(2, 0, 1, targetHeaders.length).values = [targetHeaders];
target.getRangeByIndexes(3, 0, targetRows.length, targetHeaders.length).values = targetRows;
header(target.getRangeByIndexes(2, 0, 1, targetHeaders.length));
styleBody(target.getRangeByIndexes(3, 0, targetRows.length, targetHeaders.length));
target.getRangeByIndexes(3, 1, 1, targetHeaders.length - 1).format.numberFormat = "#,##0.00";
target.getRange("A6:N6").merge();
target.getRange("A6").values = [[reportLines.at(-1)?.[0] ?? ""]];
target.getRange("A6:N6").format = { fill: pale, font: bodyFont, wrapText: true, verticalAlignment: "top" };
target.getRange("A6:N6").format.rowHeight = 72;
target.getRangeByIndexes(2, 0, 2, targetHeaders.length).format.columnWidth = 15;
target.getRange("A:A").format.columnWidth = 14;
target.freezePanes.freezeRows(3);
console.log("stage: target-ready");

const frequency = workbook.worksheets.add("Bang tan suat");
frequency.showGridLines = false;
title(frequency, frequency.getRange("A1:J1"), "BẢNG TẦN SUẤT THEO QUY TẮC STURGES");
const freqHeaders = payload.frequency_columns;
const freqRows = payload.frequency_rows.map((row) => freqHeaders.map((column) => row[column]));
frequency.getRangeByIndexes(2, 0, 1, freqHeaders.length).values = [freqHeaders];
frequency.getRangeByIndexes(3, 0, freqRows.length, freqHeaders.length).values = freqRows;
header(frequency.getRangeByIndexes(2, 0, 1, freqHeaders.length));
styleBody(frequency.getRangeByIndexes(3, 0, freqRows.length, freqHeaders.length));
frequency.getRangeByIndexes(3, 1, freqRows.length, 1).format.numberFormat = "0";
frequency.getRangeByIndexes(3, 3, freqRows.length, 3).format.numberFormat = "#,##0.00";
frequency.getRangeByIndexes(3, 6, freqRows.length, 1).format.numberFormat = "#,##0";
frequency.getRangeByIndexes(3, 7, freqRows.length, 1).format.numberFormat = "0.00";
frequency.getRangeByIndexes(3, 8, freqRows.length, 1).format.numberFormat = "#,##0";
frequency.getRangeByIndexes(3, 9, freqRows.length, 1).format.numberFormat = "0.00";
frequency.getRangeByIndexes(2, 0, freqRows.length + 1, freqHeaders.length).format.columnWidth = 17;
frequency.getRange("A:A").format.columnWidth = 14;
frequency.getRange("C:C").format.columnWidth = 25;
frequency.freezePanes.freezeRows(3);
console.log("stage: frequency-ready");

await fs.mkdir(outputDir, { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(`${outputDir}/thong_ke_mo_ta_aapl.xlsx`);
console.log("stage: workbook-exported");

const inspection = await workbook.inspect({
  kind: "table",
  range: "Muc tieu close!A1:N6",
  include: "values,formulas",
  tableMaxRows: 8,
  tableMaxCols: 14,
});
console.log(inspection.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);
