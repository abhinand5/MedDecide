// Logic test for tools/audit/audit.html — extracts the page's own functions and exercises them.
//
// Run: node tests/test_audit_page.mjs
// The page is a single self-contained file (no network, no libraries), so this test does not
// duplicate its logic: it evaluates the script block from the HTML itself, which is what the
// operator will run in a browser.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import assert from "node:assert/strict";

const here = dirname(fileURLToPath(import.meta.url));
const htmlPath = join(here, "..", "tools", "audit", "audit.html");
const html = readFileSync(htmlPath, "utf8");

const match = html.match(/<script>([\s\S]*?)<\/script>/);
assert.ok(match, "audit.html must contain a <script> block");
const source = match[1];

// The page code touches `document`/`localStorage` at load; provide inert stubs so the pure
// functions can be reached, then export them through a tiny CommonJS shim (as the page does).
const documentStub = {
  getElementById: () => ({
    addEventListener() {}, style: {}, classList: { add() {}, remove() {} },
    set innerHTML(v) {}, get innerHTML() { return ""; },
    set textContent(v) {}, get textContent() { return ""; },
    hidden: false, value: "", focus() {}, blur() {}, click() {},
  }),
  addEventListener() {},
  createElement: () => ({ click() {}, remove() {}, style: {} }),
  body: { appendChild() {} },
  activeElement: null,
  title: "",
};
const localStorageStub = {
  _data: {},
  getItem(k) { return this._data[k] ?? null; },
  setItem(k, v) { this._data[k] = v; },
  removeItem(k) { delete this._data[k]; },
};
const stubs = { document: documentStub, localStorage: localStorageStub, confirm: () => false, URL: { createObjectURL: () => "blob:", revokeObjectURL() {} }, Blob: class {}, module: { exports: {} } };
const names = Object.keys(stubs);
const fn = new Function(...names, source + "\n;return module.exports;");
const api = fn(...names.map((n) => stubs[n]));

assert.equal(typeof api.parseJsonl, "function");
assert.equal(typeof api.buildExport, "function");
assert.equal(typeof api.toJsonl, "function");
assert.equal(typeof api.sampleKey, "function");
assert.equal(typeof api.summarise, "function");

// ---- parseJsonl -----------------------------------------------------------------------
const parsed = api.parseJsonl('{"item_id":"a"}\n\n{"item_id":"b"}\n');
assert.deepEqual(parsed.map((r) => r.item_id), ["a", "b"], "blank lines are skipped");
assert.throws(() => api.parseJsonl('{"a":1}\nnot json\n'), /line 2/, "bad line reports its number");

// ---- buildExport ----------------------------------------------------------------------
const rows = [
  { item_id: "a", template_id: "t1", source: "s" },
  { item_id: "b", template_id: "t1", source: "s" },
  { item_id: "c", template_id: "t2", source: "s" },
];
const verdicts = { a: "accept", c: "reject" };
const notes = { a: "  looks right  ", c: "" };
const out = api.buildExport(rows, verdicts, notes, "2026-10-06T00:00:00Z");
assert.equal(out.length, 2, "only reviewed items are exported");
assert.deepEqual(out.map((r) => r.item_id), ["a", "c"], "export follows sample order");
assert.deepEqual(Object.keys(out[0]).sort(), ["index_in_sample", "item_id", "note", "source", "template_id", "timestamp", "verdict"].sort());
assert.equal(out[0].verdict, "accept");
assert.equal(out[0].note, "looks right", "notes are trimmed");
assert.equal(out[0].index_in_sample, 0);
assert.equal(out[1].index_in_sample, 2, "index reflects position in the sample");
assert.equal(out[0].timestamp, "2026-10-06T00:00:00Z");

const jsonl = api.toJsonl(out);
const roundTrip = api.parseJsonl(jsonl);
assert.deepEqual(roundTrip, out, "export JSONL round-trips");
assert.ok(jsonl.endsWith("\n"), "JSONL ends with a newline");

// ---- summarise ------------------------------------------------------------------------
const summary = api.summarise(rows, verdicts);
assert.deepEqual(summary, { total: 3, reviewed: 2, accept: 1, reject: 1, remaining: 1 });
assert.deepEqual(api.summarise(rows, {}), { total: 3, reviewed: 0, accept: 0, reject: 0, remaining: 3 });

// ---- sampleKey ------------------------------------------------------------------------
const k1 = api.sampleKey(rows);
const k2 = api.sampleKey([...rows].reverse());
assert.equal(k1, k2, "key does not depend on row order");
assert.notEqual(k1, api.sampleKey(rows.slice(0, 2)), "different samples get different keys");

// ---- the real sample file, if present (gitignored; skipped when absent) ---------------
const samplePath = join(here, "..", "outputs", "bench_v0", "T11", "audit_sample.jsonl");
try {
  const sample = api.parseJsonl(readFileSync(samplePath, "utf8"));
  assert.ok(sample.length > 0, "sample has rows");
  for (const row of sample) {
    assert.ok(row.item_id && row.template_id && row.source, "row has identity fields");
    assert.ok(Array.isArray(row.options) && row.options.length >= 2, "row has options");
    assert.ok(row.options.some((o) => o.key === row.gold), "gold is one of the options");
    assert.ok(row.state && row.question, "row has state and question");
    assert.ok(row.gold_provenance && Object.keys(row.gold_provenance).length >= 0, "row carries gold provenance");
  }
  const exported = api.buildExport(sample, Object.fromEntries(sample.map((r) => [r.item_id, "accept"])), {}, "2026-10-06T00:00:00Z");
  assert.equal(exported.length, sample.length, "every sample row can be exported");
  console.log(`sample file checked: ${sample.length} items`);
} catch (err) {
  if (err.code === "ENOENT") console.log("sample file not present; skipped the sample-specific checks");
  else throw err;
}

console.log("audit.html logic: all checks passed");
