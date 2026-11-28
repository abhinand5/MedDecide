// SSR smoke test: renders every entry mode, tab, view, metric and filter state
// to catch runtime crashes on missing/NOT MEASURED cells that tsc cannot see.
import { createServer } from 'vite';
import React from 'react';
import { renderToString } from 'react-dom/server';

const server = await createServer({ server: { middlewareMode: true }, appType: 'custom', logLevel: 'error' });
const h = React.createElement;
let failures = 0;
const check = (name, el, mustContain = []) => {
  try {
    const html = renderToString(el).replace(/<!-- -->/g, '');
    const text = html.replace(/<[^>]+>/g, '');
    const missing = mustContain.filter((s) => !html.includes(s) && !text.includes(s));
    if (html.includes('NaN')) throw new Error('rendered NaN');
    if (missing.length) throw new Error(`missing: ${missing.join(', ')}`);
    console.log(`ok   ${name} (${html.length.toLocaleString()} chars)`);
  } catch (e) {
    failures++;
    console.error(`FAIL ${name}: ${e.message}`);
  }
};
try {
  const App = (await server.ssrLoadModule('/src/App.tsx')).default;
  const { Leaderboard } = await server.ssrLoadModule('/src/components/Leaderboard.tsx');
  const { Pareto } = await server.ssrLoadModule('/src/components/Pareto.tsx');
  const { Analytics, __test } = await server.ssrLoadModule('/src/components/Analytics.tsx');
  const { Explorer } = await server.ssrLoadModule('/src/components/Explorer.tsx');
  const { data } = await server.ssrLoadModule('/src/lib/data.ts');

  check('App full', h(App, { mode: 'full' }), ['0.7801', '33.7 ms / 17.3 GB', '0.0229', 'CLEAR', '2.7×', '3.1×']);
  check('App leaderboard', h(App, { mode: 'leaderboard' }), ['Full site', 'Pareto']);
  for (const size of ['all', 'le10', 'le4']) check(`Pareto ${size}`, h(Pareto, { initialSize: size }));
  for (const view of ['all', 'full', 'public'])
    for (const metric of ['accuracy', 'brier', 'ece', 'latency'])
      for (const size of ['all', 'le10', 'le4'])
        check(`Leaderboard ${view}/${metric}/${size}`, h(Leaderboard, { initial: { view, metric, size, extra: true } }));
  for (const m of data.models) check(`Detail ${m.id}`, h(Leaderboard, { initial: { open: m.id, extra: true } }));
  for (const t of ['calibration', 'selective', 'cardinality', 'privacy']) check(`Analytics ${t}`, h(Analytics, { initialTab: t }));
  for (const s of __test.REL_SETS) check(`Calibration ${s}`, h(__test.Calibration, { initialSet: s }));
  for (const s of __test.SEL_SETS) check(`Selective ${s}`, h(__test.Selective, { initialSet: s }));
  for (const s of __test.OPT_SETS) check(`OptionScaling ${s}`, h(__test.OptionScaling, { initialSet: s }));
  for (const e of data.clinical_examples) check(`Explorer ${e.id}`, h(Explorer, { initialId: e.id }), ['expected']);
  check('Selective mr_verdict callouts', h(__test.Selective, { initialSet: 'mr_verdict' }), ['98.2%', '99.8%']);
  check('Selective mimic callouts', h(__test.Selective, { initialSet: 'mimic_suite' }), ['90.3%']);
  check('Privacy numbers', h(Analytics, { initialTab: 'privacy' }), ['+0.0468', '+0.0248', '0/300', 'lm_head']);
} finally {
  await server.close();
}
console.log(failures ? `\n${failures} FAILED` : '\nALL SMOKE CHECKS PASSED');
process.exit(failures ? 1 : 0);
