import { useTheme } from './lib/theme';
import { data } from './lib/data';
import { Nav } from './components/Nav';
import { Hero } from './components/Hero';
import { Explorer } from './components/Explorer';
import { Pareto } from './components/Pareto';
import { Leaderboard } from './components/Leaderboard';
import { Analytics } from './components/Analytics';
import { Footer, Quickstart, Submit } from './components/Bottom';
import { links } from './lib/data';

export type Mode = 'full' | 'leaderboard';

function CompactIntro() {
  return (
    <div className="section -mb-10 pt-8">
      <div className="eyebrow mb-3">MedDecide-Bench · Leaderboard</div>
      <h1 className="h-display max-w-4xl text-[2.1rem] leading-[1.05] sm:text-[2.6rem]">
        Calibrated medical decision models, <span className="italic text-accent">measured honestly.</span>
      </h1>
      <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-muted">
        12 models on 9 frozen sets. The public tier (<code className="num">confirm_general</code>) is open for submissions, and the private EHR
        tier is run locally under the PhysioNet DUA. Missing cells are never imputed.{' '}
        <a href={links.landing_page} className="link">
          Read about Osler →
        </a>
      </p>
    </div>
  );
}

export default function App({ mode }: { mode: Mode }) {
  const [theme, toggle] = useTheme();
  const compact = mode === 'leaderboard';
  return (
    <div id="top" className="min-h-screen">
      <Nav theme={theme} onToggle={toggle} compact={compact} />
      <main>
        {compact ? (
          <CompactIntro />
        ) : (
          <>
            <Hero />
            <div className="hairline" />
            <Explorer />
            <div className="hairline" />
          </>
        )}
        <Pareto />
        <Leaderboard />
        <Analytics />
        {!compact && <Quickstart />}
        <Submit />
      </main>
      <Footer generated={data.generated_utc} />
    </div>
  );
}
