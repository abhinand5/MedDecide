import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';

// `?view=leaderboard` on index.html opens the compact leaderboard-first view.
const view = new URLSearchParams(window.location.search).get('view');
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App mode={view === 'leaderboard' ? 'leaderboard' : 'full'} />
  </StrictMode>,
);
