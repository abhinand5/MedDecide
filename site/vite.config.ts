import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Multi-page static build. `base: './'` keeps every asset path relative so
// dist/ works on GitHub Pages, a Hugging Face Static Space, or file://.
export default defineConfig({
  base: './',
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        index: 'index.html',
        leaderboard: 'leaderboard.html',
      },
    },
  },
});
