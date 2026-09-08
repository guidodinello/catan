import { svelte } from '@sveltejs/vite-plugin-svelte'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [svelte()],
  server: {
    // server/app.py (uv run uvicorn server.app:app) during development.
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
