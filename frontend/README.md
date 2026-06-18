# Ghostrader frontend

A light Next.js + shadcn/ui read-only dashboard for the Ghostrader engine. It does
**not** talk to the Python code directly — it calls the existing Flask API
(`src/web/server.py`) and renders it. Panels: system health, config, market briefing,
price chart, daily P&L, recent trades.

## Requirements

- **Node 20+** (Next 16 + Tailwind v4 need it; Node 18 will fail to install the
  native `@tailwindcss/oxide` binary). An `.nvmrc` pins Node 20:

  ```bash
  nvm use            # -> Node 20
  ```

## Run (two processes)

1. **Backend** — from the repo root, start Flask (serves the JSON API on :5000):

   ```bash
   source .venv/bin/activate
   python -m src web
   ```

2. **Frontend** — in this folder:

   ```bash
   nvm use
   npm run dev        # http://localhost:3000
   ```

The browser calls same-origin `/api/*`, which `next.config.ts` rewrites to the Flask
backend — so there's no CORS setup. Point at a different backend with:

```bash
BACKEND_URL=http://host:port npm run dev
```

## Build

```bash
npm run build && npm start
```

## Layout

```
src/
  app/page.tsx               dashboard composition
  lib/api.ts                 typed Flask API client + response types
  lib/use-api.ts             tiny fetch hook (no SWR/react-query)
  components/dashboard/       one file per panel + a reusable <Panel> + SVG <LineChart>
  components/ui/              shadcn/ui primitives
```

The chart is a dependency-free inline SVG (no charting library) to keep the bundle small.
