import type { NextConfig } from "next";

// Proxy /api/* to the existing Flask backend so the browser makes same-origin
// calls (no CORS). Override the target with BACKEND_URL when needed.
const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:5000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${BACKEND_URL}/api/:path*` }];
  },
};

export default nextConfig;
