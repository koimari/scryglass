import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  turbopack: {
    root: __dirname,
  },
  async redirects() {
    return [
      { source: "/ratings", destination: "/elo", permanent: true },
    ];
  },
  async headers() {
    return [
      {
        source: "/packs/manifest.json",
        headers: [
          { key: "Access-Control-Allow-Origin", value: "*" },
          { key: "Cache-Control", value: "public, max-age=21600, stale-while-revalidate=3600" },
        ],
      },
      {
        source: "/packs/:path*",
        headers: [
          { key: "Access-Control-Allow-Origin", value: "*" },
          { key: "Cache-Control", value: "public, max-age=21600, stale-while-revalidate=3600" },
        ],
      },
    ];
  },
};

export default nextConfig;
