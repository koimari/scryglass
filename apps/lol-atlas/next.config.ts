import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  serverExternalPackages: ["@duckdb/duckdb-wasm"],
  // API routes read versioned v2 artifacts from the repo at request time;
  // trace them into the lambda so the routes work in production.
  outputFileTracingIncludes: {
    "/api/v2/tierlist": ["public/v2/tierlists/**/*.json"],
  },
  turbopack: {
    root: __dirname,
  },
  async redirects() {
    return [
      {
        source: "/grubs",
        destination: "/articles/void-grubs-contest-or-leave",
        permanent: true,
      },
      { source: "/ratings", destination: "/elo", permanent: true },
      { source: "/matches", destination: "/browse", permanent: true },
      { source: "/matches/head-to-head", destination: "/browse/head-to-head", permanent: true },
      { source: "/data", destination: "/reproduce", permanent: true },
    ];
  },
  async headers() {
    return [
      {
        source: "/packs/manifest.json",
        headers: [
          { key: "Access-Control-Allow-Origin", value: "*" },
          { key: "Cache-Control", value: "public, max-age=60" },
        ],
      },
      {
        source: "/packs/latest.json",
        headers: [
          { key: "Access-Control-Allow-Origin", value: "*" },
          { key: "Cache-Control", value: "public, max-age=60" },
        ],
      },
      {
        source: "/packs/:pack(v\\d{4}\\.\\d{2}\\.\\d{2})/:path*",
        headers: [
          { key: "Access-Control-Allow-Origin", value: "*" },
          { key: "Cache-Control", value: "public, max-age=31536000, immutable" },
        ],
      },
      {
        source: "/packs/:path*",
        headers: [
          { key: "Access-Control-Allow-Origin", value: "*" },
          { key: "Cache-Control", value: "public, max-age=3600" },
        ],
      },
    ];
  },
};

export default nextConfig;
