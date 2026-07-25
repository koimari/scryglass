import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  serverExternalPackages: ["@duckdb/duckdb-wasm"],
  turbopack: {
    root: __dirname,
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
        // Immutable versioned packs (pack ids look like v2026.07.25)
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
