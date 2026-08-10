import type { NextConfig } from "next";

const tierListDisplayUrl = process.env.SCRYGLASS_TIERLIST_DISPLAY_URL?.trim()
  || "https://97gks2fobqkgppwx.public.blob.vercel-storage.com/rankings/tierlists.json";

const nextConfig: NextConfig = {
  turbopack: {
    root: __dirname,
  },
  async redirects() {
    return [
      { source: "/ratings", destination: "/elo", permanent: true },
    ];
  },
  async rewrites() {
    return [
      { source: "/data/tierlists.json", destination: tierListDisplayUrl },
    ];
  },
  async headers() {
    return [
      {
        source: "/data/tierlists.json",
        headers: [
          { key: "Access-Control-Allow-Origin", value: "*" },
          { key: "Cache-Control", value: "public, max-age=0, s-maxage=21600, stale-while-revalidate=3600" },
        ],
      },
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
