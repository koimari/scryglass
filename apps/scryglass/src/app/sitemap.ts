import type { MetadataRoute } from "next";

const ROUTES = [
  "/",
  "/elo",
  "/matches",
  "/tiers",
  "/methodology",
  "/chat",
  "/privacy",
  "/sources",
  "/legal",
  "/security",
] as const;

export default function sitemap(): MetadataRoute.Sitemap {
  return ROUTES.map((route) => ({
    url: `https://scryglass.xyz${route}`,
    changeFrequency: route === "/elo" || route === "/matches" ? "daily" : "monthly",
    priority: route === "/elo" ? 1 : route === "/matches" || route === "/tiers" ? 0.8 : 0.5,
  }));
}
