import type { MetadataRoute } from "next";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: { userAgent: "*", allow: "/", disallow: ["/api/"] },
    sitemap: "https://scryglass.xyz/sitemap.xml",
    host: "https://scryglass.xyz",
  };
}
