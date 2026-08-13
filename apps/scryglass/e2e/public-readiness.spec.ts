import { expect, test } from "@playwright/test";

test("ratings load Tier 1 records and keep the initial document within budget", async ({ page, request }) => {
  const response = await request.get("/elo");
  expect(response.ok()).toBe(true);
  expect((await response.body()).byteLength).toBeLessThanOrEqual(500 * 1024);
  const allPlayers = await request.get("/elo?tab=players&leagues=ALL");
  expect(allPlayers.ok()).toBe(true);
  expect((await allPlayers.body()).byteLength).toBeLessThanOrEqual(500 * 1024);

  await page.goto("/elo");
  const summary = page.getByRole("region", { name: "Rating summary" });
  await expect(summary).toContainText("Tier 1");
  const ranked = await summary.locator("p").first().innerText();
  expect(Number(ranked.match(/\d+/)?.[0] ?? 0)).toBeGreaterThan(0);
  const firstTeamMark = page.locator('img[alt$=" mark"]').first();
  await expect.poll(() => firstTeamMark.evaluate((image: HTMLImageElement) => image.complete && image.naturalWidth > 0)).toBe(true);
});

test("rating tabs support arrow keys, direct URLs, filters, and fail-closed draft output", async ({ page }) => {
  await page.goto("/elo");
  const teams = page.getByRole("tab", { name: "Teams" });
  await teams.focus();
  await teams.press("ArrowRight");
  await expect(page).toHaveURL(/tab=players/);
  await expect(page.getByRole("tab", { name: "Players" })).toHaveAttribute("aria-selected", "true");
  await expect(page.getByRole("region", { name: "Rating summary" })).toContainText("players");

  await page.getByRole("button", { name: "All", exact: true }).click();
  await expect(page).toHaveURL(/leagues=ALL/);
  await expect(page.getByRole("region", { name: "Rating summary" })).toContainText("All levels");

  await page.getByRole("tab", { name: "Draft" }).click();
  await expect(page).toHaveURL(/tab=draft/);
  await expect(page.getByRole("heading", { name: "Draft Score is unavailable" })).toBeVisible();
  await expect(page.getByText("independent promotion receipt", { exact: false })).toBeVisible();
});

test("HTML responses use nonce-based scripts and publish launch metadata", async ({ request }) => {
  const response = await request.get("/elo");
  const policy = response.headers()["content-security-policy"] ?? "";
  const scripts = policy.split(";").find((directive) => directive.trim().startsWith("script-src")) ?? "";
  expect(scripts).toMatch(/'nonce-[a-f0-9]+'/);
  expect(scripts).toContain("'strict-dynamic'");
  expect(scripts).not.toContain("'unsafe-inline'");

  const security = await request.get("/.well-known/security.txt");
  expect(security.headers()["content-type"]).toContain("text/plain");
  expect(await security.text()).toContain("Canonical: https://scryglass.xyz/.well-known/security.txt");
  expect((await request.get("/robots.txt")).ok()).toBe(true);
  expect((await request.get("/sitemap.xml")).ok()).toBe(true);
});

test("legal, source, privacy, security, and not-found states stay reachable", async ({ page }) => {
  for (const route of ["/privacy", "/sources", "/legal", "/security"]) {
    const response = await page.goto(route);
    expect(response?.ok()).toBe(true);
    await expect(page.locator("h1")).toBeVisible();
  }

  await page.goto("/elo/player/this-player-does-not-exist-000");
  await expect(page.getByRole("heading", { name: "This record is outside the glass" })).toBeVisible();
  await expect(page.locator('meta[name="robots"][content*="noindex"]').first()).toBeAttached();
});
