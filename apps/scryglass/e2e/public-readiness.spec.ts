import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

const PUBLIC_ROUTES = [
  ["/elo", "Team and player ratings"],
  ["/elo/player/Faker", "Faker"],
  ["/elo/team/T1", "T1"],
  ["/matches", "Matches"],
  ["/matches/e2e-game-1", "T1 defeated Gen.G"],
  ["/tiers", "Tier Lists"],
  ["/methodology", "What the rankings mean"],
  ["/chat", "Ask Scryglass"],
  ["/privacy", "Privacy"],
  ["/sources", "Data Sources"],
  ["/legal", "Legal"],
  ["/security", "Security"],
] as const;

const RELEASE_BOUND_ROUTES = [
  "/elo",
  "/elo/player/Faker",
  "/elo/team/T1",
  "/matches",
  "/matches/e2e-game-1",
  "/tiers",
] as const;

const E2E_RELEASE_ID = "v2026.08.13.000001";

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
  expect(policy).toContain("frame-ancestors 'none'");
  expect(policy).toContain("object-src 'none'");
  expect(response.headers()["x-content-type-options"]).toBe("nosniff");
  expect(response.headers()["referrer-policy"]).toBe("strict-origin-when-cross-origin");
  expect(response.headers()["permissions-policy"]).toContain("camera=()");
  expect(response.headers()["cross-origin-opener-policy"]).toBe("same-origin");

  const apiNotFound = await request.get("/api/no-such-route");
  expect(apiNotFound.status()).toBe(404);
  const apiPolicy = apiNotFound.headers()["content-security-policy"] ?? "";
  const apiNonce = apiPolicy.match(/'nonce-([a-f0-9]+)'/)?.[1];
  expect(apiNonce).toBeTruthy();
  const apiHtml = await apiNotFound.text();
  const apiScripts = [...apiHtml.matchAll(/<script\b([^>]*)>/gi)];
  expect(apiScripts.length).toBeGreaterThan(0);
  for (const script of apiScripts) {
    expect(script[1]).toContain(`nonce="${apiNonce}"`);
  }
  const internalError = await request.get("/_global-error");
  expect(internalError.status()).toBe(404);
  expect(internalError.headers()["content-type"]).toContain("text/plain");
  expect(await internalError.text()).toBe("Not found\n");
  for (const path of ["/favicon.ico/nope", "/robots.txt/nope", "/sitemap.xml/nope", "/.well-known/security.txt/nope"]) {
    const nestedNotFound = await request.get(path);
    const nestedPolicy = nestedNotFound.headers()["content-security-policy"] ?? "";
    expect(nestedPolicy, path).toMatch(/'nonce-[a-f0-9]+'/);
  }
  const prefetchHeaders: Array<Record<string, string>> = [
    { purpose: "prefetch" },
    { "next-router-prefetch": "1" },
  ];
  for (const headers of prefetchHeaders) {
    const prefetch = await request.get("/elo", { headers });
    const prefetchPolicy = prefetch.headers()["content-security-policy"] ?? "";
    const prefetchNonce = prefetchPolicy.match(/'nonce-([a-f0-9]+)'/)?.[1];
    expect(prefetchNonce).toBeTruthy();
    const prefetchHtml = await prefetch.text();
    for (const script of prefetchHtml.matchAll(/<script\b([^>]*)>/gi)) {
      expect(script[1]).toContain(`nonce="${prefetchNonce}"`);
    }
  }

  const security = await request.get("/.well-known/security.txt");
  expect(security.headers()["content-type"]).toContain("text/plain");
  expect(await security.text()).toContain("Canonical: https://scryglass.xyz/.well-known/security.txt");
  expect((await request.get("/robots.txt")).ok()).toBe(true);
  expect((await request.get("/sitemap.xml")).ok()).toBe(true);
});

test("every data page publishes its exact release marker", async ({ request }) => {
  for (const route of RELEASE_BOUND_ROUTES) {
    const response = await request.get(route);
    expect(response.ok(), route).toBe(true);
    expect(await response.text(), route).toContain(
      `data-scryglass-release="${E2E_RELEASE_ID}"`,
    );
  }
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

test("every public page has one named main heading and no WCAG A or AA violations", async ({ page }) => {
  for (const [route, heading] of PUBLIC_ROUTES) {
    const response = await page.goto(route, { waitUntil: "networkidle" });
    expect(response?.ok(), route).toBe(true);
    await expect(page.getByRole("heading", { level: 1, name: heading, exact: true })).toHaveCount(1);
    await expect(page.locator("main")).toHaveCount(1);
    if (route === "/tiers") {
      await expect(page.getByText("3 champions across all roles", { exact: false })).toBeVisible();
    }
    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"])
      .analyze();
    expect(results.violations, `${route}: ${results.violations.map((item) => item.id).join(", ")}`).toEqual([]);
  }
});

test("result tables remain keyboard reachable and scroll inside a 390 px viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/chat");
  await page.getByLabel("Ask a question").fill("who has better rating, Inspired or Faker?");
  await page.getByRole("button", { name: "Ask", exact: true }).click();
  const scrollRegion = page.getByTestId("player-query-result").locator('[class*="tableScroll"]');
  await expect(scrollRegion).toBeVisible();
  const geometry = await scrollRegion.evaluate((element) => ({
    clientWidth: element.clientWidth,
    scrollWidth: element.scrollWidth,
    overflowX: getComputedStyle(element).overflowX,
  }));
  expect(geometry.scrollWidth).toBeGreaterThanOrEqual(geometry.clientWidth);
  expect(["auto", "scroll"]).toContain(geometry.overflowX);

  const profile = page.getByRole("link", { name: "Inspired", exact: true });
  await profile.focus();
  await expect(profile).toBeFocused();
  await page.goto("/elo");
  await page.keyboard.press("Tab");
  const focusStyle = await page.evaluate(() => {
    const style = getComputedStyle(document.activeElement as Element);
    return { outlineStyle: style.outlineStyle, outlineWidth: style.outlineWidth };
  });
  expect(focusStyle.outlineStyle).not.toBe("none");
  expect(focusStyle.outlineWidth).not.toBe("0px");
});

test("reduced motion disables animated scrolling and transitions", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce", colorScheme: "dark" });
  await page.goto("/elo");
  const motion = await page.evaluate(() => {
    const styles = getComputedStyle(document.documentElement);
    const element = document.querySelector("button");
    const button = element ? getComputedStyle(element) : null;
    return {
      scrollBehavior: styles.scrollBehavior,
      transitionDuration: button?.transitionDuration ?? "",
      animationDuration: button?.animationDuration ?? "",
    };
  });
  expect(["auto", ""]).toContain(motion.scrollBehavior);
  const durationSeconds = (value: string) => value.split(",").map((item) => {
    const clean = item.trim();
    return clean.endsWith("ms") ? Number.parseFloat(clean) / 1_000 : Number.parseFloat(clean);
  });
  expect(durationSeconds(motion.transitionDuration).every((value) => value <= 0.001)).toBe(true);
  expect(durationSeconds(motion.animationDuration).every((value) => value <= 0.001)).toBe(true);
});
