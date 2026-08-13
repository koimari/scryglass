import { expect, test, type Page } from "@playwright/test";

async function ask(page: Page, question: string) {
  const results = page.getByTestId("player-query-result");
  const previousCount = await results.count();
  await page.getByLabel("Ask a question").fill(question);
  await page.getByRole("button", { name: "Ask", exact: true }).click();
  await expect(results).toHaveCount(previousCount + 1);
  return results.nth(previousCount);
}

test("chat route, scrolling, expansion, and floating resize work", async ({ page }) => {
  await page.goto("/support");
  await expect(page).toHaveURL(/\/chat$/);
  await expect(page.getByRole("link", { name: "Chat", exact: true })).toHaveAttribute("aria-current", "page");
  await expect(page.getByRole("heading", { name: "Ask Scryglass", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Open Ask Scryglass" })).toHaveCount(0);

  const chat = page.getByRole("region", { name: "Ask Scryglass" });
  const thread = page.getByRole("log", { name: "Chat messages" });
  await expect(thread).toHaveAttribute("data-native-scroll", "true");

  await page.getByRole("button", { name: "Expand chat" }).click();
  await expect(page.getByRole("button", { name: "Restore chat size" })).toBeVisible();
  await expect(chat).toHaveCSS("position", "fixed");
  await page.getByRole("button", { name: "Restore chat size" }).click();

  await page.goto("/");
  await page.getByRole("button", { name: "Open Ask Scryglass" }).click();
  const floatingChat = page.getByRole("region", { name: "Ask Scryglass" });
  await expect(floatingChat).toHaveCSS("resize", "both");
});

test("reported player questions return constrained answers and proof rows", async ({ page }) => {
  await page.goto("/chat");

  const comparison = await ask(page, "who has better rating, Inspired or Faker?");
  await expect(comparison.getByTestId("player-query-headline")).toContainText(/Inspired|Faker/);
  await expect(comparison.getByRole("link", { name: "Inspired", exact: true })).toBeVisible();
  await expect(comparison.getByRole("link", { name: "Faker", exact: true })).toBeVisible();
  await expect(comparison.locator("tbody tr")).toHaveCount(2);

  const possessive = await ask(page, "what is Inspired's rating");
  await expect(possessive.getByTestId("player-query-headline")).toContainText("Inspired");
  await expect(possessive.locator("tbody tr")).toHaveCount(1);
  await expect(possessive.locator("tbody tr").first()).toContainText("Inspired");

  const champion = await ask(page, "who is the best Galio player");
  await expect(champion.getByTestId("player-query-headline")).toContainText("Galio");
  await expect(champion.getByTestId("player-query-basis")).toContainText("95% Wilson lower bound");
  await expect(champion.getByTestId("player-query-caveat")).toContainText("descriptive player-champion record");
  await expect(champion.locator("thead")).toContainText("Champion games");

  const filtered = await ask(page, "best Tier 1 LCK mid with at least 100 games");
  await expect(filtered.getByTestId("player-query-basis")).toContainText("mid role, LCK, Tier 1, at least 100 games");
  await expect(filtered.locator("tbody tr").first()).toContainText("Mid");
  await expect(filtered.locator("tbody tr").first()).toContainText("LCK");
  await expect(filtered.locator("tbody tr").first()).toContainText("Tier 1");

  const thread = page.getByRole("log", { name: "Chat messages" });
  await expect.poll(() => thread.evaluate((element) => element.scrollHeight > element.clientHeight)).toBe(true);
  await expect.poll(() => thread.evaluate((element) => element.scrollTop > 0)).toBe(true);
});
