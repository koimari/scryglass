import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import {
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import test from "node:test";
import { articleChartInputsValid } from "../components/ArticleContestCharts.tsx";
import { ARTICLES, VOID_GRUBS_PUBLICATION } from "./articles.ts";
import { readValidatedGrubsArticlePublication } from "./grubsArticlePublication.server.ts";
import {
  GRUBS_ARTICLE_SCHEMA_VERSION,
  GRUBS_MECHANICS_PATCH,
  PSTAR_FX,
  articlePStarAtGoldB,
  validateArticlePublication,
} from "./pstar.ts";
import type { ResolvedPackManifest } from "./serverPack.ts";

const execFileAsync = promisify(execFile);
const appRoot = process.cwd();
const repoRoot = path.resolve(appRoot, "../..");
let fixtureRoot = "";
let fixturePath = "";
let fixtureBytes: Buffer;
let fixture: Record<string, unknown>;

test.before(async () => {
  fixtureRoot = await mkdtemp(path.join(os.tmpdir(), "scryglass-grubs-"));
  fixturePath = path.join(fixtureRoot, "grubs_article_contest_ev.json");
  await execFileAsync(
    "python3",
    [
      "-m",
      "lol_kills.research.grubs_article_publication",
      "--output",
      fixturePath,
    ],
    {
      cwd: repoRoot,
      env: { ...process.env, PYTHONPATH: repoRoot },
    },
  );
  fixtureBytes = await readFile(fixturePath);
  fixture = JSON.parse(fixtureBytes.toString("utf8")) as Record<
    string,
    unknown
  >;
});

test.after(async () => {
  if (fixtureRoot) await rm(fixtureRoot, { recursive: true, force: true });
});

function clone<T>(value: T): T {
  return structuredClone(value);
}

function manifestFor(
  packId: string,
  sha256: string,
): ResolvedPackManifest {
  return {
    pack_id: packId,
    schema_version: "test",
    created_utc: "2026-07-27T00:00:00Z",
    filters: { years: [2026], leagues: "test" },
    attribution: "test",
    excluded: [],
    base_url: null,
    total_bytes: fixtureBytes.byteLength,
    total_files: 1,
    files: [
      {
        path: VOID_GRUBS_PUBLICATION.articlePath,
        rows: null,
        cols: null,
        bytes: fixtureBytes.byteLength,
        sha256,
      },
    ],
    source: "bundled",
    degraded: false,
    degraded_reason: null,
    clocks: {
      publication: {
        value: "2026-07-27T00:00:00Z",
        field: "created_utc",
        status: "available",
      },
      data_through: {
        value: null,
        field: "data_as_of",
        status: "not_declared",
      },
    },
    artifact_ids: {
      data_pack_id: packId,
      model_pack_id: null,
      strength_snapshot_sha256: null,
      calibration_sha256: null,
      one_clock_verified: false,
      status: "not_declared",
    },
    source_provenance: {
      sources: [],
      attribution: "test",
      canonicalization: "test",
      overlap_precedence: "test",
      overlap_precedence_status: "not_declared",
    },
  };
}

test("Python writer and TypeScript validator agree on current mechanics", () => {
  const result = validateArticlePublication(
    fixture,
    VOID_GRUBS_PUBLICATION,
  );
  assert.equal(result.ok, true);
  if (!result.ok) return;

  assert.equal(result.article.schema_version, GRUBS_ARTICLE_SCHEMA_VERSION);
  assert.equal(result.article.mechanics.patch, GRUBS_MECHANICS_PATCH);
  assert.equal(result.article.mechanics.objective_gold_equivalent, 124.13);
  assert.equal(result.article.p_star_pct, 58.24);
  assert.equal(result.article.edge_at_50_pp, -1.94);
  assert.equal(result.atFifty.model_preference, "LEAVE");
  assert.equal(PSTAR_FX.objectiveGold, 124.13);
  assert.ok(Math.abs((articlePStarAtGoldB(0) ?? 0) - 0.582420118347764) < 1e-12);
});

test("schema additions and pre-26.11 mechanics fail closed", () => {
  const extra = clone(fixture);
  extra.auxiliary = {};
  assert.equal(
    validateArticlePublication(extra, VOID_GRUBS_PUBLICATION).ok,
    false,
  );

  const stale = clone(fixture);
  const mechanics = stale.mechanics as Record<string, unknown>;
  mechanics.objective_gold_equivalent = 115.6;
  assert.equal(
    validateArticlePublication(stale, VOID_GRUBS_PUBLICATION).ok,
    false,
  );
});

test("article artifact rejects OE sister fields and headline drift", () => {
  const auxiliary = clone(fixture);
  auxiliary.oe_sister = { breakeven_p_win_fight: 0.24 };
  assert.equal(
    validateArticlePublication(auxiliary, VOID_GRUBS_PUBLICATION).ok,
    false,
  );

  const drifted = clone(fixture);
  drifted.p_star_pct = 58.9;
  assert.equal(
    validateArticlePublication(drifted, VOID_GRUBS_PUBLICATION).ok,
    false,
  );
});

test("manifest hash mismatch withholds an otherwise valid article", async () => {
  const packId = "vtest-grubs-current";
  const articleDir = path.join(
    fixtureRoot,
    packId,
    "studies",
    "grubs",
  );
  await mkdir(articleDir, { recursive: true });
  await writeFile(
    path.join(articleDir, "grubs_article_contest_ev.json"),
    fixtureBytes,
  );
  const sha = createHash("sha256").update(fixtureBytes).digest("hex");
  const manifest = manifestFor(packId, sha);

  const loaded = await readValidatedGrubsArticlePublication(manifest, {
    packRoot: fixtureRoot,
  });
  assert.ok(loaded);
  assert.equal(loaded.sha256, sha);

  const badManifest = manifestFor(packId, "0".repeat(64));
  assert.equal(
    await readValidatedGrubsArticlePublication(badManifest, {
      packRoot: fixtureRoot,
    }),
    null,
  );
});

test("article and reproduce surfaces expose no PDF or OE decision artifact", async () => {
  const sources = await Promise.all(
    [
      path.join(
        appRoot,
        "src",
        "app",
        "articles",
        "void-grubs-contest-or-leave",
        "page.tsx",
      ),
      path.join(appRoot, "src", "app", "reproduce", "page.tsx"),
      path.join(appRoot, "src", "lib", "articles.ts"),
    ].map((source) => readFile(source, "utf8")),
  );
  const combined = sources.join("\n");
  assert.doesNotMatch(
    combined,
    /grubs_decision_numbers|void_grubs_scrap_value_and_contest_rationality\.pdf|PdfEmbed|fn-24|~24%/i,
  );
  const reproduce = sources[1];
  assert.doesNotMatch(reproduce, /missing from pack/i);
  assert.match(reproduce, /expectedSha256:\s*runtimeMetadata\.artifact_sha256/);
  assert.match(reproduce, /compositionVerified/);
});

test("article charts reject blank formula or degenerate quantitative input", () => {
  const valid = {
    pStar: 0.5,
    curve: [
      {
        p_win_fight: 0.5,
        ev_contest_pp: 1,
        ev_leave_pp: 2,
        edge_contest_minus_leave_pp: -1,
        model_preference: "LEAVE" as const,
      },
      {
        p_win_fight: 0.6,
        ev_contest_pp: 3,
        ev_leave_pp: 2,
        edge_contest_minus_leave_pp: 1,
        model_preference: "CONTEST" as const,
      },
    ],
    byLeaveFarm: [
      {
        label: "two_waves",
        leave_farm_gold: 250,
        p_star_at_parity: 0.5,
        p_star_at_parity_pct: 50,
        p_star_at_B_plus_1183: 0.6,
        p_star_at_B_plus_1183_pct: 60,
      },
    ],
    byGoldB: [
      {
        B_gold: 0,
        leave_farm_gold: 250,
        objective_gold: 125,
        p_star: 0.5,
        p_star_pct: 50,
      },
      {
        B_gold: 1000,
        leave_farm_gold: 250,
        objective_gold: 125,
        p_star: 0.6,
        p_star_pct: 60,
      },
    ],
    formulaHtml: { pStar: "<span>p</span>", winProb: "<span>w</span>", params: "<span>x</span>" },
  };
  assert.equal(articleChartInputsValid(valid), true);
  assert.equal(
    articleChartInputsValid({
      ...valid,
      formulaHtml: { ...valid.formulaHtml, pStar: "   " },
    }),
    false,
  );
  assert.equal(
    articleChartInputsValid({
      ...valid,
      curve: [valid.curve[0], { ...valid.curve[0] }],
    }),
    false,
  );
});

test("article index metadata has no independent numerical publication clock", async () => {
  const article = ARTICLES.find(
    (item) => item.slug === "void-grubs-contest-or-leave",
  );
  assert.ok(article);
  assert.doesNotMatch(article.title, /\d+(?:\.\d+)?%|50\/50/);
  assert.doesNotMatch(article.dek, /\d+(?:\.\d+)?%|50\/50/);

  const homeSource = await readFile(
    path.join(appRoot, "src", "app", "page.tsx"),
    "utf8",
  );
  assert.doesNotMatch(homeSource, /\b58\.(?:9|24)%|\bAt 50\/50\b/);
});

test("/grubs uses Next's permanent redirect primitive", async () => {
  const source = await readFile(
    path.join(appRoot, "src", "app", "grubs", "page.tsx"),
    "utf8",
  );
  assert.match(source, /import \{ permanentRedirect \} from "next\/navigation"/);
  assert.match(
    source,
    /permanentRedirect\("\/articles\/void-grubs-contest-or-leave"\)/,
  );
  assert.doesNotMatch(source, /\bredirect\(/);
});
