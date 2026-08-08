import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const START_DATE = "2025-01-01T00:00:00Z";
const LIVE_WINDOW_START = "2026-07-18T00:00:00Z";

function authorized(req: Request): boolean {
  const secret = process.env.CRON_SECRET?.trim();
  if (!secret) return false;
  return req.headers.get("authorization") === `Bearer ${secret}`;
}

function workerUrl(value: string): URL | null {
  try {
    const url = new URL(value);
    if (url.protocol !== "https:" || url.username || url.password || url.hash) return null;
    return url;
  } catch {
    return null;
  }
}

function githubDispatchUrl(repository: string, workflow: string): URL | null {
  if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(repository)) return null;
  if (!/^[A-Za-z0-9_.\/-]+$/.test(workflow)) return null;
  return new URL(`https://api.github.com/repos/${repository}/actions/workflows/${workflow}/dispatches`);
}

async function dispatchGitHubWorkflow(): Promise<NextResponse> {
  const token = process.env.SCRYGLASS_TIERLIST_GITHUB_TOKEN?.trim();
  const repository = process.env.SCRYGLASS_TIERLIST_GITHUB_REPOSITORY?.trim() || "koimari/scryglass";
  const workflow = process.env.SCRYGLASS_TIERLIST_GITHUB_WORKFLOW?.trim() || "tierlist-refresh.yml";
  const ref = process.env.SCRYGLASS_TIERLIST_GITHUB_REF?.trim() || "main";
  const target = githubDispatchUrl(repository, workflow);
  if (!token || !target || !/^[^\s]+$/.test(ref)) {
    return NextResponse.json(
      {
        status: "unavailable",
        code: "refresh_github_dispatch_not_configured",
        reason: "GitHub Actions dispatch credentials or target are not configured",
        window_start: START_DATE,
      },
      { status: 503 },
    );
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15_000);
  try {
    const response = await fetch(target, {
      method: "POST",
      headers: {
        Accept: "application/vnd.github+json",
        Authorization: `Bearer ${token}`,
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        ref,
        inputs: {
          as_of: new Date().toISOString(),
          window_start: START_DATE,
          live_window_start: LIVE_WINDOW_START,
          source_mode: "oe_only",
        },
      }),
      cache: "no-store",
      signal: controller.signal,
    });
    return NextResponse.json(
      {
        status: response.ok ? "accepted" : "unavailable",
        target: "github_actions",
        repository,
        workflow,
        ref,
        window_start: START_DATE,
      },
      { status: response.ok ? 202 : 502 },
    );
  } catch (error) {
    return NextResponse.json(
      {
        status: "unavailable",
        code: "refresh_github_dispatch_failed",
        reason: error instanceof Error ? error.message : "GitHub Actions dispatch failed",
      },
      { status: 502 },
    );
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Trigger the external, durable tier-list worker.
 *
 * Vercel functions cannot rewrite the deployed repository. The worker must
 * publish an immutable index and cells, then move the manifest pointer.
 */
export async function GET(req: Request) {
  if (!authorized(req)) {
    return NextResponse.json({ status: "unauthorized" }, { status: 401 });
  }

  const target = process.env.SCRYGLASS_TIERLIST_INGEST_URL?.trim();
  if (!target) {
    return dispatchGitHubWorkflow();
  }

  const targetUrl = workerUrl(target);
  if (!targetUrl) {
    return NextResponse.json(
      {
        status: "unavailable",
        code: "refresh_target_invalid",
        reason: "SCRYGLASS_TIERLIST_INGEST_URL must be an HTTPS URL",
        window_start: START_DATE,
      },
      { status: 503 },
    );
  }

  const workerToken = process.env.SCRYGLASS_TIERLIST_INGEST_TOKEN?.trim();
  if (!workerToken) {
    return NextResponse.json(
      {
        status: "unavailable",
        code: "refresh_worker_auth_not_configured",
        reason: "SCRYGLASS_TIERLIST_INGEST_TOKEN is required when the worker is configured",
        window_start: START_DATE,
      },
      { status: 503 },
    );
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15_000);
  try {
    const response = await fetch(targetUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        Authorization: `Bearer ${workerToken}`,
      },
      body: JSON.stringify({
        window_start: START_DATE,
        live_window_start: LIVE_WINDOW_START,
        as_of: new Date().toISOString(),
        source_mode: "oe_only",
        trigger: "vercel_cron",
      }),
      cache: "no-store",
      signal: controller.signal,
    });
    const body = await response.text();
    return new NextResponse(body || JSON.stringify({ status: response.ok ? "accepted" : "unavailable" }), {
      status: response.ok ? 202 : 502,
      headers: { "Content-Type": response.headers.get("content-type") ?? "application/json" },
    });
  } catch (error) {
    return NextResponse.json(
      {
        status: "unavailable",
        code: "refresh_worker_unreachable",
        reason: error instanceof Error ? error.message : "refresh worker request failed",
      },
      { status: 502 },
    );
  } finally {
    clearTimeout(timeout);
  }
}
