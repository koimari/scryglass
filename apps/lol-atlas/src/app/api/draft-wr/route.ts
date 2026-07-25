import { NextResponse } from "next/server";
import { spawn } from "child_process";
import { existsSync } from "fs";
import path from "path";

export const runtime = "nodejs";

type Body = {
  blue?: string[];
  red?: string[];
  league?: string | null;
  elo_diff?: number | null;
  blue_roles?: string[] | null;
  red_roles?: string[] | null;
};

function findRepoRoot(): string {
  const candidates = [
    path.resolve(process.cwd(), "../.."),
    path.resolve(process.cwd()),
    path.resolve(process.cwd(), "../../.."),
  ];
  for (const c of candidates) {
    if (existsSync(path.join(c, "lol_kills", "draft_score.py"))) return c;
  }
  return candidates[0];
}

function runPython(payload: Body): Promise<Record<string, unknown>> {
  const root = findRepoRoot();
  const code = `
import json, sys
sys.path.insert(0, ${JSON.stringify(root)})
from lol_kills.draft_score import draft_score
body = json.load(sys.stdin)
out = draft_score(
    body["blue"], body["red"],
    league=body.get("league"),
    elo_diff=body.get("elo_diff"),
    blue_roles=body.get("blue_roles"),
    red_roles=body.get("red_roles"),
)
print(json.dumps(out))
`;

  return new Promise((resolve, reject) => {
    const child = spawn("python3", ["-c", code], {
      cwd: root,
      env: { ...process.env, PYTHONPATH: root },
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (d) => {
      stdout += String(d);
    });
    child.stderr.on("data", (d) => {
      stderr += String(d);
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr.trim() || `python exit ${code}`));
        return;
      }
      try {
        const line = stdout.trim().split("\n").filter(Boolean).pop() || "{}";
        resolve(JSON.parse(line));
      } catch {
        reject(new Error(`bad json: ${stdout.slice(0, 200)}`));
      }
    });
    child.stdin.write(JSON.stringify(payload));
    child.stdin.end();
  });
}

export async function POST(req: Request) {
  try {
    const body = (await req.json()) as Body;
    if (!body.blue?.length || !body.red?.length) {
      return NextResponse.json({ error: "blue and red picks required" }, { status: 400 });
    }
    if (body.blue.length !== 5 || body.red.length !== 5) {
      return NextResponse.json({ error: "need 5 picks per side" }, { status: 400 });
    }
    const result = await runPython(body);
    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : String(e) },
      { status: 500 },
    );
  }
}
