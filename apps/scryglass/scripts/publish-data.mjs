#!/usr/bin/env node
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { upload } from "@vercel/blob/client";

function argument(name) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

const packDir = path.resolve(argument("--pack-dir") || "");
const tierPath = path.resolve(argument("--tierlists") || "");
const site = (process.env.SCRYGLASS_PUBLISH_ORIGIN || "https://scryglass.xyz").replace(/\/$/, "");
const secret = process.env.SCRYGLASS_DATA_PUBLISH_TOKEN;
if (!secret || !argument("--pack-dir") || !argument("--tierlists")) {
  throw new Error("Set SCRYGLASS_DATA_PUBLISH_TOKEN and provide --pack-dir and --tierlists");
}

const headers = { authorization: `Bearer ${secret}` };
const endpoint = `${site}/api/data-upload`;
const manifest = JSON.parse(await readFile(path.join(packDir, "manifest.json"), "utf8"));
if (!/^v\d{4}\.\d{2}\.\d{2}\.\d{6}$/.test(manifest.pack_id || "")) {
  throw new Error("Pack manifest has an invalid pack ID");
}

const sha256 = (body) => createHash("sha256").update(body).digest("hex");
const uploaded = [];
for (const item of manifest.files) {
  const body = await readFile(path.join(packDir, item.path));
  if (body.byteLength !== item.bytes || sha256(body) !== item.sha256) {
    throw new Error(`Pack checksum failed before upload: ${item.path}`);
  }
  const result = await upload(`packs/${manifest.pack_id}/${item.path}`, body, {
    access: "public",
    contentType: "application/json",
    handleUploadUrl: endpoint,
    headers,
    multipart: body.byteLength > 4_000_000,
  });
  uploaded.push(result.url);
}

const marker = `/packs/${manifest.pack_id}/`;
const sample = uploaded[0];
if (!sample?.includes(marker)) throw new Error("Blob returned an unexpected pack URL");
const baseUrl = `${sample.split(marker)[0]}/packs/${manifest.pack_id}`;
const publishedManifest = { ...manifest, base_url: baseUrl };
const immutableManifest = `${JSON.stringify(publishedManifest, null, 2)}\n`;
await upload(`packs/${manifest.pack_id}/manifest.json`, immutableManifest, {
  access: "public", contentType: "application/json", handleUploadUrl: endpoint, headers,
});

for (const item of manifest.files) {
  const response = await fetch(`${baseUrl}/${item.path}`);
  if (!response.ok || sha256(Buffer.from(await response.arrayBuffer())) !== item.sha256) {
    throw new Error(`Blob readback failed: ${item.path}`);
  }
}

const latest = { pack_id: manifest.pack_id, base_url: baseUrl, created_utc: manifest.created_utc };
await upload("packs/manifest.json", immutableManifest, {
  access: "public", contentType: "application/json", handleUploadUrl: endpoint, headers,
});
await upload("packs/latest.json", `${JSON.stringify(latest, null, 2)}\n`, {
  access: "public", contentType: "application/json", handleUploadUrl: endpoint, headers,
});
const tierBody = await readFile(tierPath);
const tierPayload = JSON.parse(tierBody.toString("utf8"));
await upload("rankings/tierlists.json", tierBody, {
  access: "public", contentType: "application/json", handleUploadUrl: endpoint, headers,
  multipart: tierBody.byteLength > 4_000_000,
});
const patchOrder = (value) => {
  const [major, minor] = String(value || "").split(".").map(Number);
  return (Number.isFinite(major) ? major : 0) * 1000 + (Number.isFinite(minor) ? minor : 0);
};
const latestPatch = [...(tierPayload.options?.patches || [])]
  .map(String)
  .sort((left, right) => patchOrder(right) - patchOrder(left))[0];
if (!latestPatch) throw new Error("Tier list has no latest patch");
const latestTierBody = Buffer.from(`${JSON.stringify({
  ...tierPayload,
  latest_patch: latestPatch,
  rows: (tierPayload.rows || []).filter((row) => row.patch === latestPatch),
  scopes: (tierPayload.scopes || []).filter((scope) => scope.patch === latestPatch),
})}\n`);
await upload("rankings/tierlists-latest.json", latestTierBody, {
  access: "public", contentType: "application/json", handleUploadUrl: endpoint, headers,
});

const cleared = await fetch(`${site}/api/data-published`, { method: "POST", headers });
if (!cleared.ok) throw new Error(`Cache refresh failed with HTTP ${cleared.status}`);
console.log(JSON.stringify({ pack_id: manifest.pack_id, base_url: baseUrl, files: manifest.files.length, tier_bytes: tierBody.byteLength }));
