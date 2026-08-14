import { NextResponse, type NextRequest } from "next/server";

function contentSecurityPolicy(nonce: string): string {
  const development = process.env.NODE_ENV !== "production";
  return [
    "default-src 'self'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
    "object-src 'none'",
    `script-src 'self' 'nonce-${nonce}' 'strict-dynamic'${development ? " 'unsafe-eval'" : ""}`,
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob: https://cdn.communitydragon.org https://static.wikia.nocookie.net",
    "font-src 'self' data:",
    `connect-src 'self'${development ? " ws://localhost:* ws://127.0.0.1:*" : ""}`,
    "media-src 'self'",
    "manifest-src 'self'",
    "worker-src 'self' blob:",
    ...(development ? [] : ["upgrade-insecure-requests"]),
  ].join("; ");
}

export function proxy(request: NextRequest) {
  const nonce = crypto.randomUUID().replaceAll("-", "");
  const policy = contentSecurityPolicy(nonce);
  if (request.nextUrl.pathname === "/_global-error") {
    return new NextResponse("Not found\n", {
      status: 404,
      headers: {
        "Cache-Control": "private, no-store",
        "Content-Security-Policy": policy,
        "Content-Type": "text/plain; charset=utf-8",
      },
    });
  }
  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("x-nonce", nonce);
  requestHeaders.set("Content-Security-Policy", policy);

  const response = NextResponse.next({ request: { headers: requestHeaders } });
  response.headers.set("Content-Security-Policy", policy);
  return response;
}

export const config = {
  matcher: [
    "/((?!_next/static(?:/|$)|_next/image(?:/|$)|favicon\\.ico$|robots\\.txt$|sitemap\\.xml$|\\.well-known/security\\.txt$).*)",
  ],
};
