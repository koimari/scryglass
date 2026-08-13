const BODY = `Contact: https://github.com/koimari/scryglass/security/advisories/new
Expires: 2027-02-13T00:00:00Z
Preferred-Languages: en
Canonical: https://scryglass.xyz/.well-known/security.txt
Policy: https://scryglass.xyz/security
`;

export function GET() {
  return new Response(BODY, {
    headers: {
      "Cache-Control": "public, max-age=3600, s-maxage=3600",
      "Content-Type": "text/plain; charset=utf-8",
    },
  });
}
