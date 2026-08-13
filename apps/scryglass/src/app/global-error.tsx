"use client";

export default function GlobalError({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <html lang="en">
      <body>
        <main className="route-state" role="alert">
          <p className="blog-kicker">Application error</p>
          <h1>Scryglass could not open this page</h1>
          <p>Try the request again. The accepted release stays unchanged.</p>
          <div className="route-state-actions">
            <button type="button" onClick={reset}>Try Again</button>
            <a href="/elo">Open Ratings</a>
          </div>
        </main>
      </body>
    </html>
  );
}
