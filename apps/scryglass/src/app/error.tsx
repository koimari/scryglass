"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

export default function ErrorPage({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    const update = () => setOffline(!navigator.onLine);
    update();
    window.addEventListener("online", update);
    window.addEventListener("offline", update);
    return () => {
      window.removeEventListener("online", update);
      window.removeEventListener("offline", update);
    };
  }, []);

  return (
    <section className="route-state" role="alert">
      <p className="blog-kicker">{offline ? "Offline" : "Data error"}</p>
      <h1>{offline ? "Scryglass needs a connection" : "This page missed the snapshot"}</h1>
      <p>{offline ? "Reconnect, then try the page again." : "The previous accepted data remains safe. Try the request again."}</p>
      <div className="route-state-actions">
        <button type="button" onClick={reset}>Try again</button>
        <Link href="/elo">Open ratings</Link>
      </div>
    </section>
  );
}
