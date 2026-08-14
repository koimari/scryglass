import Link from "next/link";

export default function NotFound() {
  return (
    <section className="route-state">
      <p className="blog-kicker">404</p>
      <h1>This record is outside the glass</h1>
      <p>The link may point to an old player, team, match, or release.</p>
      <div className="route-state-actions">
        <Link href="/elo">Browse ratings</Link>
        <Link href="/matches">Browse matches</Link>
      </div>
    </section>
  );
}
