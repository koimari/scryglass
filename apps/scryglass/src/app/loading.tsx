export default function Loading() {
  return (
    <section className="route-state" role="status" aria-live="polite">
      <p className="blog-kicker">Scryglass</p>
      <h1>Loading the latest accepted data</h1>
      <p>Ratings and match records will appear when the snapshot is ready.</p>
      <div className="route-state-bars" aria-hidden="true"><i /><i /><i /></div>
    </section>
  );
}
