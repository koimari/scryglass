import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Security",
  description: "How to report a Scryglass security issue privately.",
};

export default function SecurityPage() {
  return (
    <article className="page-prose public-note">
      <header className="prose-head">
        <p className="blog-kicker">Private disclosure</p>
        <h1>Security</h1>
        <p>Send a vulnerability report through GitHub&apos;s private advisory form.</p>
      </header>
      <div className="method-content">
        <section>
          <h2>Report An Issue</h2>
          <p>Include the affected URL, the impact, steps to reproduce, and a minimal proof. Remove credentials and personal data from the report.</p>
          <p><a className="row-link" href="https://github.com/koimari/scryglass/security/advisories/new" target="_blank" rel="noreferrer">Open a private report</a></p>
        </section>
        <section>
          <h2>Safe Handling</h2>
          <p>Use test data and the smallest request that proves the issue. Avoid service disruption, access to other people&apos;s data, persistence, social engineering, and public disclosure before a fix is ready.</p>
        </section>
        <section>
          <h2>What Happens Next</h2>
          <p>The report receives a private review. Confirmed issues stay on the public-release closure ledger until a regression test and deployment check pass.</p>
        </section>
      </div>
    </article>
  );
}
