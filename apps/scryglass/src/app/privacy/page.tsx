import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Privacy",
  description: "How Scryglass handles browser settings, questions, and request data.",
};

export default function PrivacyPage() {
  return (
    <article className="page-prose public-note">
      <header className="prose-head">
        <p className="blog-kicker">Public release</p>
        <h1>Privacy</h1>
        <p>Scryglass has no user accounts. The site keeps one theme choice in your browser.</p>
      </header>
      <div className="method-content">
        <section>
          <h2>What Reaches The Server</h2>
          <p>Page requests and questions sent to Ask Scryglass reach the hosting service. The app uses a question to return an answer and does not add it to a user profile.</p>
          <p>Hosting and security services can process standard request data such as an IP address, browser details, requested path, and time. Their logs support delivery, abuse control, and incident response.</p>
        </section>
        <section>
          <h2>What Stays In Your Browser</h2>
          <p>The light or dark theme choice stays in local browser storage. Scryglass does not use that setting to identify you.</p>
        </section>
        <section>
          <h2>External Links</h2>
          <p>Links to source sites, team pages, X, Discord, and GitHub follow each service&apos;s privacy terms after you leave Scryglass.</p>
          <p>For a private security report, use the <Link className="row-link" href="/security">security channel</Link>.</p>
        </section>
      </div>
    </article>
  );
}
