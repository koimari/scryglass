import Link from "next/link";

export const metadata = {
  title: "Method — Scryglass",
  description: "How Scryglass calculates team ratings, player ratings, and champion tier lists.",
};

export default function MethodologyPage() {
  return (
    <article className="page-prose methodology-page">
      <header className="prose-head">
        <p className="blog-kicker">Scryglass method</p>
        <h1>What the rankings mean</h1>
        <p>
          Scryglass uses completed professional games. Each public number is a
          descriptive estimate from the latest accepted data update.
        </p>
      </header>

      <div className="method-content">
        <section id="team-ratings">
          <h2>Team ratings</h2>
          <p>
            The team ladder estimates organization strength across regional and
            international play. A series counts once. This stops a five-game
            series from carrying five times the weight of a short series.
          </p>
          <p>
            The default order uses an adjusted rating. The adjustment lowers
            estimates with wide uncertainty. Established teams with connected
            regional and international evidence receive a smaller adjustment.
          </p>
          <p className="font-mono text-sm">
            adjusted rating = raw rating − uncertainty above the accepted floor
          </p>
        </section>

        <section id="player-ratings">
          <h2>Player ratings</h2>
          <p>
            Player ratings use one connected results scale across every
            accepted competition tier. Complete ten-player maps enter one
            regularized fit. Cross-league events and player transfers connect
            domestic pools. Competition tier is not a fixed bonus or penalty.
          </p>
          <p>
            The displayed adjustment is more cautious when a player&apos;s current
            tier has little direct evidence against a stronger tier. Earlier
            maps in stronger competitions reduce that extra uncertainty. This
            changes the evidence width, not the fitted rating.
          </p>
          <p>
            This rating tracks team results with the player in the lineup. It
            does not isolate the player&apos;s individual contribution. The map
            grades below provide that separate performance view. The current
            team, role, and league come from the latest accepted domestic
            appearance.
          </p>
          <p>
            The evidence label checks precision, stability, freshness, sample
            support, and active status. Thin or stale evidence stays visible so
            readers can judge the rating with its limits.
          </p>
        </section>

        <section id="map-grades">
          <h2>Player map grades</h2>
          <p>
            A map grade describes one player&apos;s performance. It compares nine
            match statistics with four references: the player&apos;s earlier maps,
            their teammates in that map, the opposing player in the same role,
            and earlier league results for that role. Each reference has equal
            weight in the final score.
          </p>
          <p>
            The calculation uses only reference games from earlier calendar
            days. It does not use the map winner. A grade appears after the
            player has ten earlier comparable maps and the source contains two
            complete five-player lineups. Missing inputs make the grade
            unavailable.
          </p>
          <p>
            These grades describe recorded map performance. They do not change
            the long-term player rating and do not predict the next match.
          </p>
        </section>

        <section id="tier-lists">
          <h2>Champion tier lists</h2>
          <p>
            Each patch and role has its own board. Eligible professional games
            from all regions, leagues, and tournaments enter the same patch
            pool. The calculation controls for team strength and side. A
            champion needs verified appearances in the selected patch and role
            to enter a performance board.
          </p>
          <p>
            Blind tiers show the expected edge in a champion&apos;s weakest common
            same-role matchup. Good Into counts favorable model comparisons
            against five common role opponents. The matchup matrix compares every
            eligible same-role pair. It gives each response an S-to-D grade and
            keeps the evidence interval available in the cell details.
          </p>
          <p>
            The region filter changes which observed champions appear. The model
            remains patch-wide. These views describe matchup shape after general
            champion strength is accounted for. They do not describe draft order.
          </p>
          <p>
            Unpicked, but viable starts from champions with zero accepted maps in
            the selected patch and role. It compares their role, function, and
            mechanic profiles with champions that teams did play. The region
            filter changes the played comparison pool. An unpicked champion must
            still have zero patch-wide appearances in that role.
          </p>
          <p>
            Structural matches are leads for preparation and testing. They do not
            give an unpicked champion a tier, predicted WR, or draft recommendation.
            Family profile marks the broad ontology basis. Detailed atom profile
            marks champions with deeper mechanic coverage.
          </p>
          <p>
            Rank movement compares the current accepted board with its prior
            snapshot. The public site receives only the finished display rows.
            Training data, coefficients, and evaluation files stay in the local
            research workspace.
          </p>
        </section>

        <section id="updates">
          <h2>Updates and sources</h2>
          <p>
            Ratings refresh on a six-hour schedule. Oracle&apos;s Elixir is the
            baseline source. A completed-game bridge can fill the delay before
            the next export. Identity checks run before a new snapshot becomes
            public. A failed check keeps the previous accepted snapshot online.
          </p>
          <p>
            Return to <Link className="row-link" href="/elo">ratings</Link> or
            view the <Link className="row-link" href="/tiers">tier lists</Link>.
          </p>
        </section>
      </div>
    </article>
  );
}
