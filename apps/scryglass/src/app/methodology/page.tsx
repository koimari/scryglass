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
            Each completed series tells the model which team beat which opponent.
            Beating a stronger opponent carries more information. Losing to a
            weaker opponent carries more information in the other direction. A
            series counts once, regardless of whether it lasted two or five games.
          </p>
          <p>
            Regional results establish domestic strength. International events
            and roster movement connect those regions on one scale. The default
            order uses an adjusted rating, which lowers estimates that still have
            wide uncertainty.
          </p>
          <p className="font-mono text-sm">
            adjusted rating = raw rating − uncertainty above the accepted floor
          </p>
          <aside className="method-example">
            <strong>Example: T1 1559, Gen.G 1557</strong>
            <p>The two-point gap says their accepted results are close. It comes from the full series history, the strength of each opponent, the links between competitions, and the uncertainty around each estimate. It does not mean that T1 won two more games.</p>
          </aside>
        </section>

        <section id="player-ratings">
          <h2>Player ratings</h2>
          <p>
            Every completed game compares two five-player lineups. A player moves
            up after a win and down after a loss. The change is larger when the
            result was less expected from the strength of both lineups. Cross-league
            events and player transfers connect domestic competitions.
          </p>
          <p>
            All five teammates receive the lineup result, with each player&apos;s
            uncertainty controlling how quickly their estimate can move. KDA,
            damage, gold, farm, and vision do not enter this rating. Those statistics
            belong to the separate game-grade calculation.
          </p>
          <p>
            The displayed adjustment becomes more cautious when a player has wide
            uncertainty or weak links to stronger competitions. Earlier games in
            connected competitions reduce that uncertainty.
          </p>
          <p>
            The evidence label checks precision, stability, freshness, sample
            support, and active status. These checks are not averaged into the
            rating and do not carry equal weights. They decide how confidently the
            site can describe the estimate. Current rankings include active teams
            and players. Historical profile pages remain available by direct link.
          </p>
        </section>

        <section id="game-grades">
          <h2>Player game grades</h2>
          <p>
            A game grade describes one player&apos;s performance. It compares kill
            participation, survival, damage, gold, farm, and vision with four
            references: the player&apos;s earlier games, their teammates in that game,
            the opposing player in the same role,
            and earlier league results for that role. Each reference has equal
            weight in the final score.
          </p>
          <p>
            The calculation uses only reference games from earlier calendar
            days. It does not use the game winner. A grade appears after the
            player has ten earlier comparable games and the source contains two
            complete five-player lineups. Missing inputs make the grade
            unavailable.
          </p>
          <p>
            Champion choice is not a comparison baseline. A supportive mid can
            score below the role baseline despite a clean KDA. These grades
            describe recorded game performance. They do not change the long-term
            player rating or predict the next match.
          </p>
        </section>

        <section id="composition-signal">
          <h2>Composition signal</h2>
          <p>
            A completed game can show a descriptive signal for the two
            compositions. The model reads the champions and roles that were
            available before the game. It also controls for the pre-game team
            strength gap, uncertainty, league, and patch.
          </p>
          <p>
            Each pick receives a signed contribution when the model has at least
            forty earlier games for that champion in that role. A positive value
            helps that side&apos;s composition. The team totals add the ten pick
            contributions. Thin evidence leaves the total unavailable and shows
            the prior role-game count instead.
          </p>
          <p>
            This signal describes the composition input. It does not grade the
            player&apos;s execution, change team or player ratings, or give a betting
            probability. The private research model and its training data stay
            outside the public pack.
          </p>
        </section>

        <section id="tier-lists">
          <h2>Champion tier lists</h2>
          <p>
            Each patch and role has its own board. Eligible professional games
            from all regions, leagues, and tournaments enter the same patch
            pool. The calculation controls for team strength and side. A
            champion needs verified appearances in the selected patch and role
            to enter a performance board. The minimum-games filter uses this
            patch-wide appearance count and removes thin samples from the visible
            performance and matchup views.
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
            Unpicked, but viable starts from champions with zero accepted games in
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
            Rank movement compares the current tier ladder with the prior Sunday
            and the positions one, three, and twelve calendar months earlier.
            A positive change means that the player climbed. “New” means that the
            player did not meet the ranking rules on the comparison date. The
            public site receives only the finished display rows.
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
