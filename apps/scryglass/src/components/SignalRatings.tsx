"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  adjustedRating,
  formatWr,
  playerMatchesQuery,
  playerSlug,
  recordMatchesLeagues,
  softMu,
  TEAM_SIGMA_MIN,
  PLAYER_SIGMA_MIN,
  teamMatchesQuery,
  teamSlug,
  TIER_FILTERS,
  type PlayerMetadata,
  type PlayerRating,
  type PlayerRecord,
  type PlayerWeeklyRanks,
  type ProfileGame,
  type ProfileParticipant,
  type ProfileRecords,
  type TeamRating,
  type TeamRecord,
  type TeamWeeklyRanks,
} from "@/lib/pack";
import { evidenceFields, evidenceInfo, formatEvidenceCell } from "@/lib/evidence";
import { TeamMark } from "./TeamMark";
import styles from "./SignalRatings.module.css";

type Props = {
  teams: TeamRating[];
  players: PlayerRating[];
  teamRecords: Record<string, TeamRecord>;
  teamWeeklyRanks: TeamWeeklyRanks;
  playerRecords: Record<string, PlayerRecord>;
  playerWeeklyRanks: PlayerWeeklyRanks;
  playerMetadata: Record<string, PlayerMetadata>;
  availableLeagues: string[];
  profileRecords: ProfileRecords | null;
};

type Tab = "teams" | "players";
type Sort = "rating" | "movement" | "games" | "name";

const PLAYER_ROLES = [
  ["top", "Top"],
  ["jungle", "Jungle"],
  ["mid", "Mid"],
  ["bot", "Bot"],
  ["support", "Support"],
] as const;

function formatTier(tier: string | null | undefined): string {
  if (tier === "tier1") return "Tier 1";
  if (tier === "tier2") return "Tier 2";
  if (tier === "tier3") return "Tier 3";
  return "All levels";
}

function formatAffiliation(tier: string | null | undefined, league: string | null | undefined): string {
  const tierLabel = formatTier(tier);
  return league ? `${tierLabel} · ${league}` : tierLabel;
}

function roleLabel(role: string | null | undefined): string {
  return PLAYER_ROLES.find(([value]) => value === role)?.[1] ?? "Unknown role";
}

function rankDeltaLabel(delta: number | null | undefined): string {
  if (delta == null || delta === 0) return "—";
  return delta > 0 ? `+${delta}` : `${delta}`;
}

function parseLeagues(value: string | null): string[] {
  return value?.split(",").map((item) => item.trim()).filter(Boolean) ?? [];
}

function gameList(records: ProfileRecords | null, ids: string[] | undefined): ProfileGame[] {
  if (!records || !ids) return [];
  return ids
    .map((id) => records.games[id])
    .filter((game): game is ProfileGame => Boolean(game))
    .sort((a, b) => Date.parse(b.date) - Date.parse(a.date));
}

function teamWon(game: ProfileGame, team: string): boolean {
  if (game.blue_team.toLowerCase() === team.toLowerCase()) return game.blue_win === 1;
  return game.blue_win === 0;
}

function participantFor(game: ProfileGame, player: string): ProfileParticipant | undefined {
  return game.players.find((participant) => participant.player.toLowerCase() === player.toLowerCase());
}

function playerWon(game: ProfileGame, player: string): boolean | null {
  const participant = participantFor(game, player);
  if (!participant) return null;
  return participant.side === "Blue" ? game.blue_win === 1 : game.blue_win === 0;
}

function recentForm(games: ProfileGame[], entity: string, tab: Tab): boolean[] {
  return games.slice(0, 10).flatMap((game) => {
    const won = tab === "teams" ? teamWon(game, entity) : playerWon(game, entity);
    return won == null ? [] : [won];
  });
}

function FormStrip({ form }: { form: boolean[] }) {
  if (!form.length) return <span className={styles.formEmpty}>Awaiting recent maps</span>;
  return (
    <span className={styles.formStrip} aria-label={`Recent form: ${form.map((won) => won ? "win" : "loss").join(", ")}`}>
      {form.map((won, index) => <i key={index} className={won ? styles.formWin : styles.formLoss}>{won ? "W" : "L"}</i>)}
    </span>
  );
}

function ChampionStrip({ games, entity, tab, images, limit = 5 }: { games: ProfileGame[]; entity: string; tab: Tab; images: Record<string, string>; limit?: number }) {
  const champions: string[] = [];
  for (const game of games) {
    const participants = tab === "players"
      ? [participantFor(game, entity)].filter((participant): participant is ProfileParticipant => Boolean(participant))
      : game.players.filter((participant) => {
          const team = participant.side === "Blue" ? game.blue_team : game.red_team;
          return team.toLowerCase() === entity.toLowerCase();
        });
    for (const participant of participants) {
      if (participant.champion && !champions.includes(participant.champion)) champions.push(participant.champion);
      if (champions.length >= limit) break;
    }
    if (champions.length >= limit) break;
  }
  if (!champions.length) return null;
  return (
    <span className={styles.championStrip} aria-label={`Recent champions: ${champions.join(", ")}`}>
      {champions.map((champion) => images[champion] ? (
        // CommunityDragon supplies the champion portraits in the published pack.
        // eslint-disable-next-line @next/next/no-img-element
        <img key={champion} src={images[champion]} alt={champion} title={champion} loading="lazy" />
      ) : null)}
    </span>
  );
}

export function SignalRatings({
  teams,
  players,
  teamRecords,
  teamWeeklyRanks,
  playerRecords,
  playerWeeklyRanks,
  playerMetadata,
  availableLeagues,
  profileRecords,
}: Props) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [tab, setTab] = useState<Tab>(searchParams.get("tab") === "players" ? "players" : "teams");
  const [query, setQuery] = useState(searchParams.get("q") ?? "");
  const [leagues, setLeagues] = useState<string[]>(() => {
    const parsed = parseLeagues(searchParams.get("leagues"));
    return parsed.length ? parsed : ["TIER1"];
  });
  const [role, setRole] = useState(searchParams.get("role") ?? "");
  const [minGames, setMinGames] = useState(Math.max(5, Number(searchParams.get("min") ?? 20)));
  const [sort, setSort] = useState<Sort>((searchParams.get("sort") as Sort) || "rating");
  const [expanded, setExpanded] = useState(false);
  const [selected, setSelected] = useState("");

  useEffect(() => {
    const params = new URLSearchParams();
    if (tab === "players") params.set("tab", "players");
    if (query.trim()) params.set("q", query.trim());
    if (leagues.length) params.set("leagues", leagues.join(","));
    if (tab === "players" && role) params.set("role", role);
    if (tab === "players" && minGames !== 20) params.set("min", String(minGames));
    if (sort !== "rating") params.set("sort", sort);
    const suffix = params.toString();
    router.replace(suffix ? `${pathname}?${suffix}` : pathname, { scroll: false });
  }, [tab, query, leagues, role, minGames, sort, pathname, router]);

  const teamDelta = (team: string) => teamWeeklyRanks.by_team[team]?.delta;
  const playerDelta = (player: string) => {
    const record = playerRecords[player];
    const tier = record?.current_tier ?? "all";
    return playerWeeklyRanks.by_player[player]?.[tier]?.delta ?? playerWeeklyRanks.by_player[player]?.all?.delta;
  };

  const filteredTeams = useMemo(() => {
    const list = teams.filter((team) => teamMatchesQuery(team.team, query) && recordMatchesLeagues(teamRecords[team.team], leagues));
    return [...list].sort((a, b) => {
      if (sort === "name") return a.team.localeCompare(b.team);
      if (sort === "games") return (b.n_maps ?? 0) - (a.n_maps ?? 0);
      if (sort === "movement") return (teamDelta(b.team) ?? -999) - (teamDelta(a.team) ?? -999);
      return adjustedRating(b, TEAM_SIGMA_MIN) - adjustedRating(a, TEAM_SIGMA_MIN);
    });
  // Weekly rank maps are immutable inputs for this render.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [teams, teamRecords, leagues, query, sort, teamWeeklyRanks]);

  const filteredPlayers = useMemo(() => {
    const list = players
      .filter((player) => player.n_maps >= minGames)
      .filter((player) => {
        const record = playerRecords[player.player];
        return !role || (record?.primary_role ?? record?.roles?.[0]) === role;
      })
      .filter((player) => playerMatchesQuery(player.player, playerRecords[player.player]?.current_team ?? player.last_team, query))
      .filter((player) => {
        const record = playerRecords[player.player];
        const team = record?.current_team ?? player.last_team;
        return recordMatchesLeagues(record, leagues) || recordMatchesLeagues(team ? teamRecords[team] : undefined, leagues);
      });
    return [...list].sort((a, b) => {
      if (sort === "name") return a.player.localeCompare(b.player);
      if (sort === "games") return b.n_maps - a.n_maps;
      if (sort === "movement") return (playerDelta(b.player) ?? -999) - (playerDelta(a.player) ?? -999);
      return softMu(b.mu_total, b.sigma, PLAYER_SIGMA_MIN) - softMu(a.mu_total, a.sigma, PLAYER_SIGMA_MIN);
    });
  // Weekly rank maps are immutable inputs for this render.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [players, playerRecords, teamRecords, leagues, query, role, minGames, sort, playerWeeklyRanks]);

  const entities = tab === "teams" ? filteredTeams : filteredPlayers;
  const entityName = (entity: TeamRating | PlayerRating) => tab === "teams" ? (entity as TeamRating).team : (entity as PlayerRating).player;
  const ratingOf = (entity: TeamRating | PlayerRating) => tab === "teams"
    ? adjustedRating(entity as TeamRating, TEAM_SIGMA_MIN)
    : softMu((entity as PlayerRating).mu_total, (entity as PlayerRating).sigma, PLAYER_SIGMA_MIN);
  const deltaOf = (name: string) => tab === "teams" ? teamDelta(name) : playerDelta(name);

  const featured = entities.find((entity) => entityName(entity) === selected) ?? entities[0];
  const featuredName = featured ? entityName(featured) : "";
  const featuredRecord = featuredName ? (tab === "teams" ? teamRecords[featuredName] : playerRecords[featuredName]) : undefined;
  const featuredGames = featuredName
    ? gameList(profileRecords, tab === "teams" ? profileRecords?.teams[featuredName] : profileRecords?.players[featuredName])
    : [];
  const featuredForm = recentForm(featuredGames, featuredName, tab);
  const featuredTrust = featured
    ? evidenceInfo(
        evidenceFields(featured as unknown as Record<string, unknown>),
        featured.sigma,
        tab === "teams" ? (featuredRecord as TeamRecord | undefined)?.games : (featured as PlayerRating).n_maps,
      )
    : null;
  const visible = expanded ? entities : entities.slice(0, 18);
  const movers = [...entities]
    .filter((entity) => deltaOf(entityName(entity)) != null)
    .sort((a, b) => (deltaOf(entityName(b)) ?? 0) - (deltaOf(entityName(a)) ?? 0))
    .slice(0, 5);
  const selectedTiers = leagues.filter((league) => league.startsWith("TIER"));
  const selectedLeagues = leagues.filter((league) => !league.startsWith("TIER"));
  const scope = [...selectedTiers.map((tier) => TIER_FILTERS.find((item) => item.value === tier)?.label ?? tier), ...selectedLeagues].join(" · ") || "All levels";
  const topScore = featured ? ratingOf(entities[0]) : null;
  const featuredTeam = tab === "teams"
    ? featuredName
    : (featuredRecord as PlayerRecord | undefined)?.current_team ?? (featured as PlayerRating | undefined)?.last_team;

  const setTier = (tier: string | null) => {
    setLeagues((current) => [...current.filter((league) => !league.startsWith("TIER")), ...(tier ? [tier] : [])]);
    setExpanded(false);
  };

  const toggleLeague = (league: string) => {
    setLeagues((current) => current.includes(league) ? current.filter((item) => item !== league) : [...current, league]);
    setExpanded(false);
  };

  return (
    <div className={styles.root}>
      <section className={styles.controls} aria-label="Rating controls">
        <div className={styles.controlMain}>
          <div className={styles.tabs} aria-label="Rating type">
            {(["teams", "players"] as const).map((value) => (
              <button key={value} type="button" className={tab === value ? styles.tabActive : ""} aria-pressed={tab === value} onClick={() => { setTab(value); setExpanded(false); setSelected(""); }}>
                {value === "teams" ? "Teams" : "Players"}
              </button>
            ))}
          </div>
          <label className={styles.search}><span>Search</span><input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder={tab === "teams" ? "Team or alias" : "Player or team"} /></label>
          {tab === "players" ? (
            <label><span>Role</span><select value={role} onChange={(event) => setRole(event.target.value)}><option value="">All roles</option>{PLAYER_ROLES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
          ) : null}
          <label><span>Sort</span><select value={sort} onChange={(event) => setSort(event.target.value as Sort)}><option value="rating">Rating</option><option value="movement">Movement</option><option value="games">Games</option><option value="name">Name</option></select></label>
          {tab === "players" ? <label className={styles.minGames}><span>Min games</span><input type="number" min={5} value={minGames} onChange={(event) => setMinGames(Math.max(5, Number(event.target.value) || 5))} /></label> : null}
        </div>
        <div className={styles.scopeRow} role="group" aria-label="Competition level">
          <span>Level</span>
          <button type="button" className={!selectedTiers.length ? styles.scopeActive : ""} onClick={() => setTier(null)}>All</button>
          {TIER_FILTERS.map((tier) => <button key={tier.value} type="button" className={leagues.includes(tier.value) ? styles.scopeActive : ""} onClick={() => setTier(tier.value)}>{tier.label}</button>)}
          <details><summary>Leagues {selectedLeagues.length ? `(${selectedLeagues.length})` : ""}</summary><div>{availableLeagues.map((league) => <button key={league} type="button" className={leagues.includes(league) ? styles.scopeActive : ""} onClick={() => toggleLeague(league)}>{league}</button>)}</div></details>
        </div>
      </section>

      {featured ? (
        <>
          <section className={styles.summaryLine} aria-label="Rating summary">
            <p><span>Ranked</span><strong>{entities.length} {tab}</strong></p>
            <p><span>Leader</span><strong>{entityName(entities[0])}</strong><small>{topScore?.toFixed(0)}</small></p>
            <p><span>Scope</span><strong>{scope}</strong></p>
          </section>

          <section className={styles.focusBoard}>
            <aside className={styles.rankingRail}>
              <header><span>Leaders</span><b>→</b></header>
              {entities.slice(0, 5).map((entity, index) => {
                const name = entityName(entity);
                const entityTeam = tab === "teams" ? name : playerRecords[name]?.current_team ?? (entity as PlayerRating).last_team;
                return <button type="button" key={name} className={name === featuredName ? styles.railActive : ""} onClick={() => setSelected(name)}><span>{String(index + 1).padStart(2, "0")}</span><span className={styles.railIdentity}><TeamMark team={entityTeam} /><strong>{name}</strong></span><b>{ratingOf(entity).toFixed(0)}</b></button>;
              })}
            </aside>

            <article className={styles.featured}>
              <header>
                <div className={styles.featureIdentity}><TeamMark team={featuredTeam} size="large" /><div><h2>{tab === "players" && playerMetadata[featuredName]?.flag ? `${playerMetadata[featuredName].flag} ` : ""}{featuredName}</h2><p>{tab === "teams" ? formatAffiliation((featuredRecord as TeamRecord | undefined)?.current_tier, (featuredRecord as TeamRecord | undefined)?.current_league ?? (featuredRecord as TeamRecord | undefined)?.primary) : `${(featuredRecord as PlayerRecord | undefined)?.current_team ?? (featured as PlayerRating).last_team ?? "Independent"} · ${roleLabel((featuredRecord as PlayerRecord | undefined)?.primary_role)}`}</p></div></div>
                <dl><div><dt>Rating</dt><dd>{ratingOf(featured).toFixed(0)}</dd></div><div><dt>Movement</dt><dd className={(deltaOf(featuredName) ?? 0) > 0 ? styles.positive : (deltaOf(featuredName) ?? 0) < 0 ? styles.negative : ""}>{rankDeltaLabel(deltaOf(featuredName))}</dd></div></dl>
              </header>
              <div className={styles.featureBody}>
                <div className={styles.featureForm}><span>Recent form</span><FormStrip form={featuredForm} /><ChampionStrip games={featuredGames} entity={featuredName} tab={tab} images={profileRecords?.champion_images ?? {}} /></div>
              </div>
              <footer>
                <span>{featuredTrust ? formatEvidenceCell(featuredTrust) : "Confidence unavailable"}</span>
                <span>{tab === "teams" ? `${(featuredRecord as TeamRecord | undefined)?.games ?? featured.n_maps ?? 0} games · ${formatWr((featuredRecord as TeamRecord | undefined)?.wr)}` : `${(featured as PlayerRating).n_maps} games · ${formatWr((featuredRecord as PlayerRecord | undefined)?.wr)}`}</span>
                <Link href={tab === "teams" ? `/elo/team/${teamSlug(featuredName)}` : `/elo/player/${playerSlug(featuredName)}`}>Open profile →</Link>
              </footer>
            </article>

            <aside className={styles.rankingRail}>
              <header><span>Movers</span><b>Δ</b></header>
              {(movers.length ? movers : entities.slice(0, 5)).map((entity) => {
                const name = entityName(entity);
                const entityTeam = tab === "teams" ? name : playerRecords[name]?.current_team ?? (entity as PlayerRating).last_team;
                return <button type="button" key={name} className={name === featuredName ? styles.railActive : ""} onClick={() => setSelected(name)}><span className={(deltaOf(name) ?? 0) > 0 ? styles.positive : (deltaOf(name) ?? 0) < 0 ? styles.negative : ""}>{rankDeltaLabel(deltaOf(name))}</span><span className={styles.railIdentity}><TeamMark team={entityTeam} /><strong>{name}</strong></span><b>{ratingOf(entity).toFixed(0)}</b></button>;
              })}
            </aside>
          </section>

          <section className={styles.gallerySection}>
            <header><h2>{tab === "teams" ? "Team ratings" : "Player ratings"}</h2><p>{entities.length} shown · {scope}</p></header>
            <div className={styles.gallery}>
              {visible.map((entity, index) => {
                const name = entityName(entity);
                const record = tab === "teams" ? teamRecords[name] : playerRecords[name];
                const games = gameList(profileRecords, tab === "teams" ? profileRecords?.teams[name] : profileRecords?.players[name]);
                const form = recentForm(games, name, tab);
                const entityTeam = tab === "teams" ? name : (record as PlayerRecord | undefined)?.current_team ?? (entity as PlayerRating).last_team;
                return (
                  <Link className={styles.ratingCard} href={tab === "teams" ? `/elo/team/${teamSlug(name)}` : `/elo/player/${playerSlug(name)}`} key={name}>
                    <span className={styles.cardRank}>{String(index + 1).padStart(2, "0")}</span>
                    <div className={styles.cardIdentity}><TeamMark team={entityTeam} size="medium" /><div><h3>{tab === "players" && playerMetadata[name]?.flag ? `${playerMetadata[name].flag} ` : ""}{name}</h3><p>{tab === "teams" ? formatAffiliation((record as TeamRecord | undefined)?.current_tier, (record as TeamRecord | undefined)?.current_league ?? (record as TeamRecord | undefined)?.primary) : `${(record as PlayerRecord | undefined)?.current_team ?? (entity as PlayerRating).last_team ?? "Independent"} · ${roleLabel((record as PlayerRecord | undefined)?.primary_role)}`}</p></div></div>
                    <strong>{ratingOf(entity).toFixed(0)}</strong>
                    <span className={(deltaOf(name) ?? 0) > 0 ? styles.positive : (deltaOf(name) ?? 0) < 0 ? styles.negative : ""}>{rankDeltaLabel(deltaOf(name))}</span>
                    <div className={styles.cardVisuals}><ChampionStrip games={games} entity={name} tab={tab} images={profileRecords?.champion_images ?? {}} limit={3} /><FormStrip form={form.slice(0, 5)} /></div>
                  </Link>
                );
              })}
            </div>
            {entities.length > 18 ? <button type="button" className={styles.showAll} onClick={() => setExpanded((current) => !current)}>{expanded ? "Show first 18" : `Show all ${entities.length}`}</button> : null}
          </section>
        </>
      ) : <p className={styles.empty}>No {tab} match these filters.</p>}
    </div>
  );
}
