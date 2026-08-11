const TEAM_MARKS: Record<string, string> = {
  "bilibili gaming": "/team-marks/bilibili-gaming.png",
  cloud9: "/team-marks/cloud9.png",
  dignitas: "/team-marks/dignitas.png",
  disguised: "/team-marks/disguised.png",
  "dplus kia": "/team-marks/dplus-kia.png",
  "gen.g": "/team-marks/gen-g.png",
  giantx: "/team-marks/giantx.png",
  "ground zero": "/team-marks/ground-zero-gaming.png",
  "ground zero gaming": "/team-marks/ground-zero-gaming.png",
  "hanwha life esports": "/team-marks/hanwha-life-esports.png",
  los: "/team-marks/los.png",
  "løs": "/team-marks/los.png",
  t1: "/team-marks/t1.png",
  "team we": "/team-marks/team-we.png",
};

export function teamMarkUrl(team: string | null | undefined): string | null {
  if (!team) return null;
  return TEAM_MARKS[team.trim().toLowerCase()] ?? null;
}
