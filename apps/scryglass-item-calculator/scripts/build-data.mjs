import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const appRoot = path.resolve(here, "..");
const repoRoot = path.resolve(appRoot, "../..");
const cdragonItemsPath = path.join(repoRoot, "data/lol/knowledge/patch-packets/cdragon/2026/26.15/raw/items.json");
const mechanicsIndexPath = path.join(repoRoot, "data/lol/knowledge/patch-packets/cdragon/2026/26.15/mechanics-index.json");
const axwordChampionsPath = path.resolve(repoRoot, "../Projects/lol-strength-analysis/public/data/lolwiki/champions-full.json");
const ddragonBase = "https://ddragon.leagueoflegends.com/cdn/16.15.1/data/en_US";

const [championPayload, championFullPayload, itemPayload, cdragonItems, mechanicsIndex, axwordChampions] = await Promise.all([
  fetch(`${ddragonBase}/champion.json`).then((response) => response.json()),
  fetch(`${ddragonBase}/championFull.json`).then((response) => response.json()),
  fetch(`${ddragonBase}/item.json`).then((response) => response.json()),
  readFile(cdragonItemsPath, "utf8").then(JSON.parse),
  readFile(mechanicsIndexPath, "utf8").then(JSON.parse),
  readFile(axwordChampionsPath, "utf8").then(JSON.parse).catch(() => ({})),
]);

const close = (a, b) => Math.abs(a - b) <= 1e-5 * Math.max(1, Math.abs(a), Math.abs(b));
const compactNumber = (value) => Math.abs(value) < 1e-9 ? 0 : Number(value.toFixed(6));
const valueAtRank = (spell, name, rank) => {
  const row = spell.data_values?.find((entry) => entry.name === name);
  const value = row?.values?.[rank];
  if (!Number.isFinite(value)) throw new Error(`missing ${name}[${rank}]`);
  return value;
};

function statValue(part, stats) {
  const code = Number(part.mStat ?? 6);
  const formula = part.mStatFormula;
  if (code === 6 && formula == null) return stats.ap;
  if (code === 2 && formula == null) return stats.ad;
  if (code === 2 && formula === 2) return stats.bonusAd;
  throw new Error(`unsupported stat ${code}/${formula ?? "base"}`);
}

function breakpointValue(part, level) {
  let value = Number(part.mLevel1Value || 0);
  let perLevel = Number(part.mInitialBonusPerLevel || 0);
  const breakpoints = [...(part.mBreakpoints || [])].sort((a, b) => a.mLevel - b.mLevel);
  for (let current = 2; current <= level; current += 1) {
    const atLevel = breakpoints.filter((entry) => Number(entry.mLevel) === current);
    const rateChange = atLevel.find((entry) => Number.isFinite(entry.mBonusPerLevelAtAndAfter));
    if (rateChange) perLevel = Number(rateChange.mBonusPerLevelAtAndAfter);
    value += perLevel;
    for (const entry of atLevel) value += Number(entry.mAdditionalBonusAtThisLevel || 0);
  }
  return value;
}

function evaluatePart(part, context) {
  const type = part?.__type;
  if (type === "NumberCalculationPart") return Number(part.mNumber || 0);
  if (type === "NamedDataValueCalculationPart") return valueAtRank(context.spell, part.mDataValue, context.rank);
  if (type === "StatByCoefficientCalculationPart") return statValue(part, context.stats) * Number(part.mCoefficient || 0);
  if (type === "StatByNamedDataValueCalculationPart") return statValue(part, context.stats) * valueAtRank(context.spell, part.mDataValue, context.rank);
  if (type === "StatBySubPartCalculationPart") return statValue(part, context.stats) * evaluatePart(part.mSubpart, context);
  if (type === "ByCharLevelInterpolationCalculationPart") return Number(part.mStartValue || 0) + (Number(part.mEndValue || 0) - Number(part.mStartValue || 0)) * ((context.level - 1) / 17);
  if (type === "ByCharLevelBreakpointsCalculationPart") return breakpointValue(part, context.level);
  if (type === "SumOfSubPartsCalculationPart") return (part.mSubparts || []).reduce((sum, entry) => sum + evaluatePart(entry, context), 0);
  if (type === "ProductOfSubPartsCalculationPart") {
    const entries = part.mSubparts || [part.mPart1, part.mPart2];
    return entries.reduce((product, entry) => product * evaluatePart(entry, context), 1);
  }
  throw new Error(`unsupported part ${type || "missing"}`);
}

function evaluateCalculation(spell, calculation, rank, level, stats, seen = new Set()) {
  if (!calculation || seen.size > 8) throw new Error("unresolved calculation");
  const context = { spell, rank, level, stats };
  if (calculation.__type === "GameCalculationModified") {
    const key = calculation.mModifiedGameCalculation;
    if (seen.has(key)) throw new Error("calculation cycle");
    const base = calculationByName(spell, key);
    const value = evaluateCalculation(spell, base, rank, level, stats, new Set([...seen, key]));
    return value * (calculation.mMultiplier ? evaluatePart(calculation.mMultiplier, context) : 1);
  }
  if (calculation.__type !== "GameCalculation") throw new Error(`unsupported calculation ${calculation.__type}`);
  let value = (calculation.mFormulaParts || []).reduce((sum, entry) => sum + evaluatePart(entry, context), 0);
  if (calculation.mMultiplier) value *= evaluatePart(calculation.mMultiplier, context);
  return value;
}

function calculationByName(spell, rawName) {
  const wanted = String(rawName).toLowerCase();
  const match = Object.entries(spell.spell_calculations || {}).find(([name]) => name.toLowerCase() === wanted);
  if (match) return match[1];
  const data = spell.data_values?.find((entry) => entry.name.toLowerCase() === wanted);
  return data ? { __type: "GameCalculation", mFormulaParts: [{ __type: "NamedDataValueCalculationPart", mDataValue: data.name }] } : null;
}

function formulaTable(spell, calculation, maxRank) {
  const zero = { ap: 0, ad: 0, bonusAd: 0 };
  const fields = { ap: [], ad: [], bonusAd: [] };
  const base = [];
  let levelDependent = false;
  for (let rank = 1; rank <= maxRank; rank += 1) {
    const baseLevels = [];
    const coefficientLevels = { ap: [], ad: [], bonusAd: [] };
    for (let level = 1; level <= 18; level += 1) {
      const origin = evaluateCalculation(spell, calculation, rank, level, zero);
      baseLevels.push(compactNumber(origin));
      for (const field of Object.keys(fields)) {
        const oneStats = { ...zero, [field]: 1 };
        const twoStats = { ...zero, [field]: 2 };
        const oneValue = evaluateCalculation(spell, calculation, rank, level, oneStats) - origin;
        const twoValue = evaluateCalculation(spell, calculation, rank, level, twoStats) - origin;
        if (!close(twoValue, oneValue * 2)) throw new Error("non-linear stat formula");
        coefficientLevels[field].push(compactNumber(oneValue));
      }
      const combined = evaluateCalculation(spell, calculation, rank, level, { ap: 1, ad: 1, bonusAd: 1 });
      const expected = origin + Object.values(coefficientLevels).reduce((sum, values) => sum + values.at(-1), 0);
      if (!close(combined, expected)) throw new Error("cross-stat formula");
    }
    const baseFlat = baseLevels.every((value) => close(value, baseLevels[0]));
    levelDependent ||= !baseFlat;
    base.push(baseFlat ? baseLevels[0] : baseLevels);
    for (const field of Object.keys(fields)) {
      const values = coefficientLevels[field];
      const flat = values.every((value) => close(value, values[0]));
      levelDependent ||= !flat;
      fields[field].push(flat ? values[0] : values);
    }
  }
  const result = { base };
  for (const [field, values] of Object.entries(fields)) {
    if (values.some((value) => Array.isArray(value) ? value.some(Boolean) : value !== 0)) result[field] = values;
  }
  if (levelDependent) result.levelDependent = true;
  return result;
}

function tooltipDamageReferences(tooltip) {
  const references = [];
  const tagPattern = /<(magicDamage|physicalDamage|trueDamage)>([\s\S]*?)<\/\1>/gi;
  for (const tag of tooltip.matchAll(tagPattern)) {
    const clauseStart = Math.max(tooltip.lastIndexOf("<br", tag.index), tooltip.lastIndexOf(".", tag.index), 0);
    const clause = tooltip.slice(clauseStart, tag.index);
    if (!/(deal|damage|take|hit|burn|explode|detonat|strike|inflict)/i.test(clause)) continue;
    const type = tag[1].toLowerCase().startsWith("magic") ? "magical" : tag[1].toLowerCase().startsWith("physical") ? "physical" : "true";
    const placeholders = [...tag[2].matchAll(/{{\s*([^}]+?)\s*}}/g)];
    if (!placeholders.length || placeholders.some((placeholder) => !/^[a-zA-Z0-9_{}]+$/.test(placeholder[1]))) {
      references.push({ unsupported: true });
      continue;
    }
    const healthText = tag[2].replace(/<[^>]+>/g, " ");
    const targetKind = /missing Health/i.test(healthText) ? "targetMissingHp" : /current Health/i.test(healthText) ? "targetCurrentHp" : /max(?:imum)? Health/i.test(healthText) ? "targetMaxHp" : null;
    for (const placeholder of placeholders) {
      const suffix = tag[2].slice((placeholder.index || 0) + placeholder[0].length);
      const percentDisplayed = /^\s*%/.test(suffix);
      references.push({ key: placeholder[1], type, ...(targetKind ? { targetScale: targetKind, targetScaleDivisor: percentDisplayed ? 100 : 1 } : {}) });
    }
  }
  return references;
}

const mechanicsById = new Map(mechanicsIndex.champions.map((entry) => [Number(entry.id), entry]));

function genericAbilityKit(champion, fullChampion) {
  const packetChampion = mechanicsById.get(Number(champion.key));
  if (!packetChampion) return { abilities: [], abilityCoverage: { supported: 0, total: 4, withheld: ["Q", "W", "E", "R"] } };
  const abilities = [];
  const withheld = [];
  const spellNames = packetChampion.mechanics?.spell_names || [];
  for (let index = 0; index < 4; index += 1) {
    const slot = "QWER"[index];
    const fullSpell = fullChampion?.spells?.[index];
    const expectedPath = spellNames[index];
    const spell = packetChampion.mechanics?.spells?.find((entry) => entry.path?.endsWith(`/Spells/${expectedPath}`) || entry.path?.endsWith(`/${expectedPath}`));
    if (!spell || !fullSpell) { withheld.push(slot); continue; }
    const tooltipRefs = tooltipDamageReferences(fullSpell.tooltip || "");
    if (tooltipRefs.some((entry) => entry.unsupported)) { withheld.push(slot); continue; }
    const refs = tooltipRefs.filter((entry) => calculationByName(spell, entry.key));
    if (refs.length !== tooltipRefs.length) { withheld.push(slot); continue; }
    const uniqueKeys = [...new Set(refs.map((entry) => entry.key.toLowerCase()))];
    if (uniqueKeys.length !== 1 || refs.length === 0) { withheld.push(slot); continue; }
    try {
      const calculation = calculationByName(spell, uniqueKeys[0]);
      const formula = formulaTable(spell, calculation, Number(fullSpell.maxrank || (slot === "R" ? 3 : 5)));
      abilities.push({
        slot,
        name: fullSpell.name,
        icon: fullSpell.image.full,
        maxRank: Number(fullSpell.maxrank || (slot === "R" ? 3 : 5)),
        variants: [{ name: "Listed hit", packets: refs.map(({ type, targetScale, targetScaleDivisor }) => ({ type, ...(targetScale ? { targetScale, targetScaleDivisor } : {}) })), ...formula }],
      });
    } catch {
      withheld.push(slot);
    }
  }
  return {
    abilities,
    abilityCoverage: { supported: abilities.length, total: 4, withheld },
    source: { label: "Patch 26.15 client formula graph", clientPatch: "16.15", indexSha256: mechanicsIndex.source?.sha256 || null },
  };
}

const repeat = (value, count) => Array.from({ length: count }, () => value);
const ziggsFull = championFullPayload.data.Ziggs;
const ziggsAbilityKit = {
  source: {
    label: "League Wiki snapshot · patch 26.15",
    pages: [
      { title: "Short Fuse", revision: 4005222 },
      { title: "Bouncing Bomb", revision: 4008669 },
      { title: "Satchel Charge", revision: 4008672 },
      { title: "Hexplosive Minefield", revision: 4008673 },
      { title: "Mega Inferno Bomb", revision: 4008677 },
    ],
  },
  abilityCoverage: { supported: 5, total: 5, withheld: [] },
  abilities: [
    {
      slot: "P", name: "Short Fuse", icon: ziggsFull.passive.image.full, maxRank: 1,
      variants: [{ name: "Champion hit", type: "magical", levelBase: [20,24,28,32,36,40,48,56,64,72,80,88,100,112,124,136,148,160], ap: repeat(.5, 18) }],
    },
    {
      slot: "Q", name: "Bouncing Bomb", icon: ziggsFull.spells[0].image.full, maxRank: 5,
      variants: [{ name: "Explosion", type: "magical", base: [80,130,180,230,280], ap: [.6,.65,.7,.75,.8] }],
    },
    {
      slot: "W", name: "Satchel Charge", icon: ziggsFull.spells[1].image.full, maxRank: 5,
      variants: [{ name: "Explosion", type: "magical", base: [70,105,140,175,210], ap: repeat(.5, 5) }],
    },
    {
      slot: "E", name: "Hexplosive Minefield", icon: ziggsFull.spells[2].image.full, maxRank: 5, maxHits: 11, subsequentMultiplier: .4,
      variants: [{ name: "Mine contact", type: "magical", base: [30,70,110,150,190], ap: [.25,.3,.35,.4,.45] }],
    },
    {
      slot: "R", name: "Mega Inferno Bomb", icon: ziggsFull.spells[3].image.full, maxRank: 3,
      variants: [
        { name: "Epicenter", type: "magical", base: [300,500,700], ap: repeat(1, 3) },
        { name: "Outer blast", type: "magical", base: [195,325,455], ap: repeat(.65, 3) },
      ],
    },
  ],
};

const cdragonById = new Map(cdragonItems.map((item) => [Number(item.id), item]));
const numberFromDescription = (description, label) => {
  const pattern = new RegExp(`<attention>\\s*([0-9.]+)%?\\s*</attention>\\s*${label}(?:<|$)`, "i");
  return Number(description.match(pattern)?.[1] || 0);
};
const penetrationFromDescription = (description, label) => {
  const pattern = new RegExp(`<attention>\\s*([0-9.]+)(%)?\\s*</attention>\\s*${label}(?:<|$)`, "i");
  const match = description.match(pattern);
  return { flat: match && !match[2] ? Number(match[1]) : 0, percent: match?.[2] ? Number(match[1]) : 0 };
};

const champions = Object.values(championPayload.data).filter((champion) => Number(champion.key) < 10000).map((champion) => {
  const genericKit = genericAbilityKit(champion, championFullPayload.data[champion.id]);
  return {
    name: champion.name,
    key: champion.id,
    id: Number(champion.key),
    title: champion.title,
    tags: champion.tags,
    resource: champion.partype,
    hp: champion.stats.hp,
    hpPerLevel: champion.stats.hpperlevel,
    mana: champion.stats.mp,
    manaPerLevel: champion.stats.mpperlevel,
    ad: champion.stats.attackdamage,
    adPerLevel: champion.stats.attackdamageperlevel,
    armor: champion.stats.armor,
    armorPerLevel: champion.stats.armorperlevel,
    mr: champion.stats.spellblock,
    mrPerLevel: champion.stats.spellblockperlevel,
    attackSpeed: champion.stats.attackspeed,
    attackSpeedRatio: Number(axwordChampions[champion.id]?.stats?.attackSpeedRatio?.flat || champion.stats.attackspeed),
    attackSpeedPerLevel: champion.stats.attackspeedperlevel,
    moveSpeed: champion.stats.movespeed,
    range: champion.stats.attackrange,
    ...genericKit,
    ...(champion.id === "Ziggs" ? ziggsAbilityKit : {}),
  };
}).sort((a, b) => a.name.localeCompare(b.name));

const items = Object.entries(itemPayload.data).flatMap(([rawId, item]) => {
  const id = Number(rawId);
  const cdragon = cdragonById.get(id);
  if (id >= 100000 || !cdragon?.inStore || !item.maps?.[11] || cdragon.requiredChampion || cdragon.specialRecipe) return [];
  const description = item.description || cdragon.description || "";
  const statBlock = description.match(/<stats>([\s\S]*?)<\/stats>/i)?.[1] || description;
  const passiveText = description.split("</stats>").at(-1).replace(/<br\s*\/?\s*>/gi, " ").replace(/<[^>]+>/g, " ").replace(/&nbsp;/g, " ").replace(/\s+/g, " ").trim();
  const magicPen = penetrationFromDescription(statBlock, "Magic Penetration");
  const armorPen = penetrationFromDescription(statBlock, "Armor Penetration");
  const moveSpeed = penetrationFromDescription(statBlock, "Move Speed");
  return [{
    id,
    name: item.name,
    ap: Number(item.stats.FlatMagicDamageMod || numberFromDescription(statBlock, "Ability Power")),
    hp: Number(item.stats.FlatHPPoolMod || numberFromDescription(statBlock, "Health")),
    mana: Number(item.stats.FlatMPPoolMod || numberFromDescription(statBlock, "Mana")),
    ad: Number(item.stats.FlatPhysicalDamageMod || numberFromDescription(statBlock, "Attack Damage")),
    armor: Number(item.stats.FlatArmorMod || numberFromDescription(statBlock, "Armor")),
    mr: Number(item.stats.FlatSpellBlockMod || numberFromDescription(statBlock, "Magic Resist")),
    haste: numberFromDescription(statBlock, "Ability Haste"),
    pen: magicPen.flat,
    percentPen: magicPen.percent,
    lethality: numberFromDescription(statBlock, "Lethality"),
    percentArmorPen: armorPen.percent,
    attackSpeed: Number((item.stats.PercentAttackSpeedMod || 0) * 100 || numberFromDescription(statBlock, "Attack Speed")),
    moveSpeed: Number(item.stats.FlatMovementSpeedMod || moveSpeed.flat),
    moveSpeedPercent: moveSpeed.percent,
    crit: Number((item.stats.FlatCritChanceMod || 0) * 100 || numberFromDescription(statBlock, "Critical Strike Chance")),
    price: item.gold?.total || cdragon.priceTotal || 0,
    into: cdragon.to || [],
    categories: cdragon.categories || item.tags || [],
    passiveText,
  }];
}).sort((a, b) => a.name.localeCompare(b.name));

const abilityCoverage = champions.reduce((total, champion) => total + Number(champion.abilityCoverage?.supported || 0), 0);
await writeFile(path.join(appRoot, "data.json"), `${JSON.stringify({ patch: "26.15", champions, items, coverage: { championAbilities: abilityCoverage, possibleChampionAbilities: champions.length * 4 + 1 } })}\n`);
console.log(`Wrote ${champions.length} champions, ${items.length} items, and ${abilityCoverage} supported champion ability packets.`);
