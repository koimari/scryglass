const DDRAGON = "https://ddragon.leagueoflegends.com/cdn/16.15.1/img";

let DATA;
let pickerContext = null;
let bisContext = null;

const state = {
  attacker: {
    champion: null,
    level: 1,
    role: null,
    roleQuestComplete: false,
    buildA: [0, 0, 0, 0, 0, 0],
    buildAStacks: [0, 0, 0, 0, 0, 0],
    buildB: [0, 0, 0, 0, 0, 0],
    buildBStacks: [0, 0, 0, 0, 0, 0],
    questBootA: 0,
    questBootB: 0,
    comparisonEnabled: false,
    baseDamage: 0,
    apRatio: 0,
    physicalDamage: 0,
    adRatio: 0,
    abilityInputs: {},
  },
  targets: [],
  fight: { rotations: 1, duration: 10, aaUptime: 0 },
  optimizer: { running: false, summary: null },
};

const TIER_TWO_BOOTS = [3006, 3009, 3008, 3158, 3111, 3047, 3020];
const TIER_THREE_BOOTS = [3172, 3170, 3168, 3171, 3173, 3174, 3175];
const ALL_ROLE_BOOTS = new Set([...TIER_TWO_BOOTS, ...TIER_THREE_BOOTS]);

const $ = (id) => document.getElementById(id);
const fmt = (value) => Math.round(value).toLocaleString("en-US");
const one = (value) => Number(value).toFixed(1).replace(/\.0$/, "");
const percent = (value) => `${value.toFixed(value < 10 ? 1 : 0)}%`;
const plural = (count, singular, pluralForm = `${singular}s`) => count === 1 ? singular : pluralForm;
const escapeHtml = (value) => String(value).replace(/[&<>"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[char]);

function invalidateOptimization() {
  if (!state.optimizer.running) state.optimizer.summary = null;
}

function getChampion(name) {
  return name ? DATA.champions.find((entry) => entry.name === name) || null : null;
}

function getItem(id) {
  return Number(id) ? DATA.items.find((entry) => entry.id === Number(id)) || null : null;
}

function activeAbilityKit() {
  return getChampion(state.attacker.champion)?.abilities || [];
}

function resetAbilityInputs() {
  state.attacker.abilityInputs = Object.fromEntries(activeAbilityKit().map((ability) => [ability.slot, {
    rank: 1,
    casts: 1,
    hits: 1,
    variant: 0,
  }]));
}

function abilityInput(slot) {
  return state.attacker.abilityInputs[slot] || { rank: 0, casts: 0, hits: 1, variant: 0 };
}

function championImage(name) {
  const champion = getChampion(name);
  return champion ? `${DDRAGON}/champion/${champion.key}.png` : "";
}

function itemImage(id) {
  return `${DDRAGON}/item/${Number(id)}.png`;
}

function abilityImage(ability) {
  return `${DDRAGON}/${ability.slot === "P" ? "passive" : "spell"}/${ability.icon}`;
}

function itemName(id, fallback = "Empty slot") {
  return getItem(id)?.name || fallback;
}

function pathValue(path) {
  return path.split(".").reduce((value, part) => value?.[Number.isNaN(Number(part)) ? part : Number(part)], state);
}

function setPath(path, nextValue) {
  const parts = path.split(".");
  const last = parts.pop();
  const parent = parts.reduce((value, part) => value[Number.isNaN(Number(part)) ? part : Number(part)], state);
  parent[Number.isNaN(Number(last)) ? last : Number(last)] = nextValue;
  invalidateOptimization();
}

function stackSpec(id) {
  if (Number(id) === 1082) return { max: 10, ap: 4, moveSpeedPercentAt: null };
  if (Number(id) === 3041) return { max: 25, ap: 5, moveSpeedPercentAt: 10 };
  return null;
}

function stackValue(path) {
  const parts = path.split(".");
  if (parts[0] === "attacker" && parts[1] === "buildA") return state.attacker.buildAStacks[Number(parts[2])] || 0;
  if (parts[0] === "attacker" && parts[1] === "buildB") return state.attacker.buildBStacks[Number(parts[2])] || 0;
  if (parts[0] === "targets") return state.targets[Number(parts[1])]?.itemStacks?.[Number(parts[3])] || 0;
  return 0;
}

function setStackValue(path, value) {
  const parts = path.split(".");
  if (parts[0] === "attacker" && parts[1] === "buildA") state.attacker.buildAStacks[Number(parts[2])] = value;
  else if (parts[0] === "attacker" && parts[1] === "buildB") state.attacker.buildBStacks[Number(parts[2])] = value;
  else if (parts[0] === "targets") state.targets[Number(parts[1])].itemStacks[Number(parts[3])] = value;
  invalidateOptimization();
}

function buildStats(itemIds, stackCounts = []) {
  const entries = itemIds.map((id, index) => ({ item: getItem(id), stacks: Number(stackCounts[index] || 0) })).filter((entry) => entry.item);
  const total = entries.reduce((sum, entry) => {
    const { item, stacks } = entry;
    for (const key of ["ap", "hp", "mana", "ad", "armor", "mr", "haste", "pen", "percentPen", "lethality", "percentArmorPen", "attackSpeed", "moveSpeed", "moveSpeedPercent", "crit"]) {
      sum[key] += Number(item[key] || 0);
    }
    const spec = stackSpec(item.id);
    if (spec) {
      sum.ap += Math.min(stacks, spec.max) * spec.ap;
      if (spec.moveSpeedPercentAt && stacks >= spec.moveSpeedPercentAt) sum.moveSpeedPercent += 10;
    }
    return sum;
  }, { ap: 0, hp: 0, mana: 0, ad: 0, armor: 0, mr: 0, haste: 0, pen: 0, percentPen: 0, lethality: 0, percentArmorPen: 0, attackSpeed: 0, moveSpeed: 0, moveSpeedPercent: 0, crit: 0 });
  total.apBeforeMultiplier = total.ap;
  total.apMultiplier = entries.some(({ item }) => item.id === 3089) ? 1.3 : 1;
  total.ap *= total.apMultiplier;
  return total;
}

function championStats(name, level, itemIds = [], stackCounts = []) {
  const champion = getChampion(name);
  const boundedLevel = Math.max(1, Math.min(18, Number(level) || 1));
  const scale = (boundedLevel - 1) * (0.7025 + 0.0175 * (boundedLevel - 1));
  const build = buildStats(itemIds, stackCounts);
  const baseHp = (champion?.hp || 0) + (champion?.hpPerLevel || 0) * scale;
  return {
    baseHp,
    bonusHp: build.hp,
    hp: baseHp + build.hp,
    mana: (champion?.mana || 0) + (champion?.manaPerLevel || 0) * scale + build.mana,
    ad: (champion?.ad || 0) + (champion?.adPerLevel || 0) * scale + build.ad,
    ap: build.ap,
    armor: (champion?.armor || 0) + (champion?.armorPerLevel || 0) * scale + build.armor,
    mr: (champion?.mr || 0) + (champion?.mrPerLevel || 0) * scale + build.mr,
    haste: build.haste,
    pen: build.pen,
    percentPen: build.percentPen,
    lethality: build.lethality,
    percentArmorPen: build.percentArmorPen,
    attackSpeed: (champion?.attackSpeed || 0) + (champion?.attackSpeedRatio || champion?.attackSpeed || 0) * ((((champion?.attackSpeedPerLevel || 0) * scale) + build.attackSpeed) / 100),
    crit: Math.min(100, build.crit),
    moveSpeed: ((champion?.moveSpeed || 0) + build.moveSpeed) * (1 + build.moveSpeedPercent / 100),
    range: champion?.range || 0,
  };
}

function attackerChampionStats(itemIds = [], stackCounts = []) {
  const stats = championStats(state.attacker.champion, state.attacker.level, itemIds, stackCounts);
  if (state.attacker.role === "mid" && state.attacker.roleQuestComplete) {
    const build = buildStats(itemIds, stackCounts);
    const baseAd = stats.ad - build.ad;
    stats.ad = baseAd + build.ad * 1.08;
    stats.ap *= 1.08;
  }
  return stats;
}

function usesQuestBootSlot() {
  return state.attacker.roleQuestComplete && ["mid", "bottom"].includes(state.attacker.role);
}

function ordinarySlotCount() {
  return state.attacker.role === "mid" && state.attacker.roleQuestComplete ? 5 : 6;
}

function questBootIds() {
  return state.attacker.role === "mid" ? TIER_THREE_BOOTS : TIER_TWO_BOOTS;
}

function magicPenLabel(stats) {
  if (stats.percentPen && stats.pen) return `${one(stats.percentPen)}% + ${one(stats.pen)}`;
  if (stats.percentPen) return `${one(stats.percentPen)}%`;
  return one(stats.pen);
}

function armorPenLabel(stats) {
  if (stats.percentArmorPen && stats.lethality) return `${one(stats.percentArmorPen)}% + ${one(stats.lethality)}`;
  if (stats.percentArmorPen) return `${one(stats.percentArmorPen)}%`;
  return one(stats.lethality);
}

function statMatrix(stats, compareStats = null, compact = false) {
  const rows = [
    ["Base health", fmt(stats.baseHp), compareStats && fmt(compareStats.baseHp)],
    ["Bonus health", fmt(stats.bonusHp), compareStats && fmt(compareStats.bonusHp)],
    ["Total health", fmt(stats.hp), compareStats && fmt(compareStats.hp)],
    ["Resource", fmt(stats.mana), compareStats && fmt(compareStats.mana)],
    ["Attack damage", one(stats.ad), compareStats && one(compareStats.ad)],
    ["Ability power", one(stats.ap), compareStats && one(compareStats.ap)],
    ["Armor", one(stats.armor), compareStats && one(compareStats.armor)],
    ["Magic resist", one(stats.mr), compareStats && one(compareStats.mr)],
    ["Attack speed", one(stats.attackSpeed), compareStats && one(compareStats.attackSpeed)],
    ["Move speed", one(stats.moveSpeed), compareStats && one(compareStats.moveSpeed)],
    ["Ability haste", one(stats.haste), compareStats && one(compareStats.haste)],
    ["Critical chance", `${one(stats.crit)}%`, compareStats && `${one(compareStats.crit)}%`],
    ["Armor pen", armorPenLabel(stats), compareStats && armorPenLabel(compareStats)],
    ["Magic pen", magicPenLabel(stats), compareStats && magicPenLabel(compareStats)],
  ];
  return `<div class="stat-matrix ${compact ? "compact" : ""}">${rows.map(([label, a, b]) => `<div class="stat-cell"><span>${label}</span><strong>${a}</strong>${b !== null && b !== false ? `<b>${b}</b>` : ""}</div>`).join("")}</div>`;
}

function itemStatsLine(item) {
  if (!item) return "Remove item";
  const stats = [];
  if (item.ap) stats.push(`${item.ap} AP`);
  if (item.id === 3089) stats.push("+30% total AP");
  if (item.ad) stats.push(`${item.ad} AD`);
  if (item.hp) stats.push(`${item.hp} HP`);
  if (item.armor) stats.push(`${item.armor} armor`);
  if (item.mr) stats.push(`${item.mr} MR`);
  if (item.haste) stats.push(`${item.haste} haste`);
  if (item.pen) stats.push(`${item.pen} pen`);
  if (item.percentPen) stats.push(`${item.percentPen}% pen`);
  if (item.attackSpeed) stats.push(`${item.attackSpeed}% AS`);
  if (item.crit) stats.push(`${item.crit}% crit`);
  if (item.lethality) stats.push(`${item.lethality} lethality`);
  if (item.percentArmorPen) stats.push(`${item.percentArmorPen}% armor pen`);
  return stats.join(" · ") || "Item effect";
}

function itemSlot(path, id, compact = false, allowBis = false) {
  const item = getItem(id);
  const bisReady = Boolean(state.attacker.champion && state.targets.length && state.targets.every((target) => target.champion));
  return `<div class="slot-wrap ${compact ? "compact" : ""}">
    <button class="item-slot" type="button" data-picker="item" data-path="${path}" aria-label="${item ? `Change ${escapeHtml(item.name)}` : "Add item"}">
      <span class="item-icon ${item ? "" : "empty"}">${item ? `<img src="${itemImage(id)}" alt="" />` : `<span aria-hidden="true">+</span>`}</span>
      ${compact ? "" : `<small>${escapeHtml(item?.name || "Add item")}</small>`}
    </button>
    ${stackSpec(id) ? stackControl(path, id, compact) : ""}
    ${allowBis ? `<button class="bis-trigger" type="button" data-bis-path="${path}" title="Rank this slot" ${bisReady ? "" : "disabled"}>BIS</button>` : ""}
  </div>`;
}

function stackControl(path, id, compact = false) {
  const spec = stackSpec(id);
  const value = Math.min(stackValue(path), spec.max);
  return `<div class="stack-control ${compact ? "compact" : ""}" aria-label="${escapeHtml(itemName(id))} stacks">
    <button type="button" data-stack-path="${path}" data-delta="-1" aria-label="Decrease stacks">−</button>
    <output>${value}/${spec.max}</output>
    <button type="button" data-stack-path="${path}" data-delta="1" aria-label="Increase stacks">+</button>
  </div>`;
}

function targetCard(target, index) {
  const stats = championStats(target.champion, target.level, target.items, target.itemStacks);
  const champion = getChampion(target.champion);
  return `<article class="target-card">
    <header>
      <button class="target-pick ${champion ? "" : "empty-pick"}" type="button" data-picker="champion" data-path="targets.${index}.champion" aria-label="${champion ? `Change ${escapeHtml(target.champion)}` : "Choose enemy champion"}">${champion ? `<img src="${championImage(target.champion)}" alt="" />` : "+"}</button>
      <div class="target-title"><button type="button" data-picker="champion" data-path="targets.${index}.champion">${escapeHtml(target.champion || "Choose champion")}</button><span>${escapeHtml(champion?.title || "Enemy slot")}</span></div>
      <div class="target-level"><button type="button" data-level="targets.${index}.level" data-delta="-1" aria-label="Decrease level">−</button><output>Lv ${target.level}</output><button type="button" data-level="targets.${index}.level" data-delta="1" aria-label="Increase level">+</button></div>
      <button class="remove-target" type="button" data-remove-target="${index}" aria-label="Remove ${escapeHtml(target.champion || "enemy slot")}">×</button>
    </header>
    <div class="target-build">${target.items.map((id, slot) => itemSlot(`targets.${index}.items.${slot}`, id, true)).join("")}</div>
    ${champion ? statMatrix(stats, null, true) : `<div class="matrix-placeholder">Choose a champion to show the full stat matrix.</div>`}
  </article>`;
}

function segmented(label, values, active, attribute) {
  return `<div class="control-group"><span>${label}</span><div class="segmented">${values.map(([value, text]) => `<button type="button" data-fight="${attribute}" data-value="${value}" class="${String(active) === String(value) ? "active" : ""}">${text}</button>`).join("")}</div></div>`;
}

function stepper(label, value, attributes, minusDisabled = false, plusDisabled = false) {
  return `<div class="ability-step"><span>${label}</span><div><button type="button" ${attributes} data-delta="-1" ${minusDisabled ? "disabled" : ""}>−</button><output>${value}</output><button type="button" ${attributes} data-delta="1" ${plusDisabled ? "disabled" : ""}>+</button></div></div>`;
}

function renderAbilityPackage(champion) {
  const coverage = champion?.abilityCoverage;
  if (!champion?.abilities?.length) return `<div class="ability-empty"><strong>Exact skill formulas withheld.</strong><span>${coverage?.withheld?.length ? `${coverage.withheld.join("/")} could not be resolved unambiguously · ` : ""}Use the manual package below.</span></div>`;
  const rows = champion.abilities.map((ability) => {
    const input = abilityInput(ability.slot);
    const rankLabel = ability.slot === "P" ? "Level scales" : "Rank";
    const rankControl = ability.slot === "P"
      ? `<div class="ability-step fixed"><span>${rankLabel}</span><strong>Lv ${state.attacker.level}</strong></div>`
      : stepper(rankLabel, input.rank, `data-ability-rank="${ability.slot}"`, input.rank <= 0, input.rank >= ability.maxRank);
    const hitControl = ability.maxHits
      ? stepper("Mines hit", input.hits, `data-ability-hits="${ability.slot}"`, input.hits <= 1, input.hits >= ability.maxHits)
      : "";
    const variants = ability.variants.length > 1 ? `<div class="ability-variants">${ability.variants.map((variant, index) => `<button type="button" data-ability-variant="${ability.slot}" data-value="${index}" class="${input.variant === index ? "active" : ""}">${escapeHtml(variant.name)}</button>`).join("")}</div>` : `<small>${escapeHtml(ability.variants[0].name)}</small>`;
    return `<article class="ability-row">
      <div class="ability-name"><img src="${abilityImage(ability)}" alt="" /><b>${ability.slot}</b><span><strong>${escapeHtml(ability.name)}</strong>${variants}</span></div>
      ${rankControl}
      ${stepper(ability.slot === "P" ? "Procs" : "Casts", input.casts, `data-ability-casts="${ability.slot}"`, input.casts <= 0, input.casts >= 10)}
      ${hitControl}
    </article>`;
  }).join("");
  const withheld = coverage?.withheld?.length ? ` · withheld ${coverage.withheld.join("/")}` : "";
  return `<div class="ability-package"><div class="ability-package-head"><div><strong>Damage rotation</strong><span>${coverage ? `${coverage.supported}/${coverage.total} sourced${withheld}` : "Ranks and hits are explicit"}</span></div><small>${escapeHtml(champion.source?.label || "Patch data")}</small></div><div class="ability-rows" style="--ability-count:${Math.min(champion.abilities.length, 5)}">${rows}</div></div>`;
}

function roleQuestNote() {
  if (!state.attacker.role) return "Choose a role to apply its completed quest rules.";
  if (!state.attacker.roleQuestComplete) return "Role quest not completed.";
  if (state.attacker.role === "mid") return "+8% bonus AD and AP · Tier 3 boots use one of six slots.";
  if (state.attacker.role === "bottom") return "Six item slots · boots move into the dedicated quest slot.";
  if (state.attacker.role === "support") return "Reserved ward / support quest slot · excluded from damage scoring.";
  return "No item-slot change for this role.";
}

function renderRoleControls() {
  const roles = [["top", "Top"], ["jungle", "Jungle"], ["mid", "Mid"], ["bottom", "Bottom"], ["support", "Support"]];
  return `<div class="role-rules">
    <div class="role-picker"><span>Role</span><div>${roles.map(([value, label]) => `<button type="button" data-role="${value}" class="${state.attacker.role === value ? "active" : ""}">${label}</button>`).join("")}</div></div>
    <button class="quest-toggle ${state.attacker.roleQuestComplete ? "active" : ""}" type="button" data-role-quest aria-pressed="${state.attacker.roleQuestComplete}" ${state.attacker.role ? "" : "disabled"}><i></i><span>Role quest complete</span></button>
    <p>${escapeHtml(roleQuestNote())}</p>
  </div>`;
}

function buildArray(side) {
  return side === "A" ? state.attacker.buildA : state.attacker.buildB;
}

function buildStackArray(side) {
  return side === "A" ? state.attacker.buildAStacks : state.attacker.buildBStacks;
}

function questBootPath(side) {
  return `attacker.questBoot${side}`;
}

function renderBuildStrip(side) {
  const build = buildArray(side);
  const normalSlots = Array.from({ length: ordinarySlotCount() }, (_, index) => itemSlot(`attacker.build${side}.${index}`, build[index], false, true)).join("");
  const boot = usesQuestBootSlot()
    ? `<div class="quest-item"><span>${state.attacker.role === "mid" ? "Tier 3 boots" : "Boots slot"}</span>${itemSlot(questBootPath(side), state.attacker[`questBoot${side}`], false, true)}</div>`
    : "";
  const support = state.attacker.role === "support" && state.attacker.roleQuestComplete
    ? `<div class="utility-slot" title="This slot is not included in damage scoring"><span>Quest / wards</span><b>◆</b><small>Utility</small></div>`
    : "";
  return `<div class="complete-build build-${side.toLowerCase()}">
    <div class="complete-build-head"><strong>Build ${side}</strong><span>${ordinarySlotCount() + (usesQuestBootSlot() ? 1 : 0)} combat ${plural(ordinarySlotCount() + (usesQuestBootSlot() ? 1 : 0), "slot")}${support ? " + utility" : ""}</span></div>
    <div class="hero-slots">${normalSlots}${boot}${support}</div>
  </div>`;
}

function renderBuilder() {
  const attacker = state.attacker;
  const statsA = attackerChampionStats(buildAIds(), buildAStacks());
  const statsB = attackerChampionStats(buildBIds(), buildBStacks());
  const champion = getChampion(attacker.champion);
  const optimizePackageReady = optimizerDamagePackageReady();
  const optimizeReady = Boolean(attacker.champion && optimizePackageReady && state.targets.length && state.targets.every((target) => target.champion));
  const optimizerSummary = state.optimizer.summary
    ? `<div class="optimizer-summary"><strong>${state.optimizer.summary.tested.toLocaleString("en-US")} builds · ${one(state.optimizer.summary.elapsedMs)} ms</strong><span>Build A now shows the complete highest-damage build.</span></div>`
    : "";
  $("builder").innerHTML = `
    <section class="board-section hero-section">
      <div class="section-bar"><h2>Champion to optimize</h2><small><i class="legend-a"></i>A <i class="legend-b"></i>B values in the matrix</small></div>
      <div class="hero-board">
        <div class="hero-identity">
          <button class="hero-pick ${champion ? "" : "empty-hero"}" type="button" data-picker="champion" data-path="attacker.champion">${champion ? `<img src="${championImage(attacker.champion)}" alt="" />` : `<b>+</b>`}<span>${champion ? "Change champion" : "Choose champion"}</span></button>
          <div><p>${escapeHtml(champion?.title || "Start here")}</p><h2>${escapeHtml(attacker.champion || "Choose a champion")}</h2>${champion ? `<div class="level-control"><span>Level</span><button type="button" data-level="attacker.level" data-delta="-1">−</button><output>${attacker.level}</output><button type="button" data-level="attacker.level" data-delta="1">+</button></div>` : ""}</div>
        </div>
        <div class="hero-matrix"><div class="matrix-head"><strong>Complete champion stats</strong><span>${attacker.comparisonEnabled ? "Build A / Build B" : "Build A"}</span></div>${champion ? statMatrix(statsA, attacker.comparisonEnabled ? statsB : null) : `<div class="matrix-placeholder hero-placeholder">Select the champion whose build you want to compare or optimize.</div>`}</div>
      </div>
      ${champion ? renderAbilityPackage(champion) : ""}
      <div class="hero-build">
        ${renderRoleControls()}
        <div class="build-label"><div><strong>Complete builds</strong><span>${optimizePackageReady ? "Every occupied slot is scored into the full enemy roster" : "Add a sourced skill rotation or manual damage package to optimize"}</span></div><button class="optimize-build" type="button" data-optimize-build title="${optimizePackageReady ? "Search the strongest complete Build A" : "No sourced damage package for this champion yet"}" ${optimizeReady && !state.optimizer.running ? "" : "disabled"}>${state.optimizer.running ? "Optimizing…" : "Optimize Build A"}</button></div>
        ${optimizerSummary}
        ${renderBuildStrip("A")}
        <button class="compare-toggle" type="button" data-toggle-compare>${attacker.comparisonEnabled ? "Hide Build B" : "+ Compare another full build"}</button>
        ${attacker.comparisonEnabled ? renderBuildStrip("B") : ""}
      </div>
    </section>
    <section class="board-section">
      <div class="section-bar"><h2>Enemy roster</h2><div class="section-actions"><small>${state.targets.length}/10 · every card includes full stats</small><button class="text-button" type="button" data-add-target ${state.targets.length >= 10 ? "disabled" : ""}>+ Add enemy</button></div></div>
      <div class="target-grid">${state.targets.length ? state.targets.map(targetCard).join("") : `<div class="empty-roster">Add an enemy champion to begin.</div>`}</div>
    </section>
    <section class="board-section">
      <div class="section-bar"><h2>Time window</h2><small>Applied to every target</small></div>
      <div class="fight-controls">
        ${segmented("Rotations", [[1,"1"],[2,"2"],[3,"3"],[4,"4"],[5,"5"],[6,"6"]], state.fight.rotations, "rotations")}
        <div class="control-group"><span>Window per rotation</span><div class="duration-control"><div class="segmented">${[[3.5,"3.5s"],[8,"8s"],[16,"16s"]].map(([value, text]) => `<button type="button" data-fight="duration" data-value="${value}" class="${state.fight.duration === value ? "active" : ""}">${text}</button>`).join("")}</div><label><input type="range" min="1" max="40" step="0.5" value="${state.fight.duration}" data-fight-range="duration" /><output>${one(state.fight.duration)}s</output></label></div></div>
        <div class="control-group"><span>Auto-attack uptime</span><label class="uptime-control"><input type="range" min="0" max="100" step="5" value="${Math.round(state.fight.aaUptime * 100)}" data-fight-range="aaUptime" /><output>${Math.round(state.fight.aaUptime * 100)}%</output></label></div>
        <div class="auto-count"><span>Expected autos per rotation</span><strong><i class="legend-a"></i>A ${autoAttacksForStats(statsA)}${attacker.comparisonEnabled ? ` <i class="legend-b"></i>B ${autoAttacksForStats(statsB)}` : ""}</strong><small>attack speed × time × 0.92 × uptime</small></div>
      </div>
    </section>`;
}

function ludenDamage(ap, targetCount, targetIndex, hasLuden) {
  if (!hasLuden || targetIndex >= Math.min(6, targetCount)) return 0;
  const proc = 75 + 0.05 * ap;
  return targetIndex === 0 ? proc * (1 + 0.2 * Math.max(6 - targetCount, 0)) : proc;
}

function autoAttacksForStats(stats) {
  if (!state.attacker.champion || state.fight.aaUptime <= 0) return 0;
  const activeSeconds = state.fight.duration * Math.max(0, Math.min(1, state.fight.aaUptime));
  return Math.floor(Math.max(0.4, stats.attackSpeed) * activeSeconds * 0.92);
}

function effectiveMagicResistance(targetMr, build) {
  return Math.max(targetMr * (1 - build.percentPen / 100) - build.pen, 0);
}

function effectivePhysicalArmor(targetArmor, build) {
  return Math.max(targetArmor * (1 - build.percentArmorPen / 100) - build.lethality, 0);
}

function rankedValue(values, rank, levelScaled = false) {
  if (!Array.isArray(values) || !values.length) return 0;
  const index = levelScaled ? state.attacker.level - 1 : rank - 1;
  const selected = values[Math.max(0, Math.min(values.length - 1, index))];
  if (Array.isArray(selected)) return Number(selected[Math.max(0, Math.min(selected.length - 1, state.attacker.level - 1))] || 0);
  return Number(selected || 0);
}

function abilityDamageRows(attackerStats, targetStats) {
  const champion = getChampion(state.attacker.champion);
  const baseChampion = championStats(state.attacker.champion, state.attacker.level);
  return (champion?.abilities || []).flatMap((ability) => {
    const input = abilityInput(ability.slot);
    if (input.casts <= 0 || (ability.slot !== "P" && input.rank <= 0)) return [];
    const variant = ability.variants[input.variant] || ability.variants[0];
    const levelScaled = ability.slot === "P" && variant.levelBase;
    const rank = ability.slot === "P" ? 1 : input.rank;
    const base = rankedValue(variant.levelBase || variant.base, rank, levelScaled);
    const apRatio = rankedValue(variant.ap, rank, levelScaled);
    const adRatio = rankedValue(variant.ad, rank, levelScaled);
    const bonusAdRatio = rankedValue(variant.bonusAd, rank, levelScaled);
    const oneHit = base + apRatio * attackerStats.ap + adRatio * attackerStats.ad + bonusAdRatio * Math.max(0, attackerStats.ad - baseChampion.ad);
    const hits = ability.maxHits ? input.hits : 1;
    const hitFactor = ability.subsequentMultiplier ? 1 + Math.max(0, hits - 1) * ability.subsequentMultiplier : hits;
    const raw = oneHit * hitFactor * input.casts;
    const detailParts = [`${input.casts} ${plural(input.casts, ability.slot === "P" ? "proc" : "cast")}`];
    if (ability.maxHits) detailParts.push(`${hits} ${plural(hits, "mine")}`);
    if (ability.variants.length > 1) detailParts.push(variant.name);
    const packets = variant.packets?.length ? variant.packets : [{ type: variant.type }];
    return packets.map((packet) => {
      const spec = typeof packet === "string" ? { type: packet } : packet;
      const targetValue = spec.targetScale === "targetMaxHp" || spec.targetScale === "targetCurrentHp" ? targetStats.hp : spec.targetScale === "targetMissingHp" ? 0 : 1;
      return { source: `${ability.slot} · ${ability.name}`, detail: detailParts.join(" · "), raw: raw * targetValue / Number(spec.targetScaleDivisor || 1), type: spec.type };
    });
  });
}

function addBreakdown(map, source, detail, damage, kind = "ability") {
  if (!(damage > 0)) return;
  const key = `${kind}:${source}:${detail}`;
  const row = map.get(key) || { source, detail, damage: 0, kind };
  row.damage += damage;
  map.set(key, row);
}

function calculateBuild(buildIds, rotations = state.fight.rotations, stackCounts = [], options = {}) {
  const build = buildStats(buildIds, stackCounts);
  const hasLiandry = buildIds.includes(6653);
  const hasShadowflame = buildIds.includes(4645);
  const hasLuden = buildIds.includes(6655);
  const attackerStats = attackerChampionStats(buildIds, stackCounts);
  const autosPerRotation = autoAttacksForStats(attackerStats);
  const profile = options.profile || state.attacker;
  let cumulative = 0;
  const ledger = [];
  const breakdown = new Map();
  const recordBreakdown = (source, detail, damage, kind = "ability") => {
    if (!options.summaryOnly) addBreakdown(breakdown, source, detail, damage, kind);
  };

  for (let rotation = 1; rotation <= rotations; rotation += 1) {
    const steady = rotation > 1;
    const sufferingAmp = hasLiandry ? (steady ? 1.06 : 1) : 1;
    const burnTicks = steady ? 3 * 1.06 : 1.02 + 1.04 + 1.06;
    let rotationDamage = 0;

    state.targets.forEach((target, targetIndex) => {
      const targetStats = options.targetStats?.[targetIndex] || championStats(target.champion, target.level, target.items, target.itemStacks);
      const effectiveMr = effectiveMagicResistance(targetStats.mr, build);
      const effectiveArmor = effectivePhysicalArmor(targetStats.armor, build);
      const magicMultiplier = 100 / (100 + effectiveMr);
      const physicalMultiplier = 100 / (100 + effectiveArmor);
      const rows = options.profile?.useAbilities === false ? [] : abilityDamageRows(attackerStats, targetStats);
      if (profile.baseDamage || profile.apRatio) rows.push({ source: "Manual · magic package", detail: "Entered below", raw: profile.baseDamage + profile.apRatio * attackerStats.ap, type: "magical" });
      if (profile.physicalDamage || profile.adRatio) rows.push({ source: "Manual · physical package", detail: "Entered below", raw: profile.physicalDamage + profile.adRatio * attackerStats.ad, type: "physical" });
      const triggersAbilityItems = rows.length > 0;
      if (targetIndex === 0 && autosPerRotation > 0) {
        const expectedCritMultiplier = 1 + (attackerStats.crit / 100) * 0.75;
        rows.push({ source: "Auto attacks", detail: `${autosPerRotation} ${plural(autosPerRotation, "attack")} · ${one(state.fight.duration)}s at ${Math.round(state.fight.aaUptime * 100)}% uptime`, raw: autosPerRotation * attackerStats.ad * expectedCritMultiplier, type: "physical", kind: "auto" });
      }
      let targetMagicDamage = 0;

      rows.forEach((row) => {
        const multiplier = row.type === "magical" ? magicMultiplier : row.type === "physical" ? physicalMultiplier : 1;
        const mitigated = row.raw * multiplier;
        recordBreakdown(row.source, row.detail, mitigated, row.kind || "ability");
        rotationDamage += mitigated;
        if (row.type === "magical") targetMagicDamage += mitigated;
        if (hasLiandry && sufferingAmp > 1 && row.type !== "true") {
          const ampDamage = mitigated * (sufferingAmp - 1);
          rotationDamage += ampDamage;
          if (row.type === "magical") targetMagicDamage += ampDamage;
          recordBreakdown("Liandry’s Torment · Suffering", `${percent((sufferingAmp - 1) * 100)} ability amplification`, ampDamage, "item");
        }
      });

      const burnRaw = hasLiandry && triggersAbilityItems ? 0.02 * targetStats.hp * burnTicks : 0;
      const burn = burnRaw * magicMultiplier;
      if (burn > 0) {
        rotationDamage += burn;
        targetMagicDamage += burn;
        recordBreakdown("Liandry’s Torment · Torment", `${one(burnTicks)} burn ticks`, burn, "item");
      }
      const ludenRaw = ludenDamage(attackerStats.ap, state.targets.length, targetIndex, hasLuden && triggersAbilityItems);
      const luden = ludenRaw * magicMultiplier;
      if (luden > 0) {
        rotationDamage += luden;
        targetMagicDamage += luden;
        recordBreakdown("Luden’s Echo · Echo", targetIndex === 0 ? "Primary / returned charges" : "Secondary charge", luden, "item");
      }
      if (hasShadowflame && options.lowHealth && targetMagicDamage > 0) {
        const cinderbloom = targetMagicDamage * .2;
        rotationDamage += cinderbloom;
        recordBreakdown("Shadowflame · Cinderbloom", "Conditional: target below 40% HP", cinderbloom, "item");
      }
    });

    cumulative += rotationDamage;
    ledger.push({ rotation, rotationDamage, cumulative });
  }
  return { build, attackerStats, ledger, cumulative, breakdown: [...breakdown.values()] };
}

function buildIdsForSide(side) {
  const ids = buildArray(side).slice(0, ordinarySlotCount());
  if (usesQuestBootSlot()) ids.push(state.attacker[`questBoot${side}`]);
  return ids;
}

function buildStacksForSide(side) {
  const stacks = buildStackArray(side).slice(0, ordinarySlotCount());
  if (usesQuestBootSlot()) stacks.push(0);
  return stacks;
}

function buildAIds() { return buildIdsForSide("A"); }
function buildBIds() { return buildIdsForSide("B"); }
function buildAStacks() { return buildStacksForSide("A"); }
function buildBStacks() { return buildStacksForSide("B"); }

function resultReason(winnerName, winnerIds, winner, loser) {
  const apLead = winner.build.ap - loser.build.ap;
  const adLead = winner.build.ad - loser.build.ad;
  const attackSpeedLead = winner.build.attackSpeed - loser.build.attackSpeed;
  const critLead = winner.build.crit - loser.build.crit;
  const penLead = winner.build.pen - loser.build.pen;
  const percentPenLead = winner.build.percentPen - loser.build.percentPen;
  const lethalityLead = winner.build.lethality - loser.build.lethality;
  const armorPenLead = winner.build.percentArmorPen - loser.build.percentArmorPen;
  const averageHp = state.targets.length ? state.targets.reduce((sum, target) => sum + championStats(target.champion, target.level, target.items, target.itemStacks).hp, 0) / state.targets.length : 0;
  if (winnerIds.includes(6653) && !winnerName.includes(itemName(6653))) {
    return `<strong>${escapeHtml(winnerName)}</strong> wins because Liandry’s max-health burn repeats across ${state.targets.length} ${plural(state.targets.length, "enemy")} averaging ${fmt(averageHp)} HP.`;
  }
  if (winnerIds.includes(6653) && !winnerIds.includes(4645)) {
    return `<strong>${escapeHtml(winnerName)}</strong> wins because its max-health burn repeats across ${state.targets.length} ${plural(state.targets.length, "enemy")} averaging ${fmt(averageHp)} HP.`;
  }
  const advantages = [];
  if (apLead > 0) advantages.push(`${one(apLead)} more AP`);
  if (adLead > 0) advantages.push(`${one(adLead)} more AD`);
  if (attackSpeedLead > 0) advantages.push(`${one(attackSpeedLead)}% more attack speed`);
  if (critLead > 0) advantages.push(`${one(critLead)}% more crit`);
  if (percentPenLead > 0) advantages.push(`${one(percentPenLead)}% more magic penetration`);
  if (penLead > 0) advantages.push(`${one(penLead)} more flat magic penetration`);
  if (armorPenLead > 0) advantages.push(`${one(armorPenLead)}% more armor penetration`);
  if (lethalityLead > 0) advantages.push(`${one(lethalityLead)} more lethality`);
  return `<strong>${escapeHtml(winnerName)}</strong> wins ${advantages.length ? `through ${advantages.join(" and ")}` : "on the selected stats and damage package"}.`;
}

function renderResistanceOutput(aBuild, bBuild) {
  const host = $("resistanceOutput");
  if (!aBuild || !state.targets.length) {
    host.innerHTML = "";
    return;
  }
  const rows = state.targets.map((target) => {
    const stats = championStats(target.champion, target.level, target.items, target.itemStacks);
    const armor = `${one(effectivePhysicalArmor(stats.armor, aBuild))}${bBuild ? ` / ${one(effectivePhysicalArmor(stats.armor, bBuild))}` : ""}`;
    const mr = `${one(effectiveMagicResistance(stats.mr, aBuild))}${bBuild ? ` / ${one(effectiveMagicResistance(stats.mr, bBuild))}` : ""}`;
    return `<tr><td>${escapeHtml(target.champion)}</td><td>${one(stats.armor)} → ${armor}</td><td>${one(stats.mr)} → ${mr}</td></tr>`;
  }).join("");
  host.innerHTML = `<div class="resistance-head"><span>Effective defenses</span><b>% pen → flat pen</b></div><div class="resistance-table-wrap"><table><thead><tr><th>Enemy</th><th>Armor → A${bBuild ? " / B" : ""}</th><th>MR → A${bBuild ? " / B" : ""}</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function renderDamageBreakdown(aResult, bResult) {
  const host = $("damageBreakdown");
  if (!aResult) {
    host.innerHTML = "";
    host.hidden = true;
    return;
  }
  host.hidden = false;
  const rows = new Map();
  const ingest = (result, side) => result.breakdown.forEach((entry) => {
    const key = `${entry.source}:${entry.detail}`;
    const row = rows.get(key) || { source: entry.source, detail: entry.detail, a: 0, b: 0, kind: entry.kind };
    row[side] += entry.damage;
    rows.set(key, row);
  });
  ingest(aResult, "a");
  if (bResult) ingest(bResult, "b");

  const lowA = buildAIds().includes(4645) ? calculateBuild(buildAIds(), state.fight.rotations, buildAStacks(), { lowHealth: true }) : null;
  const lowB = bResult && buildBIds().includes(4645) ? calculateBuild(buildBIds(), state.fight.rotations, buildBStacks(), { lowHealth: true }) : null;
  const conditionalA = lowA?.breakdown.find((row) => row.source.includes("Cinderbloom"))?.damage || 0;
  const conditionalB = lowB?.breakdown.find((row) => row.source.includes("Cinderbloom"))?.damage || 0;
  if (conditionalA || conditionalB) rows.set("conditional:cinderbloom", { source: "Shadowflame · Cinderbloom", detail: "Conditional bonus below 40% HP · not included in TDD above", a: conditionalA, b: conditionalB, kind: "conditional" });

  const body = [...rows.values()].map((row) => {
    const delta = row.a - row.b;
    const value = (amount) => row.kind === "conditional" && amount > 0 ? `+${fmt(amount)}` : fmt(amount);
    const comparison = bResult ? `<td>${value(row.b)}</td><td class="${delta > .5 ? "delta-a" : delta < -.5 ? "delta-b" : ""}">${Math.abs(delta) < .5 ? "—" : `${delta > 0 ? "+" : ""}${fmt(delta)}`}</td>` : "";
    return `<tr class="${row.kind === "item" || row.kind === "conditional" ? "item-damage-row" : ""}"><td><strong>${escapeHtml(row.source)}</strong><small>${escapeHtml(row.detail)}</small></td><td>${value(row.a)}</td>${comparison}</tr>`;
  }).join("");
  const comparisonHead = bResult ? `<th><i class="legend-b"></i>Build B</th><th>A − B</th>` : "";
  const comparisonTotal = bResult ? `<td>${fmt(bResult.cumulative)}</td><td>${Math.abs(aResult.cumulative - bResult.cumulative) < .5 ? "—" : `${aResult.cumulative > bResult.cumulative ? "+" : ""}${fmt(aResult.cumulative - bResult.cumulative)}`}</td>` : "";
  host.innerHTML = `<header><div><p class="eyebrow">Damage breakdown</p><h2>Every skill, proc and burn</h2></div><span>${state.targets.length} ${plural(state.targets.length, "target")} · ${state.fight.rotations} ${plural(state.fight.rotations, "rotation")}</span></header>
    <div class="damage-table-wrap"><table class="damage-table"><thead><tr><th>Source</th><th><i class="legend-a"></i>Build A</th>${comparisonHead}</tr></thead><tbody>${body}<tr class="damage-total"><td><strong>Total damage dealt</strong><small>After selected enemy resistances</small></td><td>${fmt(aResult.cumulative)}</td>${comparisonTotal}</tr></tbody></table></div>`;
}

function renderResults() {
  $("resultFootnote").textContent = `DPS uses ${one(state.fight.duration)}s per rotation · autos use ${Math.round(state.fight.aaUptime * 100)}% uptime · TDD sums selected targets.`;
  const scenarioReady = state.attacker.champion && state.targets.length && state.targets.every((target) => target.champion);
  const hasA = buildAIds().some(Boolean);
  const hasB = state.attacker.comparisonEnabled && buildBIds().some(Boolean);
  if (!scenarioReady || !hasA) {
    $("resultContext").textContent = "Waiting for scenario";
    $("winnerVisual").innerHTML = `<div class="result-empty-mark">+</div><div><span>Nothing calculated</span><strong>Build a scenario</strong><b>Champion · builds · enemies</b></div>`;
    $("scoreGrid").innerHTML = `<div class="empty-score">Build A</div><div class="empty-score">${state.attacker.comparisonEnabled ? "Build B" : "Comparison optional"}</div>`;
    $("why").textContent = "Choose a champion, add Build A and select at least one enemy.";
    $("threshold").innerHTML = `<span>Crossover</span><strong>Waiting for complete inputs</strong>`;
    renderResistanceOutput(null, null);
    renderMechanicsOutput(0, 0);
    renderDamageBreakdown(null, null);
    $("rotationTable").innerHTML = "";
    return;
  }
  const aCurrent = calculateBuild(buildAIds(), state.fight.rotations, buildAStacks());
  const aAll = state.fight.rotations === 6 ? aCurrent : calculateBuild(buildAIds(), 6, buildAStacks());
  const aTotal = aCurrent.cumulative;
  const seconds = state.fight.rotations * state.fight.duration;
  $("resultContext").textContent = `${state.targets.length} ${plural(state.targets.length, "enemy")} · ${state.fight.rotations} ${plural(state.fight.rotations, "rotation")}`;

  if (!hasB) {
    $("winnerVisual").innerHTML = `<img src="${championImage(state.attacker.champion)}" alt="" /><div><span>Modeled output</span><strong>Build A</strong><b>${buildAIds().filter(Boolean).length} items scored</b></div>`;
    $("scoreGrid").innerHTML = `<div class="score winner single-score"><header><img src="${championImage(state.attacker.champion)}" alt="" /><span>Build A</span></header><strong>${fmt(aTotal)}</strong><small>TDD · ${fmt(aTotal / seconds)} DPS</small></div>`;
    renderResistanceOutput(aCurrent.build, null);
    $("why").innerHTML = `<strong>Build A</strong> is scored as one complete build. Enable Build B only for a full-build comparison.`;
    $("threshold").innerHTML = `<span>Six-rotation total</span><strong>${fmt(aAll.cumulative)} modeled damage</strong>`;
    $("tableA").textContent = "Build A";
    $("tableB").textContent = "—";
    $("rotationTable").innerHTML = aAll.ledger.map((row) => `<tr><td>${row.rotation}</td><td>${fmt(row.cumulative)}</td><td>—</td><td>Build A</td></tr>`).join("");
    renderMechanicsOutput(aTotal, 0);
    renderDamageBreakdown(aCurrent, null);
    return;
  }

  const bCurrent = calculateBuild(buildBIds(), state.fight.rotations, buildBStacks());
  const bAll = state.fight.rotations === 6 ? bCurrent : calculateBuild(buildBIds(), 6, buildBStacks());
  const bTotal = bCurrent.cumulative;
  const tied = Math.abs(aTotal - bTotal) < 0.5;
  const aWins = aTotal > bTotal;
  const winner = aWins ? { ...aAll, cumulative: aTotal } : { ...bAll, cumulative: bTotal };
  const loser = aWins ? { ...bAll, cumulative: bTotal } : { ...aAll, cumulative: aTotal };
  const winnerIds = aWins ? buildAIds() : buildBIds();
  const edge = loser.cumulative > 0 ? Math.abs(winner.cumulative / loser.cumulative - 1) * 100 : 0;
  const aName = "Build A";
  const bName = "Build B";

  $("winnerVisual").innerHTML = tied
    ? `<div></div><div><span>Dead even</span><strong>Tie</strong><b>Less than 1 damage apart</b></div>`
    : `<img src="${championImage(state.attacker.champion)}" alt="" /><div><span>Better here</span><strong>${aWins ? "Build A" : "Build B"}</strong><b>${percent(edge)} more total damage</b></div>`;
  $("scoreGrid").innerHTML = [
    { total: aTotal, name: aName, winner: aWins && !tied },
    { total: bTotal, name: bName, winner: !aWins && !tied },
  ].map((entry) => `<div class="score ${entry.winner ? "winner" : ""}"><header><img src="${championImage(state.attacker.champion)}" alt="" /><span>${entry.name}</span></header><strong>${fmt(entry.total)}</strong><small>TDD · ${fmt(entry.total / seconds)} DPS</small></div>`).join("");
  renderResistanceOutput(aCurrent.build, bCurrent.build);
  $("why").innerHTML = tied ? "Both builds deal effectively the same damage in this setup." : resultReason(aWins ? "Build A" : "Build B", winnerIds, winner, loser);

  const leads = aAll.ledger.map((row, rowIndex) => Math.sign(row.cumulative - bAll.ledger[rowIndex].cumulative));
  const firstLead = leads[0];
  const crossoverIndex = leads.findIndex((lead, rowIndex) => rowIndex > 0 && lead !== 0 && lead !== firstLead);
  let thresholdText;
  if (crossoverIndex >= 0) {
    thresholdText = `${escapeHtml(leads[crossoverIndex] > 0 ? aName : bName)} takes over at rotation ${crossoverIndex + 1}`;
  } else {
    thresholdText = firstLead === 0 ? "No meaningful difference through 6 rotations" : `${escapeHtml(firstLead > 0 ? aName : bName)} stays ahead through 6 rotations`;
  }
  $("threshold").innerHTML = `<span>Crossover</span><strong>${thresholdText}</strong>`;
  $("tableA").textContent = aName;
  $("tableB").textContent = bName;
  $("rotationTable").innerHTML = aAll.ledger.map((row, rowIndex) => {
    const bRow = bAll.ledger[rowIndex];
    const lead = Math.abs(row.cumulative - bRow.cumulative) < .5 ? "Tie" : row.cumulative > bRow.cumulative ? aName : bName;
    return `<tr><td>${row.rotation}</td><td>${fmt(row.cumulative)}</td><td>${fmt(bRow.cumulative)}</td><td>${escapeHtml(lead)}</td></tr>`;
  }).join("");
  renderMechanicsOutput(aTotal, bTotal);
  renderDamageBreakdown(aCurrent, bCurrent);
}

function itemOccurrence(id) {
  const occurrences = [];
  state.attacker.buildA.forEach((itemId, index) => { if (itemId === id) occurrences.push({ path: `attacker.buildA.${index}`, build: "A" }); });
  if (state.attacker.questBootA === id) occurrences.push({ path: "attacker.questBootA", build: "A" });
  if (state.attacker.comparisonEnabled) {
    state.attacker.buildB.forEach((itemId, index) => { if (itemId === id) occurrences.push({ path: `attacker.buildB.${index}`, build: "B" }); });
    if (state.attacker.questBootB === id) occurrences.push({ path: "attacker.questBootB", build: "B" });
  }
  return occurrences[0] || null;
}

function mechanicDetail(id, occurrence, aTotal, bTotal) {
  const stacks = occurrence ? stackValue(occurrence.path) : 0;
  if (id === 1082) return `${stacks}/10 Glory · +${stacks * 4} AP from stacks`;
  if (id === 3041) return `${stacks}/25 Glory · +${stacks * 5} AP${stacks >= 10 ? " · 10% bonus move speed active" : " · move speed activates at 10"}`;
  if (id === 3089) {
    const apA = buildStats(buildAIds(), buildAStacks());
    const apB = buildStats(buildBIds(), buildBStacks());
    const ap = occurrence?.build === "B" ? apB : apA;
    const finalAp = occurrence?.build === "B" ? attackerChampionStats(buildBIds(), buildBStacks()).ap : attackerChampionStats(buildAIds(), buildAStacks()).ap;
    const quest = state.attacker.role === "mid" && state.attacker.roleQuestComplete ? ` · Mid quest × 1.08 = ${one(finalAp)} final AP` : "";
    return `Magical Opus: ${one(ap.apBeforeMultiplier)} AP before the passive × 1.30 = ${one(ap.ap)} AP${quest}.`;
  }
  if (id === 6653) return "Torment burn and the 2% → 6% Suffering ramp are applied automatically over time.";
  if (id === 6655) return `Six Echo charges are distributed across ${state.targets.length} ${plural(state.targets.length, "enemy")}; unused charges return to the primary target at 20% damage.`;
  if (id === 4645) {
    if (occurrence?.build === "A") {
      const low = calculateBuild(buildAIds(), state.fight.rotations, buildAStacks(), { lowHealth: true }).cumulative;
      return `Above 40%: ${fmt(aTotal)} TDD · Below 40%: ${fmt(low)} TDD with Cinderbloom.`;
    }
    if (occurrence?.build === "B") {
      const low = calculateBuild(buildBIds(), state.fight.rotations, buildBStacks(), { lowHealth: true }).cumulative;
      return `Above 40%: ${fmt(bTotal)} TDD · Below 40%: ${fmt(low)} TDD with Cinderbloom.`;
    }
    return "Cinderbloom’s 20% damage increase is conditional on the enemy already being below 40% health.";
  }
  return getItem(id)?.passiveText || "No conditional item effect in the patch data.";
}

function renderMechanicsOutput(aTotal, bTotal) {
  const ids = [...new Set([...buildAIds(), ...(state.attacker.comparisonEnabled ? buildBIds() : [])].filter(Boolean))];
  const modeled = new Set([1082, 3041, 3089, 6653, 6655, 4645]);
  $("mechanicsOutput").innerHTML = ids.length ? `<div class="mechanics-head"><span>Item-specific output</span><b>${ids.length} selected</b></div>${ids.map((id) => {
    const item = getItem(id);
    const occurrence = itemOccurrence(id);
    return `<article class="mechanic-row"><img src="${itemImage(id)}" alt="" /><div><strong>${escapeHtml(item.name)}</strong><p>${escapeHtml(mechanicDetail(id, occurrence, aTotal, bTotal))}</p></div><span class="${modeled.has(id) ? "modelled" : "text-only"}">${modeled.has(id) ? "Calculated" : "Patch text"}</span></article>`;
  }).join("")}` : "";
}

function scenarioSentence() {
  if (!state.attacker.champion) return "Choose a champion, a build and an enemy roster to begin.";
  const buildA = buildAIds().map(getItem).filter(Boolean).map((item) => item.name);
  const names = state.targets.map((target) => target.champion);
  const roster = names.length <= 1 ? (names[0] || "no enemies") : `${names.slice(0, -1).join(", ")} and ${names.at(-1)}`;
  const compareText = state.attacker.comparisonEnabled && buildBIds().some(Boolean) ? `, comparing <strong>Build A</strong> with <strong>Build B</strong>` : "";
  const targetText = names.length ? `, into ${escapeHtml(roster)}` : "";
  return `<strong>${escapeHtml(state.attacker.champion)} level ${state.attacker.level}</strong>${buildA.length ? ` with ${escapeHtml(buildA.join(" + "))}` : ""}${compareText}${targetText} over ${state.fight.rotations} ${plural(state.fight.rotations, "rotation")} · ${one(state.fight.duration)}s each · ${Math.round(state.fight.aaUptime * 100)}% auto uptime.`;
}

function render() {
  renderBuilder();
  renderResults();
  $("scenarioSentence").innerHTML = scenarioSentence();
}

function openPicker(type, path) {
  pickerContext = { type, path };
  $("pickerKind").textContent = type === "champion" ? "Champion roster" : "Item catalogue";
  $("pickerTitle").textContent = type === "champion" ? "Choose a champion" : "Choose an item";
  $("pickerSearch").value = "";
  renderPicker("");
  $("picker").showModal();
  requestAnimationFrame(() => $("pickerSearch").focus());
}

function renderPicker(query) {
  if (!pickerContext) return;
  const normalized = query.trim().toLowerCase();
  const selected = pathValue(pickerContext.path);
  const entries = (pickerContext.type === "champion" ? DATA.champions : DATA.items).filter((entry) => {
    if (!entry.name.toLowerCase().includes(normalized)) return false;
    if (pickerContext.type !== "item") return true;
    if (pickerContext.path.includes("questBoot")) return questBootIds().includes(entry.id);
    if (usesQuestBootSlot() && pickerContext.path.match(/^attacker\.build[AB]\./)) return !ALL_ROLE_BOOTS.has(entry.id);
    return true;
  });
  const empty = pickerContext.type === "item" && !normalized ? `<button class="picker-option ${Number(selected) === 0 ? "selected" : ""}" type="button" data-picker-value="0"><span class="empty-icon">×</span><span><strong>Empty slot</strong><small>Remove item</small></span></button>` : "";
  $("pickerGrid").innerHTML = empty + entries.map((entry) => {
    const value = pickerContext.type === "champion" ? entry.name : entry.id;
    const image = pickerContext.type === "champion" ? championImage(entry.name) : itemImage(entry.id);
    const detail = pickerContext.type === "champion" ? `${entry.tags.join(" · ")} · ${entry.resource}` : itemStatsLine(entry);
    return `<button class="picker-option ${String(selected) === String(value) ? "selected" : ""}" type="button" data-picker-value="${escapeHtml(value)}"><img src="${image}" alt="" loading="lazy" /><span><strong>${escapeHtml(entry.name)}</strong><small>${escapeHtml(detail)}</small></span></button>`;
  }).join("") || `<p class="picker-empty">No matches for “${escapeHtml(query)}”.</p>`;
}

function closePicker() {
  $("picker").close();
  pickerContext = null;
}

function idsForBis(path, candidateId) {
  const side = path.includes("buildB") || path.includes("questBootB") ? "B" : "A";
  const ids = buildArray(side).slice(0, ordinarySlotCount());
  if (path.match(/^attacker\.build[AB]\./)) ids[Number(path.split(".").at(-1))] = candidateId;
  if (usesQuestBootSlot()) ids.push(path.includes("questBoot") ? candidateId : state.attacker[`questBoot${side}`]);
  return ids;
}

function stacksForBis(path, candidateId) {
  const side = path.includes("buildB") || path.includes("questBootB") ? "B" : "A";
  const paths = Array.from({ length: ordinarySlotCount() }, (_, index) => `attacker.build${side}.${index}`);
  const stacks = paths.map((itemPath) => itemPath === path && Number(pathValue(path)) !== Number(candidateId) ? 0 : stackValue(itemPath));
  if (usesQuestBootSlot()) stacks.push(0);
  return stacks;
}

function bisProfile() {
  const selectedAbilities = activeAbilityKit().filter((ability) => abilityInput(ability.slot).casts > 0 && (ability.slot === "P" || abilityInput(ability.slot).rank > 0));
  if (selectedAbilities.length) return { baseDamage: 0, apRatio: 0, physicalDamage: 0, adRatio: 0, useAbilities: true, label: "Selected skill rotation", exact: true };
  const exact = state.attacker.baseDamage > 0 || state.attacker.apRatio > 0 || state.attacker.physicalDamage > 0 || state.attacker.adRatio > 0;
  if (exact) return { ...state.attacker, label: "Exact damage package", exact: true };
  const champion = getChampion(state.attacker.champion);
  const primary = champion?.tags?.[0] || "Fighter";
  const magical = primary === "Mage" || primary === "Support";
  return magical
    ? { baseDamage: 100, apRatio: 1, physicalDamage: 0, adRatio: 0, label: "Stat-only AP preview", exact: false }
    : { baseDamage: 0, apRatio: 0, physicalDamage: 0, adRatio: 1, label: "Stat-only AD preview", exact: false };
}

function optimizerDamagePackageReady() {
  const champion = getChampion(state.attacker.champion);
  const coverage = champion?.abilityCoverage;
  const completeSourcedKit = Boolean(coverage && coverage.supported === coverage.total && coverage.total > 0);
  const selectedAbilities = completeSourcedKit && activeAbilityKit().some((ability) => abilityInput(ability.slot).casts > 0 && (ability.slot === "P" || abilityInput(ability.slot).rank > 0));
  const manualPackage = state.attacker.baseDamage > 0 || state.attacker.apRatio > 0 || state.attacker.physicalDamage > 0 || state.attacker.adRatio > 0;
  return selectedAbilities || manualPackage;
}

function bisCandidates(path, profile) {
  const selectedTypes = profile.useAbilities ? activeAbilityKit().flatMap((ability) => ability.variants.flatMap((variant) => variant.packets?.map((packet) => typeof packet === "string" ? packet : packet.type) || [variant.type])) : [];
  const physical = profile.physicalDamage > 0 || profile.adRatio > 0 || selectedTypes.includes("physical");
  const magical = profile.baseDamage > 0 || profile.apRatio > 0 || selectedTypes.includes("magical");
  const currentIds = idsForBis(path, 0).filter(Boolean);
  if (path.includes("questBoot")) return DATA.items.filter((item) => questBootIds().includes(item.id) && !currentIds.includes(item.id));
  return DATA.items.filter((item) => {
    const completed = item.price >= 2200 && item.into.length === 0;
    const relevant = (magical && (item.ap || item.pen || item.percentPen || [6653, 6655].includes(item.id))) || (physical && (item.ad || item.lethality || item.percentArmorPen));
    return completed && relevant && !ALL_ROLE_BOOTS.has(item.id) && !currentIds.includes(item.id);
  });
}

const optimizerExclusiveGroups = [
  new Set([3135, 3137]), // Void Staff / Cryptbloom
  new Set([3033, 3036, 6694]), // Last Whisper upgrades
];

function legalOptimizerBuild(ids) {
  if (new Set(ids).size !== ids.length) return false;
  return optimizerExclusiveGroups.every((group) => ids.filter((id) => group.has(id)).length <= 1);
}

function optimizerCandidatePool(profile, targetStats) {
  const selectedTypes = profile.useAbilities ? activeAbilityKit().flatMap((ability) => ability.variants.flatMap((variant) => variant.packets?.map((packet) => typeof packet === "string" ? packet : packet.type) || [variant.type])) : [];
  const hasMagicalAbilities = profile.baseDamage > 0 || profile.apRatio > 0 || selectedTypes.includes("magical");
  const hasPhysicalAbilities = profile.physicalDamage > 0 || profile.adRatio > 0 || selectedTypes.includes("physical");
  const physical = hasPhysicalAbilities;
  const magical = hasMagicalAbilities;
  const relevant = DATA.items.filter((item) => {
    const completed = item.price >= 2200 && item.into.length === 0;
    const damageStat = (magical && (item.ap || item.pen || item.percentPen || [6653, 6655].includes(item.id)))
      || (physical && (item.ad || item.lethality || item.percentArmorPen || item.attackSpeed || item.crit));
    return completed && damageStat;
  });
  const scoreOptions = { profile, summaryOnly: true, targetStats };
  const scored = relevant.map((item) => ({
    item,
    total: calculateBuild([item.id], state.fight.rotations, [0], scoreOptions).cumulative,
  })).sort((a, b) => b.total - a.total);
  if (!(magical && physical)) return scored.slice(0, 15);

  const isMagicalItem = ({ item }) => item.ap || item.pen || item.percentPen || [6653, 6655].includes(item.id);
  const isPhysicalItem = ({ item }) => item.ad || item.lethality || item.percentArmorPen || item.attackSpeed || item.crit;
  const magicalQuota = hasPhysicalAbilities ? 8 : 11;
  const physicalQuota = 15 - magicalQuota;
  const chosen = [];
  const chosenIds = new Set();
  const take = (entries, count) => {
    for (const entry of entries) {
      if (chosen.length >= 15 || count <= 0) break;
      if (chosenIds.has(entry.item.id)) continue;
      chosen.push(entry);
      chosenIds.add(entry.item.id);
      count -= 1;
    }
  };
  take(scored.filter(isMagicalItem), magicalQuota);
  take(scored.filter(isPhysicalItem), physicalQuota);
  take(scored, 15 - chosen.length);
  return chosen;
}

function optimizeFullBuild() {
  const started = performance.now();
  const profile = bisProfile();
  const targetStats = state.targets.map((target) => championStats(target.champion, target.level, target.items, target.itemStacks));
  const pool = optimizerCandidatePool(profile, targetStats);
  const slotCount = ordinarySlotCount();
  if (pool.length < slotCount) throw new Error("Not enough completed damage items for this package.");
  const ids = pool.map((entry) => entry.item.id);
  const boots = usesQuestBootSlot() ? questBootIds() : [0];
  const evaluated = [];
  const selected = [];
  const scoreOptions = { profile, summaryOnly: true, targetStats };

  function search(from) {
    if (selected.length === slotCount) {
      if (!legalOptimizerBuild(selected)) return;
      for (const bootId of boots) {
        const buildIds = bootId ? [...selected, bootId] : [...selected];
        evaluated.push({ ids: [...selected], questBoot: bootId, total: calculateBuild(buildIds, state.fight.rotations, [], scoreOptions).cumulative });
      }
      return;
    }
    const needed = slotCount - selected.length;
    for (let index = from; index <= ids.length - needed; index += 1) {
      selected.push(ids[index]);
      if (legalOptimizerBuild(selected)) search(index + 1);
      selected.pop();
    }
  }

  search(0);
  evaluated.sort((a, b) => b.total - a.total);
  const best = evaluated[0];
  if (!best) throw new Error("No legal complete build matched this package.");
  const elapsedMs = performance.now() - started;
  return { build: best.ids, questBoot: best.questBoot, tested: evaluated.length, elapsedMs, total: best.total, candidateCount: pool.length };
}

function startOptimizeBuild() {
  if (state.optimizer.running || !state.attacker.champion || !optimizerDamagePackageReady() || !state.targets.length || !state.targets.every((target) => target.champion)) return;
  state.optimizer.running = true;
  state.optimizer.summary = null;
  renderBuilder();
  requestAnimationFrame(() => requestAnimationFrame(() => {
    try {
      const result = optimizeFullBuild();
      state.attacker.buildA = [...result.build, ...Array(Math.max(0, 6 - result.build.length)).fill(0)].slice(0, 6);
      state.attacker.buildAStacks = [0, 0, 0, 0, 0, 0];
      state.attacker.questBootA = result.questBoot || 0;
      state.optimizer.summary = result;
    } finally {
      state.optimizer.running = false;
      render();
    }
  }));
}

function openBis(path) {
  if (!state.attacker.champion || !state.targets.length || !state.targets.every((target) => target.champion)) return;
  bisContext = { path };
  const profile = bisProfile();
  const ranked = bisCandidates(path, profile).filter((item) => legalOptimizerBuild(idsForBis(path, item.id).filter(Boolean))).map((item) => {
    const result = calculateBuild(idsForBis(path, item.id), state.fight.rotations, stacksForBis(path, item.id), { profile });
    return { item, result, total: result.cumulative };
  }).sort((a, b) => b.total - a.total).slice(0, 12);
  const baselineId = Number(pathValue(path)) || 0;
  const baseline = calculateBuild(idsForBis(path, baselineId), state.fight.rotations, stacksForBis(path, baselineId), { profile }).cumulative;
  const side = path.includes("buildB") || path.includes("questBootB") ? "B" : "A";
  $("bisTitle").textContent = path.includes("questBoot") ? `Best boots for Build ${side}` : `Best item for Build ${side} · slot ${Number(path.split(".").at(-1)) + 1}`;
  $("bisSummary").textContent = `${state.attacker.champion} · ${profile.label} · ${state.targets.length} ${plural(state.targets.length, "enemy")}`;
  $("bisList").innerHTML = ranked.map((entry, index) => {
    const gain = baseline > 0 ? (entry.total / baseline - 1) * 100 : 0;
    return `<article class="bis-row"><span class="bis-rank">${String(index + 1).padStart(2, "0")}</span><img src="${itemImage(entry.item.id)}" alt="" /><div><strong>${escapeHtml(entry.item.name)}</strong><small>${escapeHtml(itemStatsLine(entry.item))}</small></div><p><strong>${fmt(entry.total)}</strong><span>TDD · ${gain >= 0 ? "+" : ""}${one(gain)}%</span></p><button type="button" data-bis-value="${entry.item.id}">Use</button></article>`;
  }).join("") || `<p class="picker-empty">No valid completed items match this damage package.</p>`;
  $("bis").showModal();
}

function closeBis() {
  $("bis").close();
  bisContext = null;
}

function updateDamagePackage() {
  invalidateOptimization();
  state.attacker.baseDamage = Math.max(0, Number($("baseDamage").value) || 0);
  state.attacker.apRatio = Math.max(0, Number($("apRatio").value) || 0);
  state.attacker.physicalDamage = Math.max(0, Number($("physicalDamage").value) || 0);
  state.attacker.adRatio = Math.max(0, Number($("adRatio").value) || 0);
  render();
}

document.addEventListener("click", (event) => {
  if (event.target.closest("[data-optimize-build]")) return startOptimizeBuild();
  const roleButton = event.target.closest("[data-role]");
  if (roleButton) {
    state.attacker.role = roleButton.dataset.role;
    state.attacker.questBootA = 0;
    state.attacker.questBootB = 0;
    invalidateOptimization();
    return render();
  }
  if (event.target.closest("[data-role-quest]")) {
    state.attacker.roleQuestComplete = !state.attacker.roleQuestComplete;
    state.attacker.questBootA = 0;
    state.attacker.questBootB = 0;
    invalidateOptimization();
    return render();
  }
  if (event.target.closest("[data-toggle-compare]")) {
    state.attacker.comparisonEnabled = !state.attacker.comparisonEnabled;
    if (state.attacker.comparisonEnabled && !state.attacker.buildB.some(Boolean)) {
      state.attacker.buildB = [...state.attacker.buildA];
      state.attacker.buildBStacks = [...state.attacker.buildAStacks];
      state.attacker.questBootB = state.attacker.questBootA;
    }
    return render();
  }
  const pickerButton = event.target.closest("[data-picker]");
  if (pickerButton) return openPicker(pickerButton.dataset.picker, pickerButton.dataset.path);
  const bisButton = event.target.closest("[data-bis-path]");
  if (bisButton) return openBis(bisButton.dataset.bisPath);
  const levelButton = event.target.closest("[data-level]");
  if (levelButton) {
    setPath(levelButton.dataset.level, Math.max(1, Math.min(18, Number(pathValue(levelButton.dataset.level)) + Number(levelButton.dataset.delta))));
    return render();
  }
  const stackButton = event.target.closest("[data-stack-path]");
  if (stackButton) {
    const path = stackButton.dataset.stackPath;
    const spec = stackSpec(pathValue(path));
    setStackValue(path, Math.max(0, Math.min(spec.max, stackValue(path) + Number(stackButton.dataset.delta))));
    return render();
  }
  const abilityRankButton = event.target.closest("[data-ability-rank]");
  if (abilityRankButton) {
    invalidateOptimization();
    const slot = abilityRankButton.dataset.abilityRank;
    const ability = activeAbilityKit().find((entry) => entry.slot === slot);
    const input = abilityInput(slot);
    input.rank = Math.max(0, Math.min(ability.maxRank, input.rank + Number(abilityRankButton.dataset.delta)));
    state.attacker.abilityInputs[slot] = input;
    return render();
  }
  const abilityCastButton = event.target.closest("[data-ability-casts]");
  if (abilityCastButton) {
    invalidateOptimization();
    const slot = abilityCastButton.dataset.abilityCasts;
    const input = abilityInput(slot);
    input.casts = Math.max(0, Math.min(10, input.casts + Number(abilityCastButton.dataset.delta)));
    state.attacker.abilityInputs[slot] = input;
    return render();
  }
  const abilityHitButton = event.target.closest("[data-ability-hits]");
  if (abilityHitButton) {
    invalidateOptimization();
    const slot = abilityHitButton.dataset.abilityHits;
    const ability = activeAbilityKit().find((entry) => entry.slot === slot);
    const input = abilityInput(slot);
    input.hits = Math.max(1, Math.min(ability.maxHits, input.hits + Number(abilityHitButton.dataset.delta)));
    state.attacker.abilityInputs[slot] = input;
    return render();
  }
  const abilityVariantButton = event.target.closest("[data-ability-variant]");
  if (abilityVariantButton) {
    invalidateOptimization();
    const slot = abilityVariantButton.dataset.abilityVariant;
    const input = abilityInput(slot);
    input.variant = Number(abilityVariantButton.dataset.value);
    state.attacker.abilityInputs[slot] = input;
    return render();
  }
  const fightButton = event.target.closest("[data-fight]");
  if (fightButton) {
    invalidateOptimization();
    const key = fightButton.dataset.fight;
    state.fight[key] = Number(fightButton.dataset.value);
    return render();
  }
  const removeButton = event.target.closest("[data-remove-target]");
  if (removeButton) {
    invalidateOptimization();
    state.targets.splice(Number(removeButton.dataset.removeTarget), 1);
    return render();
  }
  if (event.target.closest("[data-add-target]")) {
    invalidateOptimization();
    if (state.targets.length < 10) {
      const index = state.targets.length;
      state.targets.push({ champion: null, level: 1, items: [0, 0, 0, 0, 0, 0], itemStacks: [0, 0, 0, 0, 0, 0] });
      render();
      return openPicker("champion", `targets.${index}.champion`);
    }
    return;
  }
  const option = event.target.closest("[data-picker-value]");
  if (option && pickerContext) {
    const selectedPath = pickerContext.path;
    if (pickerContext.type === "item") setStackValue(pickerContext.path, 0);
    setPath(pickerContext.path, pickerContext.type === "item" ? Number(option.dataset.pickerValue) : option.dataset.pickerValue);
    if (pickerContext.type === "champion" && selectedPath === "attacker.champion") resetAbilityInputs();
    closePicker();
    return render();
  }
  const bisOption = event.target.closest("[data-bis-value]");
  if (bisOption && bisContext) {
    setStackValue(bisContext.path, 0);
    setPath(bisContext.path, Number(bisOption.dataset.bisValue));
    closeBis();
    return render();
  }
});

document.addEventListener("input", (event) => {
  const range = event.target.closest("[data-fight-range]");
  if (!range) return;
  invalidateOptimization();
  const key = range.dataset.fightRange;
  state.fight[key] = key === "aaUptime" ? Number(range.value) / 100 : Number(range.value);
  const output = range.parentElement.querySelector("output");
  if (output) output.textContent = key === "aaUptime" ? `${Math.round(state.fight.aaUptime * 100)}%` : `${one(state.fight.duration)}s`;
  const statsA = attackerChampionStats(buildAIds(), buildAStacks());
  const statsB = attackerChampionStats(buildBIds(), buildBStacks());
  const autoCount = document.querySelector(".auto-count strong");
  if (autoCount) autoCount.innerHTML = `<i class="legend-a"></i>A ${autoAttacksForStats(statsA)}${state.attacker.comparisonEnabled ? ` <i class="legend-b"></i>B ${autoAttacksForStats(statsB)}` : ""}`;
  renderResults();
  $("scenarioSentence").innerHTML = scenarioSentence();
});

document.addEventListener("change", (event) => {
  if (event.target.closest("[data-fight-range]")) renderBuilder();
});

$("pickerSearch").addEventListener("input", (event) => renderPicker(event.target.value));
$("pickerClose").addEventListener("click", closePicker);
$("picker").addEventListener("click", (event) => { if (event.target === $("picker")) closePicker(); });
$("bisClose").addEventListener("click", closeBis);
$("bis").addEventListener("click", (event) => { if (event.target === $("bis")) closeBis(); });
for (const id of ["baseDamage", "apRatio", "physicalDamage", "adRatio"]) $(id).addEventListener("input", updateDamagePackage);
$("themeToggle").addEventListener("click", () => { document.documentElement.dataset.theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; });

fetch("data.json")
  .then((response) => { if (!response.ok) throw new Error("Patch snapshot failed to load"); return response.json(); })
  .then((data) => { DATA = data; render(); })
  .catch(() => { $("builder").innerHTML = `<p class="empty-roster">The patch snapshot could not load. Refresh to try again.</p>`; });
