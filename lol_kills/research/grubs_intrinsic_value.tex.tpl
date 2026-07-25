\documentclass[10pt,letterpaper]{article}
\usepackage[margin=0.80in,top=0.70in,bottom=0.70in]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,microtype,tabularx,array,enumitem,fancyhdr,colortbl}
\usepackage[hidelinks]{hyperref}
\usepackage{xcolor}
\usepackage{titlesec}

\definecolor{ink}{HTML}{1C1E24}
\definecolor{muted}{HTML}{5C606A}
\definecolor{paper}{HTML}{FFFFFF}
\definecolor{rule}{HTML}{C8CAD0}
\definecolor{steel}{HTML}{3D4A5C}
\definecolor{steelfill}{HTML}{E6EBEF}
\definecolor{danger}{HTML}{6B3A32}
\definecolor{dangerfill}{HTML}{F3E9E6}
\pagecolor{paper}
\color{ink}
\hypersetup{
  pdftitle={Contesting Void Grubs under uncertainty},
  pdfauthor={Mari Cabral Bonfim (Koi)},
  pdfsubject={Statistical decision analysis of Void Grub contests},
  pdfkeywords={Void Grubs, League of Legends, logistic regression, expected utility, sensitivity analysis}
}
\renewcommand{\familydefault}{\rmdefault}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.46em}
\setlength{\abovedisplayskip}{6pt}
\setlength{\belowdisplayskip}{6pt}
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.15}
\setlist[itemize]{leftmargin=1.2em,itemsep=0.20em,topsep=0.25em}
\titleformat{\section}{\large\bfseries\color{ink}}{\thesection}{0.72em}{}
\titlespacing*{\section}{0pt}{1.30em}{0.43em}
\titleformat{\subsection}{\normalsize\bfseries\color{ink}}{\thesubsection}{0.62em}{}
\titlespacing*{\subsection}{0pt}{0.85em}{0.18em}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\newcommand{\fineprint}[1]{{\footnotesize\color{muted}#1}}
\newcommand{\source}[1]{{\scriptsize\color{muted}#1}}
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\color{muted}\footnotesize\thepage}
\renewcommand{\headrulewidth}{0pt}

\begin{document}

\begin{center}
{\small\sffamily\color{steel}\MakeUppercase{Preprint} \textperiodcentered\ Statistical decision analysis\par}
\vspace{0.75em}
{\fontsize{23}{27}\selectfont\bfseries Contesting Void Grubs under uncertainty\par}
\vspace{0.38em}
{\large\itshape\color{muted}Break-even fight probabilities from competitive-map calibration and terminal-state valuation\par}
\vspace{0.90em}
{\large Mari Cabral Bonfim (Koi)\par}
\vspace{0.22em}
{\small\color{muted}19 July 2026\par}
\vspace{0.22em}
{\small\color{muted}LCS, LCK, LEC, LPL, and CBLOL \textperiodcentered\ 2026 reward era\par}
\vspace{0.72em}
\rule{0.92\linewidth}{0.55pt}
\end{center}

\begin{center}
\begin{minipage}{0.91\linewidth}
\begin{center}\textbf{Abstract}\end{center}
\small
This paper computes the minimum fight-win probability $p^\star$ at which contesting Void Grubs is preferred to conceding without fighting, using competitive gold@10 map-win calibration ($n=\textrm{<<N_FIT>>}$ fits from <<N_MAPS>> 2026 reward-era maps in LCS, LCK, LEC, LPL, CBLOL; LPL lacks gold@10) plus mechanical reward and leave-farm scenarios. In the reference state---two waves preserved by conceding without fighting, brief Touch of the Void package, secure-if-win---a fight at $p=50\%$ is $<<EV_AT_50>>$ map-win pp worse than conceding; indifference requires $p^\star=<<REFUSAL_PCT>>\%$. Leave-farm opportunity cost, rather than the Touch gold-equivalent, is the binding sensitivity. A ranked fight-win-probability pilot is too weak for operational decision use. The result is a conditional hurdle, not a causal effect of contesting.

\vspace{0.45em}
\noindent\textbf{Keywords:} Void Grubs; break-even probability; opportunity cost; expected utility; competitive League of Legends.
\end{minipage}
\end{center}

\section{Introduction}
The decision problem is whether to contest a river objective when the probability of winning the resulting fight is uncertain. The relevant actions are $C$, contest, and $L$, concede without fighting. Action $L$ is not a zero-payoff control: it may preserve lane farm, wave position, health, cooldowns, summoners, and a later timing window. Consequently, the sign of the expected-value contrast
\begin{equation}
\Delta_{\mathrm{EV}}(p)=\operatorname{EV}(C\mid p)-\operatorname{EV}(L)
\end{equation}
cannot be inferred from the intrinsic value of the objective alone.

Two public discussions supply the motivating claims \cite{coach,analyst}. The coach-side argument treats champion properties, first move, and tempo as reasons to enter river; the analyst-side argument treats observed outcomes of teams that end 3--0 in grubs as evidence about objective value. The first set of variables belongs inside the fight probability and terminal-state payoffs. The second is a selected observational regime and does not identify the counterfactual effect of the package. The study therefore asks a narrower statistical question:

\begin{quote}
\small
For a specified grub package, fight payoff, capture branch, and concede-without-fighting payoff, what fight-win probability $p^\star$ satisfies $\Delta_{\mathrm{EV}}(p^\star)=0$?
\end{quote}

The analysis has four layers. Riot mechanics define the reward vector. A competitive-map model supplies an associational conversion from early gold state to map-win probability. A decision model integrates contest terminal states and the concede-without-fighting outside option over fight and capture probabilities. Finally, a strictly pre-outcome ranked pilot tests whether a narrow conditional fight-win probability can be forecast without leaking the realized objective.

\clearpage
\section{Data and statistical methods}

\subsection{Analysis sample}
The unit of analysis is one completed competitive map, not one team-row. The prespecified population is restricted to LCS, LCK, LEC, LPL, and CBLOL. Within those leagues, the raw 2026 Oracle's Elixir source resolves to $n=\textrm{<<N_RAW_2026>>}$ unique reward-era maps, dated <<DATE_MIN>> through <<DATE_MAX>> \cite{oe}. Of these, <<N_EXACT_THREE>> record exactly three total grubs and <<N_FEWER_THREE>> record fewer. Both groups remain in the calibration because the gold-state conversion should not condition on realized objective completion. The primary model excludes <<N_GOLD_MISSING>> maps with non-finite gold@10 and <<N_GOLD_OUTCAP>> maps outside $|G_{10}|\leq 3000$, leaving $n=\textrm{<<N_FIT>>}$ maps across <<N_FIT_LEAGUES>> leagues; no additional maps are lost to missing binary outcomes ($n=\textrm{<<N_OUTCOME_MISSING>>}$). All <<N_GOLD_MISSING>> missing-gold maps are from <<MISSING_GOLD_LEAGUES>>, so that league is absent from the fitted calibration. The joint gold-XP model has $n=\textrm{<<N_JOINT>>}$ complete maps with $|G_{10}|\leq 3000$ and $|X_{10}|\leq 2000$. The caps limit leverage from extreme early states; they are model-design restrictions, not estimates from the grub decision itself.

The single three-grub encounter began in Patch 25.09, when the camp moved to one spawn without respawn \cite{riot2509}. This paper begins its calibration at Patch 26.1 because that patch changed the full-camp reward vector to 90 local gold and 195 XP \cite{riot261}. Thus $n=\textrm{<<N_FIT>>}$ is the analytic sample for the 2026 reward specification, not a count of every map played since the three-grub format began.

\subsection{Target and identification boundary}
Let $Y_i(C)$ and $Y_i(L)$ denote potential map outcomes under contest and concede-without-fighting for the same pre-river state $S_i=s$. The causal target would be
\begin{equation}
\tau(s)=\mathbb E\!\left[Y_i(C)-Y_i(L)\mid S_i=s\right].
\end{equation}
The available map-level data contain neither random assignment nor a sufficiently complete pre-contest state and action log to identify $\tau(s)$. Let $B$ be own-team gold difference immediately before the decision, $A$ the action, and $D$ a contest-fight loss indicator. Two descriptive quantities requested by the motivating discussion are $P(D=1\mid A=C,B<0)$ and $P(A=C,D=1\mid B<0)$; neither is recoverable from the competitive source. The estimand used here is instead the structural threshold $p^\star(B,O,F,K,s_W,s_D;\tilde q)$ implied by a specified terminal-state model and a side-neutral associational calibration function $\tilde q$.

\subsection{Gold-state calibration}
For map $i$, let $Y_i=1$ if blue wins and $G_i$ be blue gold difference at 10 minutes. The primary model is
\begin{align}
Y_i &\sim \operatorname{Bernoulli}(\pi_i),\\
\operatorname{logit}(\pi_i) &= \alpha+\beta G_i.
\end{align}
The nearly unpenalized logistic fit gives $\hat\alpha=<<BETA0>>$ and $\hat\beta=<<BETA1>>$ per gold. The direct fit predicts blue-side win probability,
\begin{equation}
\hat q_B(g)=\left[1+\exp\{-(\hat\alpha+\hat\beta g)\}\right]^{-1}.
\end{equation}
Applying $\hat q_B$ directly to a generic ``own-team'' gold state would import the fitted blue-side intercept. Scenario valuation therefore averages the blue- and red-side representations of the same own-team lead,
\begin{equation}
\tilde q(g)=\tfrac12\{\hat q_B(g)+1-\hat q_B(-g)\}.
\end{equation}
This gives $\tilde q(0)=0.5$ while retaining the fitted gold slope. For a mechanical or scenario increment $d$ evaluated from baseline $g_0$, the reported map-win conversion is
\begin{equation}
\widehat{\Delta}_{q}(d;g_0)
=100\{\tilde q(g_0+d)-\tilde q(g_0)\}\quad\text{pp}.
\end{equation}
Sampling intervals use the observed-information covariance matrix and the delta method under a map-level independence working assumption. They quantify uncertainty in the fitted association only; they do not account for repeated teams, series, patches, or leagues, do not cover uncertainty in the game-mechanical assumptions, and do not identify a causal grub effect.

\subsection{Out-of-fold diagnostics}
Model performance was evaluated with <<CV_FOLDS>>-fold stratified map-level cross-validation using a fixed split seed. Out-of-fold area under the ROC curve was <<CV_AUC>>, Brier score was <<CV_BRIER>> versus <<CV_NULL_BRIER>> for the prevalence-only predictor, and log loss was <<CV_LOGLOSS>>. Regressing outcomes on the out-of-fold prediction logits gave calibration intercept <<CV_CAL_INTERCEPT>> and slope <<CV_CAL_SLOPE>>. The folds are not grouped by team or series, so these are descriptive diagnostics, not transport estimates. They evaluate the univariate gold-state conversion, not the unobserved pre-fight probability $p$.

\subsection{Joint gold-XP specification}
The XP-inclusive scenario uses
\begin{equation}
\operatorname{logit}(\pi_i)
=\alpha_J+\beta_G G_i+\beta_X X_i,
\end{equation}
with $\hat\alpha_J=<<BETA_J0>>$, $\hat\beta_G=<<BETA_JG>>$ per gold, and $\hat\beta_X=<<BETA_JX>>$ per XP. This model converts the joint 90g-plus-195-XP state to a common fitted probability scale. Because local XP eligibility and subsequent conversion are state dependent, the joint result is reported separately from the direct-cash specification.

\subsection{Mechanical and scenario inputs}
Three Voidgrubs form a single spawn group in the baron pit: 90g killer-local gold (30g each; no global share), 195 listed XP shared equally among living allies within 2000 units, and one Touch of the Void stack per grub killed \cite{riot261,wiki,wikitouch,wikihunger}. Touch is a team-wide buff. Basic attacks against structures apply a four-second true-damage burn that ticks every 0.5 seconds; later structure attacks refresh that duration rather than stacking overlapping full burn packets, and the burn applies only when the triggering instance can damage the structure \cite{wikitouch}. Patch 26.11 melee ticks are 4 / 12 / 16 true damage at one / two / three stacks (ranged 2 / 6 / 8); a complete three-stack melee cycle is therefore $8\times16=128$ true damage \cite{riot2611,wikitouch}. At three stacks the team also gains Hunger of the Void, which summons one allied Voidmite while in combat with a targetable enemy structure; mite attacks apply and refresh the summoner's Touch and do not disable turret Reinforced Armor \cite{wikihunger}.

The pre-26.11 brief-Touch ceiling assumes eight seconds of structure access: 192 true damage, or $192/900$ of the \emph{first} outer plate (outer turrets: 9000 HP; plates claimed at 10/25/45/70/100\% missing HP under Patch 26.1). Valued at 120g per plate, this is 25.6g of undiscounted plate progress, giving $O=115.6$ rather than 90 \cite{riot261,wiki,wikiturret}. The post-26.11 analogue uses 256 true damage and 34.13g of progress ($O=124.13$) \cite{riot2611}. These are upper-bound Touch-burn progress equivalents, not guaranteed immediate gold; Hunger mite summons are omitted from $O$.

Leave-farm gold is locked to the Void Grub clock. Grubs are alive from 08:00 to 14:45, and the article assumes outer turrets are still standing, so any plate gold is outer plating at 120g local per plate \cite{wikiturret}. Before 14:00, every wave is $3\times20+3\times14=102$g of melee/caster gold, with a siege/cannon minion every third wave whose bounty is $50+\lfloor t/90\rfloor$ gold \cite{wiki,wikiminion}. At the 10:00 mid-grub reference clock the cannon is worth 56g, so
\begin{equation}
\mathbb{E}[\text{wave}\mid t=10{:}00]
=102+56/3=\frac{362}{3}=120.\overline{6}\text{g},\qquad
3(62)+3(31)+75/3=304\text{ XP}.
\end{equation}
A non-cannon wave is exactly 102g / 279 XP; a 10:00 cannon wave is exactly 158g / 354 XP. (The wiki's 0:30 average of $102+50/3=118.\overline{6}$g is the spawn-row identity, not the grub-window unit.) One outer plate remains exactly 120g; plating is no longer removed at 14:00 \cite{riot261,wikiturret}. These constants generate the five $F$ values used in the sensitivity grid:
$F\in\{0,\,120.\overline{6},\,241.\overline{3},\,361.\overline{3},\,482\}$.

\subsection{Worked siege example as a mechanics check}
To make the Touch term concrete, consider level-15 Zaahen striking a 5000-HP, 60-armor structure, matching a current inner turret's nominal state \cite{riot261,wikiturret}. Wiki base statistics at level 15 give 116.06 base AD, 0.625 base attack speed, and $+33.16\%$ bonus attack speed from levels. The combat state adds Doran's Blade, Trinity Force, Hexdrinker, Mercury's Treads, Sundered Sky, and Caulfield's Warhammer ($+136$ flat AD and Trinity's further $+30\%$ attack speed), for $116.06+136=<<SIEGE_TOTAL_AD>>$ total AD and $0.625\times(1+0.3316+0.30)=<<SIEGE_AS>>$ attacks per second. Armor is held fixed; Bulwark, backdoor protection, regeneration, minions, allied attackers, Hunger mites, Cultivation of War / Determination (champion damage), The Darkin Glaive double-strike and attack resets, latency, and animation timing are excluded. Structure damage uses autoattacks and Trinity Spellblade only; Sundered Sky's Lightshield Strike does not apply to structures. This is a controlled mechanics example, not a live turret forecast.

For non-negative armor $R$, the physical-damage multiplier is \cite{wikiarmor}
\begin{equation}
m_R=\frac{100}{100+R}=\frac{100}{160}=<<SIEGE_ARMOR_MULT>>.
\end{equation}
An ordinary attack therefore deals $<<SIEGE_TOTAL_AD>>m_R=<<SIEGE_NORMAL_DMG>>$ physical damage. Trinity Force's Spellblade adds $200\%$ \emph{base} AD, not $100\%$, so a proc contributes $(2\times116.06)m_R=<<SIEGE_SPELL_DMG>>$ more and triggers against structures \cite{wikitrinity}. The attack interval is $1/<<SIEGE_AS>>=<<SIEGE_ATTACK_PERIOD>>$ seconds; against a 1.5-second post-proc cooldown, perfect ability availability permits a proc on every <<SIEGE_PROC_EVERY>>nd attack.

Patch 26.11 sets melee Touch to 4, 12, and 16 true damage per 0.5-second tick at one, two, and three stacks \cite{riot2611,wikitouch}. The model pre-primes the first Spellblade attack; Touch first ticks after 0.5 seconds and remains active through attack refreshes (one maintained burn, not stacked 128-damage packets). A discrete event calculation gives:

\begin{center}
\scriptsize
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}rrrrrrrrr@{}}
\toprule
\textbf{Touch stacks} & \textbf{Dmg/tick} & \textbf{True DPS} & \textbf{Time (s)} & \textbf{Attacks} & \textbf{Touch true} & \textbf{Zaahen (no Touch)} & \textbf{Touch share} & \textbf{Sec.\ saved} \\
\midrule
<<SIEGE_ROWS>>
\bottomrule
\end{tabular*}
\vspace{0.22em}

\fineprint{\textbf{Level-15 Zaahen siege clock.} Deterministic single-attacker clock for the stated idealized structure under the listed build. ``Touch true'' is the flat true damage that actually lands before the structure dies; ``Zaahen (no Touch)'' is the full 5000 HP that Zaahen alone must deal with zero stacks. ``Touch share'' is Touch true $\div$ Zaahen (no Touch). Hunger mites and Determination omitted.}
\end{center}

At three stacks, level-15 Zaahen drops the structure at <<SIEGE_T3>> seconds after <<SIEGE_A3>> attacks, versus <<SIEGE_T0>> seconds and <<SIEGE_A0>> attacks without Touch: <<SIEGE_TSAVE>> seconds and <<SIEGE_ASAVE>> attacks saved, a <<SIEGE_TREDUCTION>>\% reduction in elapsed time under these assumptions. Three-stack Touch contributes <<SIEGE_TOUCH_DMG3>> true damage against Zaahen's <<SIEGE_ZAAHEN0>> without Touch (<<SIEGE_TOUCH_SHARE3>>\% of the structure). The often-quoted 128 damage is one complete four-second three-stack burn cycle ($8\times16$), not 128 damage on every attack. This example validates the burn accounting but is not inserted into the fight threshold or treated as observed map-win value; the main model continues to value only explicitly bounded windows of structure access.

\subsection{Terminal states and break-even probability}
Let $B$ denote own-team gold difference immediately before commitment, $O$ the grub package in gold-equivalent units, $F$ the gold preserved by conceding without fighting, and $K=600$ the symmetric two-kill swing. Let $S=1$ indicate that the focal contesting team secures the camp; in the exhaustive two-team outcome model, $S=0$ means the opponent secures it. The four contest cells and the outside-option state are
\begin{equation}
g_{W1}=B+K+O,\quad g_{W0}=B+K-O,\quad
g_{D1}=B-K+O,\quad g_{D0}=B-K-O,\quad
g_L=B+F-O,
\end{equation}
where $W$ and $D$ denote fight win and fight loss. Define $s_W=P(S=1\mid W)$ and $s_D=P(S=1\mid D)$. The conditional contest values are
\begin{align}
Q_W(s_W)&=s_W\tilde q(g_{W1})+(1-s_W)\tilde q(g_{W0}),\\
Q_D(s_D)&=s_D\tilde q(g_{D1})+(1-s_D)\tilde q(g_{D0}).
\end{align}
For subjective fight-win probability $p$,
\begin{align}
\operatorname{EV}_{C}(p,s_W,s_D)&=pQ_W(s_W)+(1-p)Q_D(s_D),\\
\operatorname{EV}_{L}&=\tilde q(g_L),\\
\Delta_{\mathrm{EV}}(p,s_W,s_D)&=100\{\operatorname{EV}_{C}-\operatorname{EV}_{L}\}\quad\text{pp}.
\end{align}
Whenever $Q_W(s_W)>Q_D(s_D)$ and $Q_D(s_D)\leq\tilde q(g_L)\leq Q_W(s_W)$, the unique break-even probability in $[0,1]$ is
\begin{equation}
p^\star(s_W,s_D)
=\frac{\tilde q(g_L)-Q_D(s_D)}{Q_W(s_W)-Q_D(s_D)}.
\end{equation}
$p^\star$ is the break-even fight-win probability: the minimum estimated chance of winning the river fight at which contest and concede without fighting have equal expected map-win value. For $p<p^\star$, concede without fighting has the higher modelled value; for $p>p^\star$, contest has the higher modelled value, conditional on the stated terminal states. The model treats action $C$ as committing to a decisive fight lottery; non-decisive contacts are outside this node (the ranked pilot finds only about one fifth of first-grub episodes become decisive kill exchanges).
The main sensitivity grid evaluates $p\in[0,1]$ at $B=0$, both reward packages ($O=90$ and $O=115.6$), five concede-without-fighting states ($F\in\{0,120.\overline{6},241.\overline{3},361.\overline{3},482\}$), and all four corners $(s_W,s_D)\in\{0,1\}^2$. A separate deficit analysis evaluates $B\in\{0,-500,-1000,-2000\}$ in the reference branch. The corners bound the continuous capture-probability square because $p^\star$ is monotone in each capture probability over the displayed states. The reference branch $(1,0)$ means secure if the fight is won; opponent secures if the fight is lost.

\begin{center}
\small
\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{1.45cm}>{\raggedleft\arraybackslash}p{3.0cm}Y@{}}
\toprule
\textbf{Symbol} & \textbf{Values} & \textbf{Role in the decision model} \\
\midrule
$O$ & 90g; 115.6g & Own-team objective package; cash only or cash plus brief-Touch ceiling (Touch-burn only; Hunger omitted). \\
$B$ & 0g; $-500$g; $-1000$g; $-2000$g & Own-team gold immediately before commitment; main grid uses parity. \\
$K$ & 600g & Net fight swing on a contest win; $-K$ on a contest loss. \\
$F$ & 0g; $120.\overline{6}$g; $241.\overline{3}$g; $361.\overline{3}$g; 482g & Own-team farm preserved by conceding without fighting (grub-era waves + outer plate). \\
$p$ & $[0,1]$ & Subjective probability of winning the river fight. \\
$s_W,s_D$ & $[0,1]^2$ & Camp-secure probabilities conditional on fight win and fight loss. \\
$\tilde q(g)$ & side-neutral fitted probability & Associational conversion from own-team terminal gold to map-win probability. \\
\bottomrule
\end{tabularx}
\vspace{0.22em}

\fineprint{\textbf{Table 1. Prespecified sensitivity parameters.} $B$, $O$, $K$, and $F$ define terminal states; $p$, $s_W$, and $s_D$ are varied rather than selected.}
\end{center}

\section{Resolved contest payoffs and ex ante expected value}
All effects in this section are measured relative to one explicit outside option: concede the camp without fighting and preserve two average early waves. That action is defined as $0$ pp. The reference contest state is evaluated at gold parity with the 115.6g brief-Touch ceiling, a symmetric $\pm600$g fight swing, and camp secure if the fight is won (opponent secures if the fight is lost). Figure~1 reports the leave-relative payoffs that enter the reference mixture.

\begin{center}
\includegraphics[width=0.99\linewidth]{<<FIG1>>}\\[-0.2em]
\fineprint{\textbf{Figure 1. Resolved payoffs and pre-fight expected value.} Panel A reports each terminal contest outcome relative to conceding without fighting. Open points are fight losses; filled points are fight wins. Panel B mixes the reference win-and-secure and loss-and-opponent-secure branches over the pre-fight fight-win probability $p$. These are structural scenario values, not observed cell frequencies.}
\end{center}

\subsection{The fight dominates the camp reward}
Securing the camp changes either fight outcome by $<<CAMP_OWNERSHIP_PP>>$ pp. Changing the fight result from loss to win while holding camp ownership fixed changes fitted map-win value by $<<FIGHT_RESULT_PP>>$ pp, approximately $<<FIGHT_TO_CAMP_RATIO>>$ times as much. Under the reference calibration, fight-result contrast dominates camp-ownership contrast.

For the reference branch, contest-win-and-secure is $<<PD_VS_LEAVE_WIN_YES>>$ pp relative to the outside option and contest-loss-and-opponent-secure is $<<PD_VS_LEAVE_LOSE_NO>>$ pp. The ex ante contest edge is
\begin{equation}
\Delta_{\mathrm{EV}}(p)
=p(<<PD_VS_LEAVE_WIN_YES>>)+(1-p)(<<PD_VS_LEAVE_LOSE_NO>>)
=<<REFUSAL_SLOPE>>p<<PD_VS_LEAVE_LOSE_NO>>\quad\text{pp}.
\end{equation}
Every 10-point increase in fight-win probability adds $<<EV_PER_10>>$ map-win pp. At $p=50\%$, contest is $<<EV_AT_50>>$ pp worse than conceding without fighting. The edge reaches zero only at $p^\star=<<REFUSAL_PCT>>\%$ and is $<<EV_AT_70>>$ pp at $p=70\%$.

\begin{center}
\small
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Pre-fight fight-win probability} & \textbf{Contest minus concede} & \textbf{Decision} \\
\midrule
$50.0\%$ & $<<EV_AT_50>>$ pp & Concede \\
$<<REFUSAL_PCT>>\%$ & $0.00$ pp & Indifferent \\
$70.0\%$ & $<<EV_AT_70>>$ pp & Contest \\
\bottomrule
\end{tabular}
\par\vspace{0.14em}
\parbox{0.88\linewidth}{\centering\fineprint{\textbf{Table 2. Exact reference-branch decision points.} Values use the same complete terminal states as Figure~1; rounding is to two decimal places.}}
\end{center}

\section{Opportunity cost and required fight-win probability}
The break-even probability $p^\star$ is the minimum fight-win probability that makes contest and concede equal under a specified state. It is a hurdle, not a forecast. Figure~2 varies the farm preserved by conceding and displays the reference secure-if-win threshold as a filled point. The band is the range over the four deterministic capture corners. Those corners are sensitivity bounds, not four strategies and not row--column choices.

\begin{center}
\includegraphics[width=0.98\linewidth]{<<FIG2>>}\\[-0.2em]
\fineprint{\textbf{Figure 2. Break-even probability by opportunity cost.} Filled points use the brief-Touch ceiling ($O=115.6$g under the corrected first-plate HP) and the secure-if-win branch. Hollow points use cash only. Bands span all deterministic values of $(s_W,s_D)$. The fight swing is fixed at $\pm600$g. ``No farm recovered'' means conceding preserves 0g; it does not mean that the contesting team always secures the camp.}
\end{center}

The reference threshold rises from $<<TOUCH_FIRST_PCT>>\%$ when conceding recovers no farm to $<<TOUCH_LAST_PCT>>\%$ when it preserves three waves and a plate. At the two-wave outside option, $p^\star=<<REFUSAL_PCT>>\%$. This is a structural indifference point under fixed scenario knobs $(K,F,O,s_W,s_D)$; the main sensitivity is the $F$-ladder (and related capture-branch bands), not a map-iid sampling interval on $\widehat\beta$ alone. Replacing the brief-Touch ceiling with cash only changes the displayed reference thresholds by about one percentage point or less. The dominant sensitivity is what the team can recover by conceding without fighting, not the disputed gold-equivalent assigned to Touch.

\subsection{A pre-fight probability can be forecast, but the present pilot is not decision-grade}
For operational use, $p^\star$ must be compared with a forecast constructed only from information available before commitment. Define
\begin{equation}
\widehat p_{\mathrm{dec}}(x)
=\widehat P(\text{focal team wins}\mid\text{a decisive local kill exchange occurs},X=x).
\end{equation}
The cached high-elo ranked pilot contains <<FIGHT_EPISODES>> deduplicated first-grub episodes under a strict cutoff: the predictor frame is at least 30 seconds before the first grub, while the outcome window begins 20 seconds before it. The first contiguous grub cluster is capped at 90 seconds. At the prespecified 2200-unit radius, only <<FIGHT_N>> episodes (<<FIGHT_DECISIVE_RATE>>\%) produce a non-tied local kill exchange. Each contributes two mirrored team perspectives, kept together in grouped 10-fold validation.

The transparent gold-only model is
\begin{equation}
\operatorname{logit}(\widehat p_{\mathrm{dec}})=<<FIGHT_BETA>>\,B/1000,
\end{equation}
with intercept fixed at zero by team symmetry. Its out-of-fold AUC is <<FIGHT_AUC>>, Brier score <<FIGHT_BRIER>> versus a <<FIGHT_NULL_BRIER>> null, and log-loss performance remains weak. Adding the pre-cutoff local player-count difference lowers AUC to <<FIGHT_PRESENCE_AUC>>. Figure~3 shows both the label sensitivity and the narrow conditional forecast.

\begin{center}
\includegraphics[width=0.99\linewidth]{<<FIG3>>}\\[-0.2em]
\fineprint{\textbf{Figure 3. Ranked pilot qualification.} Panel A shows that the number of labelled episodes changes materially with the pit radius. Panel B reports the gold-only forecast conditional on a decisive local kill exchange. This forecast is not directly comparable with the unconditional binary $p$ in the structural threshold: most observed episodes are non-decisive, the exact expanded-cache Diamond/Master+ manifest is unavailable, and the model omits health, items, cooldowns, summoners, vision, lane priority, and Smite access.}
\end{center}

\begin{center}
\small
\begin{tabular}{@{}lr@{}}
\toprule
\textbf{Pre-fight gold state} & \textbf{$\widehat p_{\mathrm{dec}}$} \\
\midrule
<<FIGHT_GOLD_ROWS>>
\bottomrule
\end{tabular}
\par\vspace{0.14em}
\parbox{0.88\linewidth}{\centering\fineprint{\textbf{Table 3. Gold-only conditional fight forecast.} These probabilities describe only decisive kill-producing ranked episodes. They are reported to demonstrate estimability, not to authorize a professional contest call.}}
\end{center}

The pilot shows weak gold-only discrimination among decisive kill-producing episodes (AUC <<FIGHT_AUC>>). It does not authorize substituting $\widehat p_{\mathrm{dec}}$ for the unconditional $p$ in $p^\star$. A decision-grade forecast would need a preserved rank/platform/anchor manifest, audited event labels, combat-state features, and separate estimation of the probability that an episode becomes decisive. Until then, $p^\star$ under stated scenario knobs is the quantitative result; $\widehat p_{\mathrm{dec}}$ is a method prototype only.

\section{Model estimates and inferential scope}
Table 4 separates fitted probability contrasts from mechanical inputs. The working intervals propagate estimated logistic-model covariance through the probability-difference transformation. They do not incorporate repeated-map dependence or uncertainty in the gold-equivalent assigned to Touch, the two-kill sensitivity, capture probabilities, or the concede-without-fighting specification.

\begin{center}
\small
{\setlength{\tabcolsep}{3.5pt}
\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{3.65cm}>{\raggedleft\arraybackslash}p{3.35cm}>{\raggedleft\arraybackslash}p{1.10cm}Y@{}}
\toprule
\textbf{Contrast} & \textbf{Estimate [working 95\% CI], pp} & \textbf{$n$} & \textbf{Statistical status} \\
\midrule
90g at $G_{10}=0$ & <<PP_GOLD>> [<<CI_CASH_LO>>, <<CI_CASH_HI>>] & <<N_FIT>> & Univariate gold fit. \\
90g + pre-26.11 brief Touch & <<PP_PREF>> [<<CI_PREF_LO>>, <<CI_PREF_HI>>] & <<N_FIT>> & Gold-scenario transform. \\
90g + post-26.11 brief Touch & <<PP_PREF_POST>> [<<CI_PREF_POST_LO>>, <<CI_PREF_POST_HI>>] & <<N_FIT>> & Gold-scenario transform. \\
90g + 195 XP at $(0,0)$ & <<PP_JOINT>> [<<CI_JOINT_LO>>, <<CI_JOINT_HI>>] & <<N_JOINT>> & Joint gold-XP fit. \\
\bottomrule
\end{tabularx}}
\vspace{0.25em}

\fineprint{\textbf{Table 4. Side-neutral fitted probability contrasts.} All estimates are percentage-point differences in fitted map-win probability. Intervals use the map-level independence working covariance and therefore exclude repeated-team, series, league, patch, and scenario-parameter dependence.}
\end{center}

\subsection{Selection induced by the realized 3--0 regime}
Let $A=1$ denote the event that blue ends the three-grub sequence 3--0. Conditioning on $A$ selects maps in which blue has already demonstrated some combination of lane priority, jungle position, vision, first move, smite access, and execution. These variables also affect $Y$. Therefore
\begin{equation}
\Pr(Y=1\mid A=1,G_{10},X_{10})-\Pr(Y=1\mid A=0,G_{10},X_{10})
\end{equation}
is not the effect of assigning the grub package while holding the pre-contest state fixed. No 3--0 conditional coefficient is used in the decision model: without a pre-contest action model and a defensible dependence structure, attaching a precise interval to that selected contrast would create false inferential precision.

\subsection{Threats to validity}
\begin{itemize}
\item \textbf{Counterfactual identification.} Each map records only the action actually taken. It does not reveal what would have happened to the same teams, in the same pre-fight state, under the unchosen action. The analysis therefore does not estimate the causal effect of contesting. It calculates the fight-win threshold implied by explicit terminal payoffs and separately examines whether pre-fight information can forecast a narrowly defined ranked fight outcome.
\item \textbf{Proxy timing.} Gold@10 is an early-state proxy for the professional conversion model. The ranked pilot uses the final participant frame at least 30 seconds before the first grub and begins outcome measurement 20 seconds before it, but it cannot recover the team's subjective probability or professional combat state.
\item \textbf{Outcome-tree restriction.} The $\pm600$g swing excludes assists, shutdowns, deaths with non-standard bounties, objective steals, recalls, plates, and subsequent objective sequences.
\item \textbf{Dependence and parameter uncertainty.} Working intervals treat maps as independent despite repeated teams, series, leagues, and patches; ranked anchors also contribute repeated matches. The reported $p^\star$ interval propagates only this working logistic covariance. No interval is propagated for uncertainty in $F$, $O$, $K$, $s_W$, or $s_D$.
\item \textbf{Coverage and transportability.} The source cohort is deliberately restricted to LCS, LCK, LEC, LPL, and CBLOL; regional, academy, challenger, and developmental leagues are excluded. The fitted calibration covers only <<N_FIT_LEAGUES>> leagues because all <<N_GOLD_MISSING>> LPL maps lack gold@10. This concentrated missingness is not missing at random and limits geographic transportability. The equal-quota Diamond/Master+ anchor audit is analyzed separately and is not pooled with the competitive coefficient.
\item \textbf{Model form.} A linear log-odds relation is assumed within the truncation window. Calibration diagnostics and nonlinear alternatives would be required before treating $\tilde q(g)$ as a general-purpose win-probability model.
\end{itemize}

The study uses $\tilde q$ as a monotone conversion scale for bounded scenario comparisons. It does not claim that gold difference alone is a sufficient model of competitive map outcomes.

\clearpage
\section{D{}iamond and Master+ solo-queue calibration audit (non-reproducible appendix)}

\subsection{Sampling frame}
The external audit targeted <<SQ_ATTEMPTED>> ranked solo-queue match IDs, matching the order of magnitude of the <<N_MAPS>>-map competitive source cohort. It retained <<SQ_USABLE>> maps with at least one parseable HORDE event and excluded <<SQ_ATTRITION>> attempted IDs without a usable event record. The audit is split into disjoint current-rank anchor buckets: Diamond ($n=<<SQ_DIAMOND_N>>$) and Master+ ($n=<<SQ_MASTER_N>>$), with Master+ taking precedence when the same match is reached from both buckets. Ranks below Diamond are excluded. The pooled row is an approximately equal-quota mixture of rank-by-platform anchor strata, not a prevalence-weighted estimate of the Diamond+ population.

Sampling covers NA1, KR, EUW1, EUN1, and BR1. These correspond to the public Riot platforms available for North America, Korea, Europe, and Brazil. Mainland-China platforms are not exposed in the public routing list \cite{riotapi}; consequently, the ranked audit cannot supply an LPL-region analogue. The collector requested each anchor's most recent queue-420 matches at the 19 July 2026 collection snapshot. Timeline payloads do not preserve match-creation dates, so explicit patch filtering was unavailable. Anchor rank is current at collection time and does not establish that all ten players held that rank at the historical match time. The original collector retained only aggregates, not the sampled match-ID and anchor manifest; the audit is therefore not independently reconstructible. It is retained as a transport check: whether Diamond+ ranked gold@10 conversion can stand in for the competitive coefficient when computing $p^\star$.

\begin{center}
\footnotesize
{\setlength{\tabcolsep}{4.5pt}
\begin{tabularx}{0.99\linewidth}{@{}>{\raggedright\arraybackslash}p{3.10cm}>{\raggedleft\arraybackslash}p{1.35cm}>{\raggedleft\arraybackslash}p{1.30cm}>{\raggedleft\arraybackslash}p{3.10cm}>{\raggedleft\arraybackslash}Y@{}}
\toprule
\textbf{Population} & \textbf{Maps $n$} & \textbf{Fit $n$} & \textbf{$\Delta_{90g}$ pp [working 95\% CI]} & \textbf{AUC} \\
\midrule
<<CONTROL_POPULATION_ROWS>>
\bottomrule
\end{tabularx}}
\vspace{0.18em}

\fineprint{\textbf{Table 5. Competitive and solo-queue calibration samples.} Maps $n$ is the source cohort for competitive play and usable HORDE maps for ranked play. Intervals and AUCs are map-level working quantities; the ranked rows are anchor-sample descriptions, not population estimates.}
\end{center}

\subsection{Matched $p^\star$ comparison}
Both samples use the same side-neutral gold@10 specification, $|G_{10}|\leq3000$, 90g cash state, symmetric $\pm600$g fight swing, and outside-option grid. The decision-relevant quantity is not the raw 90g gold contrast itself---competitive exceeds the pooled equal-quota anchor by $<<PRO_MINUS_SQ_90G>>$ pp on that scale---but the induced break-even fight-win probability $p^\star$ under identical terminal-state branches. Across the five cash-only concede-without-fighting branches in Table~6, the absolute gap $|\Delta p^\star|$ is at most $<<SQ_MAX_ABS_DPSTAR>>$~pp (anchor audit minus competitive). Discrimination is also close: competitive AUC <<SQ_PRO_AUC>> versus Diamond <<SQ_DIAMOND_AUC>>, Master+ <<SQ_MASTER_AUC>>, and pooled <<SQ_AUC>>.

\begin{center}
\small
{\setlength{\tabcolsep}{4.5pt}
\begin{tabularx}{0.99\linewidth}{@{}Y>{\raggedleft\arraybackslash}p{2.15cm}>{\raggedleft\arraybackslash}p{2.15cm}>{\raggedleft\arraybackslash}p{1.75cm}@{}}
\toprule
\textbf{Concede-without-fight branch} & \textbf{Competitive $p^\star$} & \textbf{Anchor audit $p^\star$} & \textbf{$\Delta p^\star$} \\
\midrule
<<CONTROL_THRESHOLD_ROWS>>
\bottomrule
\end{tabularx}}
\vspace{0.18em}

\fineprint{\textbf{Table 6. Side-neutral cash-only reference-branch thresholds under a common state specification.} $\Delta p^\star$ is anchor audit minus competitive. Differences isolate the fitted gold-to-win conversion; they are neither empirical fight-win rates nor population estimates.}
\end{center}

\subsection{Interpretation: transport for simulation, and a competitive-specificity check}
Within this audit, Diamond+ solo queue is a usable stand-in for the competitive gold@10 conversion that enters $p^\star$: swapping the competitive coefficient for the equal-quota Diamond/Master+ fit moves the cash-only threshold by less than one tenth of a map-win percentage point on every displayed outside option. That agreement licenses large-scale ranked timeline simulation for structural sensitivity work---state grids, Touch ceilings, leave-farm ladders, and fight-swing scenarios---where competitive map volume alone would be too thin, provided the simulation inherits the same side-neutral gold@10 specification and the same terminal-state tree.

The same agreement has a sharper substantive reading. If professional Void Grub play imposed a materially different early-gold-to-map-win conversion around this objective, the competitive $p^\star$ ladder would separate from the Diamond+ ladder under a common state grid. It does not. On the margin that determines the break-even fight hurdle, 2026 competitive maps still look like high-elo solo queue: the audit supplies no evidence of a distinct competitive-tier grub valuation once cash, leave-farm, and $\pm600$g fight outcomes are held fixed. The conversion scale used to price those fights is, to within $<<SQ_MAX_ABS_DPSTAR>>$~pp in $p^\star$, interchangeable with Diamond+.

No cross-population hypothesis test is reported: maps repeat teams or anchors, and the retained ranked aggregate does not permit the clustered resampling needed for defensible population inference. The transport claim is therefore descriptive agreement under a shared specification, not a formal equivalence test.

\clearpage
\section{D{}iscussion}

\subsection{Comparative statics}\leavevmode\par
For fixed $O$, $K$, $s_W$, and $s_D$, the concede-without-fighting payoff enters only through $\tilde q(F-O)$. Differentiating the break-even expression gives
\begin{equation}
\frac{\partial p^\star}{\partial F}
=\frac{\tilde q'(F-O)}{Q_W(s_W)-Q_D(s_D)}>0.
\end{equation}
Thus any additional farm preserved by conceding strictly increases the fight-win probability required to justify contest within this model. This is the dominant relationship in Figure 2. In the reference branch, increasing $O$ from 90g to the 115.6g brief-Touch ceiling moves the estimated threshold by about one percentage point or less across the five displayed outside options.

\subsection{Resolution of the motivating claims}
Champion statistics, composition, item states, first move, vision, cooldowns, and smite access can alter $p$, $K$, $F$, $O$, $s_W$, $s_D$, and potentially the conversion function $\tilde q$ itself. Naming those variables does not determine the sign of $\Delta_{\mathrm{EV}}$; they must be quantified in the pre-fight state. Absent numerical values for those inputs, the coach-side argument does not determine the sign of $\Delta_{\mathrm{EV}}$.

The data-side claim has a different limitation. More observations can reduce sampling uncertainty in $\tilde q$, but sample size does not repair selection into the realized 3--0 regime or dependence among repeated teams. A conditional 3--0 association is therefore not substituted for the mechanical package value.

\subsection{Decision use and required extension}
The present study supplies $p^\star$ conditional on a state specification and a leakage-controlled prototype for $\widehat p_{\mathrm{dec}}$ in ranked play. The prototype is conditional on a decisive kill-producing exchange and is too weak and incomplete to substitute for the team-specific $p$, $s_W$, or $s_D$. Operational use requires a calibrated model based only on information available before commitment to river. If these inputs are distributions rather than point estimates, the coherent comparison is
\begin{equation}
\mathbb{E}\!\left[\Delta_{\mathrm{EV}}(p,s_W,s_D)\mid\mathcal I_{\mathrm{pre}}\right],
\end{equation}
where $\mathcal I_{\mathrm{pre}}$ contains only pre-decision information. A subsequent validation study should estimate this quantity from timestamped competitive event data and report calibration, discrimination, decision-curve performance, and sensitivity to the spatial and temporal fight definition.

\section{Conclusion}
Under the reference state---conceding without fighting preserves two waves; contesting uses the brief-Touch package and secures the camp if the fight is won---a fight at $p=50\%$ is $<<EV_AT_50>>$ map-win pp worse than conceding. Indifference requires $p^\star=<<REFUSAL_PCT>>\%$. When conceding without fighting preserves three waves and one outer plate, that threshold rises to $<<TOUCH_LAST_PCT>>\%$.

Main quantitative contrasts:
\begin{itemize}[leftmargin=1.2em,itemsep=0.28em,topsep=0.20em]
\item \textbf{Fight outcome dominates camp ownership.} Winning versus losing the river fight swings about $<<FIGHT_RESULT_PP>>$ map-win pp; who secures the camp after the fight swings only about $<<CAMP_OWNERSHIP_PP>>$ pp.
\item \textbf{Leave-farm opportunity cost binds more than the Touch equivalent.} Each extra wave or outer plate preserved by conceding without fighting raises required $p^\star$ by roughly nine to ten points. Switching from cash-only to the brief-Touch ceiling moves thresholds by about one point.
\item \textbf{Local grub gold does not offset lost early waves.} Three Void Grubs pay 90g locally. One grub-era average wave is already $120.\overline{6}$g; two are $241.\overline{3}$g. Listed cash is below one missed average wave and below a 100g jungle-camp proxy before counting Touch; rotating laners who drop waves are usually worse off on gold alone.
\end{itemize}

\begin{center}
\small
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Pre-contest gold state $B$} & \textbf{Required fight win $p^\star$} & \textbf{$1-p^\star$} \\
\midrule
<<DEFICIT_ROWS>>
\bottomrule
\end{tabular}
\end{center}

Lower $p^\star$ when behind follows from the fitted conversion curve; it is not evidence that trailing teams win more river fights. How often trailing teams contest and lose remains unidentified: public sources do not reliably record the choice to contest the river objective.

\begin{thebibliography}{99}
\scriptsize
\setlength{\parskip}{0pt}
\setlength{\itemsep}{0pt}
\setlength{\parsep}{0pt}
\bibitem{coach} LS. \textit{Heated Argument w/ Tier2 Coach, Data, Analysis}. Discussion transcript, 17 July 2026.
\bibitem{analyst} LS. \textit{Void Grubs Analysis vs a Twitter Data Analyst}. Discussion transcript, 16 July 2026.
\bibitem{riot2509} Riot Games. \href{https://www.leagueoflegends.com/en-us/news/game-updates/patch-25-09-notes/}{\textit{Patch 25.09 Notes}}. 2025.
\bibitem{riot261} Riot Games. \href{https://www.leagueoflegends.com/en-us/news/game-updates/patch-26-1-notes/}{\textit{Patch 26.1 Notes}}. 2026.
\bibitem{riot2611} Riot Games. \href{https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-11-notes/}{\textit{Patch 26.11 Notes}}. 2026.
\bibitem{wiki} League of Legends Wiki contributors. \href{https://wiki.leagueoflegends.com/en-us/Voidgrub_camp}{\textit{Voidgrub camp}}. Accessed 19 July 2026.
\bibitem{wikiminion} League of Legends Wiki contributors. \href{https://wiki.leagueoflegends.com/en-us/Minion}{\textit{Minion}}. Accessed 20 July 2026.
\bibitem{wikitouch} League of Legends Wiki contributors. \href{https://wiki.leagueoflegends.com/en-us/Touch_of_the_Void}{\textit{Touch of the Void}}. Accessed 20 July 2026.
\bibitem{wikihunger} League of Legends Wiki contributors. \href{https://wiki.leagueoflegends.com/en-us/Hunger_of_the_Void}{\textit{Hunger of the Void}}. Accessed 20 July 2026.
\bibitem{wikiarmor} League of Legends Wiki contributors. \href{https://wiki.leagueoflegends.com/en-us/Armor}{\textit{Armor}}. Accessed 19 July 2026.
\bibitem{wikiturret} League of Legends Wiki contributors. \href{https://wiki.leagueoflegends.com/en-us/Turret}{\textit{Turret}}. Accessed 20 July 2026.
\bibitem{wikitrinity} League of Legends Wiki contributors. \href{https://wiki.leagueoflegends.com/en-us/Trinity_Force}{\textit{Trinity Force}}. Accessed 19 July 2026.
\bibitem{oe} Oracle's Elixir. \textit{2026 competitive match data}. Accessed 18 July 2026.
\bibitem{riotapi} Riot Games. \textit{League of Legends Developer API: Routing Values}. Accessed 19 July 2026.
\end{thebibliography}

\end{document}
