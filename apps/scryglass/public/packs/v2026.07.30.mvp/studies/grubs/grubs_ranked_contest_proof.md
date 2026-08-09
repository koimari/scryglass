# High-elo ranked SOLO queue HORDE contest proof (personal Riot key)

**n=1326** ranked matches with ≥1 void grub (HORDE) · contest rate **12.1%**

## Why ranked

OE pro `LOLTMNT*` gameids are unavailable through a personal Match-V5 key. This is therefore a distinct ranked SOLO queue layer, not a pro dataset. It samples Diamond and Masters+ anchors separately across all supported platforms; ranks below Diamond are excluded from the sampling frame.

## Results

- All-3 sweeper WR: **59.6%** (n=1094)
- Contested all-3 sweeper WR: **61.9%**
- Free all-3 sweeper WR: **59.3%**
- Δ contested−free: **2.540636287857745** pp
- Dog steal (behind@8, got all3): n=301 · contest_rate=0.12624584717607973 · WR=0.3853820598006645
- Fav take (ahead@8, got all3): n=522 · contest_rate=0.10153256704980843 · WR=0.7432950191570882
- Mean first grub minute: **8.99**
- Diamond anchor cohort: n=665; Masters+ anchor cohort: n=661

## Gold@8 bins (sweeper POV)

| Bin | n contested | n free | WR contested | WR free | Δpp |
|-----|-------------|--------|--------------|---------|-----|
| even | 27 | 244 | 0.6296296296296297 | 0.5368852459016393 | 9.274438372799032 |
| behind2k | 7 | 72 | 0.14285714285714285 | 0.2222222222222222 | -7.936507936507936 |
| behind | 31 | 191 | 0.6129032258064516 | 0.418848167539267 | 19.40550582671846 |
| ahead | 29 | 291 | 0.5172413793103449 | 0.6941580756013745 | -17.691669629102968 |
| ahead2k | 24 | 178 | 0.875 | 0.8426966292134831 | 3.2303370786516905 |

## Next

1. Scale `--n-matches` while retaining the separate Diamond and Masters+ reports.
2. Add historical all-lobby rank reconstruction before making a lobby-wide tier claim.
3. When Grid is justified: use a separate pro series rather than blending pro and ranked estimates.
