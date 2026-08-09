# Champion portraits

This directory contains 173 local 128×128 PNG champion-square portraits: one for every unique champion name in `src/data/drake-study.json` across the pilot compositions, explorer champion catalog, and role catalog.

## Source and rights

Retrieved from the [League of Legends Wiki](https://wiki.leagueoflegends.com/en-us/) on 2026-07-28. Each resolved Wiki file page identifies the image as a Riot Games asset. The files remain copyrighted by Riot Games, Inc. and are used subject to [Riot Games' Legal Jibber Jabber](https://www.riotgames.com/en/legal).

These are stored locally for the app; the runtime does not hotlink Wiki images.

## Retrieval method

1. Extract and deduplicate the champion names in `drake-study.json`.
2. Resolve `File:<Champion name>Square.png` through the Wiki's MediaWiki API at `https://wiki.leagueoflegends.com/en-us/api.php`, following file redirects to the current `OriginalSquare` asset.
3. Fetch the API-resolved image URL with a one-time `raw` cache-buster so Cloudflare's PNG polishing layer cannot replace the original file bytes.
4. Require the downloaded SHA-1 to match MediaWiki's original-file SHA-1, then record the stronger local SHA-256 below.
5. Fully decode every PNG with Pillow and require PNG format plus 128×128 dimensions.

Filename rule: Unicode NFKD normalization; remove apostrophes and periods; replace `&` with `and`; lowercase; replace remaining non-alphanumeric runs with `-`; append `.png`.

## Validation record

- Expected champion names: 173
- Local PNG files: 173
- Fully decoded: 173
- Missing names: none
- Decode or dimension failures: none
- Total PNG payload: 3,619,101 bytes (3.45 MiB)
- Color modes: 168 RGBA, 5 RGB

The exact-name runtime mapping, including local paths and the same SHA-256 values, is in `src/data/champion-images.json`.

## File manifest

| Champion | Local file | League of Legends Wiki file page | SHA-256 |
| --- | --- | --- | --- |
| Aatrox | `aatrox.png` | [File:Aatrox OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Aatrox_OriginalSquare.png) | `d6e6332a080f9b6ce9a5520d00d20c88b8d0163a8042ec92a6a5ad160a8c99f0` |
| Ahri | `ahri.png` | [File:Ahri OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Ahri_OriginalSquare.png) | `70fbb226ff9f13f24cf558144b6b79f7d736b224086c589d4032b975989e97b6` |
| Akali | `akali.png` | [File:Akali OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Akali_OriginalSquare.png) | `bddaacb8a2712279dd0ec69533848035aed4d3c122252b07d29fb0bf96e6b6d0` |
| Akshan | `akshan.png` | [File:Akshan OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Akshan_OriginalSquare.png) | `ea9bbc787b506c66718f562f107b97657f56473f6b690a2e66038772ffc93196` |
| Alistar | `alistar.png` | [File:Alistar OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Alistar_OriginalSquare.png) | `bee01ca25ee2d80166b52baf7eeec9db88cb01f4179cba5258cc360fd53191a1` |
| Ambessa | `ambessa.png` | [File:Ambessa OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Ambessa_OriginalSquare.png) | `65820ba9c81cdfefb2e9b08f7604214d8318e84abfa50b621b9a508c5c3dc588` |
| Amumu | `amumu.png` | [File:Amumu OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Amumu_OriginalSquare.png) | `16d1fc1db3ed28adbe63fe6a3dcea521154e2c60334797a28bfc5454f7da1cf8` |
| Anivia | `anivia.png` | [File:Anivia OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Anivia_OriginalSquare.png) | `5d038474f67ed4436178b4b2ec9fd37db6b35781c91e9c1a99c191b7eecc5522` |
| Annie | `annie.png` | [File:Annie OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Annie_OriginalSquare.png) | `8b8e5c9d45396822a3f468b66ed9e75295e962b208130b70dd4bbdb00613d360` |
| Aphelios | `aphelios.png` | [File:Aphelios OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Aphelios_OriginalSquare.png) | `f77850566090b0cf0f584c4c1d4b9bb67ccf27302e2991b055106819b8562e0f` |
| Ashe | `ashe.png` | [File:Ashe OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Ashe_OriginalSquare.png) | `bb1883f44b38313af756686fe95659efd2c24ac2b66be2ffb4a076a70d555821` |
| Aurelion Sol | `aurelion-sol.png` | [File:Aurelion Sol OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Aurelion_Sol_OriginalSquare.png) | `3a11d95dd8f70a184e823c2151a298a6887d685275df9167d1377d0689e917c3` |
| Aurora | `aurora.png` | [File:Aurora OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Aurora_OriginalSquare.png) | `adcbc7e96e3d8e5828a7e1449861fbd11504e01b2b2a09afa5a0b3cc22ee4331` |
| Azir | `azir.png` | [File:Azir OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Azir_OriginalSquare.png) | `d98a698ba1edf4f378595f621df91780a6ce16a42ed5871a269b1455efd629cf` |
| Bard | `bard.png` | [File:Bard OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Bard_OriginalSquare.png) | `bd9b0d46e6b4e429fff3913579e31f71c2dc8bc7feb8888d2fc507f9cec5fd77` |
| Bel'Veth | `belveth.png` | [File:Bel'Veth OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Bel%27Veth_OriginalSquare.png) | `632107fa223e065d50042455cd468a21beee2facc810d9300b1690034dae4238` |
| Blitzcrank | `blitzcrank.png` | [File:Blitzcrank OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Blitzcrank_OriginalSquare.png) | `bbad98afcba1bc85a2feb9d14dac934ae9c2de00d007c14cad77db0ca8cdacf5` |
| Brand | `brand.png` | [File:Brand OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Brand_OriginalSquare.png) | `c3a891942bbd761344427781057c67650757c25930c08303e7360dc661f34f6a` |
| Braum | `braum.png` | [File:Braum OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Braum_OriginalSquare.png) | `bdd450de0d4068adbfc8aaa4e57a9f4feb2a55e1517e62f83d1ceb7a5582533b` |
| Briar | `briar.png` | [File:Briar OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Briar_OriginalSquare.png) | `a1290322ec0f515bbf0243597f2855b8c5a7e27d1742b0ee4403d59efff25051` |
| Caitlyn | `caitlyn.png` | [File:Caitlyn OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Caitlyn_OriginalSquare.png) | `8f352d4033069369e829dd576a74917a588809ac826f7c0c5e423eec2be205a6` |
| Camille | `camille.png` | [File:Camille OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Camille_OriginalSquare.png) | `aee853e075c85bc8156dc6cb2f9d0a1a5cd506cd5c97c2bf1901b28d4fada242` |
| Cassiopeia | `cassiopeia.png` | [File:Cassiopeia OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Cassiopeia_OriginalSquare.png) | `d4ce0af7e6ba50ffbf22d102e662c2cd514ebe14ad81d70b51311cb30d79295b` |
| Cho'Gath | `chogath.png` | [File:Cho'Gath OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Cho%27Gath_OriginalSquare.png) | `d48c31c850024d1023e864f1312e0aad3afdf796732464e8694a26810859cd27` |
| Corki | `corki.png` | [File:Corki OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Corki_OriginalSquare.png) | `e098a9988a59f68af9e2eb40bf0434db45aa6dc25eed489d14140c76c4fbde04` |
| Darius | `darius.png` | [File:Darius OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Darius_OriginalSquare.png) | `4caafad6fb82a2057640861a91a74d236437927e88e0f2d3f59e62cc91616cb0` |
| Diana | `diana.png` | [File:Diana OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Diana_OriginalSquare.png) | `1416b9b93dbbfb0a52d2e948d811af163bf8ccafcd54668759e0710abc00ea28` |
| Dr. Mundo | `dr-mundo.png` | [File:Dr. Mundo OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Dr._Mundo_OriginalSquare.png) | `621f097bc81497e9b04ab73f826935e8805181b22f6763063b71d1d1f9ac466f` |
| Draven | `draven.png` | [File:Draven OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Draven_OriginalSquare.png) | `d38d9b9d9c1260727367240dc3fc3b2481743ed5c4cbdc5be9f70ea019ffbefd` |
| Ekko | `ekko.png` | [File:Ekko OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Ekko_OriginalSquare.png) | `0af36f31b97668d821571d4fd336e80aefcadeb0e02659deffddb1ef3666b30f` |
| Elise | `elise.png` | [File:Elise OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Elise_OriginalSquare.png) | `1dd3adb106e7bd14ac74e375d7e44ee87a718702a93f7aadac8815707c8cb251` |
| Evelynn | `evelynn.png` | [File:Evelynn OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Evelynn_OriginalSquare.png) | `ca9c60821531eecc0fb92fc0d9c4e334c8e182fa2b308d4d782ea77854ca36e6` |
| Ezreal | `ezreal.png` | [File:Ezreal OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Ezreal_OriginalSquare.png) | `e9e3a15240e5a624eced34c07fdc7774ece47ca08639616ca72af209f7ddc1e2` |
| Fiddlesticks | `fiddlesticks.png` | [File:Fiddlesticks OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Fiddlesticks_OriginalSquare.png) | `f06af761173aaab64b7674a93fa425768ef67ed85b3417d400b66716f2d342c6` |
| Fiora | `fiora.png` | [File:Fiora OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Fiora_OriginalSquare.png) | `f50982f8805772e608df01dc6abeb9f29f58218c0752688c4efc212029727ffd` |
| Fizz | `fizz.png` | [File:Fizz OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Fizz_OriginalSquare.png) | `c25375cf7478b2f7ead03622abc69c609aea05eae818202aaeb67a927434c720` |
| Galio | `galio.png` | [File:Galio OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Galio_OriginalSquare.png) | `d6d634f2374c6b739177e1cc08f4ac8e655ab6c4933c964209186ae628af00e2` |
| Gangplank | `gangplank.png` | [File:Gangplank OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Gangplank_OriginalSquare.png) | `0063e2d52d015ed0d7558ec4b2374b6ebe3795fa3dc06b38b9ff5f4669112f17` |
| Garen | `garen.png` | [File:Garen OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Garen_OriginalSquare.png) | `a58a35f00f6286adfe012119fa316260c8321272195b9b9b782dc8ced2b14eb6` |
| Gnar | `gnar.png` | [File:Gnar OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Gnar_OriginalSquare.png) | `fc5e4585885e29641f11e3b1254c2603b71e3cdc79ac2c003cc91c907bc830dd` |
| Gragas | `gragas.png` | [File:Gragas OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Gragas_OriginalSquare.png) | `80a48cec4f3cbb9ee7c7c44c33e855eb694887ce79455bff291ce1ae6d9c5abd` |
| Graves | `graves.png` | [File:Graves OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Graves_OriginalSquare.png) | `a76974d48f9f42a0e391a7a0c37977881b0aef6bf3b53315d998e0eea005b234` |
| Gwen | `gwen.png` | [File:Gwen OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Gwen_OriginalSquare.png) | `cb00ae8ec061c65c1ee12a373a03af80a250dcfbd8a0153acff521572fdd1546` |
| Hecarim | `hecarim.png` | [File:Hecarim OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Hecarim_OriginalSquare.png) | `db6eef1f75e0becd4543b7b063ab7fe9f91f69fa826fe3d434ff86a04090ae10` |
| Heimerdinger | `heimerdinger.png` | [File:Heimerdinger OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Heimerdinger_OriginalSquare.png) | `f214d5a77c6cdcffcf6e712882f0c371a0c18c436863067b4099510c0719c214` |
| Hwei | `hwei.png` | [File:Hwei OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Hwei_OriginalSquare.png) | `3d5b02a84d16f0ee4aedca04d50edbdfc8b703d7e440336aa2881cd4a5114881` |
| Illaoi | `illaoi.png` | [File:Illaoi OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Illaoi_OriginalSquare.png) | `85f733d7b136b74eb600a1eeea7f81325d0bf1968d2a408e91d11b3571348f5d` |
| Irelia | `irelia.png` | [File:Irelia OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Irelia_OriginalSquare.png) | `f9cee4bab5c5c11db03c602ef97d5f4bfb1d21467e9482ab763f8816023c6e50` |
| Ivern | `ivern.png` | [File:Ivern OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Ivern_OriginalSquare.png) | `e8bcde583a1e38afbee679af8f8240f78b3d4a449319d5caad2a49a28b9ff2eb` |
| Janna | `janna.png` | [File:Janna OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Janna_OriginalSquare.png) | `80c7efc0381f75e344a199cbe458028ad85c18416de6a247aa7a473693615911` |
| Jarvan IV | `jarvan-iv.png` | [File:Jarvan IV OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Jarvan_IV_OriginalSquare.png) | `af2ee3efec1b446be487226a2e8be6dc15330ae1fa89b15f0ff88207c4d2fdab` |
| Jax | `jax.png` | [File:Jax OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Jax_OriginalSquare.png) | `ba2d20d10d65ab264c7bb57c82622aacd782b4931be31f59fcdfde9439500db9` |
| Jayce | `jayce.png` | [File:Jayce OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Jayce_OriginalSquare.png) | `0ddf575aec75944a26480a420386b69245ccd3347f26bbf5a753cb908cbe1b45` |
| Jhin | `jhin.png` | [File:Jhin OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Jhin_OriginalSquare.png) | `15ca5afd55f3d8c47f476821a61b08a86f7fe104cfd27beb359cfa5137dcb047` |
| Jinx | `jinx.png` | [File:Jinx OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Jinx_OriginalSquare.png) | `90dd753aa881c5ca80c8301eaab2fddb92a3277286d320f53b88c8e706b4f028` |
| K'Sante | `ksante.png` | [File:K'Sante OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:K%27Sante_OriginalSquare.png) | `2ee803533529231318a04a4056ec659354f7787ff7416ed332b19694320e98d9` |
| Kai'Sa | `kaisa.png` | [File:Kai'Sa OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Kai%27Sa_OriginalSquare.png) | `63ae710dc962d85cd26de3d80c4af2f5b9a508bc5668f6bd9c5d3f9ca90fb1a2` |
| Kalista | `kalista.png` | [File:Kalista OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Kalista_OriginalSquare.png) | `d016ddc04c64382c27bb5275f7495e70228b92da88794dab4221a3141c3a3c1f` |
| Karma | `karma.png` | [File:Karma OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Karma_OriginalSquare.png) | `a7459131dafcaec94026d7658ccbc6684fd41561d8826dd2b747363426c07114` |
| Karthus | `karthus.png` | [File:Karthus OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Karthus_OriginalSquare.png) | `0a58c9e4e3b1db387d1bd882a4a5b85bafd23c1faf4c9b2728db6b04fb3d31f4` |
| Kassadin | `kassadin.png` | [File:Kassadin OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Kassadin_OriginalSquare.png) | `0770fd37b1dd053ef01ae03bb1e093e3a1d31d30e2ef56eec03a18d9955580f4` |
| Katarina | `katarina.png` | [File:Katarina OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Katarina_OriginalSquare.png) | `91a31eab1bfb56c112a928700775e7a8bedd01f4d25987d0d535ea1394def372` |
| Kayle | `kayle.png` | [File:Kayle OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Kayle_OriginalSquare.png) | `eef5275ced5e931d8c69ab365b30b01ea7fc3f78945fafc9768ed07f6d12f93a` |
| Kayn | `kayn.png` | [File:Kayn OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Kayn_OriginalSquare.png) | `ce3eb186696a844e5b302e69f7f609a50a452613264745d86cff36ae5134917f` |
| Kennen | `kennen.png` | [File:Kennen OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Kennen_OriginalSquare.png) | `daa8fa9e12b12af356def35f80f1446f95e9a3ec9776f9c0d92fee17af27fffb` |
| Kha'Zix | `khazix.png` | [File:Kha'Zix OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Kha%27Zix_OriginalSquare.png) | `98bb682f0c352d9054971255c20fc8d47a39639d03a9485eecc4119acb56940c` |
| Kindred | `kindred.png` | [File:Kindred OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Kindred_OriginalSquare.png) | `56e7fc6a172e6f95b86913880a7a5adfd0891203feb68faa254001d42373c93f` |
| Kled | `kled.png` | [File:Kled OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Kled_OriginalSquare.png) | `f9c6261d7909c0afba965f0ca17a87e9a1b930d129fc27ee6f2bc6c08ec130ef` |
| Kog'Maw | `kogmaw.png` | [File:Kog'Maw OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Kog%27Maw_OriginalSquare.png) | `af775332dde08d8d3d3c3f769c28f5eda13b55ec43d5dd9118dbd735f870bc4d` |
| LeBlanc | `leblanc.png` | [File:LeBlanc OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:LeBlanc_OriginalSquare.png) | `9fd11f6ca2435edaf89d88885157fecc2ead04b1973813ed00e4d96f98acdcb9` |
| Lee Sin | `lee-sin.png` | [File:Lee Sin OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Lee_Sin_OriginalSquare.png) | `afde99ba48ffb5cdf0904e6d02bd7b19da4cd7a81538d2503fa3ceb798cd657f` |
| Leona | `leona.png` | [File:Leona OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Leona_OriginalSquare.png) | `59ad27a6caafa84d973b11acad7f24939c05f0b767a4dea2aef1bca102509ba9` |
| Lillia | `lillia.png` | [File:Lillia OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Lillia_OriginalSquare.png) | `054f77c2d25f2a2f0cc1612ef4e5cb259ad28aa3e4742223808cd47c605fdcd6` |
| Lissandra | `lissandra.png` | [File:Lissandra OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Lissandra_OriginalSquare.png) | `4f788ff7f7bb208a408305bca44fea4ec06a9b158c3054ff66afc808cc929b77` |
| Locke | `locke.png` | [File:Locke OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Locke_OriginalSquare.png) | `0ead99e0c74bbc1c42917b619cb1dcb0a1dfd11e6aebcc1bf4252a0ab74bf849` |
| Lucian | `lucian.png` | [File:Lucian OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Lucian_OriginalSquare.png) | `d8f3c4b376f5d3d3cc5d38988e53ca183e5015ae77c2c97326c95025fe82bbef` |
| Lulu | `lulu.png` | [File:Lulu OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Lulu_OriginalSquare.png) | `b3b08916efd665e496546f1c53d55407a90879d4547fbd7200ce7b642e0ecd6c` |
| Lux | `lux.png` | [File:Lux OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Lux_OriginalSquare.png) | `d342af1bb3ca0a459783c8ba180b08f1a2565e1bff2d11851e1f136b7075b8bb` |
| Malphite | `malphite.png` | [File:Malphite OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Malphite_OriginalSquare.png) | `8984cad3d943bb370a0ce4501a04e53a429d51d336f3ad64ecbaeeb72032b03a` |
| Malzahar | `malzahar.png` | [File:Malzahar OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Malzahar_OriginalSquare.png) | `2a2698034fc6993ae4bc914460f40e1339e5bf81ce213ec37177c107636629ce` |
| Maokai | `maokai.png` | [File:Maokai OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Maokai_OriginalSquare.png) | `e6bc892681a446ac229c4b97d9cbb37603ce4cf61173fc9d43e04524208e5b1d` |
| Master Yi | `master-yi.png` | [File:Master Yi OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Master_Yi_OriginalSquare.png) | `4f10a1b9f534612b4b58f62d5f03ec1c5bdb0d1fd04ec490ab2b5cd9c628a705` |
| Mel | `mel.png` | [File:Mel OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Mel_OriginalSquare.png) | `bf82d5d3cb80a5ea30d486f7470d45eb5ceb905c34df65cb509f9d188a944493` |
| Milio | `milio.png` | [File:Milio OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Milio_OriginalSquare.png) | `be51a9de96bdce92534f18550b38476204c4b046522ae00ec7249c1e3f0bcd07` |
| Miss Fortune | `miss-fortune.png` | [File:Miss Fortune OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Miss_Fortune_OriginalSquare.png) | `cc7f0562ee977c9ebfdf7bd2dd7c26c26b3f82b7f4b7d5bf3a7486f72b274dd2` |
| Mordekaiser | `mordekaiser.png` | [File:Mordekaiser OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Mordekaiser_OriginalSquare.png) | `32c814bc5d2140e32dce65a6999c9a3370dc65da62ec36f811a27c7f3ca9de58` |
| Morgana | `morgana.png` | [File:Morgana OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Morgana_OriginalSquare.png) | `d726fa3d4d51b1f814a26cf6c0ea5b22f63f1bfae74cd6a4310f20d7a1e80637` |
| Naafiri | `naafiri.png` | [File:Naafiri OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Naafiri_OriginalSquare.png) | `d2262dd8f2809c2ce9c241d08b72c1232d94a8b66244cdd79d69af0129d8f595` |
| Nami | `nami.png` | [File:Nami OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Nami_OriginalSquare.png) | `98d0605cd851beaea63865cca0acf9d8e755c089336481a8e51c0abfc1470e9a` |
| Nasus | `nasus.png` | [File:Nasus OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Nasus_OriginalSquare.png) | `154319859b07098b3966e7dc00680d8094d3697aff806f55239be5233f75fe7e` |
| Nautilus | `nautilus.png` | [File:Nautilus OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Nautilus_OriginalSquare.png) | `ceeec1827da574d4bb4f52f8bc9dbb6ce8bb81b116a635d8228ce459e4c21065` |
| Neeko | `neeko.png` | [File:Neeko OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Neeko_OriginalSquare.png) | `29d32d646623f56f0cd5a1912454ce1b9b60a7bb330f419b405067a0db15e9b9` |
| Nidalee | `nidalee.png` | [File:Nidalee OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Nidalee_OriginalSquare.png) | `401935e4e599c202362f99b7b9516498b2adc5198bb4258415c6d003fafb069c` |
| Nilah | `nilah.png` | [File:Nilah OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Nilah_OriginalSquare.png) | `0dd39a30eed176be71353fb7e0f60184acd0b878e23368aa86b6c568d93436af` |
| Nocturne | `nocturne.png` | [File:Nocturne OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Nocturne_OriginalSquare.png) | `e45c141c8063774345107e5fd6de6901b463dd5aa71ad43192b59f03ca0fc932` |
| Nunu & Willump | `nunu-and-willump.png` | [File:Nunu OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Nunu_OriginalSquare.png) | `1595e01d0f3235b30a14f59321b45c79016ccc34f58c03c9005b28e9f7991a5f` |
| Olaf | `olaf.png` | [File:Olaf OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Olaf_OriginalSquare.png) | `55c5483a7dd2e9671cc09b44af1eb0629e60d0f24caf1173746aeea54706f4dd` |
| Orianna | `orianna.png` | [File:Orianna OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Orianna_OriginalSquare.png) | `45080f64d31c6e15934bc7b0de7da8953ec282573fbd47ccbf85c7b38dddaf18` |
| Ornn | `ornn.png` | [File:Ornn OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Ornn_OriginalSquare.png) | `efc2b97db532d4f261424d70a201f69b613d1ec2e3802e2d16d8b103b8ea7b92` |
| Pantheon | `pantheon.png` | [File:Pantheon OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Pantheon_OriginalSquare.png) | `cad09e855b11a42451827e1b708c9393abf703f54e147dc7dc42ecfd1fae25be` |
| Poppy | `poppy.png` | [File:Poppy OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Poppy_OriginalSquare.png) | `882e8a016b1b2a297f88bf69770191d50f84acc6a90ea18f105228d667797bec` |
| Pyke | `pyke.png` | [File:Pyke OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Pyke_OriginalSquare.png) | `f697a528205c2042e9855a456f400267bc525438a7cf650d7c8ccc074cbc24d4` |
| Qiyana | `qiyana.png` | [File:Qiyana OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Qiyana_OriginalSquare.png) | `8654f79ce2754dc825860bb4ad2cd0671347a2e4a0babcf844486e04921a40d4` |
| Quinn | `quinn.png` | [File:Quinn OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Quinn_OriginalSquare.png) | `1b12b0ec3d9279f58cedaf163e0d4dd65cb8a37b6ab10bd4c4dfac88eb872ce0` |
| Rakan | `rakan.png` | [File:Rakan OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Rakan_OriginalSquare.png) | `a9605afb4ae32c98a2fd19ac13d10e0e5b229de80734b548d12f223e112940f5` |
| Rammus | `rammus.png` | [File:Rammus OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Rammus_OriginalSquare.png) | `a3535e816e42de50b9d0c97d09b1b3b6e20bc0228e46e67e0b0b8386df896cc5` |
| Rek'Sai | `reksai.png` | [File:Rek'Sai OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Rek%27Sai_OriginalSquare.png) | `3db0ef189268218369353066008c74b7c64c115bb2ae45cea289fe31b6640d75` |
| Rell | `rell.png` | [File:Rell OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Rell_OriginalSquare.png) | `377084f6a87f0f1b94d288d3447105a59c8a519712eb34fd8033afb21e44efa1` |
| Renata Glasc | `renata-glasc.png` | [File:Renata Glasc OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Renata_Glasc_OriginalSquare.png) | `e704355d67723c7fc404533f911ca5f8cfc52ed9235fd4f927400534daf94684` |
| Renekton | `renekton.png` | [File:Renekton OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Renekton_OriginalSquare.png) | `b510596571b05128b456936f7e385b073cc95de3b93ab7783ca6a862fae8192a` |
| Rengar | `rengar.png` | [File:Rengar OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Rengar_OriginalSquare.png) | `aa035b7534f962d8e82fb37278a04548c852ce02b293f18ebffb5a76c17d3864` |
| Riven | `riven.png` | [File:Riven OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Riven_OriginalSquare.png) | `cff3c49b5bc9f7c0714d285e197eb861f86f28472200aaac674db2e7b254a019` |
| Rumble | `rumble.png` | [File:Rumble OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Rumble_OriginalSquare.png) | `1d66147ff4d5ed432dcd58035c010523df374e4bd371ed859d40044ab76bf366` |
| Ryze | `ryze.png` | [File:Ryze OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Ryze_OriginalSquare.png) | `761be9ccbce467a69e9a7ddc4322df03ff9f7fd482ea6cf35c844ce5ee6cbc1b` |
| Samira | `samira.png` | [File:Samira OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Samira_OriginalSquare.png) | `ccb9358650bd82670632d9042dbfc2dbe13d259e988444f4f8399eac198b4e1a` |
| Sejuani | `sejuani.png` | [File:Sejuani OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Sejuani_OriginalSquare.png) | `6292c0687cce868c4de4c6af34c6d651374f98e2ce5b7e7a38b227711d5df4c2` |
| Senna | `senna.png` | [File:Senna OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Senna_OriginalSquare.png) | `8fd82d962bfa1bbfa278ba0f946fc6b2132075653cd9105e2a45b93138325fc6` |
| Seraphine | `seraphine.png` | [File:Seraphine OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Seraphine_OriginalSquare.png) | `f6cb31c3d8166018f654ca51e9f09841c6774df68f55fee3947a74a76ab115c5` |
| Sett | `sett.png` | [File:Sett OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Sett_OriginalSquare.png) | `96009031e2aa5432daef2d0a238f1808fb1c9210fdbfe5081b61e032c21bfe1b` |
| Shaco | `shaco.png` | [File:Shaco OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Shaco_OriginalSquare.png) | `4ed82bd163f7772178d5d4e70c68b4e014389d99e69d005df19cce1ceb3ee856` |
| Shen | `shen.png` | [File:Shen OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Shen_OriginalSquare.png) | `c002df86f3d581903ad1533e0d2c50cbf31e48f95ac97fa0d77173a2539c7261` |
| Shyvana | `shyvana.png` | [File:Shyvana OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Shyvana_OriginalSquare.png) | `3a136718b056d58636f4333e962ec9500624148bae03e4ce72e29d5e20743377` |
| Singed | `singed.png` | [File:Singed OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Singed_OriginalSquare.png) | `87a962be71bf016a3bf66f68139cfb58ffe7505fe6e8f19adc85dd79e7a5980b` |
| Sion | `sion.png` | [File:Sion OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Sion_OriginalSquare.png) | `1a6a3ea4181c94129e5e8863c4a5712bc827ab0b38c635beb1bc6447f2129b17` |
| Sivir | `sivir.png` | [File:Sivir OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Sivir_OriginalSquare.png) | `00dbb1d1d438561c34d6efd1198cec9396289dfa6b96b5ac1dc11f39cbb2fca8` |
| Skarner | `skarner.png` | [File:Skarner OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Skarner_OriginalSquare.png) | `48b268292312c0950dd97830ec7862474f5fce87b30708408640d5f7a3870219` |
| Smolder | `smolder.png` | [File:Smolder OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Smolder_OriginalSquare.png) | `1ed291a149013b77f035b0d3eb2807bfdc2bd739335244850968afa3c4b8fb29` |
| Sona | `sona.png` | [File:Sona OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Sona_OriginalSquare.png) | `356f127ace994b3ce2343f0e8e6430f7622741d37afcdb8dcb2829d9f0033e15` |
| Soraka | `soraka.png` | [File:Soraka OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Soraka_OriginalSquare.png) | `ddc6cac5c01888f5371c6afa588b121cd03cb933a6a6f6063b9385a64246c291` |
| Swain | `swain.png` | [File:Swain OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Swain_OriginalSquare.png) | `5083212d690f42b967b965d487e8355536fd2136dab221bba9037b10fcd84bb5` |
| Sylas | `sylas.png` | [File:Sylas OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Sylas_OriginalSquare.png) | `3f13a482f5cbf2782f1a7c08020d94c5d9cd5049c56861d80537cd8b4b47ac54` |
| Syndra | `syndra.png` | [File:Syndra OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Syndra_OriginalSquare.png) | `3f1eed902044e22a37d397d18c8f60e3177cf53803a062166ccc00fd2c42e87c` |
| Tahm Kench | `tahm-kench.png` | [File:Tahm Kench OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Tahm_Kench_OriginalSquare.png) | `863b4e874c78258117d91b687a5bffa96986824f4aad2275b2df0d78d6228b2a` |
| Taliyah | `taliyah.png` | [File:Taliyah OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Taliyah_OriginalSquare.png) | `69c5ccc27ec13e1f318c6f9733b6dd9829dbcbec0b485a5f718207263a420a7e` |
| Talon | `talon.png` | [File:Talon OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Talon_OriginalSquare.png) | `45863ebefd4d773363c69d99729a7f4e1b99aa7017ce850d24ac9bfe7221c58f` |
| Taric | `taric.png` | [File:Taric OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Taric_OriginalSquare.png) | `55b0f32b1acd66dee666f148749fca7d5e0d5700b65a38dd5eda7f313d4258b7` |
| Teemo | `teemo.png` | [File:Teemo OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Teemo_OriginalSquare.png) | `e35cb18be64f59a1f173f3547ab6617ca841d96f11b9f18b780d851fe1ed2678` |
| Thresh | `thresh.png` | [File:Thresh OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Thresh_OriginalSquare.png) | `7000cee4b4f24b041987c0f9c328e6195383b553af824b8f166cb1026cf75d39` |
| Tristana | `tristana.png` | [File:Tristana OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Tristana_OriginalSquare.png) | `2d38390c239c02b9589844f798ce8c823b068a47c7e958c6464326301e3d359e` |
| Trundle | `trundle.png` | [File:Trundle OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Trundle_OriginalSquare.png) | `57ba8efbfdc6baf77208e09123ac77a61b380bc1101f17ddcd505b6b44e390ae` |
| Tryndamere | `tryndamere.png` | [File:Tryndamere OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Tryndamere_OriginalSquare.png) | `286002b5e8d62a6bc732cd04b10bf8b1bd30c449406b78336ca9bd3dc4903943` |
| Twisted Fate | `twisted-fate.png` | [File:Twisted Fate OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Twisted_Fate_OriginalSquare.png) | `21d80f7878ddc54b8551cbde2f8fd976f12148d3c25c56e1df732d9da7fe71f9` |
| Twitch | `twitch.png` | [File:Twitch OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Twitch_OriginalSquare.png) | `d7634e486d3fd24c1cc9c5d7f4b77bdc80a022d3d462856610f5a6a279c21e23` |
| Udyr | `udyr.png` | [File:Udyr OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Udyr_OriginalSquare.png) | `002016da1607305f6f1d7b1a92e969e4e577772763feabde447995bf14016f88` |
| Urgot | `urgot.png` | [File:Urgot OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Urgot_OriginalSquare.png) | `b9d6bc9ef72b0fc1d86c5cafaffa8117809df146703224ab4496b6c317da3c4f` |
| Varus | `varus.png` | [File:Varus OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Varus_OriginalSquare.png) | `585e84015a45bbd474b74d045785120fddf4be3e31a36851b3ff89fd537fc8af` |
| Vayne | `vayne.png` | [File:Vayne OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Vayne_OriginalSquare.png) | `71ac7236faf803f6c0f05e8fa4cad77a535d8044c468c87d3849a1c460d2db4e` |
| Veigar | `veigar.png` | [File:Veigar OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Veigar_OriginalSquare.png) | `d72ad4b86664bbd6d8f5ac48badd42f8e06518d4ac980610f9bd179011ca66ed` |
| Vel'Koz | `velkoz.png` | [File:Vel'Koz OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Vel%27Koz_OriginalSquare.png) | `a280f3ea6a2b95b9c931dc1ebe6a96304c0762f7200a9d34cef6af914cf76096` |
| Vex | `vex.png` | [File:Vex OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Vex_OriginalSquare.png) | `e4687c71454b66907fc0ca5325f0e7cd04d32900e517e3afdd1a2d74df7b3dae` |
| Vi | `vi.png` | [File:Vi OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Vi_OriginalSquare.png) | `9dddea4f7dea01129a85961e07cf2a5add3afa5984fa8ace7435973f39858c4f` |
| Viego | `viego.png` | [File:Viego OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Viego_OriginalSquare.png) | `988b987ea96ac3f0abf45b18422d7b4d7c29764ab055a5d47ba81667bad16b8b` |
| Viktor | `viktor.png` | [File:Viktor OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Viktor_OriginalSquare.png) | `c86e87b3e660edd08be6e9290d6612152ec5489fc7c1b1cf3fa89446cf3cdeca` |
| Vladimir | `vladimir.png` | [File:Vladimir OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Vladimir_OriginalSquare.png) | `afa12e33a29f4131ada6db287759ac6e5c1932526814fb5b76b79553945dfa3c` |
| Volibear | `volibear.png` | [File:Volibear OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Volibear_OriginalSquare.png) | `60f9c4fedd98d4a4d923490bb04876577f34825e0cc5156aa63ff4dbc5e0aa0b` |
| Warwick | `warwick.png` | [File:Warwick OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Warwick_OriginalSquare.png) | `dbc90b5a8a2aeac9be6b9e45ee0350593be73300630bef7e67afc104180391f2` |
| Wukong | `wukong.png` | [File:Wukong OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Wukong_OriginalSquare.png) | `93d1f4ad400a7c36ebea6adac337472c86030d5b8de522972eadc24249c53746` |
| Xayah | `xayah.png` | [File:Xayah OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Xayah_OriginalSquare.png) | `1fdf3b0313fd95cee9e214196b23028fc26e01bc1890049ae8840bc93ce22942` |
| Xerath | `xerath.png` | [File:Xerath OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Xerath_OriginalSquare.png) | `0004f830cf181f4b0330b0fa14b9014d71c5c6e380140a9f4abfcf4afe66cd4a` |
| Xin Zhao | `xin-zhao.png` | [File:Xin Zhao OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Xin_Zhao_OriginalSquare.png) | `906d6f88cc712a2a42f84c8407a7f93be48a4b42b2b35f1aa43d741e2b896091` |
| Yasuo | `yasuo.png` | [File:Yasuo OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Yasuo_OriginalSquare.png) | `9e958eb083bfad05517aec14d0e8aeb1d0905dd19dbc1cbb6acc01aaf28e2ca5` |
| Yone | `yone.png` | [File:Yone OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Yone_OriginalSquare.png) | `7603ff63876683f0b458578d67225690b9359f12783558ab9894356f01889aeb` |
| Yorick | `yorick.png` | [File:Yorick OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Yorick_OriginalSquare.png) | `8b38d3ec0f7264f7c9d4a74b4ab2539e188cf911224021b4e54c71d5a79e352b` |
| Yunara | `yunara.png` | [File:Yunara OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Yunara_OriginalSquare.png) | `9f63c820f4702aeedba7ddc28ff1108950c842c7d5a987defb33ad97705ec230` |
| Yuumi | `yuumi.png` | [File:Yuumi OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Yuumi_OriginalSquare.png) | `35674e4561fdbad50ad01f3da28cb3deeb5f6644517b50eff2dcd4ca353bfd9c` |
| Zaahen | `zaahen.png` | [File:Zaahen OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Zaahen_OriginalSquare.png) | `df7d945c37aa89961cebde9b13889b370bf32e91cab880fda93f61ff47a1bfea` |
| Zac | `zac.png` | [File:Zac OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Zac_OriginalSquare.png) | `13d61a9aa0b9f73e9cd88931a1c2d677fcd8506f36e56d5e96e68239fcf95d71` |
| Zed | `zed.png` | [File:Zed OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Zed_OriginalSquare.png) | `8c666305d68c373c4d1760bcb9f4ec956749a6648a4257dbe21690ae6ffb314a` |
| Zeri | `zeri.png` | [File:Zeri OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Zeri_OriginalSquare.png) | `d1256160294fa8505e36a6915baa7223dac4cb3894fa8f7754dbecbb316ddfe5` |
| Ziggs | `ziggs.png` | [File:Ziggs OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Ziggs_OriginalSquare.png) | `192686f4cdfdd33992e7c2d37d553533a6a45c0dff8bbd25aea97dc930c8c4d3` |
| Zilean | `zilean.png` | [File:Zilean OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Zilean_OriginalSquare.png) | `bb7ee2c83d525c48265d33c4b8c8f279466dfd29b4f04c7d1e6a82718b6a0b0e` |
| Zoe | `zoe.png` | [File:Zoe OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Zoe_OriginalSquare.png) | `bf73337b43d7745bed5b8587c8de03d90e27dacf1703b8169852c941b2fd0aba` |
| Zyra | `zyra.png` | [File:Zyra OriginalSquare.png](https://wiki.leagueoflegends.com/en-us/File:Zyra_OriginalSquare.png) | `990f2e7550414cc4703f78e459493b8fe1fdbc349dbad1bc7af5845a92caf18c` |
