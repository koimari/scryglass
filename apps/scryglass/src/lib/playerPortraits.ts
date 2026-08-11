type PlayerPortrait = {
  src: string;
  source: string;
};

// Reviewed Leaguepedia page images. Keep this explicit because player handles
// are not unique and an automatic name lookup can return the wrong person.
const PLAYER_PORTRAITS: Record<string, PlayerPortrait> = {
  busio: {
    src: "https://static.wikia.nocookie.net/lolesports_gamepedia_en/images/c/c6/100.A_Busio_2022_Split_1.png/revision/latest/scale-to-width-down/384?cb=20220216155356",
    source: "https://lol.fandom.com/wiki/Busio",
  },
  canyon: {
    src: "https://static.wikia.nocookie.net/lolesports_gamepedia_en/images/5/56/DWG_Canyon_2019_Split_1.png/revision/latest/scale-to-width-down/384?cb=20190722060534",
    source: "https://lol.fandom.com/wiki/Canyon",
  },
  chovy: {
    src: "https://static.wikia.nocookie.net/lolesports_gamepedia_en/images/2/25/GRF_Chovy_2018_Split_1.png/revision/latest?cb=20250426112706",
    source: "https://lol.fandom.com/wiki/Chovy",
  },
  delight: {
    src: "https://static.wikia.nocookie.net/lolesports_gamepedia_en/images/5/59/BRO_Delight_2021_Split_1.png/revision/latest/scale-to-width-down/384?cb=20210211034022",
    source: "https://lol.fandom.com/wiki/Delight",
  },
  dhokla: {
    src: "https://static.wikia.nocookie.net/lolesports_gamepedia_en/images/1/18/Dhokla_Sin_gaming_2017.png/revision/latest/scale-to-width-down/384?cb=20171210113451",
    source: "https://lol.fandom.com/wiki/Dhokla",
  },
  doran: {
    src: "https://static.wikia.nocookie.net/lolesports_gamepedia_en/images/c/c3/GRF_Doran_2019_Split_2.png/revision/latest/scale-to-width-down/384?cb=20260516170548",
    source: "https://lol.fandom.com/wiki/Doran_(Choi_Hyeon-joon)",
  },
  duro: {
    src: "https://static.wikia.nocookie.net/lolesports_gamepedia_en/images/1/15/LSB.C_Duro_2022_Split_2.png/revision/latest?cb=20221208160018",
    source: "https://lol.fandom.com/wiki/Duro",
  },
  faker: {
    src: "https://static.wikia.nocookie.net/lolesports_gamepedia_en/images/3/31/Faker2014.jpg/revision/latest/scale-to-width-down/384?cb=20170801215036",
    source: "https://lol.fandom.com/wiki/Faker",
  },
  inspired: {
    src: "https://static.wikia.nocookie.net/lolesports_gamepedia_en/images/a/ae/REC_Inspired_2019_Split_1.png/revision/latest/scale-to-width-down/384?cb=20190122002721",
    source: "https://lol.fandom.com/wiki/Inspired",
  },
  keria: {
    src: "https://static.wikia.nocookie.net/lolesports_gamepedia_en/images/e/eb/DRX_Keria_2020_Split_1.png/revision/latest/scale-to-width-down/384?cb=20200307042644",
    source: "https://lol.fandom.com/wiki/Keria",
  },
  kiin: {
    src: "https://static.wikia.nocookie.net/lolesports_gamepedia_en/images/2/2e/AFS_Kiin_2018_Spring.png/revision/latest/scale-to-width-down/384?cb=20180127001257",
    source: "https://lol.fandom.com/wiki/Kiin",
  },
  knight: {
    src: "https://static.wikia.nocookie.net/lolesports_gamepedia_en/images/8/84/Knight_LSPL_2017.png/revision/latest/scale-to-width-down/384?cb=20171204135220",
    source: "https://lol.fandom.com/wiki/Knight_(Zhuo_Ding)",
  },
  kyeahoo: {
    src: "https://static.wikia.nocookie.net/lolesports_gamepedia_en/images/3/30/DRX_kyeahoo_2023_Split_2.png/revision/latest?cb=20230612230048",
    source: "https://lol.fandom.com/wiki/Kyeahoo",
  },
  oner: {
    src: "https://static.wikia.nocookie.net/lolesports_gamepedia_en/images/7/73/T1_Oner_2021_Split_1.png/revision/latest/scale-to-width-down/384?cb=20210211041838",
    source: "https://lol.fandom.com/wiki/Oner",
  },
  peter: {
    src: "https://static.wikia.nocookie.net/lolesports_gamepedia_en/images/6/61/NS.C_Peter_2021_Split_2.png/revision/latest/scale-to-width-down/384?cb=20210606204025",
    source: "https://lol.fandom.com/wiki/Peter_(Jeong_Yoon-su)",
  },
  ruler: {
    src: "https://static.wikia.nocookie.net/lolesports_gamepedia_en/images/1/11/Ruler_Summer_2016.png/revision/latest/scale-to-width-down/384?cb=20170802112350",
    source: "https://lol.fandom.com/wiki/Ruler",
  },
  slowq: {
    src: "https://static.wikia.nocookie.net/lolesports_gamepedia_en/images/6/66/Z10_SlowQ_2023_Split_1.png/revision/latest/scale-to-width-down/383?cb=20230216192154",
    source: "https://lol.fandom.com/wiki/SlowQ",
  },
  viper: {
    src: "https://static.wikia.nocookie.net/lolesports_gamepedia_en/images/2/20/GRF_Viper_2018_Split_1.png/revision/latest?cb=20250426112816",
    source: "https://lol.fandom.com/wiki/Viper_(Park_Do-hyeon)",
  },
  zeka: {
    src: "https://static.wikia.nocookie.net/lolesports_gamepedia_en/images/c/cf/VG_Zeka_2020_Split_1.png/revision/latest/scale-to-width-down/384?cb=20200108145011",
    source: "https://lol.fandom.com/wiki/Zeka_(Kim_Geon-woo)",
  },
  zeus: {
    src: "https://static.wikia.nocookie.net/lolesports_gamepedia_en/images/d/db/T1_Zeus_2021_Split_1.png/revision/latest/scale-to-width-down/384?cb=20210211041848",
    source: "https://lol.fandom.com/wiki/Zeus",
  },
};

function portrait(player: string | null | undefined): PlayerPortrait | null {
  if (!player) return null;
  return PLAYER_PORTRAITS[player.trim().toLowerCase()] ?? null;
}

export function playerPortraitUrl(player: string | null | undefined): string | null {
  return portrait(player)?.src ?? null;
}

export function playerPortraitSource(player: string | null | undefined): string | null {
  return portrait(player)?.source ?? null;
}
