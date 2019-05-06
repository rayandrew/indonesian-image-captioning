export { components } from 'mdx-deck-code-surfer';

export { default as theme } from './theme';

// Slides
import intro from './slides/01-intro.mdx';
import outline from './slides/02-outline.mdx';
import latarBelakang from './slides/03-latar-belakang.mdx';
import rumusanMasalah from './slides/04-rumusan-masalah.mdx';
import tujuan from './slides/05-tujuan.mdx';
import batasanMasalah from './slides/06-batasan-masalah.mdx';
import rancanganSolusi from './slides/07-rancangan-solusi.mdx';
import skenarioEksperimen from './slides/08-skenario-eksperimen.mdx';
import hasilEksperimen from './slides/09-hasil-eksperimen.mdx';
import analisis from './slides/10-analisis.mdx';
import kesimpulan from './slides/11-kesimpulan.mdx';

export default [
    ...intro,
    ...outline,
    ...latarBelakang,
    ...rumusanMasalah,
    ...tujuan,
    ...batasanMasalah,
    ...rancanganSolusi,
    ...skenarioEksperimen,
    ...hasilEksperimen,
    ...analisis,
    ...kesimpulan,
];
