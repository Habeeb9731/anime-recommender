// Catalog matching checks: real titles resolve to the right entry, newer or made-up ones don't.
import assert from 'node:assert/strict';
import CATALOG from '../src/catalog.js';
import { findTitle, describeContext } from '../src/index.js';

const name = (title, year) => { const i = findTitle(title, year); return i < 0 ? null : CATALOG[i][0]; };

assert.equal(name('Your Name', 2016), 'Your Name.');
assert.equal(name('Haikyuu!!', 2014), 'Haikyu!!');
assert.equal(name('Shingeki no Kyojin', 2013), 'Attack on Titan');
assert.equal(name('Clannad: After Story', 2008), 'Clannad ~After Story~');
assert.equal(name('Mushishi', 2005), 'Mushi-Shi');
assert.equal(CATALOG[findTitle('Berserk', 2016)][2], 2016);
assert.equal(CATALOG[findTitle('Berserk', 1997)][2], 1997);
assert.equal(name('Chainsaw Man', 2022), null);
assert.equal(name('Demon Slayer', 2019), null);
assert.equal(name('A Totally Made Up Show', 2010), null);
assert.match(describeContext({ liked: ['Death Note', 'not a real title'], avoid: ['Horror', 'x'] }), /Death Note\.\s+Genres they skip: Horror\./);
assert.doesNotMatch(describeContext({ liked: ['not a real title'] }), /not a real/);
console.log('ok');
