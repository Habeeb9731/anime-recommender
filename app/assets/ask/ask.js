// "Ask AnimeRec": a small mascot in the corner that opens a chat answered by /api/ask (deploy/ask-worker).
// It suggests titles from the catalog, and one button hands them to the recommender as picks, through the
// same ?p= preferences token the app reads. Inlined into Streamlit's boot page by deploy/brand_static.py,
// so it lives outside the app and survives every rerun.
(() => {
  if (document.querySelector('.ask')) return;

  // `make app` runs Streamlit on localhost, where the Worker isn't routed, so it calls the live one (CORS allows localhost).
  const API = /^(localhost|127\.0\.0\.1)$/.test(location.hostname) ? 'https://anime.abdulhabeeb.com/api/ask' : '/api/ask';
  const STORAGE_KEY = 'animerec_prefs';
  const MAX_PICKS = 30;
  const LINKS = {
    source_code: ['Source code', 'https://github.com/Habeeb9731/anime-recommender', 'github'],
    case_study: ['Case study', 'https://abdulhabeeb.com/projects/anime-recommender/', 'doc'],
    author: ['Abdul Habeeb', 'https://abdulhabeeb.com/', 'person'],
  };
  const T = {
    greet: 'Hi! I’m AnimeRec’s AI. Tell me what you’re in the mood for and I’ll find you something to watch.',
    title: 'Ask AnimeRec', sub: 'Picks from the catalog, explained',
    help: 'What are you in the mood for?',
    placeholder: 'Something like Death Note…',
    offTopic: 'I can only help with finding anime here and with how AnimeRec works. Try “something cozy and funny”.',
    limited: 'To prevent abuse, chats are limited. Come back later to ask more. Your picks and the search button still work as usual.',
    error: 'Something went wrong on my side. Please try again in a moment.',
  };
  // Suggested questions. Ones with an answer are answered right here, free; the rest go to the assistant.
  const CHIPS = [
    ['Like Death Note', 'Something like Death Note, with clever mind games'],
    ['A cozy movie', 'A cozy, feel-good anime movie for tonight'],
    ['How does it work?', 'How does AnimeRec work?', 'It blends two signals. Collaborative filtering (EASE) learns from what 23 million list entries say people watch together, and a content model reads synopses, genres, studio and year. Each title leans on whichever it has more evidence for, so even shows nobody has rated yet can be ranked. Your picks are folded in instantly, without retraining.', ['case_study', 'source_code']],
    ['Is it accurate?', 'How accurate is it?', 'On a held-out three-month window it reaches NDCG@10 0.152, about four times all-time popularity (0.037) and 63% above recent popularity. It’s level with EASE alone overall; its edge is handling brand-new titles. The catalog ends in early 2018, so newer shows aren’t in it.', ['case_study']],
  ];
  const store = {
    get: (s, k) => { try { return s.getItem(k); } catch { return null; } },
    set: (s, k, v) => { try { s.setItem(k, v); } catch { /* storage blocked */ } },
    del: (s, k) => { try { s.removeItem(k); } catch { /* storage blocked */ } },
  };

  // The mascot: a little white mochi with cat ears, a red bow and big anime eyes. Matte 3D from layered gradients.
  const eye = x => `<g class="ask-eye" data-cx="${x}" data-cy="60"><g class="ask-pupil">
      <ellipse cx="${x}" cy="60" rx="7.2" ry="9.4" fill="url(#amIris)"/>
      <ellipse cx="${x}" cy="56.4" rx="5.2" ry="4.2" fill="#000" opacity=".35"/>
      <circle cx="${x - 2.3}" cy="56.6" r="2.7" fill="#fff"/><circle cx="${x + 2.5}" cy="63.6" r="1.2" fill="#fff" opacity=".85"/>
    </g></g>`;
  const mascot = `<svg class="ask-body" viewBox="0 0 100 100" aria-hidden="true">
    <defs>
      <radialGradient id="amBody" cx=".4" cy=".3" r=".8"><stop offset="0" stop-color="#fff"/><stop offset=".55" stop-color="#f3f1f6"/><stop offset="1" stop-color="#cfcadb"/></radialGradient>
      <linearGradient id="amIris" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#120914"/><stop offset=".55" stop-color="#3b0b1d"/><stop offset="1" stop-color="#ef3a48"/></linearGradient>
      <radialGradient id="amRed" cx=".38" cy=".3" r=".8"><stop offset="0" stop-color="#ff5a63"/><stop offset=".55" stop-color="#e50914"/><stop offset="1" stop-color="#a1050d"/></radialGradient>
      <radialGradient id="amBlush" cx=".5" cy=".5" r=".5"><stop offset="0" stop-color="#ff7d9f" stop-opacity=".85"/><stop offset="1" stop-color="#ff7d9f" stop-opacity="0"/></radialGradient>
      <filter id="amSoft" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="3"/></filter>
    </defs>
    <ellipse class="ask-paw-l" cx="14" cy="72" rx="6.5" ry="8" fill="url(#amBody)" transform="rotate(18 14 72)"/>
    <g class="ask-head">
      <path d="M21 46Q17 17 26 10.5Q30 8 34.5 12L49 30Z" fill="url(#amBody)"/>
      <path d="M26 37Q24.5 21 28.5 16.5Q30.5 15 32.5 17.5L41.5 29Z" fill="#ffc4d3"/>
      <path d="M79 46Q83 17 74 10.5Q70 8 65.5 12L51 30Z" fill="url(#amBody)"/>
      <path d="M74 37Q75.5 21 71.5 16.5Q69.5 15 67.5 17.5L58.5 29Z" fill="#ffc4d3"/>
      <path d="M50 24C74 24 90 40 90 62C90 82 74 93 50 93C26 93 10 82 10 62C10 40 26 24 50 24Z" fill="url(#amBody)"/>
      <ellipse cx="37" cy="37" rx="15" ry="7" fill="#fff" opacity=".8" filter="url(#amSoft)"/>
      <g class="ask-face">
        ${eye(36)}${eye(64)}
        <ellipse cx="25" cy="71" rx="6.5" ry="3.6" fill="url(#amBlush)"/><ellipse cx="75" cy="71" rx="6.5" ry="3.6" fill="url(#amBlush)"/>
        <path d="M45 69q2.5 3 5 0q2.5 3 5 0" fill="none" stroke="#3a1b2b" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
      </g>
      <g class="ask-bow"><path d="M70 28 60.5 21.5Q58 28 60.5 34.5Z" fill="url(#amRed)"/><path d="M70 28 79.5 21.5Q82 28 79.5 34.5Z" fill="url(#amRed)"/><circle cx="70" cy="28" r="3.4" fill="#c80711"/></g>
    </g>
    <g class="ask-paw"><ellipse cx="86" cy="71" rx="6.5" ry="8" fill="url(#amBody)" transform="rotate(-18 86 71)"/></g>
  </svg>`;

  const root = document.createElement('div');
  root.className = 'ask';
  root.innerHTML = `
    <div class="ask-panel" role="dialog" aria-modal="false" aria-labelledby="askTitle" hidden>
      <div class="ask-head-row">
        <div><p class="ask-title" id="askTitle">${T.title}</p><p class="ask-sub">${T.sub}</p></div>
        <div class="ask-tools">
          <button class="ask-clear ask-x" type="button" aria-label="Clear chat" title="Clear chat"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3"/><path d="M4.5 4.5v4h4"/></svg></button>
          <button class="ask-close ask-x" type="button" aria-label="Close"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg></button>
        </div>
      </div>
      <div class="ask-stage">
        <div class="ask-log" aria-live="polite"></div>
        <div class="ask-intro"><p class="ask-help">${T.help}</p><div class="ask-chips"></div></div>
        <form class="ask-form">
          <input type="text" maxlength="300" autocomplete="off" placeholder="${T.placeholder}" aria-label="Ask AnimeRec" />
          <button class="ask-send" type="submit" aria-label="Send"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 19V5M5.5 11.5 12 5l6.5 6.5"/></svg></button>
        </form>
      </div>
    </div>
    <div class="ask-bubble" hidden><p></p><button class="ask-bubble-x" type="button" aria-label="Dismiss"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 7l10 10M17 7 7 17"/></svg></button></div>
    <button class="ask-button" type="button" aria-expanded="false" aria-label="Ask AnimeRec’s AI assistant">${mascot}</button>`;
  document.body.append(root);

  const $ = s => root.querySelector(s);
  const panel = $('.ask-panel'), bubble = $('.ask-bubble'), petBtn = $('.ask-button');
  const log = $('.ask-log'), chips = $('.ask-chips');
  const form = $('.ask-form'), input = form.querySelector('input'), send = $('.ask-send');
  chips.append(...CHIPS.map(([label], i) => {
    const b = Object.assign(document.createElement('button'), { type: 'button', textContent: label });
    b.dataset.i = i;
    return b;
  }));

  // --- Appear once the app has drawn its first screen, then greet once per visit ---
  function bubbleSay(text, ms = 10000) {
    bubble.querySelector('p').textContent = text;
    bubble.hidden = false;
    requestAnimationFrame(() => bubble.classList.add('is-in'));
    clearTimeout(bubble._t);
    bubble._t = setTimeout(() => {
      if (bubble.hidden) return;
      if (bubble.matches(':hover')) bubble.addEventListener('pointerleave', hideBubble, { once: true });
      else hideBubble();
    }, ms);
  }
  const hideBubble = () => { bubble.classList.remove('is-in'); store.set(sessionStorage, 'ask-greeted', '1'); setTimeout(() => { bubble.hidden = true; }, 250); };
  const ready = () => document.querySelector('.hero');
  const start = setInterval(() => {
    if (!ready()) return;
    clearInterval(start);
    setTimeout(() => {
      root.classList.add('is-in', 'is-waving');
      setTimeout(() => root.classList.remove('is-waving'), 1900);
      const applied = store.get(sessionStorage, 'ask-applied');
      if (applied) {
        store.del(sessionStorage, 'ask-applied');
        setTimeout(() => bubbleSay(applied, 7000), 600);
      } else if (!store.get(sessionStorage, 'ask-greeted')) {
        setTimeout(() => { if (panel.hidden) bubbleSay(T.greet); }, 700);
      }
    }, 1400);
  }, 250);

  function open() {
    hideBubble();
    panel.hidden = false;
    requestAnimationFrame(() => panel.classList.add('is-open'));
    root.classList.add('is-open');
    petBtn.setAttribute('aria-expanded', 'true');
    // On phones, wait for a tap on the field so the keyboard doesn't cover the panel straight away.
    if (!input.disabled && matchMedia('(pointer: fine)').matches) input.focus({ preventScroll: true });
  }
  function close() {
    panel.classList.remove('is-open');
    root.classList.remove('is-open');
    petBtn.setAttribute('aria-expanded', 'false');
    setTimeout(() => { if (!panel.classList.contains('is-open')) panel.hidden = true; }, 260);
  }
  petBtn.addEventListener('click', () => (panel.hidden ? open() : close()));
  bubble.addEventListener('click', e => { if (e.target.closest('.ask-bubble-x')) hideBubble(); else open(); });
  $('.ask-close').addEventListener('click', () => { close(); petBtn.focus({ preventScroll: true }); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && !panel.hidden) { close(); petBtn.focus({ preventScroll: true }); } });

  // --- Eyes follow the cursor and the head tilts a little; on touch screens it glances around ---
  const svg = $('.ask-body'), head = $('.ask-head');
  const eyes = [...root.querySelectorAll('.ask-eye')].map(g => ({ g, pupil: g.querySelector('.ask-pupil') }));
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let target = null, frame = 0;
  function look() {
    frame = 0;
    if (!target) return;
    const box = svg.getBoundingClientRect();
    const scale = box.width / 100;
    for (const { g, pupil } of eyes) {
      const ex = box.left + g.dataset.cx * scale, ey = box.top + g.dataset.cy * scale;
      const dx = target.x - ex, dy = target.y - ey, d = Math.hypot(dx, dy) || 1;
      const r = Math.min(3, d / 40);
      pupil.setAttribute('transform', `translate(${(dx / d * r).toFixed(2)} ${(dy / d * r).toFixed(2)})`);
    }
    const tilt = Math.max(-6, Math.min(6, (target.x - (box.left + box.width / 2)) / 70));
    head.style.transform = `rotate(${tilt.toFixed(2)}deg)`;
  }
  addEventListener('pointermove', e => { if (e.pointerType === 'mouse') { target = { x: e.clientX, y: e.clientY }; if (!frame) frame = requestAnimationFrame(look); } }, { passive: true });
  if (!matchMedia('(pointer: fine)').matches && !reduced) {
    setInterval(() => {
      if (document.hidden) return;
      const box = svg.getBoundingClientRect();
      target = { x: box.left + (Math.random() - .6) * 400, y: box.top + (Math.random() - .7) * 300 };
      look();
    }, 2600);
  }
  if (!reduced) {
    (function blink() {
      setTimeout(() => { root.classList.add('is-blinking'); setTimeout(() => { root.classList.remove('is-blinking'); blink(); }, 140); }, 2500 + Math.random() * 3500);
    })();
  }

  // --- Preferences: the same base64 JSON token the app keeps in the URL and local storage ---
  function readPrefs() {
    const token = new URLSearchParams(location.search).get('p') || store.get(localStorage, STORAGE_KEY);
    if (!token) return {};
    try {
      const bin = atob(token.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - token.length % 4) % 4));
      const data = JSON.parse(new TextDecoder().decode(Uint8Array.from(bin, c => c.charCodeAt(0))));
      return data && typeof data === 'object' ? data : {};
    } catch { return {}; }
  }
  function writePrefs(p) {
    const bytes = new TextEncoder().encode(JSON.stringify(p));
    return btoa(String.fromCharCode(...bytes)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  }
  function context() {
    const p = readPrefs();
    return { liked: Array.isArray(p.l) ? p.l.slice(-30) : [], avoid: Array.isArray(p.a) ? p.a : [], format: p.f, era: p.e };
  }
  // Hands the chosen titles (and any settings) to the recommender and reloads with them. `fresh` starts from
  // empty preferences, as onboarding does; otherwise they are merged into what the visitor already has.
  function apply(titles, settings, { fresh = false, note = '' } = {}) {
    const p = fresh ? {} : readPrefs();
    const list = k => (Array.isArray(p[k]) ? p[k] : []);
    const next = {
      l: [...list('l').filter(t => !titles.includes(t)), ...titles].slice(-MAX_PICKS),
      d: list('d').filter(t => !titles.includes(t)),
      g: [...new Set([...list('g'), ...(settings.genres || [])])],
      a: [...new Set([...list('a'), ...(settings.avoid || [])])],
      f: settings.format || p.f || 'Both',
      e: settings.era || p.e || 'Any time',
    };
    const token = writePrefs(next);
    store.set(localStorage, STORAGE_KEY, token);
    const n = titles.length;
    store.set(sessionStorage, 'ask-applied', note || (n ? `Added ${n} title${n === 1 ? '' : 's'} to your picks. Here’s your new top 10.` : 'Done. Your recommendations are updated.'));
    store.set(sessionStorage, 'ask-greeted', '1');
    location.assign(`${location.pathname}?p=${token}`);
  }

  // --- Chat ---
  // While the cursor is over the panel the page behind never scrolls. The wheel only moves the
  // conversation, and only while it has room to move in that direction.
  panel.addEventListener('wheel', e => {
    const canScroll = log.contains(e.target) && (e.deltaY < 0 ? log.scrollTop > 0 : log.scrollTop + log.clientHeight < log.scrollHeight - 1);
    if (!canScroll) e.preventDefault();
  }, { passive: false });
  const history = [];
  let busy = false;

  // Owner pass: opening the site once with ?ask_owner=<key> stores the key in this browser, and the
  // chat sends it so the server skips the limits. ?ask_owner=off removes it.
  let ownerKey = null;
  try {
    const url = new URL(location.href);
    const given = url.searchParams.get('ask_owner');
    if (given) {
      if (given === 'off') localStorage.removeItem('ask-owner'); else localStorage.setItem('ask-owner', given);
      url.searchParams.delete('ask_owner');
      window.history.replaceState(window.history.state, '', url.pathname + url.search + url.hash);
    }
    ownerKey = localStorage.getItem('ask-owner');
  } catch { /* storage blocked */ }

  function say(role, text, extra = '') {
    const el = document.createElement('p');
    el.className = `ask-msg is-${role}${extra ? ' ' + extra : ''}`;
    if (text) el.textContent = text;
    log.append(el);
    root.classList.add('has-chat');
    log.scrollTop = log.scrollHeight;
    return el;
  }
  const after = (el, node) => { el.after(node); log.scrollTop = log.scrollHeight; return node; };

  const ICONS = {
    github: '<path class="f" d="M12 3.3a8.8 8.8 0 0 0-2.78 17.15c.44.08.6-.19.6-.42v-1.5c-2.45.53-2.97-1.18-2.97-1.18-.4-1.02-.98-1.29-.98-1.29-.8-.55.06-.54.06-.54.88.06 1.35.91 1.35.91.79 1.35 2.07.96 2.57.73.08-.57.31-.96.56-1.18-1.96-.22-4.01-.98-4.01-4.35 0-.96.34-1.75.91-2.37-.09-.22-.39-1.12.09-2.33 0 0 .74-.24 2.42.9a8.4 8.4 0 0 1 4.4 0c1.68-1.14 2.42-.9 2.42-.9.48 1.21.18 2.11.09 2.33.57.62.91 1.41.91 2.37 0 3.38-2.06 4.13-4.02 4.34.32.27.6.81.6 1.63v2.42c0 .23.16.5.6.42A8.8 8.8 0 0 0 12 3.3Z"/>',
    doc: '<path d="M14 3.5H7a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-10z"/><path d="M14 3.5v5h5M9 13h6M9 16.5h4"/>',
    person: '<circle cx="12" cy="8.5" r="3.5"/><path d="M5 19.5a7 7 0 0 1 14 0"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
  };
  const icon = name => `<svg class="ai" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name]}</svg>`;

  function linkRow(el, ids) {
    const row = document.createElement('div');
    row.className = 'ask-links';
    for (const id of ids) {
      const l = LINKS[id];
      if (!l) continue;
      const a = Object.assign(document.createElement('a'), { href: l[1], target: '_blank', rel: 'noopener' });
      a.innerHTML = icon(l[2]);
      a.append(l[0]);
      a.insertAdjacentHTML('beforeend', ' <span aria-hidden="true">↗</span>');
      row.append(a);
    }
    if (row.children.length) after(el, row);
  }

  // One title card: cover, name, year · type · score and the reason; opens MyAnimeList.
  function cardEl(p) {
    const a = Object.assign(document.createElement('a'), { className: 'ask-card', href: p.url, target: '_blank', rel: 'noopener' });
    a.title = `${p.title} on MyAnimeList`;
    const art = document.createElement('span');
    art.className = 'ask-art';
    art.textContent = (p.title.match(/[\p{L}\p{N}]/u) || ['?'])[0].toUpperCase();
    if (p.cover) {
      const img = Object.assign(document.createElement('img'), { src: p.cover.replace(/\.jpg$/, 't.jpg'), alt: '', loading: 'lazy', referrerPolicy: 'no-referrer' });
      img.addEventListener('error', () => img.remove());
      art.append(img);
    }
    const text = document.createElement('span');
    text.className = 'ask-card-text';
    text.append(
      Object.assign(document.createElement('b'), { textContent: p.title }),
      Object.assign(document.createElement('small'), { textContent: [p.year, p.type, p.score ? `★ ${p.score.toFixed(1)}` : ''].filter(Boolean).join(' · ') }),
      Object.assign(document.createElement('em'), { textContent: p.why }),
    );
    a.append(art, text);
    return a;
  }

  // Title cards under an answer, plus the button that hands them to the recommender.
  function pickCards(el, picks, settings) {
    const wrap = document.createElement('div');
    wrap.className = 'ask-picks';
    for (const p of picks) wrap.append(cardEl(p));
    const changes = [];
    if (picks.length) changes.push(`Adds ${picks.length} title${picks.length === 1 ? '' : 's'} to your picks`);
    if (settings.format) changes.push(settings.format === 'Both' ? 'Series and movies' : `${settings.format} only`);
    if (settings.era) changes.push(settings.era);
    if (settings.avoid?.length) changes.push(`Skips ${settings.avoid.join(', ')}`);
    if (changes.length) {
      const go = Object.assign(document.createElement('button'), { type: 'button', className: 'ask-apply' });
      go.innerHTML = icon('plus');
      go.append(picks.length ? 'Add to my picks' : 'Apply to my recommendations');
      const note = Object.assign(document.createElement('span'), { className: 'ask-apply-note', textContent: changes.join(' · ') });
      go.addEventListener('click', () => { go.disabled = true; go.classList.add('is-busy'); apply(picks.map(p => p.title), settings); });
      wrap.append(go, note);
    }
    if (wrap.children.length) after(el, wrap);
  }

  function lock() {
    if (input.disabled) return;
    input.disabled = true;
    send.disabled = true;
    input.placeholder = '';
    say('bot', T.limited, 'is-note');
  }

  // Clear: back to the start screen. The daily limit is kept on the server, so a locked chat stays locked.
  $('.ask-clear').addEventListener('click', () => {
    if (busy) return;
    log.replaceChildren();
    history.length = 0;
    root.classList.remove('has-chat');
    if (input.disabled) say('bot', T.limited, 'is-note');
    else if (matchMedia('(pointer: fine)').matches) input.focus({ preventScroll: true });
  });

  function remember(question, answer) {
    history.push({ role: 'user', content: question }, { role: 'assistant', content: answer.slice(0, 1400) });
    history.splice(0, Math.max(0, history.length - 8));
  }

  function waiting() {
    const wait = say('bot', '', 'is-wait');
    wait.innerHTML = '<i></i><i></i><i></i>';
    return wait;
  }

  async function ask(question) {
    question = question.trim().slice(0, 300);
    if (!question || busy || input.disabled) return;
    busy = true;
    send.disabled = true;
    say('user', question);
    input.value = '';
    const wait = waiting();
    root.classList.add('is-thinking');
    try {
      const res = await fetch(API, {
        method: 'POST',
        headers: { 'content-type': 'application/json', ...(ownerKey ? { 'x-ask-owner': ownerKey } : {}) },
        body: JSON.stringify({ question, history, context: context() }),
      });
      const data = await res.json().catch(() => ({}));
      wait.remove();
      if (res.status === 429) return lock();
      if (!res.ok) { say('bot', T.error, 'is-note'); return; }
      if (data.on_topic) {
        const msg = say('bot', data.answer);
        const picks = Array.isArray(data.picks) ? data.picks.filter(p => p && typeof p.title === 'string' && /^https:\/\/myanimelist\.net\//.test(p.url)) : [];
        const settings = data.settings && typeof data.settings === 'object' ? data.settings : {};
        pickCards(msg, picks, settings);
        linkRow(log.lastElementChild, Array.isArray(data.links) ? data.links : []);
        remember(question, data.answer + (picks.length ? ` Suggested: ${picks.map(p => `${p.title} (${p.year})`).join('; ')}.` : ''));
      } else {
        say('bot', T.offTopic, 'is-note');
      }
      if (data.remaining === 0) lock();
    } catch {
      wait.remove();
      say('bot', T.error, 'is-note');
    } finally {
      busy = false;
      root.classList.remove('is-thinking');
      if (!input.disabled) { send.disabled = false; if (matchMedia('(pointer: fine)').matches) input.focus({ preventScroll: true }); }
    }
  }

  form.addEventListener('submit', e => { e.preventDefault(); ask(input.value); });
  chips.addEventListener('click', e => {
    const b = e.target.closest('button');
    if (!b || busy) return;
    const [, question, answer, links] = CHIPS[b.dataset.i];
    if (!answer) { ask(question); return; }
    // Answered instantly from the text above, free and outside the daily limit.
    busy = true;
    say('user', question);
    const wait = waiting();
    setTimeout(() => {
      wait.remove();
      linkRow(say('bot', answer), links);
      remember(question, answer);
      busy = false;
    }, 550);
  });
  // --- AI onboarding: the app draws an empty .ai-ob-mount on its "Describe it" screen and this fills it ---
  const EXAMPLES = [
    ['mask', 'Mind games', 'Loved Death Note and Steins;Gate. Twists, nothing too gory.', 'I loved Death Note and Steins;Gate. Mind games, twists, nothing too gory.'],
    ['cup', 'Cozy and funny', 'Slice of life to unwind with. Not into horror.', 'Cozy and funny slice of life to unwind with. Not into horror.'],
    ['heart', 'Big feelings', 'Emotional movies like Your Name, with gorgeous animation.', 'Big emotional movies like Your Name, with beautiful animation.'],
  ];
  const EX_ICONS = {
    mask: '<path d="M4 6.5c2.6-1 5.3-1.5 8-1.5s5.4.5 8 1.5v4.5c0 4.4-3.6 8-8 8s-8-3.6-8-8z"/><path d="M8 11.2c.8-.7 1.9-.7 2.6 0M13.4 11.2c.8-.7 1.9-.7 2.6 0M9.5 15c1.5 1 3.5 1 5 0"/>',
    cup: '<path d="M5 9h11v5a5 5 0 0 1-5 5h-1a5 5 0 0 1-5-5z"/><path d="M16 10.5h1.5a2.5 2.5 0 0 1 0 5H16M9 3.5c-.8 1 .8 2 0 3M12.5 3.5c-.8 1 .8 2 0 3"/>',
    heart: '<path d="M12 19.5s-7.5-4.4-7.5-10A4.2 4.2 0 0 1 12 7a4.2 4.2 0 0 1 7.5 2.5c0 5.6-7.5 10-7.5 10z"/>',
  };
  function mountOnboarding(host) {
    host.dataset.mounted = '1';
    host.innerHTML = `
      <div class="aiob">
        <form class="aiob-box">
          <div class="aiob-strip"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M13 2.5 5.5 13.5H11l-1 8 7.5-11H12z"/></svg>Powered by AI <span>· Picks from 10,720 titles</span></div>
          <div class="aiob-inner">
            <textarea maxlength="300" rows="3" aria-label="Describe what you like to watch" placeholder="Describe what you like to watch…"></textarea>
            <div class="aiob-row">
              <span class="aiob-pill"><span class="aiob-count">0 / 300</span></span>
              <button class="aiob-go" type="submit" aria-label="Find my anime"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 19V5M5.5 11.5 12 5l6.5 6.5"/></svg></button>
            </div>
          </div>
        </form>
        <div class="aiob-examples"></div>
        <div class="aiob-result" aria-live="polite"></div>
      </div>`;
    const formEl = host.querySelector('.aiob-box'), area = host.querySelector('textarea'), go = host.querySelector('.aiob-go');
    const count = host.querySelector('.aiob-count'), result = host.querySelector('.aiob-result'), ex = host.querySelector('.aiob-examples');
    // Suggestion cards send straight away.
    for (const [ic, title, sub, text] of EXAMPLES) {
      const b = Object.assign(document.createElement('button'), { type: 'button' });
      b.innerHTML = `<b><svg viewBox="0 0 24 24" aria-hidden="true">${EX_ICONS[ic]}</svg></b>`;
      b.querySelector('b').append(title);
      b.append(Object.assign(document.createElement('span'), { textContent: sub }));
      b.addEventListener('click', () => { area.value = text; area.dispatchEvent(new Event('input')); formEl.requestSubmit(); });
      ex.append(b);
    }
    area.addEventListener('input', () => { count.textContent = `${area.value.length} / 300`; });
    area.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); formEl.requestSubmit(); } });
    const note = text => { result.replaceChildren(Object.assign(document.createElement('p'), { className: 'aiob-note', textContent: text })); };

    formEl.addEventListener('submit', async e => {
      e.preventDefault();
      const text = area.value.trim();
      if (!text || go.disabled) { area.focus(); return; }
      go.disabled = true;
      result.replaceChildren();
      const loader = juggle();
      try {
        const res = await fetch(API, {
          method: 'POST',
          headers: { 'content-type': 'application/json', ...(ownerKey ? { 'x-ask-owner': ownerKey } : {}) },
          body: JSON.stringify({ question: text, mode: 'onboard', context: {} }),
        });
        const data = await res.json().catch(() => ({}));
        if (res.status === 429) throw new Error('The AI is resting for today to prevent abuse. Go back and answer step by step instead, it takes under a minute.');
        if (!res.ok) throw new Error('Something went wrong on our side. Try again in a moment, or go back and answer step by step.');
        if (!data.on_topic) throw new Error('Tell me about anime you like: shows you loved, genres, or the mood you’re after.');
        const s = data.settings || {};
        const picks = Array.isArray(data.picks) ? data.picks.filter(p => p && /^https:\/\/myanimelist\.net\//.test(p.url)) : [];
        const loved = Array.isArray(s.loved) ? s.loved : [];
        const seeds = loved.length ? loved : picks.map(p => p.title);
        if (!seeds.length && !(s.genres || []).length) throw new Error('I couldn’t pin that down. Try naming a show or two you loved, or a genre.');
        // The juggling cards turn into the picks, then the recommender takes over.
        await loader.found(picks);
        apply(seeds, s, { fresh: true, note: 'Here’s your top 10. Ask me anytime for more picks.' });
      } catch (err) {
        loader.close();
        note(err instanceof TypeError ? 'Something went wrong on our side. Try again in a moment, or go back and answer step by step.' : err.message);
        go.disabled = false;
      }
    });
  }

  // Full-screen loader: a reel of posters that steps past a lit, enlarged centre card on a red glow. It shows
  // well-known catalog titles while the AI reads the description, then the visitor's own picks.
  const REEL = [["Cowboy Bebop",1998,"TV",8.7,"4/19644"],["Spirited Away",2001,"Movie",8.8,"6/79597"],["Attack on Titan",2013,"TV",8.5,"10/47347"],["Your Name.",2016,"Movie",8.8,"5/87048"],["Fullmetal Alchemist: Brotherhood",2009,"TV",9,"1223/96541"],["One Punch Man",2015,"TV",8.5,"12/76049"],["Steins;Gate",2011,"TV",9,"5/73199"],["Hunter x Hunter",2011,"TV",9,"1337/99013"],["Mob Psycho 100",2016,"TV",8.4,"8/80356"],["Princess Mononoke",1997,"Movie",8.7,"7/75919"],["Toradora!",2008,"TV",8.1,"13/22128"],["Haikyu!!",2014,"TV",8.4,"7/76014"]]
    .map(([title, year, type, score, c]) => ({ title, year, type, score, cover: `https://cdn.myanimelist.net/images/anime/${c}.jpg` }));
  function juggle() {
    const SLOTS = 9; // cards in the DOM: one centre, four each side (the outer ones sit off screen)
    const el = document.createElement('div');
    el.className = 'aiob-loader';
    el.setAttribute('role', 'status');
    el.innerHTML = `<div class="aiob-glow"></div><div class="aiob-reel"></div><p class="aiob-msg">Reading your taste</p><div class="aiob-bar"><i></i></div>`;
    document.body.append(el);
    const reel = el.querySelector('.aiob-reel'), msg = el.querySelector('.aiob-msg');
    const fill = (card, p) => {
      card.style.backgroundImage = p.cover ? `url("${p.cover.replace(/\.jpg$/, 'l.jpg')}")` : '';
      card.innerHTML = '';
      const score = Object.assign(document.createElement('span'), { className: 'aiob-score', textContent: p.score ? `${p.score.toFixed(1)} ★` : '' });
      const foot = document.createElement('span');
      foot.className = 'aiob-foot';
      foot.append(Object.assign(document.createElement('b'), { textContent: p.title }), Object.assign(document.createElement('small'), { textContent: [p.year, p.type].filter(Boolean).join(' · ') }));
      card.append(score, foot);
    };
    let items = REEL, next = 0;
    const cards = Array.from({ length: SLOTS }, (_, i) => {
      const c = document.createElement('div');
      c.className = 'aiob-reel-card';
      c.dataset.slot = i - (SLOTS >> 1);
      fill(c, items[next++ % items.length]);
      reel.append(c);
      return c;
    });
    const place = () => cards.forEach(c => { c.style.setProperty('--s', c.dataset.slot); c.classList.toggle('is-center', c.dataset.slot === '0'); });
    place();
    requestAnimationFrame(() => el.classList.add('is-in'));
    // Every step each card moves one slot left; the one leaving on the left wraps to the far right with the next poster.
    const step = () => {
      for (const c of cards) {
        let s = +c.dataset.slot - 1;
        if (s < -(SLOTS >> 1)) {
          s = SLOTS >> 1;
          c.classList.add('is-wrap');
          fill(c, items[next++ % items.length]);
          requestAnimationFrame(() => requestAnimationFrame(() => c.classList.remove('is-wrap')));
        }
        c.dataset.slot = s;
      }
      place();
    };
    const timer = reduced ? 0 : setInterval(step, 900);
    const steps = ['Reading your taste', 'Searching 10,720 titles', 'Checking what fits'];
    let k = 0;
    const tick = setInterval(() => { k = (k + 1) % steps.length; swap(steps[k]); }, 1800);
    function swap(text) {
      msg.classList.remove('is-swap');
      void msg.offsetWidth;
      msg.textContent = text;
      msg.classList.add('is-swap');
    }
    return {
      found(picks) {
        clearInterval(tick);
        const own = picks.filter(p => p.cover);
        // The whole reel fades over to the visitor's picks (repeated to fill it when there are few).
        if (own.length) {
          items = own;
          next = 0;
          reel.classList.add('is-swapping');
          setTimeout(() => {
            const order = [...cards].sort((x, y) => Math.abs(x.dataset.slot) - Math.abs(y.dataset.slot) || x.dataset.slot - y.dataset.slot);
            for (const c of order) fill(c, items[next++ % items.length]);
            reel.classList.remove('is-swapping');
          }, 320);
        }
        swap(picks.length ? `Found ${picks.length} you’ll love. Lining up your top 10` : 'Lining up your top 10');
        el.classList.add('is-found');
        return new Promise(r => setTimeout(r, reduced ? 600 : 3800));
      },
      close() {
        clearInterval(tick);
        clearInterval(timer);
        el.classList.remove('is-in');
        setTimeout(() => el.remove(), 300);
      },
    };
  }

  const scan = () => document.querySelectorAll('.ai-ob-mount:not([data-mounted])').forEach(mountOnboarding);
  new MutationObserver(scan).observe(document.body, { childList: true, subtree: true });
  scan();
})();
