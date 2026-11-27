# Ask AnimeRec

A small Cloudflare Worker behind the demo's chat. Claude Haiku answers questions about AnimeRec (from `src/knowledge.js`) and suggests titles; every suggestion is matched against the demo's catalog (`src/catalog.js`) before it is returned, so the chat only shows titles the recommender knows. The page can then add them to the visitor's picks, and optionally change the format, era and skipped genres, through the same `?p=` preferences token the app already reads.

It runs only on `anime.abdulhabeeb.com/api/ask`; every other request goes to the origin unchanged. The chat widget itself lives in `app/assets/ask/` and is inlined into Streamlit's boot page by `deploy/brand_static.py`.

## Guardrails

- Only requests from the site (Origin check), questions up to 300 characters, the last 4 turns of history.
- 5 questions per visitor per day (salted hash of the IP), 40 a day overall, counted in KV under `anime:`.
- Haiku returns `on_topic`; off-topic answers are dropped by the Worker.
- The Anthropic key lives in a workspace with a monthly spend limit, shared with the portfolio and depower chats.
- An owner token (`x-ask-owner`) skips the limits and isn't logged.

## Update and deploy

```bash
npm install
python scripts/build_data.py   # after the demo bundle, README or DECISIONS change
node scripts/test.mjs          # catalog matching checks
cf deploy --secrets-file /path/outside/the/repo/secrets.env
```

The secrets file holds `ANTHROPIC_API_KEY`, `IP_SALT` and `ASK_OWNER_TOKEN`. Never commit it.
