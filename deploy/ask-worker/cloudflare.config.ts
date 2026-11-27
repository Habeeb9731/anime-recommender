// "Ask AnimeRec" Worker: answers chat questions on anime.abdulhabeeb.com/api/ask.
// Deploy with `cf deploy --secrets-file <file>` (see README.md in this folder).
import { bindings, defineConfig, triggers } from "cf/config";

export default defineConfig({
  worker: {
    name: "animerec-ask",
    entrypoint: "./src/index.js",
    compatibilityDate: "2026-11-01",
    env: {
      // Shared with the depower chat; this Worker's counters and logs are prefixed with "anime:".
      ASK_KV: bindings.kv({ id: "7c84394d83a24d40981fb11a44625b1c" }),
      ANTHROPIC_API_KEY: bindings.secret(),
      IP_SALT: bindings.secret(),
      ASK_OWNER_TOKEN: bindings.secret(),
    },
    triggers: [triggers.fetch({ pattern: "anime.abdulhabeeb.com/api/ask*", zone: "abdulhabeeb.com" })],
  },
});
