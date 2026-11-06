// Kunci LLM dibaca dari .env proyek Mao saat pm2 start; tidak disimpan di repo ini.
const fs = require("fs");
let key = "";
try {
  const m = fs.readFileSync("/home/agentuser/mao-vtuber-studio/.env", "utf8").match(/^LR_API_KEY=(.*)$/m);
  if (m) key = m[1].trim().replace(/^["']|["']$/g, "");
} catch (e) {}

module.exports = {
  apps: [{
    name: "pocket-tts-id",
    cwd: "/home/Claude/Kerja/pocket-tts-indonesian",
    script: ".venv/bin/uvicorn",
    args: "server:app --host 0.0.0.0 --port 6001 --app-dir src",
    interpreter: "none",
    env: {
      DEFAULT_VOICE: "klon",
      EOS_THRESHOLD: "-6.0",
      LR_BASE: "http://100.109.71.90:20130/v1",
      LR_MODEL: "gemini/gemini-3.5-flash-lite",
      LR_API_KEY: key,
    },
    max_restarts: 5,
    autorestart: true,
  }],
};
