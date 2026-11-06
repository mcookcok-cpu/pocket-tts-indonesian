module.exports = {
  apps: [{
    name: "pocket-tts-id",
    cwd: "/home/Claude/Kerja/pocket-tts-indonesian",
    script: ".venv/bin/uvicorn",
    args: "server:app --host 0.0.0.0 --port 6001 --app-dir src",
    interpreter: "none",
    env: { PYTHONPATH: "src", DEFAULT_VOICE: "klon", EOS_THRESHOLD: "-6.0" },
    max_restarts: 5,
    autorestart: true,
  }],
};
