<div align="center">

**English** | [العربية](README.md)

</div>

---

# Playground — Interactive Mushaf for Quran Muaalem

An interactive Quran Mushaf web app that serves as a visual playground for testing the **Quran Muaalem** API. Instead of testing with `curl` or Postman, you can recite into your microphone and see the AI-powered correction results rendered beautifully on an authentic Mushaf layout.

Built on top of the open-source [Java Quran Web](https://github.com/iTarek/Java-Quran-Web) project.

## Features

- **Search (Find My Position)** — Recite a few ayahs and the app finds your location in the Quran, navigating the Mushaf to the matched ayah
- **Recite (Correct Tajweed)** — Tap an ayah, recite it, and get instant feedback on tajweed errors with rule names in Arabic and English
- **Full Mushaf** — All 604 pages rendered with QCF4 fonts in authentic Mushaf Madina layout
- **Tafseer & Translation** — Long-press any ayah for Arabic tafseer (Al-Muyassar) and English translation (Saheeh International)
- **Audio Playback** — Listen to verse recitation by Mishary Alafasy

## Quick Start

1. **Start the engine and app servers:**

   ```bash
   # Terminal 1 — Start the engine (Wav2Vec2-BERT model)
   uv run quran-muaalem-engine --accelerator mps  # or --accelerator cuda

   # Terminal 2 — Start the app server (from the project root)
   uv run quran-muaalem-app
   ```

2. **Open the playground:**

   Navigate to **http://localhost:8001** in your browser.

   > The app server automatically serves the playground when it detects the `playground/` directory in the current working directory.

3. **Try it out:**
   - Click **Search** → recite "بسم الله الرحمن الرحيم" → the Mushaf navigates to the matched ayah
   - Tap an ayah to select it → click **Recite** → recite the ayah → see tajweed correction results

## How It Works

```
Browser (playground)
    ↓ fetch(/search, /correct-recitation)
App Server (localhost:8001) — FastAPI
    ↓
Engine Server (localhost:8000) — Wav2Vec2-BERT
```

The playground is a pure static web app (HTML + CSS + vanilla JS) that calls the Quran Muaalem API endpoints directly. When served from the app server, no proxy or CORS configuration is needed — everything runs on the same port.

## Technology

- **Zero dependencies** — Pure vanilla JavaScript, HTML, and CSS
- **QCF4 fonts** — Official Quran Complex Fonts v4 (WOFF2, ~36 MB)
- **Web Audio API** — Microphone recording with WAV encoding, no external libraries
- **Responsive** — Works on desktop and mobile with touch support

## Production Deployment

To deploy the playground on a production server (e.g., Ubuntu VPS), you need nginx to serve the static files and proxy API requests to the app server.

### 1. Install dependencies and quran-muaalem

```bash
apt-get update
apt-get install -y ffmpeg libsndfile1 curl nginx

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install quran-muaalem
mkdir -p /opt/quran-muaalem
cd /opt/quran-muaalem
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "quran-muaalem[engine]" librosa
```

### 2. Pre-download the model

```bash
/opt/quran-muaalem/.venv/bin/python -c \
  "from huggingface_hub import snapshot_download; snapshot_download('obadx/muaalem-model-v3_2')"
```

### 3. Create systemd services

**Engine** (port 8000 — Wav2Vec2-BERT model inference):

```bash
cat > /etc/systemd/system/quran-engine.service << 'EOF'
[Unit]
Description=Quran Muaalem Engine
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/quran-muaalem
ExecStart=/opt/quran-muaalem/.venv/bin/quran-muaalem-engine
Environment=ACCELERATOR=cpu
Environment=DTYPE=float32
Environment=PORT=8000
Environment=HOST=127.0.0.1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
```

**App** (port 8001 — FastAPI API server):

```bash
cat > /opt/quran-muaalem/start-app.py << 'PYEOF'
import os
os.environ.setdefault("ENGINE_URL", "http://localhost:8000/predict")
from quran_muaalem.app.serve import app
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
uvicorn.run(app, host="127.0.0.1", port=8001)
PYEOF

cat > /etc/systemd/system/quran-app.service << 'EOF'
[Unit]
Description=Quran Muaalem App
After=quran-engine.service
Requires=quran-engine.service

[Service]
Type=simple
WorkingDirectory=/opt/quran-muaalem
ExecStart=/opt/quran-muaalem/.venv/bin/python /opt/quran-muaalem/start-app.py
Environment=ENGINE_URL=http://localhost:8000/predict
Environment=MAX_WORKERS_PHONETIC_SEARCH=4
Environment=MAX_WORKERS_PHONETIZATION=4
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
```

Start the services:

```bash
systemctl daemon-reload
systemctl enable quran-engine quran-app
systemctl start quran-engine
# Wait ~30s for model to load, then:
systemctl start quran-app
```

### 4. Deploy playground files

```bash
# Copy the playground to the server
rsync -avz --exclude='.DS_Store' playground/ your-server:/var/www/mushaf/
```

### 5. Configure nginx

```bash
cat > /etc/nginx/sites-available/quran-muaalem << 'EOF'
server {
    listen 80;
    server_name your-domain.com;

    root /var/www/mushaf;
    index index.html;

    # Cache static assets
    location ~* \.(woff2|css|js|json|txt|csv)$ {
        expires 7d;
        add_header Cache-Control "public, immutable";
    }

    # Reverse proxy — /api/* → localhost:8001/*
    location /api/ {
        proxy_pass http://127.0.0.1:8001/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        client_max_body_size 25m;
        proxy_read_timeout 60s;
    }

    location / {
        try_files $uri $uri/ /index.html;
    }
}
EOF

ln -sf /etc/nginx/sites-available/quran-muaalem /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx
```

> **SSL:** For HTTPS, add SSL certificates using [certbot](https://certbot.eff.org/) or use Cloudflare with a self-signed cert in "Full" mode.

### 6. Verify

```bash
curl http://localhost:8000/health   # Engine
curl http://localhost:8001/health   # App
curl http://your-domain.com/api/health  # Through nginx
```

Open `http://your-domain.com` in your browser — you should see the Mushaf playground.

## Credits

- Mushaf viewer based on [Java Quran Web](https://github.com/iTarek/Java-Quran-Web) (GPL-3.0)
- QCF4 fonts from King Fahd Complex for the Printing of the Holy Quran
- Tafseer: Al-Tafseer Al-Muyassar | Translation: Saheeh International
- Audio: Mishary Rashid Alafasy via everyayah.com
