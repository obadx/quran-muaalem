<div align="center">

[English](README_EN.md) | **العربية**

</div>

---

# الملعب التفاعلي — مصحف تفاعلي للمعلم القرآني

تطبيق ويب تفاعلي لعرض المصحف الشريف يعمل كملعب مرئي لاختبار واجهة **المعلم القرآني** البرمجية (API). بدلاً من الاختبار باستخدام `curl` أو Postman، يمكنك التلاوة عبر الميكروفون ورؤية نتائج التصحيح الذكي معروضة بشكل جميل على تخطيط مصحف المدينة.

مبني على مشروع [Java Quran Web](https://github.com/iTarek/Java-Quran-Web) مفتوح المصدر.

## المميزات

- **البحث (أين أنا في القرآن)** — اقرأ بضع آيات وسيجد التطبيق موقعك في القرآن وينتقل بالمصحف إلى الآية المطابقة
- **التلاوة (تصحيح التجويد)** — اضغط على آية، اقرأها، واحصل على تقييم فوري لأخطاء التجويد مع أسماء القواعد بالعربية والإنجليزية
- **المصحف الكامل** — جميع الـ 604 صفحة معروضة بخطوط QCF4 بتخطيط مصحف المدينة الأصلي
- **التفسير والترجمة** — اضغط مطولاً على أي آية للحصول على التفسير الميسر والترجمة الإنجليزية (Saheeh International)
- **تشغيل الصوت** — استمع لتلاوة الآية بصوت الشيخ مشاري العفاسي

## البدء السريع

1. **شغّل خوادم المحرك والتطبيق:**

   ```bash
   # الطرفية الأولى — تشغيل المحرك (نموذج Wav2Vec2-BERT)
   uv run quran-muaalem-engine --accelerator mps  # أو --accelerator cuda

   # الطرفية الثانية — تشغيل خادم التطبيق (من جذر المشروع)
   uv run quran-muaalem-app
   ```

2. **افتح الملعب:**

   انتقل إلى **http://localhost:8001** في متصفحك.

   > خادم التطبيق يقدم الملعب تلقائياً عندما يكتشف مجلد `playground/` في مسار العمل الحالي.

3. **جرّب:**
   - اضغط **بحث** ← اقرأ "بسم الله الرحمن الرحيم" ← ينتقل المصحف إلى الآية المطابقة
   - اضغط على آية لتحديدها ← اضغط **تلاوة** ← اقرأ الآية ← شاهد نتائج تصحيح التجويد

## كيف يعمل

```
المتصفح (الملعب)
    ↓ fetch(/search, /correct-recitation)
خادم التطبيق (localhost:8001) — FastAPI
    ↓
خادم المحرك (localhost:8000) — Wav2Vec2-BERT
```

الملعب هو تطبيق ويب ثابت (HTML + CSS + JavaScript خالص) يستدعي نقاط واجهة المعلم القرآني البرمجية مباشرة. عند تقديمه من خادم التطبيق، لا حاجة لإعداد proxy أو CORS — كل شيء يعمل على نفس المنفذ.

## التقنيات

- **صفر تبعيات** — JavaScript و HTML و CSS خالص بدون مكتبات خارجية
- **خطوط QCF4** — خطوط مجمع الملك فهد لطباعة المصحف الشريف الإصدار 4 (WOFF2، ~36 ميغابايت)
- **Web Audio API** — تسجيل الميكروفون مع ترميز WAV، بدون مكتبات خارجية
- **متجاوب** — يعمل على الحاسوب والهاتف مع دعم اللمس

## النشر على خادم إنتاج

لنشر الملعب على خادم إنتاج (مثل Ubuntu VPS)، تحتاج nginx لتقديم الملفات الثابتة وتوجيه طلبات API إلى خادم التطبيق.

### 1. تثبيت المتطلبات و quran-muaalem

```bash
apt-get update
apt-get install -y ffmpeg libsndfile1 curl nginx

# تثبيت uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# تثبيت quran-muaalem
mkdir -p /opt/quran-muaalem
cd /opt/quran-muaalem
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "quran-muaalem[engine]" librosa
```

### 2. تحميل النموذج مسبقاً

```bash
/opt/quran-muaalem/.venv/bin/python -c \
  "from huggingface_hub import snapshot_download; snapshot_download('obadx/muaalem-model-v3_2')"
```

### 3. إنشاء خدمات systemd

**المحرك** (المنفذ 8000 — استدلال نموذج Wav2Vec2-BERT):

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

**التطبيق** (المنفذ 8001 — خادم FastAPI):

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

تشغيل الخدمات:

```bash
systemctl daemon-reload
systemctl enable quran-engine quran-app
systemctl start quran-engine
# انتظر ~30 ثانية لتحميل النموذج، ثم:
systemctl start quran-app
```

### 4. نشر ملفات الملعب

```bash
# انسخ الملعب إلى الخادم
rsync -avz --exclude='.DS_Store' playground/ your-server:/var/www/mushaf/
```

### 5. إعداد nginx

```bash
cat > /etc/nginx/sites-available/quran-muaalem << 'EOF'
server {
    listen 80;
    server_name your-domain.com;

    root /var/www/mushaf;
    index index.html;

    # تخزين الملفات الثابتة مؤقتاً
    location ~* \.(woff2|css|js|json|txt|csv)$ {
        expires 7d;
        add_header Cache-Control "public, immutable";
    }

    # وكيل عكسي — /api/* → localhost:8001/*
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

> **SSL:** لتفعيل HTTPS، أضف شهادات SSL باستخدام [certbot](https://certbot.eff.org/) أو استخدم Cloudflare مع شهادة ذاتية التوقيع في وضع "Full".

### 6. التحقق

```bash
curl http://localhost:8000/health   # المحرك
curl http://localhost:8001/health   # التطبيق
curl http://your-domain.com/api/health  # عبر nginx
```

افتح `http://your-domain.com` في متصفحك — يجب أن ترى ملعب المصحف.

## الشكر والتقدير

- عارض المصحف مبني على [Java Quran Web](https://github.com/iTarek/Java-Quran-Web) (GPL-3.0)
- خطوط QCF4 من مجمع الملك فهد لطباعة المصحف الشريف
- التفسير: التفسير الميسر | الترجمة: Saheeh International
- الصوت: الشيخ مشاري راشد العفاسي عبر everyayah.com
