---
layout: home
hero:
  name: المعلّم القرآني
  text: تصحيح التلاوات القرآنية من حروف وتشكيل وصفات الحروف وقواعد التجويد باستخدام الذكاء الاصطناعي المتاح المصدر لأمة الإسلام
  tagline: النموذج + الأدوات + مسار النص القرآني في مستودع واحد
  actions:
    - theme: brand
      text: ابدأ
      link: /getting-started
    - theme: alt
      text: Quran Transcript
      link: /quran-transcript/
features:
  - title: CTC متعدد المستويات
    details: يفك النموذج الشيفرة على مستوى الفونيمات وخصائص التجويد باستخدام رأس CTC متعدد المستويات.
  - title: تحليل يعتمد على المرجع
    details: الاستدلال يقارن الفونيمات المتوقعة مع رسم صوتي مرجعي من quran-transcript.
  - title: واجهة Gradio
    details: واجهة جاهزة للتشغيل موجودة في src/quran_muaalem/gradio_app.py.
---

## روابط المشروع

- GitHub: https://github.com/obadx/quran-muaalem
- PyPI: https://pypi.org/project/quran-muaalem/
- نموذج Hugging Face: https://huggingface.co/obadx/muaalem-model-v3_2
- مجموعة البيانات على Hugging Face: https://huggingface.co/datasets/obadx/muaalem-annotated-v3
- الورقة العلمية: https://arxiv.org/abs/2509.00094

## ماذا يحتوي هذا المستودع؟

- كود الاستدلال وواجهة المستخدم للمعلم القرآني تحت `src/quran_muaalem/`.
- مشروع Quran Transcript كامل تحت `quran-transcript/` (مُعبأ كحزمة `quran-transcript`).
- أدوات النشر والاختبارات ضمن `deploy/` و `tests/` و `assets/`.

## للباحثين: أين تبدأ؟

إذا كان هدفك تقييم النموذج أو إعادة إنتاج النتائج:

1. ابدأ بصفحة **المعمارية** لفهم CTC متعدد المستويات.
2. راجع **خطوات المسار** لمعرفة كيفية إنشاء الوسوم الصوتية.
3. استخدم صفحة **التقييم والقياسات** لتوحيد المقاييس.
4. راجع **المخرجات** لمعرفة مخطط البيانات المتوقع.

## للباحثين والمطورين: ملاحظات عملية

- النموذج يتوقع صوتًا بمعدل 16 kHz.
- النتائج تعتمد على جودة المرجع الذي يُنتج عبر `quran_transcript.quran_phonetizer`.
- المقاييس الموصى بها موجودة في قسم **التقييم والقياسات**.

استخدم التنقل للوصول إلى كل قسم.
