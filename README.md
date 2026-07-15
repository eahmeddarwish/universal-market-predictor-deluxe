<div align="center">

# 📈 Universal Market Predictor — Deluxe Edition

### One shared LSTM, every ticker, honest evaluation
### نموذج LSTM موحّد لجميع الأسهم، بتقييمٍ يقوم على الصدق والشفافية

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-00C896.svg)](LICENSE)

**Built by [Ahmed Darwish](mailto:eahmeddarwish@gmail.com)**

[⬅️ Original Project / المشروع الأصلي](https://github.com/eahmeddarwish/universal-market-predictor)

</div>

---

## 🌍 Overview | نظرة عامة

**[English]**
This is a rebuild of [universal-market-predictor](https://github.com/eahmeddarwish/universal-market-predictor),
designed around one idea: **a prediction system is only as trustworthy as
the baseline it's compared against.** Every number this project produces is
shown next to a "does-nothing" baseline, and the architecture is built to
grow (new stocks, new exchanges, new coins) without retraining from zero
every time.

**[العربية]**
هذا المشروع إعادة بناء لمشروع [universal-market-predictor](https://github.com/eahmeddarwish/universal-market-predictor)،
قائمٌ على فكرةٍ واحدة: **لا يفوق نظام التنبؤ في مصداقيته المقياسَ المرجعي
الذي يُقارَن به.** فكل رقمٍ يُنتجه هذا المشروع يُعرض إلى جانب مقياسٍ مرجعي
"لا يقوم بأي شيء"، والبنية مصمَّمة بحيث تتّسع (أسهمٌ جديدة، بورصاتٌ جديدة،
عملاتٌ رقمية جديدة) دون الحاجة إلى إعادة التدريب من الصفر في كل مرة.

---

## ✨ What Changed vs. the Original Project | ما الذي تغيّر عن المشروع الأصلي

| | Original / الأصلي | Deluxe |
|---|---|---|
| Training / التدريب | On-demand, every request (1-2 min wait) | Pretrained offline (Colab/local), instant inference |
| Model / النموذج | One separate LSTM per ticker | **One shared LSTM backbone + per-ticker embeddings** |
| Forecast horizon / أفق التنبؤ | Next day only | **1 / 3 / 7 trading days**, direct multi-output (not recursive) |
| Evaluation / التقييم | Raw MAE/MAPE/R² only | Same metrics **plus a naive-persistence & moving-average baseline on every single prediction, plus a statistical significance test on directional accuracy** |
| Backtesting / الاختبار الرجعي | Single train/test split | Single split + **chronological walk-forward folds** to catch regime-dependent failure |
| New tickers / أسهم جديدة | Not supported without retraining everything | **Cold-start fine-tuning**: freeze the backbone, train only the new ticker's embedding row |
| Cache / التخزين المؤقت | Ephemeral, resets on restart | Model is a versioned artifact in `model_registry/`, meant to be committed/shipped |

---

## 🎯 Why Compare Against a Baseline at All? | لماذا نُقارَن أصلًا بمقياسٍ مرجعي؟

**[English]**
Tomorrow's closing price for a large, liquid stock is *usually* close to
today's. That means almost any model — including a genuinely useless one —
can report a flattering-looking MAPE simply by leaning on that fact. The
only way to know if this LSTM has learned anything real is to place its
error and directional accuracy directly next to:

- **`naive_persistence`** — predict tomorrow = today, no ML at all
- **`moving_average_5` / `moving_average_20`** — predict the recent average, still no ML

If `shared_lstm` isn't clearly ahead of `naive_persistence` on a given
ticker/horizon — especially on **directional accuracy**, where 50% is a
coin flip — that result should be treated as noise, not signal. The app and
every training report surface this comparison directly; nothing is hidden
behind a single "our model got 98% accuracy!" headline number.

**[العربية]**
عادةً ما يكون سعر إغلاق الغد، بالنسبة لسهمٍ كبيرٍ وذي سيولةٍ عالية، قريبًا
جدًا من سعر إغلاق اليوم. وهذا يعني أن أي نموذج — حتى لو كان عديم الفائدة
فعلًا — يمكن أن يُظهر قيمة MAPE تبدو مُرضية لمجرّد اعتماده على هذه الحقيقة.
والطريقة الوحيدة لمعرفة ما إذا كان هذا النموذج (LSTM) قد تعلّم شيئًا حقيقيًا
هي وضع خطئه ودقة اتجاهه مباشرةً إلى جانب:

- **`naive_persistence`** — التنبؤ بأن سعر الغد يساوي سعر اليوم، دون أي تعلّم آلي
- **`moving_average_5` / `moving_average_20`** — التنبؤ بمتوسط الأيام الأخيرة، دون تعلّم آلي أيضًا

فإذا لم يتفوّق `shared_lstm` بوضوح على `naive_persistence` في تكرٍ/أفقٍ
معيّن — خصوصًا في **دقة الاتجاه**، حيث تمثّل نسبة 50% رميةَ عملةٍ معدنية
بحتة — فإن هذه النتيجة ينبغي التعامل معها بوصفها ضوضاء لا إشارةً حقيقية.
يعرض التطبيق، وكذلك كل تقرير تدريب، هذه المقارنة مباشرةً؛ فلا شيء يُخفى وراء
رقمٍ واحدٍ برّاق من نوع "حقق النموذج دقة 98%!".

### 🔬 Is a Directional Edge Even Real, or Just Noise? | هل تُعدّ ميزة الاتجاه حقيقيةً فعلًا، أم أنها مجرّد ضوضاء؟

**[English]**
A real training run on this project surfaced a second, subtler version of
the same "don't trust a flattering number" problem: `shared_lstm` came out
at ~52-58% directional accuracy on several tickers, next to baselines
sitting near the 50% coin-flip line. A couple of points above 50% on a few
hundred test windows can easily be sampling noise, not a real edge — so
`comparison_report.csv` now runs a two-sided binomial test against the 50%
null on every `shared_lstm` row, plus a Wilson 95% confidence interval
(`Direction_CI_95`, `Direction_p_value`, `Direction_Significant` in
`src/evaluation.py::direction_significance`). Only treat a ticker/horizon's
directional accuracy as a genuine finding when `Direction_Significant` is
`True` — i.e. the 95% CI excludes 50% entirely. The report also adds
`R2_vs_naive` (every method's R² minus `naive_persistence`'s R² for that
exact ticker/horizon) on every row, since a high absolute R² on next-day
price is mostly ordinary price autocorrelation, not proof of forecasting
skill, and the real lift over doing nothing should be readable directly
instead of requiring the reader to hunt down the baseline row and subtract
by hand.

**[العربية]**
كشف تدريبٌ فعلي على هذا المشروع عن نسخةٍ ثانية وأدقّ من المشكلة ذاتها: "لا
تُصدّق رقمًا برّاقًا". فقد أظهر `shared_lstm` دقة اتجاهٍ تتراوح بين 52%
و58% تقريبًا في عدة أسهم، إلى جانب مقاييس مرجعية تقترب من خط الـ50% (رمية
العملة). وفرقٌ من نقطتين أو ثلاث فوق الـ50%، على بضع مئاتٍ من نوافذ
الاختبار، قد يكون بسهولة ضوضاء إحصائية لا ميزةً حقيقية — لذلك يُجري
`comparison_report.csv` الآن اختبارًا ثنائي الطرف من نوع binomial ضد فرضية
الـ50% على كل صفٍّ من صفوف `shared_lstm`، إضافةً إلى فترة ثقة Wilson بنسبة
95% (`Direction_CI_95`، `Direction_p_value`، `Direction_Significant` في
`src/evaluation.py::direction_significance`). ولا ينبغي التعامل مع دقة
اتجاه أي تكرٍ/أفقٍ باعتبارها نتيجةً حقيقية إلا إذا كانت قيمة
`Direction_Significant` تساوي `True` — أي أن فترة الثقة بنسبة 95% لا تشمل
الـ50% إطلاقًا. كما يضيف التقرير عمود `R2_vs_naive` (قيمة R² لكل طريقة
مطروحًا منها R² الخاصة بـ`naive_persistence` لنفس التكر والأفق بالضبط) في
كل صف، ذلك أن ارتفاع قيمة R² المطلقة على سعر الغد غالبًا ما يكون انعكاسًا
طبيعيًا للارتباط الذاتي للسعر، لا دليلًا على مهارة تنبؤية حقيقية، وينبغي أن
تكون القيمة الحقيقية المضافة مقارنةً بعدم فعل أي شيء واضحةً بشكلٍ مباشر،
دون أن يضطر القارئ إلى البحث عن صف المقياس المرجعي وطرح القيم يدويًا.

---

## 🏗️ Architecture | المعمارية

```
universal-market-predictor-deluxe/
├── app.py                    ← Gradio UI, loads the pretrained shared model
├── train.py                  ← Full training pipeline (Colab-ready, resumable)
├── add_new_ticker.py         ← Cold-start a new ticker into the trained model
├── config.yaml               ← Ticker universe + all hyperparameters
├── requirements.txt
├── notebooks/
│   └── train_on_colab.ipynb  ← One-click Colab training notebook
├── src/
│   ├── data_fetcher.py       ← yfinance wrapper (stocks + crypto, any exchange)
│   ├── indicators.py         ← 12 technical features, pure pandas (no ta-lib)
│   ├── dataset.py            ← Per-ticker windowing + multi-ticker assembly
│   ├── model.py              ← Shared LSTM + ticker embedding + multi-horizon head
│   ├── evaluation.py         ← Baselines, metrics, significance testing, walk-forward backtesting
│   └── charts.py             ← Plotly figures
└── model_registry/            ← Generated by train.py (not in git until trained)
    ├── shared_model.keras
    ├── ticker_id_map.json
    ├── meta.json
    ├── scalers/<TICKER>.pkl
    └── metrics/*.csv
```

### The Shared Model, Visually | النموذج المشترك بصريًا

```
price_sequence (60 days × 12 features)     ticker_id
              │                                │
        LSTM(128) → Dropout → BatchNorm   Embedding(64 slots × 16 dims)
        LSTM(64)  → Dropout → BatchNorm         │
        LSTM(32)  → Dropout                     │
              └──────────────┬──────────────────┘
                       Concatenate
                        Dense(32, relu)
                        Dense(16, relu)
                    Dense(3)  ← [+1 day, +3 day, +7 day] all at once
```

**Why shared, not one-model-per-ticker:**
the backbone sees market behavior across every stock, exchange, and coin in
the universe — crashes, rallies, volatility clusters — far more than any
single ticker's own history could teach it. The embedding lets the model
specialize its output per asset without needing a separate network per
asset.

**لماذا نموذجٌ مشترك، لا نموذجٌ منفصل لكل تكر:**
يرى العمود الفقري للنموذج (backbone) سلوك السوق عبر كل سهمٍ وبورصةٍ وعملةٍ
رقمية في النطاق المُدرَّب عليه — الانهيارات، والصعودات، وتجمّعات التقلّب —
بما يفوق كثيرًا ما يستطيع تاريخ أي تكرٍ بمفرده أن يُعلّمه إياه. أما
الـembedding فيتيح للنموذج تخصيص مخرجاته لكل أصلٍ على حدة، دون الحاجة إلى
شبكةٍ منفصلة لكل أصل.

**Why multi-output instead of recursive:**
predicting day+1 and then feeding that prediction back in as "day+1's
actual price" to predict day+2 compounds error fast (this is exactly what
went wrong in the earlier prototype scripts — see the "very bad results"
iterations in the project history). A single forward pass emitting all
horizons at once avoids that entirely.

**لماذا الإخراج المتعدد (multi-output) لا الأسلوب التكراري (recursive):**
إن التنبؤ باليوم+1 ثم إعادة تغذية ذلك التنبؤ كما لو كان "السعر الفعلي
لليوم+1" للتنبؤ باليوم+2 يُراكم الخطأ بسرعة (وهذا بالضبط ما حدث خطأً في
النصوص البرمجية الأولية للمشروع — انظر تكرارات "النتائج السيئة جدًا" في
تاريخ المشروع). أما التمريرة الأمامية الواحدة التي تُخرج جميع الآفاق دفعةً
واحدة، فتتجنّب هذه المشكلة كليًا.

**Why the model predicts a % return, not a scaled price (a real bug we hit
and fixed):**
the first fully-trained version of this project predicted a
MinMax-scaled Close price directly, and a real training run exposed the
problem immediately — on trending stocks like AAPL/MSFT, `shared_lstm` lost
badly to the `naive_persistence` baseline (higher MAE, negative R²). The
cause: each ticker's scaler is fit only on its training period (by design,
to avoid look-ahead leakage), so after 10 years of upward trend, the test
period's prices sat outside the [0,1] range the model had ever seen — it
was being asked to extrapolate blind. The fix has two parts:
1. Every model input feature (`src/indicators.py`) is now expressed as a
   bounded ratio or percentage (`MACD_pct`, `BB_Upper_pct`, `ATR_pct`,
   `Price_vs_MA20`, etc.) instead of a raw price/volume level, so nothing
   fed to the network drifts out of range as a ticker's price grows.
2. The target itself is a percentage return —
   `(Close[t+h] - Close[t]) / Close[t]` — reconstructed back to a price at
   inference time as `anchor_price * (1 + predicted_return)`
   (`src/dataset.py::return_to_price`), not a scaler inverse-transform.

This is the evaluation methodology (see above) doing exactly its job:
catching a model that looked reasonable in isolation but was actually worse
than doing nothing, before anyone trusted it.

**لماذا يتنبأ النموذج بنسبة عائدٍ مئوية، لا بسعرٍ مُعايَر (خللٌ حقيقي واجهناه
وأصلحناه):**
كانت أوّل نسخةٍ مُدرَّبة بالكامل من هذا المشروع تتنبأ مباشرةً بسعر إغلاقٍ
مُعايَرٍ بطريقة MinMax، وقد كشف تدريبٌ فعلي عن المشكلة على الفور — ففي
أسهمٍ ذات اتجاهٍ صاعد مثل AAPL وMSFT، خسر `shared_lstm` خسارةً فادحة أمام
مقياس `naive_persistence` (قيمة MAE أعلى، وR² سالبة). والسبب: أن لكل تكرٍ
مقياس معايرة (scaler) خاصًا به يُدرَّب فقط على فترة التدريب الخاصة به (وهذا
بالتصميم، تجنّبًا لتسرّب معلوماتٍ مستقبلية)، فبعد عشر سنواتٍ من الاتجاه
الصاعد، وقعت أسعار فترة الاختبار خارج النطاق [0,1] الذي شهده النموذج من
قبل — أي أنه كان مطالَبًا بالاستقراء (extrapolation) بصورةٍ عمياء. ويتألف
الإصلاح من جزأين:
1. أصبحت كل ميزةٍ من الميزات المُدخَلة إلى النموذج (`src/indicators.py`)
   تُعبَّر الآن كنسبةٍ أو قيمةٍ مئويةٍ محدودة (`MACD_pct`، `BB_Upper_pct`،
   `ATR_pct`، `Price_vs_MA20`، وغيرها) بدلًا من مستوى سعرٍ أو حجم تداولٍ
   خام، بحيث لا يخرج أي مُدخَلٍ للشبكة عن نطاقه مع نمو سعر التكر.
2. أصبح الهدف نفسه نسبة عائدٍ مئوية —
   `(Close[t+h] - Close[t]) / Close[t]` — تُعاد إلى صورة سعرٍ وقت الاستدلال
   عبر `anchor_price * (1 + predicted_return)`
   (`src/dataset.py::return_to_price`)، لا عبر عكس تحويل المقياس (scaler).

وهذا بالضبط ما تقوم به منهجية التقييم (انظر أعلاه) على أكمل وجه: ضبط نموذجٍ
بدا معقولًا حين يُنظر إليه بمعزلٍ عن غيره، بينما كان في الحقيقة أسوأ من عدم
فعل أي شيء، قبل أن يثق فيه أحد.

**Cold-start for new tickers, without a full retrain:**
`add_new_ticker.py` freezes every layer except the ticker-embedding table,
then fine-tunes on just the new ticker's data. Because Keras embedding
gradients are sparse — only the row actually looked up in a batch gets
updated — and every fine-tuning batch here only ever contains the new
ticker's id, this is mathematically guaranteed not to disturb any other
ticker's learned embedding. The new ticker is folded into the live model in
minutes, not by retraining the whole system.

**بدءٌ باردٌ (Cold-start) لأسهمٍ جديدة، دون إعادة تدريبٍ كاملة:**
يُجمِّد `add_new_ticker.py` جميع الطبقات باستثناء جدول ticker-embedding، ثم
يُجري ضبطًا دقيقًا (fine-tuning) يقتصر على بيانات التكر الجديد فحسب. ولأن
تدرّجات طبقة الـembedding في Keras متفرّقة (sparse) — إذ لا يُحدَّث سوى
الصف الذي جرى استدعاؤه فعليًا ضمن الدفعة (batch) — وبما أن كل دفعةٍ من
دفعات الضبط الدقيق هنا لا تحتوي إلا على معرّف التكر الجديد، فإن ذلك
مضمونٌ رياضيًا ألّا يمسّ أي embedding تعلّمه تكرٌ آخر. وهكذا ينضم التكر
الجديد إلى النموذج الحي في دقائق معدودة، دون إعادة تدريب النظام بأكمله.

---

## 🚀 Quick Start | البدء السريع

### 1. Train the shared model (Google Colab, free GPU) | تدريب النموذج المشترك (Google Colab، وحدة معالجة رسومية مجانية)

Open `notebooks/train_on_colab.ipynb` in Colab, or from a terminal:

```bash
pip install -r requirements.txt
python train.py
```

Training is **resumable** — if a Colab session disconnects, just re-run;
`train.py` picks up from the last checkpoint (`model_registry`/checkpoint
dir, saved every `checkpoint_every_n_epochs` epochs per `config.yaml`)
instead of starting over.

التدريب **قابلٌ للاستئناف** — إذا انقطعت جلسة Colab، يكفي إعادة تشغيل النص
البرمجي؛ إذ يستكمل `train.py` العمل من آخر نقطة حفظ (checkpoint) (داخل
مجلد `model_registry`/checkpoint، الذي يُحفَظ كل `checkpoint_every_n_epochs`
حسب `config.yaml`) بدلًا من البدء من جديد.

### 2. Run the app locally | تشغيل التطبيق محليًا

```bash
python app.py
```

Open `http://localhost:7860`.

### 3. Add a new ticker later, without retraining everything | إضافة تكرٍ جديد لاحقًا، دون إعادة تدريب كل شيء

```bash
python add_new_ticker.py --ticker 2010.SR --name "SABIC" --region SA
```

Then add it to `config.yaml`'s `universe` list too, so the *next* full
retrain includes it from scratch as well.

ثم أضِفه أيضًا إلى قائمة `universe` في `config.yaml`، حتى تشمله *إعادة
التدريب الكاملة القادمة* من الصفر كذلك.

---

## 🌐 Ticker Universe | نطاق الأسهم المدعومة

**[English]**
Configured in `config.yaml`. Ships with US large caps, Gulf/MENA exchanges
(Tadawul, Boursa Kuwait, Qatar Exchange, Dubai Financial Market, EGX),
UK/Germany/Japan/Hong Kong/India, and major cryptocurrencies. Add or remove
tickers there and re-run `train.py`.

**[العربية]**
مُعرَّفٌ في `config.yaml`. يأتي المشروع مزوَّدًا بأسهمٍ أمريكيةٍ كبرى،
وبورصات الخليج والشرق الأوسط (تداول، بورصة الكويت، بورصة قطر، سوق دبي
المالي، البورصة المصرية)، إضافةً إلى بريطانيا وألمانيا واليابان وهونغ كونغ
والهند، وأبرز العملات الرقمية. أضِف التكرات أو احذفها من هناك، ثم أعِد
تشغيل `train.py`.

**Gulf/Egypt tickers were verified individually against live Yahoo Finance
data** (some exchanges have inconsistent `yfinance` coverage, so a
plausible-looking symbol can silently return nothing):

**تم التحقق من رموز الخليج ومصر فرديًا مقابل بيانات Yahoo Finance الحيّة**
(تغطية بعض البورصات في `yfinance` غير مستقرة، فقد يُرجع رمزٌ يبدو معقولًا
نتيجةً فارغة دون أي تنبيه):

| Ticker / الرمز | Company / الشركة | Exchange / البورصة |
|---|---|---|
| `2222.SR` | Saudi Aramco / أرامكو السعودية | 🇸🇦 Tadawul (Saudi) |
| `1120.SR` | Al Rajhi Bank / مصرف الراجحي | 🇸🇦 Tadawul (Saudi) |
| `NBK.KW` | National Bank of Kuwait / بنك الكويت الوطني | 🇰🇼 Boursa Kuwait |
| `KFH.KW` | Kuwait Finance House / بيت التمويل الكويتي | 🇰🇼 Boursa Kuwait |
| `QNBK.QA` | Qatar National Bank / بنك قطر الوطني | 🇶🇦 Qatar Exchange |
| `EMAAR.AE` | Emaar Properties / إعمار العقارية | 🇦🇪 Dubai Financial Market |
| `COMI.CA` | Commercial Intl. Bank / البنك التجاري الدولي | 🇪🇬 EGX (Egypt) |

Two symbols in earlier versions of this project didn't actually resolve on
Yahoo Finance and were replaced: `NBKK.KW` → `NBK.KW`, and `FAB.AD` (First
Abu Dhabi Bank isn't on Yahoo Finance at all) → `EMAAR.AE`.

رمزان في نسخٍ سابقة من هذا المشروع لم يكونا في الحقيقة يُترجَمان على Yahoo
Finance، فاستُبدلا: `NBKK.KW` ← `NBK.KW`، و`FAB.AD` (إذ إن بنك أبوظبي
الأول غير موجود على Yahoo Finance إطلاقًا) ← `EMAAR.AE`.

---

## ⚠️ Honest Limitations | القيود الصادقة

**[English]**
- **Walk-forward evaluation here re-evaluates the one trained model across
  chronological test-set folds — it does not re-fit the model per fold.**
  True walk-forward retraining (refit at every rolling window) would be far
  more rigorous but is too slow to run across a whole universe on a free
  Colab GPU. This is a deliberate, documented tradeoff, not an oversight.
- **Daily OHLCV + technical indicators carry limited predictive signal on
  their own.** No news, sentiment, macro, or order-flow data is used yet
  (see Roadmap). Treat directional accuracy in the high 50s as a modest
  edge, not a trading system.
- **A model that loses to `naive_persistence` is a real, expected possible
  outcome, not a bug to hide.** This project already caught one such case
  during development (raw-price scaling on trending stocks — now fixed, see
  Architecture above). Re-run the comparison report after every retrain and
  actually read it before trusting any forecast.
- **The directional edge is real for some tickers, absent for others — and
  weakest on the Gulf tickers specifically.** A full 23-ticker training run
  showed a statistically distinguishable directional edge (see
  "Is a directional edge even real?" above) on several large US/UK names
  (`HSBA.L`, `AAPL`, `GOOGL`, `NVDA`, `AMZN`, `MSFT`), but `2222.SR` (Saudi
  Aramco), `NBK.KW`, and `KFH.KW` came out at or below the 50% coin-flip
  line, with MAE 3.5-8% worse than `naive_persistence` too. The likely
  cause: the shared backbone's training data is still majority US/European,
  so Gulf tickers are relatively out-of-distribution for it, on top of
  having shorter/thinner trading history than the mega-caps. This is
  reported here rather than fixed, deliberately — see
  `comparison_report.csv` after every retrain rather than assuming
  performance is uniform across the universe.
- **This is a research/educational project.** Nothing here is financial
  advice. Markets involve real risk.

**[العربية]**
- **يُعيد تقييم walk-forward هنا تقييم النموذج المُدرَّب الواحد عبر فتراتٍ
  زمنيةٍ متتالية من مجموعة الاختبار — ولا يُعيد تدريب النموذج في كل فترة.**
  إن إعادة التدريب الحقيقية بأسلوب walk-forward (أي إعادة التدريب عند كل
  نافذةٍ متحركة) ستكون أكثر دقةً بكثير، لكنها بطيئة للغاية بحيث يتعذّر
  تنفيذها على كامل نطاق الأسهم باستخدام وحدة معالجة رسومية مجانية في Colab.
  وهذا خيارٌ مقصودٌ وموثَّق، لا سهو.
- **تحمل بيانات OHLCV اليومية والمؤشرات الفنية إشارةً تنبؤيةً محدودة
  بمفردها.** لا تُستخدم حتى الآن بيانات الأخبار أو المعنويات أو المؤشرات
  الكلية أو تدفّق الأوامر (انظر خطة التطوير). تعامَل مع دقة اتجاهٍ في
  النطاق العلوي من الخمسينيات بالمئة باعتبارها ميزةً متواضعة، لا نظام
  تداول.
- **كون النموذج يخسر أمام `naive_persistence` نتيجةٌ حقيقيةٌ ومتوقّعة، لا
  خللًا يجب إخفاؤه.** وقد رصد هذا المشروع بالفعل حالةً كهذه أثناء التطوير
  (معايرة السعر الخام في أسهمٍ صاعدة — وقد أُصلحت الآن، انظر قسم المعمارية
  أعلاه). شغِّل تقرير المقارنة بعد كل إعادة تدريب، واقرأه فعليًا قبل
  الوثوق بأي تنبؤ.
- **ميزة الاتجاه حقيقيةٌ في بعض الأسهم، وغائبةٌ في أسهمٍ أخرى — وهي في
  أضعف حالاتها في أسهم الخليج تحديدًا.** أظهر تدريبٌ فعليٌّ على 23 تكرًا
  ميزةً اتجاهيةً مميّزةً إحصائيًا (انظر "هل تُعدّ ميزة الاتجاه حقيقيةً
  فعلًا؟" أعلاه) في عدة أسهمٍ أمريكيةٍ وبريطانيةٍ كبرى (`HSBA.L`، `AAPL`،
  `GOOGL`، `NVDA`، `AMZN`، `MSFT`)، بينما جاءت `2222.SR` (أرامكو السعودية)
  و`NBK.KW` و`KFH.KW` عند خط الـ50% (رمية العملة) أو دونه، مع قيمة MAE
  أسوأ من `naive_persistence` بنسبة 3.5-8% أيضًا. والسبب المُرجَّح: أن
  بيانات تدريب العمود الفقري المشترك لا تزال أغلبها أمريكيًا وأوروبيًا، ما
  يجعل أسهم الخليج "خارج نطاق التوزيع" نسبيًا بالنسبة إليه، إضافةً إلى أن
  تاريخ تداولها أقصر وأقل كثافةً من الأسهم العملاقة. وقد جرى توثيق هذا
  الأمر هنا عمدًا بدلًا من إصلاحه — فراجع `comparison_report.csv` بعد كل
  إعادة تدريب، بدلًا من افتراض أن الأداء موحّدٌ عبر كامل نطاق الأسهم.
- **هذا مشروعٌ بحثيٌّ وتعليمي.** لا شيء هنا يُعدّ نصيحةً مالية. تنطوي
  الأسواق المالية على مخاطرةٍ حقيقية.

---

## 🗺️ Roadmap | خطة التطوير

- [x] Shared multi-ticker model with embeddings + multi-horizon forecasting / نموذج مشترك متعدد الأسهم بتضمينات (embeddings) وتنبؤ متعدد الآفاق
- [x] Baseline-aware evaluation + walk-forward folds + statistical significance testing / تقييمٌ واعٍ بالمقياس المرجعي + فترات walk-forward + اختبار دلالة إحصائية
- [ ] News/sentiment features (Arabic + English) / ميزاتٌ مستخرجة من الأخبار والمعنويات (بالعربية والإنجليزية)
- [ ] Macro indicators (rates, oil, FX) as auxiliary inputs / مؤشراتٌ كلية (أسعار الفائدة، النفط، العملات) كمدخلاتٍ مساعدة
- [ ] Portfolio-level view across multiple held tickers / عرضٌ على مستوى المحفظة عبر عدة أسهم
- [ ] REST API endpoint alongside the Gradio UI / نقطة نهاية REST API إلى جانب واجهة Gradio

---

## ⚠️ Disclaimer | إخلاء المسؤولية

> **This project is for educational and research purposes only.**
> Predictions generated by this tool are **not** financial advice.
> Markets involve real risk — always consult a qualified financial advisor
> before making investment decisions.

> **هذا المشروع مُعدٌّ لأغراضٍ تعليميةٍ وبحثيةٍ فقط.**
> التوقعات التي تُنتجها هذه الأداة **ليست** نصيحةً مالية.
> تنطوي الأسواق المالية على مخاطرةٍ حقيقية — استشر دائمًا مستشارًا ماليًا
> مؤهَّلًا قبل اتخاذ أي قرارٍ استثماري.

---

## 👤 Author | المطور

<div align="center">

**Ahmed Darwish**

*Electrical & Computer Engineer | Python · Arduino · Raspberry Pi · AI/ML*

[![Email](https://img.shields.io/badge/Email-eahmeddarwish%40gmail.com-EA4335?logo=gmail&logoColor=white)](mailto:eahmeddarwish@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-eahmeddarwish-181717?logo=github)](https://github.com/eahmeddarwish)

</div>

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

```
MIT License — free to use, modify, and distribute with attribution.
```

---

<div align="center">

*Made with ❤️ by Ahmed Darwish*

</div>
