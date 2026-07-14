<div align="center">

# 📈 Universal Market Predictor — Deluxe Edition

### One shared LSTM, every ticker, honest evaluation
### نموذج LSTM واحد مشترك، لكل الأسهم، بتقييم صادق

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
مبني على فكرة واحدة: **نظام التوقع مش أوثق من المقياس المرجعي اللي بيتقارن
بيه.** كل رقم بيُنتجه المشروع ده بيتعرض جنب مقياس مرجعي "من غير أي ذكاء
اصطناعي"، والمعمارية مبنية عشان تكبر (أسهم جديدة، بورصات جديدة، عملات رقمية
جديدة) من غير ما تحتاج إعادة تدريب من الصفر في كل مرة.

---

## ✨ What Changed vs. the Original Project | ما الذي تغيّر عن المشروع الأصلي

| | Original / الأصلي | Deluxe |
|---|---|---|
| Training / التدريب | On-demand, every request (1-2 min wait) | Pretrained offline (Colab/local), instant inference |
| Model / الموديل | One separate LSTM per ticker | **One shared LSTM backbone + per-ticker embeddings** |
| Forecast horizon / أفق التوقع | Next day only | **1 / 3 / 7 trading days**, direct multi-output (not recursive) |
| Evaluation / التقييم | Raw MAE/MAPE/R² only | Same metrics **plus a naive-persistence & moving-average baseline on every single prediction, plus a statistical significance test on directional accuracy** |
| Backtesting / الاختبار الخلفي | Single train/test split | Single split + **chronological walk-forward folds** to catch regime-dependent failure |
| New tickers / أسهم جديدة | Not supported without retraining everything | **Cold-start fine-tuning**: freeze the backbone, train only the new ticker's embedding row |
| Cache / التخزين المؤقت | Ephemeral, resets on restart | Model is a versioned artifact in `model_registry/`, meant to be committed/shipped |

---

## 🎯 Why Compare Against a Baseline at All? | ليه لازم نقارن بمقياس مرجعي أصلاً؟

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
سعر إغلاق الغد لسهم كبير وسائل التداول فيه عالي بيكون *عادةً* قريب جدًا من
سعر إغلاق النهاردة. ده معناه إن أي موديل — حتى لو مفيدوش أي حاجة فعلًا —
ممكن يطلّع رقم MAPE يبان كويس بس لأنه مستفيد من الحقيقة دي. الطريقة الوحيدة
عشان نعرف لو الـLSTM اتعلم حاجة حقيقية هي إننا نحط الخطأ بتاعه ودقة اتجاهه
جنب:

- **`naive_persistence`** — التنبؤ بسعر الغد = سعر النهاردة، من غير أي تعلم آلي
- **`moving_average_5` / `moving_average_20`** — التنبؤ بمتوسط الأيام الأخيرة، برضه من غير تعلم آلي

لو `shared_lstm` مش متفوق بوضوح على `naive_persistence` في تكر/أفق معين —
خصوصًا في **دقة الاتجاه**، حيث 50% معناها رمي عملة معدنية — النتيجة دي
لازم تتعامل معاها كضوضاء مش كإشارة حقيقية. التطبيق وكل تقرير تدريب بيعرضوا
المقارنة دي مباشرة؛ مفيش حاجة مخبّية وراء رقم واحد براق زي "الموديل حقق
دقة 98%!".

### 🔬 Is a Directional Edge Even Real, or Just Noise? | هل ميزة الاتجاه حقيقية فعلًا ولا مجرد ضوضاء؟

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
تدريب فعلي على المشروع ده كشف نسخة تانية وأدق من نفس مشكلة "متصدقش رقم
براق": `shared_lstm` طلّع دقة اتجاه حوالي 52-58% في عدة أسهم، جنب مقاييس
مرجعية قريبة من خط الـ50% (رمي العملة). فرق نقطتين أو تلاتة فوق الـ50% على
كام مية نافذة اختبار ممكن يكون بسهولة ضوضاء إحصائية مش ميزة حقيقية — فـ
`comparison_report.csv` دلوقتي بيشغّل اختبار binomial ثنائي الطرف ضد فرضية
الـ50% على كل صف `shared_lstm`، بالإضافة لفترة ثقة Wilson بنسبة 95%
(`Direction_CI_95`، `Direction_p_value`، `Direction_Significant` في
`src/evaluation.py::direction_significance`). متعاملش مع دقة اتجاه أي
تكر/أفق كنتيجة حقيقية إلا لو `Direction_Significant` كانت `True` — يعني
فترة الثقة 95% مش شاملة الـ50% خالص. التقرير كمان بيضيف `R2_vs_naive` (R²
بتاع كل طريقة ناقص R² بتاع `naive_persistence` لنفس التكر/الأفق بالظبط)،
لأن R² عالي على سعر الغد غالبًا انعكاس طبيعي لارتباط السعر الذاتي مش دليل
على مهارة تنبؤية حقيقية، والقيمة المضافة الحقيقية عن "عدم فعل أي حاجة"
لازم تكون واضحة مباشرة من غير ما القارئ يدور على صف المقياس المرجعي
ويطرح بنفسه.

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

### The Shared Model, Visually | الموديل المشترك بصريًا

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

**ليه موديل مشترك، مش موديل منفصل لكل تكر:**
الـbackbone بيشوف سلوك السوق عبر كل سهم وبورصة وعملة رقمية في الكون
المُدرّب عليه — انهيارات، صعودات، تجمعات تقلب — أكتر بكتير مما يقدر تاريخ
أي تكر بمفرده يعلّمه. الـembedding بيخلي الموديل يخصص مخرجاته لكل أصل من
غير ما يحتاج شبكة منفصلة لكل أصل.

**Why multi-output instead of recursive:**
predicting day+1 and then feeding that prediction back in as "day+1's
actual price" to predict day+2 compounds error fast (this is exactly what
went wrong in the earlier prototype scripts — see the "very bad results"
iterations in the project history). A single forward pass emitting all
horizons at once avoids that entirely.

**ليه multi-output مش recursive:**
التنبؤ بيوم+1 وبعدين تغذية التنبؤ ده تاني كـ"السعر الفعلي ليوم+1" عشان
تتنبأ بيوم+2 بيراكم الخطأ بسرعة (ده بالظبط اللي حصل غلط في السكريبتات
الأولية للمشروع — شوف تكرارات "النتائج السيئة جدًا" في تاريخ المشروع).
تمريرة واحدة للأمام بتُخرج كل الآفاق مرة واحدة بتتجنب المشكلة دي خالص.

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

**ليه الموديل بيتنبأ بنسبة عائد مئوية، مش سعر معاير (باج حقيقي واجهناه
وأصلحناه):**
أول نسخة كاملة التدريب من المشروع كانت بتتنبأ مباشرة بسعر إغلاق معاير
بـMinMax، وتدريب فعلي كشف المشكلة على طول — في أسهم صاعدة زي AAPL/MSFT،
`shared_lstm` خسر بشدة قدام مقياس `naive_persistence` (MAE أعلى، وR² سالب).
السبب: كل تكر عنده scaler خاص بيتدرب بس على فترة التدريب بتاعته (بالتصميم،
عشان نتجنب تسرب معلومات مستقبلية)، فبعد 10 سنين من الاتجاه الصاعد، أسعار
فترة الاختبار كانت خارج نطاق [0,1] اللي الموديل شافه أصلًا — كان مطلوب منه
يستقرئ (extrapolate) بشكل أعمى. الإصلاح فيه جزئين:
1. كل ميزة مدخلة للموديل (`src/indicators.py`) دلوقتي متعبّرة كنسبة أو
   مئوية محدودة (`MACD_pct`، `BB_Upper_pct`، `ATR_pct`، `Price_vs_MA20`،
   إلخ) بدل مستوى سعر/حجم تداول خام، عشان مفيش حاجة داخلة للشبكة تخرج عن
   النطاق مع نمو سعر التكر.
2. الهدف نفسه بقى نسبة عائد مئوية —
   `(Close[t+h] - Close[t]) / Close[t]` — بيترجع لسعر وقت الاستدلال بـ
   `anchor_price * (1 + predicted_return)`
   (`src/dataset.py::return_to_price`)، مش عن طريق عكس الـscaler.

ده بالظبط منهجية التقييم (شوف فوق) بتعمل شغلها: بتمسك موديل كان شكله معقول
لوحده لكن كان في الحقيقة أسوأ من عدم فعل أي حاجة، قبل ما حد يثق فيه.

**Cold-start for new tickers, without a full retrain:**
`add_new_ticker.py` freezes every layer except the ticker-embedding table,
then fine-tunes on just the new ticker's data. Because Keras embedding
gradients are sparse — only the row actually looked up in a batch gets
updated — and every fine-tuning batch here only ever contains the new
ticker's id, this is mathematically guaranteed not to disturb any other
ticker's learned embedding. The new ticker is folded into the live model in
minutes, not by retraining the whole system.

**بدء بارد (Cold-start) لأسهم جديدة، من غير إعادة تدريب كاملة:**
`add_new_ticker.py` بيجمّد كل الطبقات ما عدا جدول ticker-embedding، وبعدين
بيعمل fine-tuning بس على بيانات التكر الجديد. لأن تدرجات Keras embedding
تفرّقية (sparse) — بس الصف اللي اتبحث عنه فعلًا في الدفعة (batch) هو اللي
بيتحدّث — وكل دفعة fine-tuning هنا بتحتوي بس على معرّف التكر الجديد، ده
مضمون رياضيًا إنه مش هيأثر على أي embedding اتعلمه تكر تاني. التكر الجديد
بينضم للموديل الحي في دقائق، مش عن طريق إعادة تدريب النظام كله.

---

## 🚀 Quick Start | البدء السريع

### 1. Train the shared model (Google Colab, free GPU) | تدريب الموديل المشترك (Google Colab، GPU مجاني)

Open `notebooks/train_on_colab.ipynb` in Colab, or from a terminal:

```bash
pip install -r requirements.txt
python train.py
```

Training is **resumable** — if a Colab session disconnects, just re-run;
`train.py` picks up from the last checkpoint (`model_registry`/checkpoint
dir, saved every `checkpoint_every_n_epochs` epochs per `config.yaml`)
instead of starting over.

التدريب **قابل للاستكمال** — لو جلسة Colab اتقطعت، شغّل السكريبت تاني؛
`train.py` بيكمل من آخر checkpoint (فولدر `model_registry`/checkpoint،
بيتحفظ كل `checkpoint_every_n_epochs` حسب `config.yaml`) بدل ما يبدأ من
الأول.

### 2. Run the app locally | تشغيل التطبيق محليًا

```bash
python app.py
```

Open `http://localhost:7860`.

### 3. Add a new ticker later, without retraining everything | إضافة تكر جديد لاحقًا، من غير إعادة تدريب كل حاجة

```bash
python add_new_ticker.py --ticker 2010.SR --name "SABIC" --region SA
```

Then add it to `config.yaml`'s `universe` list too, so the *next* full
retrain includes it from scratch as well.

بعد كده ضيفه لقائمة `universe` في `config.yaml` كمان، عشان *إعادة التدريب
الكاملة الجاية* تشمله من الصفر برضه.

---

## 🌐 Ticker Universe | كون الأسهم

**[English]**
Configured in `config.yaml`. Ships with US large caps, Gulf/MENA exchanges
(Tadawul, Boursa Kuwait, Qatar Exchange, Dubai Financial Market, EGX),
UK/Germany/Japan/Hong Kong/India, and major cryptocurrencies. Add or remove
tickers there and re-run `train.py`.

**[العربية]**
معرّف في `config.yaml`. المشروع بيجي مزوّد بأسهم أمريكية كبيرة، بورصات
الخليج/الشرق الأوسط (تداول، بورصة الكويت، بورصة قطر، سوق دبي المالي، البورصة
المصرية)، بريطانيا/ألمانيا/اليابان/هونج كونج/الهند، وأهم العملات الرقمية.
ضيف أو احذف تكرات من هناك وشغّل `train.py` تاني.

**Gulf/Egypt tickers were verified individually against live Yahoo Finance
data** (some exchanges have inconsistent `yfinance` coverage, so a
plausible-looking symbol can silently return nothing):

**رموز الخليج/مصر اتّتأكد منها فرديًا مقابل بيانات Yahoo Finance الحية**
(بعض البورصات تغطيتها في `yfinance` مش ثابتة، فرمز شكله معقول ممكن يرجّع
فراغ من غير أي تنبيه):

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

رمزين في نسخ سابقة من المشروع مكانوش بيترجموا فعليًا على Yahoo Finance
واتستبدلوا: `NBKK.KW` → `NBK.KW`، و`FAB.AD` (بنك أبوظبي الأول مش موجود
على Yahoo Finance خالص) → `EMAAR.AE`.

---

## ⚠️ Honest Limitations | قيود صادقة

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
- **تقييم walk-forward هنا بيعيد تقييم الموديل المتدرب الواحد عبر فترات
  زمنية متتالية من مجموعة الاختبار — مش بيعيد تدريب الموديل لكل فترة.**
  إعادة تدريب walk-forward حقيقية (إعادة تدريب في كل نافذة متحركة) هتكون
  أدق بكتير لكن بطيئة جدًا عشان تتنفذ على كون كامل من الأسهم على GPU مجاني
  في Colab. ده اختيار مقصود وموثّق، مش سهو.
- **بيانات OHLCV اليومية + المؤشرات الفنية بتحمل إشارة تنبؤية محدودة
  لوحدها.** مفيش بيانات أخبار أو معنويات أو ماكرو أو تدفق أوامر مستخدمة
  لسه (شوف خطة التطوير). تعامل مع دقة اتجاه في الخمسينات العليا كميزة
  متواضعة، مش كنظام تداول.
- **موديل بيخسر قدام `naive_persistence` هي نتيجة حقيقية ومتوقعة، مش باج
  نخبيه.** المشروع ده أصلًا مسك حالة زي كده أثناء التطوير (معايرة السعر
  الخام على أسهم صاعدة — دلوقتي متصلحة، شوف المعمارية فوق). شغّل تقرير
  المقارنة بعد كل إعادة تدريب واقرأه فعلًا قبل ما تثق في أي تنبؤ.
- **ميزة الاتجاه حقيقية في بعض الأسهم، غايبة في تانية — وأضعف حاجة تحديدًا
  في أسهم الخليج.** تدريب فعلي على 23 تكر أظهر ميزة اتجاهية مميزة إحصائيًا
  (شوف "هل ميزة الاتجاه حقيقية؟" فوق) في عدة أسهم أمريكية/بريطانية كبيرة
  (`HSBA.L`، `AAPL`، `GOOGL`، `NVDA`، `AMZN`، `MSFT`)، لكن `2222.SR` (أرامكو
  السعودية)، `NBK.KW`، و`KFH.KW` طلعوا عند أو تحت خط الـ50% (رمي العملة)،
  مع MAE أسوأ من `naive_persistence` بـ3.5-8% كمان. السبب المرجّح: بيانات
  تدريب الـbackbone المشترك لسه غالبيتها أمريكية/أوروبية، فأسهم الخليج
  نسبيًا "خارج التوزيع" بالنسبة له، فوق إن تاريخ تداولها أقصر وأقل كثافة من
  الأسهم العملاقة. ده متوثّق هنا عمدًا بدل ما يتصلح، — راجع
  `comparison_report.csv` بعد كل إعادة تدريب بدل افتراض إن الأداء موحّد عبر
  كل الكون.
- **ده مشروع بحثي/تعليمي.** مفيش حاجة هنا نصيحة مالية. الأسواق المالية فيها
  مخاطرة حقيقية.

---

## 🗺️ Roadmap | خطة التطوير

- [x] Shared multi-ticker model with embeddings + multi-horizon forecasting / موديل مشترك متعدد الأسهم بـembeddings وتوقع متعدد الآفاق
- [x] Baseline-aware evaluation + walk-forward folds + statistical significance testing / تقييم واعٍ بالمقياس المرجعي + فترات walk-forward + اختبار دلالة إحصائية
- [ ] News/sentiment features (Arabic + English) / ميزات أخبار/معنويات (عربي وإنجليزي)
- [ ] Macro indicators (rates, oil, FX) as auxiliary inputs / مؤشرات ماكرو (فوائد، نفط، عملات) كمدخلات مساعدة
- [ ] Portfolio-level view across multiple held tickers / عرض على مستوى المحفظة عبر عدة أسهم
- [ ] REST API endpoint alongside the Gradio UI / نقطة نهاية REST API جنب واجهة Gradio

---

## ⚠️ Disclaimer | إخلاء المسؤولية

> **This project is for educational and research purposes only.**
> Predictions generated by this tool are **not** financial advice.
> Markets involve real risk — always consult a qualified financial advisor
> before making investment decisions.

> **هذا المشروع لأغراض تعليمية وبحثية فقط.**
> التوقعات اللي بينتجها الأداة دي **مش** نصيحة مالية.
> الأسواق المالية فيها مخاطرة حقيقية — استشر دايمًا مستشار مالي مؤهّل قبل
> ما تاخد أي قرار استثماري.

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
