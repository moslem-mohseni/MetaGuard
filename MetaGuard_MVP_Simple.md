# پروژه MetaGuard - نسخه MVP (ساده و کاربردی)

---

## چرا این پروژه؟ (خیلی واضح)

**مشکل:** سال 2024، بیش از 3 میلیارد دلار از کلاهبرداری در متاورس خسارت وارد شده. هیچ ابزار ساده‌ای برای تشخیص این تهدیدات وجود نداره.

**راه‌حل ما:** یک کتابخانه Python که با 3 خط کد، بتونه به شما بگه یک تراکنش در متاورس مشکوک هست یا نه.

**هدف نهایی:** 
```python
from metaguard import check_transaction

result = check_transaction({"amount": 1000, "user": "user123"})
if result.is_suspicious:
    print(f"⚠️ مشکوک! ریسک: {result.risk_score}")
```

همین! نه بیشتر، نه کمتر.

---

## باید و نبایدهای پروژه

### ✅ بایدها
- **ساده باشه** - اگه بیشتر از 5 دقیقه طول بکشه که کسی بفهمه چطور کار می‌کنه، اشتباه کردیم
- **کار کنه** - حتی اگه دقت 70% باشه، بهتر از هیچی هست
- **سریع باشه** - کمتر از 1 ثانیه جواب بده
- **قابل استفاده باشه** - با pip install بشه نصبش کرد

### ❌ نبایدها
- **Over-engineering نکنیم** - نیازی به 50 ماژول نداریم
- **Perfect نباشیم** - ورژن 1 باید "good enough" باشه
- **پیچیده نکنیم** - اگه توضیحش بیشتر از 1 پاراگراف بشه، زیادی پیچیده‌س
- **همه چیز رو support نکنیم** - فقط 1-2 پلتفرم کافیه

---

## دیتاست‌ها (فقط آنچه واقعاً داریم)

### 1. دیتاست Kaggle - تراکنش‌های متاورس
```bash
# دانلود با PowerShell (Windows)
Invoke-WebRequest -Uri "https://www.kaggle.com/api/v1/datasets/download/faizaniftikharjanjua/metaverse-financial-transactions-dataset" -OutFile "data.zip"
Expand-Archive -Path "data.zip" -DestinationPath "data"
```

**چی داریم:** 100,000 تراکنش با label (fraud/normal)

### 2. دیتاست تولیدی
```python
# generate_data.py - ساده و سریع
import pandas as pd
import numpy as np

def generate_simple_data(n=10000):
    """فقط 5 ویژگی مهم"""
    return pd.DataFrame({
        'amount': np.random.lognormal(3, 2, n),
        'hour': np.random.randint(0, 24, n),
        'user_age_days': np.random.randint(1, 365, n),
        'transaction_count': np.random.poisson(3, n),
        'is_fraud': np.random.binomial(1, 0.05, n)  # 5% fraud
    })

# همین! بیشتر نیاز نداریم
```

---

## ساختار پروژه (فقط ضروری‌ها)

```
MetaGuard-MVP/
├── metaguard/
│   ├── __init__.py           # فقط import‌های اصلی
│   ├── detector.py           # یک فایل برای تشخیص
│   ├── risk.py              # محاسبه ریسک
│   └── models/
│       └── model.pkl        # مدل آموزش دیده
│
├── scripts/
│   ├── train.py             # آموزش مدل
│   └── generate_data.py     # تولید داده
│
├── data/                    # دیتاست‌ها
├── setup.py                 # برای pip install
├── requirements.txt         # فقط 5 کتابخانه
└── README.md               # توضیحات 1 صفحه‌ای
```

**همین 7 فایل Python!** بیشتر نیاز نداریم.

---

## کد اصلی (کل پروژه در 3 فایل)

### فایل 1: `detector.py` - تشخیص تهدید
```python
import pickle
import pandas as pd

class SimpleDetector:
    def __init__(self):
        # بارگذاری مدل از فایل
        with open('models/model.pkl', 'rb') as f:
            self.model = pickle.load(f)
    
    def detect(self, transaction):
        # تبدیل به DataFrame
        df = pd.DataFrame([transaction])
        
        # پیش‌بینی
        prob = self.model.predict_proba(df)[0][1]
        
        return {
            'is_suspicious': prob > 0.5,
            'risk_score': prob,
            'risk_level': 'High' if prob > 0.7 else 'Medium' if prob > 0.4 else 'Low'
        }
```

### فایل 2: `risk.py` - محاسبه ریسک (ساده‌ترین فرمول)
```python
def calculate_risk(amount, user_age, transaction_count):
    """
    فرمول خیلی ساده:
    Risk = (Amount / 1000) * (5 / User_Age) * (Transaction_Count / 10)
    """
    risk = (amount / 1000) * (5 / max(user_age, 1)) * (transaction_count / 10)
    return min(100, risk * 10)  # نرمال‌سازی به 0-100
```

### فایل 3: `train.py` - آموزش مدل
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# بارگذاری داده
data = pd.read_csv('data/transactions.csv')

# آماده‌سازی
X = data[['amount', 'hour', 'user_age_days', 'transaction_count']]
y = data['is_fraud']

# تقسیم داده
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# آموزش (با پارامترهای پیش‌فرض - ساده!)
model = RandomForestClassifier(n_estimators=50, max_depth=10)
model.fit(X_train, y_train)

# ذخیره
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# نمایش دقت
accuracy = model.score(X_test, y_test)
print(f"دقت: {accuracy:.2%}")
```

---

## نحوه استفاده (در 3 قدم)

### قدم 1: نصب
```bash
# Windows PowerShell
git clone https://github.com/yourname/metaguard-mvp.git
cd metaguard-mvp
pip install -r requirements.txt
```

### قدم 2: آموزش مدل
```bash
python scripts/train.py
# خروجی: دقت: 87.5%
```

### قدم 3: استفاده
```python
from metaguard.detector import SimpleDetector

detector = SimpleDetector()
result = detector.detect({
    'amount': 5000,
    'hour': 3,
    'user_age_days': 5,
    'transaction_count': 20
})

print(result)
# {'is_suspicious': True, 'risk_score': 0.82, 'risk_level': 'High'}
```

---

## requirements.txt (فقط 5 کتابخانه!)
```
pandas==2.0.3
scikit-learn==1.3.0
numpy==1.24.3
pickle-mixin==1.0.2
joblib==1.3.1
```

---

## رودمپ 4 هفته‌ای (نه 4 ماه!)

### هفته 1: آماده‌سازی داده
- [ ] دانلود دیتاست Kaggle
- [ ] نوشتن data generator
- [ ] ایجاد train/test split

### هفته 2: مدل ساده
- [ ] آموزش Random Forest
- [ ] تست با دقت حداقل 70%
- [ ] ذخیره مدل

### هفته 3: API ساده
- [ ] نوشتن detector.py
- [ ] نوشتن risk.py
- [ ] تست end-to-end

### هفته 4: انتشار
- [ ] ایجاد setup.py
- [ ] آپلود در GitHub
- [ ] نوشتن README ساده
- [ ] انتشار در PyPI

---

## معیارهای موفقیت MVP

| معیار | حداقل قابل قبول | ایده‌آل |
|------|-----------------|---------|
| دقت | 70% | 85% |
| سرعت | < 1 ثانیه | < 100ms |
| حجم کد | < 500 خط | < 300 خط |
| Setup time | < 5 دقیقه | < 2 دقیقه |

---

## مثال کامل End-to-End

```python
# example.py - یک فایل که همه چیز رو نشون میده
import pandas as pd
from metaguard.detector import SimpleDetector

# داده‌های تست
test_transactions = [
    {'amount': 100, 'hour': 14, 'user_age_days': 30, 'transaction_count': 5},    # Normal
    {'amount': 5000, 'hour': 3, 'user_age_days': 1, 'transaction_count': 50},    # Suspicious!
    {'amount': 200, 'hour': 20, 'user_age_days': 100, 'transaction_count': 10},  # Normal
]

# تشخیص
detector = SimpleDetector()

for i, tx in enumerate(test_transactions):
    result = detector.detect(tx)
    print(f"Transaction {i+1}: {result['risk_level']} (Score: {result['risk_score']:.2f})")
    
# خروجی:
# Transaction 1: Low (Score: 0.12)
# Transaction 2: High (Score: 0.89)
# Transaction 3: Low (Score: 0.23)
```

---

## چالش‌ها و محدودیت‌ها (صادقانه)

### چالش‌ها
1. **دیتاست محدود** - فقط 100K رکورد داریم
2. **فقط 4 ویژگی** - ممکنه برخی patterns رو miss کنیم
3. **یک مدل** - بدون ensemble یا backup

### محدودیت‌های MVP
- فقط تراکنش‌های مالی (نه رفتاری)
- فقط offline detection (نه real-time)
- بدون API (فقط Python library)

### راه‌حل در نسخه‌های بعدی
- v1.1: اضافه کردن behavioral detection
- v1.2: REST API
- v2.0: Real-time monitoring

---

## خلاصه در یک پاراگراف

MetaGuard یک کتابخانه **ساده** Python است که با **3 خط کد** می‌تواند تراکنش‌های مشکوک در متاورس را با دقت **بالای 70%** تشخیص دهد. با استفاده از **Random Forest** و فقط **4 ویژگی اصلی**، در **کمتر از 1 ثانیه** نتیجه می‌دهد. کل پروژه **کمتر از 300 خط کد** است و در **4 هفته** قابل تحویل است.

---

## نکته نهایی

> "Perfection is the enemy of good" - Voltaire

این MVP فقط باید **کار کنه** و **مفید باشه**. نسخه 2 رو بعداً می‌سازیم!

---

## دستورات سریع برای شروع (Windows)

```powershell
# Clone و Setup (2 دقیقه)
git clone https://github.com/yourname/metaguard-mvp.git
cd metaguard-mvp
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Train (30 ثانیه)
python scripts/train.py

# Test (5 ثانیه)
python example.py

# Done! 🎉
```

---

*نسخه: MVP 1.0*  
*تاریخ: آبان 1403*  
*خطوط کد: < 300*  
*زمان توسعه: 4 هفته*
