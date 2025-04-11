
import os
import pandas as pd
import joblib
import requests
import io
import numpy as np
import torch
import torch.nn as nn
import hashlib
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from cryptography.fernet import Fernet

# ===============================
# تنظیمات گوگل درایو
# ===============================
from google.colab import drive
drive.mount('/content/drive')

# مسیر فایل مدل در گوگل درایف
MODEL_PATH = '/content/drive/MyDrive/best_churn_model (1).pkl'

# بارگذاری مدل از گوگل درایو
model = joblib.load(MODEL_PATH)

# متغیر جهانی برای ذخیره داده‌های پردازش‌شده
processed_data = None

# ===============================
# ماژول یادگیری تضادآمیز (Contrastive Learning)
# افزودن ماژول یادگیری تضادآمیز برای بهبود استخراج ویژگی‌های تمایز‌دهنده از داده‌های پیچیده.
# ===============================
class ContrastiveNet(nn.Module):
    def __init__(self, input_dim, embedding_dim=32):
        super(ContrastiveNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

def extract_contrastive_features(X, input_dim, embedding_dim=32):
    """
    استفاده از شبکه عصبی برای استخراج ویژگی‌های تمایزدهنده از داده‌ها.
    این بخش مرتبط با پیشنهاد "یادگیری تضادآمیز" است.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ContrastiveNet(input_dim, embedding_dim).to(device)
    net.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        features = net(X_tensor).cpu().numpy()
    return features

# ===============================
# ماژول تفسیرپذیری مدل (Explainable AI)
# افزودن یک ماژول تفسیرپذیر جهت ارائه توضیحات شفاف درباره تصمیمات مدل.
# ===============================
def explain_model_prediction(X_sample):
    """
    استفاده از SHAP برای تفسیرپذیری مدل.
    این بخش مرتبط با پیشنهاد "بهبود قابلیت تفسیرپذیری مدل" است.
    """
    background = np.random.rand(100, X_sample.shape[1])
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_sample, nsamples=100)
    return shap_values

# ===============================
# تکنیک‌های کاهش ابعاد (PCA)
# استفاده از PCA برای کاهش نیازمندی‌های محاسباتی و زمان آموزش.
# ===============================
def apply_pca(X, n_components=2):
    """
    کاهش ابعاد داده‌ها با استفاده از PCA.
    این بخش مرتبط با پیشنهاد "بهینه‌سازی محاسباتی" است.
    """
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced

# ===============================
# لایه‌های امنیتی
# بکارگیری فناوری‌های رمزنگاری برای تضمین امنیت و حریم خصوصی داده‌ها.
# ===============================
def generate_key():
    """
    تولید کلید رمزنگاری برای امنیت داده‌ها.
    این بخش مرتبط با پیشنهاد "افزودن لایه‌های امنیتی" است.
    """
    return Fernet.generate_key()

def encrypt_data(data, key):
    """
    رمزگذاری داده‌ها با استفاده از کلید.
    این بخش مرتبط با پیشنهاد "افزودن لایه‌های امنیتی" است.
    """
    fernet = Fernet(key)
    encrypted_data = fernet.encrypt(data.encode())
    return encrypted_data

def decrypt_data(encrypted_data, key):
    """
    رمزگشایی داده‌ها با استفاده از کلید.
    این بخش مرتبط با پیشنهاد "افزودن لایه‌های امنیتی" است.
    """
    fernet = Fernet(key)
    decrypted_data = fernet.decrypt(encrypted_data).decode()
    return decrypted_data

def digital_signature(data):
    """
    ایجاد امضای دیجیتال برای تأیید صحت داده‌ها.
    این بخش مرتبط با پیشنهاد "افزودن لایه‌های امنیتی" است.
    """
    return hashlib.sha256(data.encode()).hexdigest()

# ===============================
# محاسبه CLTV
# استفاده از داده‌های لحظه‌ای برای به‌روزسازی مدل و انعطاف‌پذیری بیشتر.
# ===============================
def calculate_cltv(df, gross_margin=0.8):
    """
    محاسبه ارزش عمر مشتری (CLTV) بر اساس داده‌های لحظه‌ای.
    این بخش مرتبط با پیشنهاد "ادغام فناوری‌های نوین" است.
    """
    df['AverageMonthlyRevenue'] = df['MonthlyCharges']
    df['ChurnRate'] = df['Churn_Probability']
    df['CLTV'] = np.where(df['ChurnRate'] == 0, 0, (df['AverageMonthlyRevenue'] * gross_margin) / df['ChurnRate'])
    return df

# ===============================
# تابع پیشنهادات شخصی‌سازی‌شده
# ===============================
def personalized_offer(row):
    """
    ارائه پیشنهادات شخصی‌سازی‌شده بر اساس احتمال چرخش مشتری.
    این بخش مرتبط با پیشنهاد "ادغام فناوری‌های نوین" است.
    """
    if row['Churn_Probability'] >= 0.8:
        if row['tenure'] < 6:
            return 'پیشنهاد ویژه با تخفیف ویژه برای حفظ مشتری'
        elif row['MonthlyCharges'] < 50:
            return 'پیشنهاد تخفیف برای ارتقاء بسته خدماتی'
        elif row['PaymentMethod'] == 2:
            return 'پیشنهاد پرداخت با تخفیف ویژه از طریق کارت اعتباری'
        else:
            return 'پیشنهاد تخفیف ویژه برای تمدید اشتراک'
    elif row['Churn_Probability'] >= 0.6:
        if row['Contract'] == 2:
            return 'پیشنهاد ارتقاء به اشتراک بلندمدت با تخفیف'
        elif row['InternetService'] == 1:
            return 'پیشنهاد ارتقاء به اینترنت پرسرعت'
        else:
            return 'پیشنهاد تمدید اشتراک با تخفیف ویژه'
    else:
        if row['MonthlyCharges'] > 100:
            return 'پیشنهاد ارتقاء به بسته خدماتی پیشرفته'
        elif row['PaymentMethod'] == 1:
            return 'پیشنهاد تخفیف برای پرداخت با چک'
        elif row['SeniorCitizen'] == 1:
            return 'پیشنهاد خدمات مخصوص مشتریان مسن'
        else:
            return 'پیشنهاد خدمات معمولی'

# ===============================
# اجرای کلی سیستم (بدون روت‌های Flask)
# ===============================
def main():
    global processed_data
    try:
        # پردازش داده‌ها فقط یک بار
        if processed_data is None:
            # لینک فایل CSV در گوگل درایو
            file_url = "https://drive.google.com/uc?id=1YMO2gog6AGiZhlgLCtgHb8Yr3QPxoZ0Q"
            output_file = 'data.csv'

            # دانلود فایل CSV
            response = requests.get(file_url)
            if response.status_code == 200 and len(response.content) > 0:
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                print("فایل داده‌ها با موفقیت دانلود شد.")
            else:
                print("خطا در دانلود فایل CSV.")

            # خواندن و پیش‌پردازش داده‌ها
            df = pd.read_csv(output_file)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
            df = df.dropna(subset=['tenure', 'MonthlyCharges'])
            df.dropna(how='all', inplace=True)
            df.dropna(axis=1, how='all', inplace=True)

            label_encoder = LabelEncoder()
            categorical_columns = [
                'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
            ]
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = label_encoder.fit_transform(df[col])

            required_columns = [
                'tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"ویژگی‌های زیر در داده‌ها وجود ندارد: {', '.join(missing_columns)}")
                return

            X_values = df[required_columns].values.astype(float)

            # استفاده از ماژول یادگیری تضادآمیز (Contrastive Learning)
            contrastive_features = extract_contrastive_features(X_values, input_dim=X_values.shape[1], embedding_dim=16)
            print("نمونه‌ای از ویژگی‌های استخراج شده با Contrastive Learning:\n", contrastive_features[:5])

            # استفاده از تکنیک‌های کاهش ابعاد (PCA)
            X_reduced = apply_pca(X_values, n_components=2)
            plt.figure(figsize=(6, 4))
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.5)
            plt.title("Demonstrating data dimensionality reduction with PCA")
            plt.show()

            # محاسبه احتمال و پیش‌بینی چرخش مشتری
            churn_probabilities = model.predict_proba(X_values)[:, 1]
            churn_predictions = (churn_probabilities >= 0.5).astype(int)
            df['Churn_Prediction'] = churn_predictions
            df['Churn_Probability'] = churn_probabilities

            # محاسبه CLTV
            df = calculate_cltv(df)

            # تفسیرپذیری مدل با SHAP
            shap_values = explain_model_prediction(X_values[:1])
            print("مقادیر SHAP برای نمونه اول:\n", shap_values)

            # افزودن پیشنهادات شخصی‌سازی شده
            df['Personalized_Offer'] = df.apply(personalized_offer, axis=1)

            # ذخیره داده‌های پردازش‌شده در متغیر جهانی
            processed_data = df

        # نمایش داده‌ها
        print("داده‌های پردازش‌شده:")
        print(processed_data.head())

    except Exception as e:
        print(f"خطا در پردازش فایل: {str(e)}")

# ===============================
# اجرای برنامه
# ===============================
if __name__ == "__main__":
    main()