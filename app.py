from flask import Flask, request, render_template, send_file
import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import io
from cryptography.fernet import Fernet
import threading
import numpy as np
import torch
import torch.nn as nn
import hashlib
import shap
import requests

# ===============================
# تنظیمات اولیه Flask و بارگذاری مدل
# ===============================
app = Flask(__name__, static_folder='imgs')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
model = joblib.load('best_churn_model (1).pkl')

# ===============================
# توابع رمزنگاری و امنیت
# ===============================
def generate_key():
    return Fernet.generate_key()

def encrypt_data(data, key):
    fernet = Fernet(key)
    encrypted_data = fernet.encrypt(data.encode())
    return encrypted_data

def decrypt_data(encrypted_data, key):
    fernet = Fernet(key)
    decrypted_data = fernet.decrypt(encrypted_data).decode()
    return decrypted_data

def digital_signature(data):
    return hashlib.sha256(data.encode()).hexdigest()

# ===============================
# مدل نمونه یادگیری تضادآمیز (Contrastive Learning)
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ContrastiveNet(input_dim, embedding_dim).to(device)
    net.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        features = net(X_tensor).cpu().numpy()
    return features

# ===============================
# کاهش ابعاد با PCA
# ===============================
def apply_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced

# ===============================
# تفسیرپذیری مدل با SHAP
# ===============================
def explain_model_prediction(X_sample):
    background = np.random.rand(100, X_sample.shape[1])
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_sample, nsamples=100)
    return shap_values

# ===============================
# محاسبه CLTV
# ===============================
def calculate_cltv(df, gross_margin=0.8):
    df['AverageMonthlyRevenue'] = df['MonthlyCharges']
    df['ChurnRate'] = df['Churn_Probability']
    df['CLTV'] = np.where(df['ChurnRate'] == 0, 0, (df['AverageMonthlyRevenue'] * gross_margin) / df['ChurnRate'])
    return df

# ===============================
# روت‌های اصلی Flask
# ===============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluation')
def evaluation():
    # مسیر جدید برای صفحه evaluation.html
    return render_template('evaluation.html')

@app.route('/result', methods=['GET'])
def result():
    try:
        # لینک فایل CSV در گوگل درایو
        file_url = "https://drive.google.com/uc?id=1YMO2gog6AGiZhlgLCtgHb8Yr3QPxoZ0Q"
        output_file = os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv')

        # دانلود فایل از گوگل درایو
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        response = requests.get(file_url)
        if response.status_code == 200 and len(response.content) > 0:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print("فایل داده‌ها با موفقیت دانلود شد.")
        else:
            return "خطا در دانلود فایل CSV."

        # خواندن فایل CSV
        df = pd.read_csv(output_file)
        print("نمونه‌ای از داده‌های ورودی:\n", df.head())

        # پیش‌پردازش داده‌ها
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
            return f"ویژگی‌های زیر در داده‌ها وجود ندارد: {', '.join(missing_columns)}"

        X_values = df[required_columns].values.astype(float)
        X_reduced = apply_pca(X_values, n_components=2)
        plt.figure(figsize=(6, 4))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.5)
        plt.title("نمایش کاهش ابعاد داده‌ها با PCA")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        pca_image = buf.getvalue()
        buf.close()

        contrastive_features = extract_contrastive_features(X_values, input_dim=X_values.shape[1], embedding_dim=16)
        print("نمونه‌ای از ویژگی‌های استخراج شده با Contrastive Learning:\n", contrastive_features[:5])

        churn_probabilities = model.predict_proba(X_values)[:, 1]
        churn_predictions = (churn_probabilities >= 0.5).astype(int)
        df['Churn_Prediction'] = churn_predictions
        df['Churn_Probability'] = churn_probabilities

        # محاسبه CLTV
        df = calculate_cltv(df)

        shap_values = explain_model_prediction(X_values[:1])

        def personalized_offer(row):
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

        df['Personalized_Offer'] = df.apply(personalized_offer, axis=1)

        # نمایش 50 داده اول
        data_preview = df.head(50)

        table_html = data_preview.to_html(classes='data table table-bordered', escape=False)
        return render_template('result.html', 
                               table_html=table_html,
                               titles=df.columns.values,
                               pca_image=pca_image)
    except Exception as e:
        return f"خطا در پردازش فایل: {str(e)}"

# ===============================
# اجرای برنامه
# ===============================
if __name__ == "__main__":
    app.run(debug=True)