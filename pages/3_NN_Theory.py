import streamlit as st
import joblib

st.set_page_config(page_title="NN Theory", layout="wide")
st.title("🧠 ทฤษฎี Neural Network — Sleep Disorder Detection")
st.markdown("หน้าเว็บนี้อธิบายถึงกระบวนการสร้างและสถาปัตยกรรมโครงข่ายประสาทเทียม โดยแยกหัวข้อตามข้อกำหนดอย่างชัดเจน")
st.markdown("---")

st.header("1️⃣ แหล่งอ้างอิงข้อมูลที่นำมาใช้ (Dataset Reference)")
st.markdown("""
* **ชื่อชุดข้อมูลต้นฉบับ:** Sleep Health and Lifestyle Dataset
* **แหล่งที่มา:** [Kaggle - Sleep Health Dataset (by Laksika Tharmalingam)](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)
* **โครงสร้างข้อมูล:** ข้อมูลทางการแพทย์และไลฟ์สไตล์ 374 แถว (Samples)
""")

st.header("2️⃣ รายละเอียดตัวแปรชุดข้อมูล (Dataset Features)")
st.markdown("""
* **ตัวแปรอิสระ (Input Features):** 12 ตัวแปรหลัก ประกอบด้วย:
  * *ข้อมูลพื้นฐาน:* `Age`, `Gender`, `Occupation`, `BMI Category`
  * *คุณภาพการนอน:* `Sleep Duration` (ชั่วโมงการนอน), `Quality of Sleep` (คะแนน 1-10)
  * *ไลฟ์สไตล์และสรีรวิทยา:* `Physical Activity Level`, `Stress Level`, `Heart Rate`, `Daily Steps`, `Blood Pressure`
* **ตัวแปรเป้าหมาย (Target Class):** `Sleep Disorder` ทำหน้าที่แบ่งกลุ่มความเสี่ยงเป็น 3 กลุ่ม (Multi-class Classification) ได้แก่ `None` (ปกติ), `Insomnia` (นอนไม่หลับ), `Sleep Apnea` (หยุดหายใจขณะหลับ)
""")

st.header("3️⃣ ความไม่สมบูรณ์ของชุดข้อมูล (Dataset Imperfections)")
st.markdown("""
โครงข่ายประสาทเทียมมีความอ่อนไหวต่อข้อมูลดิบมาก จากการตรวจสอบ Dataset พบปัญหาหลักดังนี้:
1. **Missing Values ที่มีความหมาย:** คอลัมน์ `Sleep Disorder` มีค่าว่าง (NaN) ปะปนอยู่ ซึ่งในบริบทนี้ ค่าว่างหมายถึง "บุคคลที่ไม่มีโรค" 
2. **ข้อมูลซ้อนทับ (Unparseable String):** คอลัมน์ `Blood Pressure` ถูกบันทึกมาในรูปแบบข้อความ เช่น `"120/80"` ซึ่งสมองกลไม่สามารถอ่านเป็นตัวเลขได้
3. **คอลัมน์ที่ไม่จำเป็น (Irrelevant Features):** มีคอลัมน์ `Person ID` ซึ่งเป็นเพียงรหัสประจำตัว ไม่ส่งผลต่อโรค
""")

st.header("4️⃣ การเตรียมข้อมูล (Data Preparation)")
st.markdown("""
เพื่อแก้ไขปัญหาความไม่สมบูรณ์ของข้อมูล จึงได้เตรียมข้อมูลดังนี้:
1. **Target Imputation & Label Encoding:** จัดการค่าว่างในตัวแปรเป้าหมายด้วยคำว่า `None` และใช้ `LabelEncoder` แปลงคลาสเป็นตัวเลข `[0, 1, 2]`
2. **Feature Extraction:** ตัดแบ่ง (Split) คอลัมน์ความดันโลหิต ออกเป็น 2 คอลัมน์ปริมาณ คือ `Systolic_BP` (ตัวบน) และ `Diastolic_BP` (ตัวล่าง)
3. **Standardization:** แปลงตัวเลขเชิงปริมาณทั้งหมดเข้าสู่สมการ Z-Score (Standard Scaler) เพื่อป้องกันปัญหา Exploding Gradients ระหว่างการเทรน
""")

st.header("5️⃣ ทฤษฎีของอัลกอริทึมที่พัฒนา (Algorithm Theory)")
st.markdown("")
st.markdown("ระบบใช้โครงข่ายประสาทเทียมแบบ **Multilayer Perceptron (MLP)** หรือ Feedforward Neural Network โดยมีสถาปัตยกรรมดังนี้:")
st.markdown("""
* **Input Layer:** ชั้นรับข้อมูล นำเข้า 12 ตัวแปร
* **Hidden Layers:** ประกอบด้วย 2 ชั้นซ่อนเร้น (64 Neurons และ 32 Neurons) 
  * **Activation Function:** ใช้ฟังก์ชัน **ReLU (Rectified Linear Unit)** เพื่อจับความสัมพันธ์แบบไม่เป็นเส้นตรง (Non-linear) และแก้ปัญหา Vanishing Gradient
* **Output Layer:** มี 3 Neurons (ตามจำนวนโรค)
  * **Activation Function:** ใช้ **Softmax** แปลงผลลัพธ์ให้อยู่ในรูปของร้อยละความน่าจะเป็น (Probability)
* **Optimizer:** ใช้อัลกอริทึม **Adam** ในการอัปเดตค่าน้ำหนัก (Weights) ให้โมเดลฉลาดขึ้นอย่างรวดเร็ว
""")

st.header("6️⃣ ขั้นตอนการพัฒนาโมเดล (Model Development Steps)")
st.code("""
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. การจัดการตัวแปรเป้าหมาย (Label Encoding)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 2. การแบ่งข้อมูล (Stratified Train-Test Split)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded)

# 3. การสร้างสถาปัตยกรรมโครงข่าย (Network Configuration)
nn_model = MLPClassifier(
    hidden_layer_sizes=(64, 32), # กำหนด 2 Hidden Layers
    activation='relu',           # ฟังก์ชันกระตุ้น
    solver='adam',               # Optimizer
    early_stopping=True,         # หยุดอัตโนมัติเพื่อป้องกัน Overfitting
    validation_fraction=0.1,     # แบ่ง 10% ไว้ตรวจเช็ค Validation Loss
    max_iter=500
)

# 4. การฝึกสอนโมเดล (Forward & Backward Propagation)
nn_model.fit(X_train_scaled, y_train)

# 5. การประเมินผล (Inference)
y_pred = nn_model.predict(X_test_scaled)
""", language="python")

st.markdown("---")
st.header("📈 ผลการประเมินประสิทธิภาพ (Performance Metrics)")
try:
    metrics = joblib.load("NN_Project/models/nn_metrics.pkl")
    st.metric("🎯 Classification Accuracy (ความแม่นยำรวมจาก Test Set)", f"{metrics['accuracy']*100:.2f}%")
except:
    st.warning("ยังไม่พบไฟล์ผลลัพธ์")