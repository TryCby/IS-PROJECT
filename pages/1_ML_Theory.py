import streamlit as st
import joblib

st.set_page_config(page_title="ML Theory", layout="wide")
st.title("📖 ทฤษฎี Machine Learning — Consumer Financial Analysis")
st.markdown("หน้าเว็บนี้อธิบายถึงกระบวนการทางวิทยาการข้อมูล (Data Science Pipeline) โดยแยกหัวข้อตามข้อกำหนดอย่างชัดเจน")
st.markdown("---")

st.header("1️⃣ แหล่งอ้างอิงข้อมูลที่นำมาใช้ (Dataset Reference)")
st.markdown("""
* **ชื่อชุดข้อมูลต้นฉบับ:** Student Spending Dataset (นำมาประยุกต์ใช้ในมุมมอง Consumer Financial Pattern)
* **แหล่งที่มา:** [Kaggle - Student Spending Dataset (by Sumanth Nimmagadda)](https://www.kaggle.com/datasets/sumanthnimmagadda/student-spending-dataset)
* **โครงสร้างข้อมูล:** ข้อมูลจำนวน 1,000 แถว (Samples)
""")

st.header("2️⃣ รายละเอียดตัวแปรชุดข้อมูล (Dataset Features)")
st.markdown("""
* **ตัวแปรอิสระ (Independent Variables - Features):** 15 ตัวแปร แบ่งเป็น:
  * *ประชากรศาสตร์:* `age` (อายุ), `gender` (เพศ), `year_in_school` (ระดับประสบการณ์), `major` (กลุ่มสายงาน)
  * *พฤติกรรมทางการเงิน:* `preferred_payment_method` (ช่องทางชำระเงินหลัก)
  * *โครงสร้างค่าใช้จ่าย 10 หมวดหมู่ (USD):* `tuition`, `housing`, `food`, `transportation`, `books_supplies`, `entertainment`, `personal_care`, `technology`, `health_wellness`, `miscellaneous`
* **ตัวแปรตาม (Dependent Variable - Target):** `monthly_income` (รายได้ต่อเดือน) ซึ่งเป็นเป้าหมายในการทำนายแบบต่อเนื่อง (Regression)
""")

st.header("3️⃣ ความไม่สมบูรณ์ของชุดข้อมูล (Dataset Imperfections)")
st.markdown("""
จากการสำรวจข้อมูลดิบ (Exploratory Data Analysis) พบปัญหาที่ต้องแก้ไขดังนี้:
1. **มีคอลัมน์ขยะ (Garbage Index):** มีคอลัมน์ `Unnamed: 0` ติดมาจากการ Export ไฟล์ CSV ซึ่งไม่มีความหมายเชิงสถิติ หากปล่อยไว้โมเดลอาจเกิดความลำเอียง (Bias)
2. **ชนิดข้อมูลผสมกัน (Mixed Data Types):** มีทั้งข้อมูลตัวเลข (ค่าใช้จ่าย) และข้อมูลข้อความ (เช่น Male/Female, Cash/Credit) ซึ่งคอมพิวเตอร์ไม่สามารถนำข้อความไปคำนวณในสมการได้
3. **ความแตกต่างของสเกล (Scale Discrepancy):** ตัวเลขในหมวดค่าใช้จ่ายมีสเกลต่างกันมาก เช่น `housing` อาจสูงถึงหลักพัน แต่ `personal_care` อยู่ในหลักสิบ
""")

st.header("4️⃣ การเตรียมข้อมูล (Data Preparation)")
st.markdown("""
เพื่อแก้ไขความไม่สมบูรณ์ด้านบน จึงได้ทำการเตรียมข้อมูลดังนี้:
1. **Data Cleansing:** ทำการลบคอลัมน์ `Unnamed: 0` ทิ้งออกจากระบบ
2. **One-Hot Encoding (OHE):** แปลงข้อมูลข้อความให้เป็นเมทริกซ์ 0 และ 1 (Binary Matrix) เพื่อให้โมเดลประมวลผลได้โดยไม่เข้าใจผิดว่าหมวดหมู่ใดมีค่ามากกว่ากัน
3. **Feature Standardization:** ปรับสเกลข้อมูลค่าใช้จ่ายให้อยู่ในมาตรฐานเดียวกัน (Mean=0, SD=1) ด้วย **StandardScaler** การทำสเกลนี้จำเป็นอย่างยิ่งสำหรับอัลกอริทึมที่อิงระยะทาง
""")

st.header("5️⃣ ทฤษฎีของอัลกอริทึมที่พัฒนา (Algorithm Theory)")
st.markdown("")
st.markdown("แบบจำลองนี้ใช้แนวคิด **Ensemble Learning** เพื่อลดความคลาดเคลื่อน โดยผสานรวม 3 อัลกอริทึมผ่านกลไก `VotingRegressor`:")
st.markdown("""
1. **Random Forest Regressor (Bagging Method):** สร้างต้นไม้ตัดสินใจจำนวนมากโดยสุ่มข้อมูล (Bootstrapping) ช่วยลดปัญหาความแปรปรวน (Variance)
2. **Gradient Boosting Regressor (Boosting Method):** สร้างต้นไม้แบบลำดับขั้น แต่ละต้นสร้างมาเพื่อแก้ไขข้อผิดพลาด (Residual Error) ของต้นก่อนหน้า ช่วยลดความลำเอียง (Bias)
3. **Support Vector Regressor (SVR):** ค้นหาสมการเส้นขอบเขต (Hyperplane) ในพื้นที่มิติสูง ที่ครอบคลุมข้อมูลส่วนใหญ่ ทนทานต่อข้อมูลค่าใช้จ่ายที่ผิดปกติ (Outliers)

**สมการการตัดสินใจร่วม (Voting Regressor Formulation):**
""")
st.latex(r"\hat{y} = \frac{RandomForest(x) + GradientBoosting(x) + SVR(x)}{3}")

st.header("6️⃣ ขั้นตอนการพัฒนาโมเดล (Model Development Steps)")
st.code("""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR

# 1. การแบ่งข้อมูล (Train/Test Split) สัดส่วน 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. การปรับสเกลข้อมูล (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. กำหนดสถาปัตยกรรมและ Hyperparameters
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
svr = SVR(kernel='linear', C=10)

# 4. รวมร่างโมเดล (Ensemble Setup)
ensemble_model = VotingRegressor(estimators=[('RF', rf), ('GB', gb), ('SVR', svr)])

# 5. ฝึกสอนโมเดล (Model Training)
ensemble_model.fit(X_train_scaled, y_train)
""", language="python")

st.markdown("---")
st.header("📈 ผลการประเมินประสิทธิภาพ (Evaluation Metrics)")
try:
    metrics = joblib.load("ML_Project/models/ml_metrics.pkl")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² (Coefficient of Determination)", f"{metrics['r2']:.4f}")
    m2.metric("MAE (Mean Absolute Error)", f"${metrics['mae']:,.2f}")
    m3.metric("RMSE (Root Mean Squared Error)", f"${metrics['rmse']:,.2f}")
except:
    st.warning("ยังไม่พบไฟล์ผลลัพธ์ กรุณาตรวจสอบกระบวนการ Training")