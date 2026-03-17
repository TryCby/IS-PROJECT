import streamlit as st
import joblib
import os

st.set_page_config(
    page_title="6704062611069 IS",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMarkdown p, .stMarkdown li { font-size: 1.15rem !important; line-height: 1.8 !important; }
    .main-title {
        text-align: center; background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 3.8rem; font-weight: 900; margin-bottom: 0.2rem;
    }
    .sub-title { text-align: center; color: #A0AEC0; font-size: 1.4rem; margin-bottom: 2.5rem; letter-spacing: 1px;}
    .project-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 35px; border-radius: 15px; 
        color: white; box-shadow: 0 10px 30px rgba(0,0,0,0.2); height: 100%; transition: transform 0.3s;
    }
    .project-card-nn {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%); padding: 35px; border-radius: 15px; 
        color: white; box-shadow: 0 10px 30px rgba(0,0,0,0.2); height: 100%; transition: transform 0.3s;
    }
    .project-card:hover, .project-card-nn:hover { transform: translateY(-8px); }
    .tech-badge {
        display: inline-block; background: rgba(255,255,255,0.2); padding: 6px 14px; 
        border-radius: 20px; margin: 4px; font-size: 0.95rem; border: 1px solid rgba(255,255,255,0.4);
    }
    .section-header { border-bottom: 2px solid #4FACFE; padding-bottom: 10px; margin-top: 30px; margin-bottom: 20px; }
    
    .stat-container { display: flex; flex-wrap: wrap; gap: 25px; justify-content: center; margin: 2.5rem 0; }
    .stat-box {
        background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08); padding: 30px 25px;
        border-radius: 20px; text-align: center; flex: 1; min-width: 220px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2); transition: all 0.4s ease;
        position: relative; overflow: hidden;
    }
    .stat-box::before {
        content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 4px;
        background: linear-gradient(90deg, #00F2FE, #4FACFE); opacity: 0; transition: opacity 0.3s;
    }
    .stat-box:hover::before { opacity: 1; }
    .stat-box:hover { transform: translateY(-8px); background: rgba(30, 41, 59, 0.9); box-shadow: 0 20px 40px rgba(0,242,254,0.15); }
    .stat-icon { font-size: 2.5rem; margin-bottom: 15px; opacity: 0.9; }
    .stat-value { font-size: 3.2rem; font-weight: 900; color: #E2E8F0; margin-bottom: 8px; line-height: 1.1; text-shadow: 0 2px 10px rgba(0,0,0,0.3); }
    .stat-value-ml { color: #38BDF8; } 
    .stat-value-nn { color: #C084FC; } 
    .stat-label { font-size: 1.15rem; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 5px; }
    .stat-desc { font-size: 0.95rem; color: #64748B; line-height: 1.4; }
    
    /* สไตล์สำหรับ Tools Card */
    .tool-card {
        background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.1);
        padding: 20px; border-radius: 15px; height: 100%;
    }

    /* สไตล์สำหรับปุ่ม GitHub */
    .github-btn-container { text-align: center; margin-bottom: 2.5rem; }
    .github-btn {
        display: inline-flex; align-items: center; justify-content: center;
        background-color: #24292e; color: #ffffff !important;
        padding: 12px 30px; border-radius: 30px; text-decoration: none;
        font-weight: bold; font-size: 1.1rem; transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1);
    }
    .github-btn:hover {
        background-color: #2b3137; transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(255,255,255,0.2); text-decoration: none;
    }
    .github-icon { font-size: 1.5rem; margin-right: 12px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🤖 Intelligence System Project</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">6704062611069 Teerayoth Chanbanyong</p>', unsafe_allow_html=True)

st.markdown("""
<div class="github-btn-container">
    <a href="https://github.com/TryCby/IS-PROJECT" target="_blank" class="github-btn">
        <span class="github-icon">💻</span> View Source Code on GitHub
    </a>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("<h2 class='section-header'>🎯 Project Objectives</h2>", unsafe_allow_html=True)
st.markdown("""
โครงการนี้จัดทำขึ้นเพื่อนำเสนอการประยุกต์ใช้วิทยาการข้อมูล (Data Science) และปัญญาประดิษฐ์ (Artificial Intelligence) ในการแก้ปัญหาจริง 2 ด้าน ได้แก่:
1. **ด้านเศรษฐศาสตร์พฤติกรรม:** ใช้ Machine Learning วิเคราะห์ว่าการจัดสรรรายจ่ายในชีวิตประจำวัน สามารถบ่งบอกถึงระดับฐานะทางการเงินและกำลังซื้อได้อย่างไร
2. **ด้านสาธารณสุขศาสตร์:** ใช้ Deep Learning (Neural Network) เพื่อคัดกรองความเสี่ยงของโรคที่เกี่ยวข้องกับการนอนหลับ จากดัชนีชี้วัดทางสรีรวิทยาและพฤติกรรม
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="project-card">
        <h2 style="margin-top:0; color:white;">💰 Consumer Financial Analysis</h2>
        <h5 style="opacity: 0.9; margin-bottom: 20px;">Machine Learning (Regression Task)</h5>
        <p>ระบบประเมินฐานะทางการเงินและกำลังซื้อของผู้บริโภคยุคใหม่ โดยเจาะลึกการวิเคราะห์โครงสร้างค่าใช้จ่ายทั้ง 10 หมวดหมู่ (เช่น ที่พัก, อาหาร, การพัฒนาตนเอง) เพื่อพยากรณ์รายได้ต่อเดือน</p>
        <p><strong>Models:</strong></p>
        <span class="tech-badge">Random Forest</span>
        <span class="tech-badge">Gradient Boosting</span>
        <span class="tech-badge">Support Vector Regressor (SVR)</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="project-card-nn">
        <h2 style="margin-top:0; color:white;">😴 Sleep Disorder Detection</h2>
        <h5 style="opacity: 0.9; margin-bottom: 20px;">Neural Network (Classification Task)</h5>
        <p>ระบบประเมินความเสี่ยงโรคการนอนหลับ (Insomnia / Sleep Apnea) โดยใช้โครงข่ายประสาทเทียมเรียนรู้ความสัมพันธ์เชิงซ้อนแบบ Non-linear จากข้อมูลสัญญาณชีพและไลฟ์สไตล์ 12 ตัวแปร</p>
        <p><strong>Model:</strong></p>
        <span class="tech-badge">Multilayer Perceptron (MLP)</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<h2 class='section-header'>📂 Folder Structure</h2>", unsafe_allow_html=True)
st.markdown("โปรเจกต์นี้ถูกออกแบบโครงสร้างแบบโมดูล (Modular Structure) เพื่อให้ง่ายต่อการดูแลรักษา (Maintainability) โดยแบ่งส่วนของ Machine Learning และ Neural Network ออกจากกันอย่างชัดเจน")

st.code("""
IS-PROJECT/
│
├── ML_Project/                         # 🤖 ส่วนงาน Machine Learning
│   ├── data/                           # เก็บชุดข้อมูลดิบและทำความสะอาดแล้ว
│   │   ├── student_spending.csv
│   │   └── student_spending_cleaned.csv
│   ├── models/                         # เก็บไฟล์โมเดลที่ผ่านการเทรนและ Scaler
│   │   ├── ensemble_model.pkl
│   │   ├── ml_features.pkl
│   │   ├── ml_metrics.pkl
│   │   └── ml_scaler.pkl
│   └── notebooks/                      # โค้ด Jupyter Notebook สำหรับทำ Data Prep. และ Model Training
│       ├── 01_ML_Data_Preparation.ipynb
│       └── 02_ML_Model_Training.ipynb
│
├── NN_Project/                         # 🧠 ส่วนงาน Neural Network
│   ├── data/                           # เก็บชุดข้อมูลดิบและทำความสะอาดแล้ว
│   │   ├── Sleep_health_and_lifestyle_dataset.csv
│   │   └── sleep_health_cleaned.csv
│   ├── models/                         # เก็บไฟล์โมเดล MLP, Label Encoder และ Scaler
│   │   ├── nn_features.pkl
│   │   ├── nn_label_encoder.pkl
│   │   ├── nn_metrics.pkl
│   │   ├── nn_mlp_model.pkl
│   │   └── nn_scaler.pkl
│   └── notebooks/                      # โค้ด Jupyter Notebook สำหรับทำ Preprocessing และออกแบบ Architecture
│       ├── 01_NN_Data_Preprocessing.ipynb
│       └── 02_NN_Architecture.ipynb
│
├── pages/                              # 🌐 หน้าเว็บแอปพลิเคชัน (แสดงผลที่ Sidebar)
│   ├── 1_ML_Theory.py                  # อธิบายทฤษฎีฝั่ง Machine Learning
│   ├── 2_ML_Test.py                    # ระบบทดสอบทำนายข้อมูลฝั่ง ML
│   ├── 3_NN_Theory.py                  # อธิบายทฤษฎีฝั่ง Neural Network
│   └── 4_NN_Test.py                    # ระบบทดสอบวิเคราะห์โรคฝั่ง NN
│
├── Homepage.py                         # 🏠 หน้าหลักของแอปพลิเคชัน (Main Entry Point)
└── requirements.txt                    # 📦 รายการไลบรารีที่จำเป็นสำหรับการรันโปรเจกต์
""", language="text")

st.markdown("<h2 class='section-header'>🛠️ Tools & Technologies</h2>", unsafe_allow_html=True)
t1, t2, t3, t4 = st.columns(4)

with t1:
    st.markdown("""
    <div class="tool-card">
        <h4 style="color: #4FACFE; margin-top:0;">💻 Languages & IDE</h4>
        <span class="tech-badge">Python 3.10+</span>
        <span class="tech-badge">Jupyter Notebook</span>
        <span class="tech-badge">VS Code</span>
    </div>
    """, unsafe_allow_html=True)

with t2:
    st.markdown("""
    <div class="tool-card">
        <h4 style="color: #4FACFE; margin-top:0;">🧠 AI & Machine Learning</h4>
        <span class="tech-badge">Scikit-learn (v1.8.0)</span>
        <span class="tech-badge">XGBoost (v3.2.0)</span>
        <span class="tech-badge">PyTorch (v2.10.0)</span>
    </div>
    """, unsafe_allow_html=True)

with t3:
    st.markdown("""
    <div class="tool-card">
        <h4 style="color: #4FACFE; margin-top:0;">📊 Data Processing</h4>
        <span class="tech-badge">Pandas (v2.3.3)</span>
        <span class="tech-badge">NumPy (v2.4.0)</span>
        <span class="tech-badge">Joblib (v1.5.3)</span>
    </div>
    """, unsafe_allow_html=True)

with t4:
    st.markdown("""
    <div class="tool-card">
        <h4 style="color: #4FACFE; margin-top:0;">🌐 Framework & Deploy</h4>
        <span class="tech-badge">Streamlit (v1.55.0)</span>
        <span class="tech-badge">Git & GitHub</span>
        <span class="tech-badge">Streamlit Cloud</span>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<h2 class='section-header'>📊 Model Performance Statistics</h2>", unsafe_allow_html=True)

ml_r2, ml_mae, nn_acc = "0.924", "$215", "89.5%"

try:
    if os.path.exists("ML_Project/models/ml_metrics.pkl"):
        ml_metrics = joblib.load("ML_Project/models/ml_metrics.pkl")
        ml_r2 = f"{ml_metrics['r2']:.3f}"
        ml_mae = f"${ml_metrics['mae']:,.0f}"
except: pass

try:
    if os.path.exists("NN_Project/models/nn_metrics.pkl"):
        nn_metrics = joblib.load("NN_Project/models/nn_metrics.pkl")
        nn_acc = f"{nn_metrics['accuracy']*100:.1f}%"
except: pass

st.markdown(f"""
<div class="stat-container">
    <div class="stat-box">
        <div class="stat-icon">🎯</div>
        <div class="stat-value stat-value-ml">{ml_r2}</div>
        <div class="stat-label">ML R² Score</div>
        <div class="stat-desc">ความแม่นยำในการพยากรณ์รายได้</div>
    </div>
    <div class="stat-box">
        <div class="stat-icon">📉</div>
        <div class="stat-value stat-value-ml">{ml_mae}</div>
        <div class="stat-label">ML MAE</div>
        <div class="stat-desc">ความคลาดเคลื่อนเฉลี่ยต่อเดือน</div>
    </div>
    <div class="stat-box">
        <div class="stat-icon">🧠</div>
        <div class="stat-value stat-value-nn">{nn_acc}</div>
        <div class="stat-label">NN Accuracy</div>
        <div class="stat-desc">ความแม่นยำในการจำแนกโรค</div>
    </div>
    <div class="stat-box">
        <div class="stat-icon">📈</div>
        <div class="stat-value text-white">1,374</div>
        <div class="stat-label">Total Data Samples</div>
        <div class="stat-desc">จำนวนข้อมูลที่ใช้ฝึกสอน AI</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<h2 class='section-header'>🗺️ System Structure</h2>", unsafe_allow_html=True)
st.markdown("""
| เมนู (Sidebar Menu) | ประเภท | คำอธิบายรายละเอียด (Description) |
| :--- | :--- | :--- |
| 🏠 **Homepage** | Overview | ภาพรวมโปรเจกต์ วัตถุประสงค์ และโครงสร้างของระบบ |
| 📖 **ML Theory** | Documentation | ทฤษฎีสมการคณิตศาสตร์เบื้องหลัง, การทำ Data Preprocessing, และ Evaluation Metrics ฝั่ง ML |
| 💰 **ML Test** | Application | ระบบจำลองการประเมินกำลังซื้อ โดยผู้ใช้สามารถกำหนดสัดส่วนรายจ่ายได้อิสระ |
| 🧠 **NN Theory** | Documentation | สถาปัตยกรรมโครงข่ายประสาทเทียม (Layers, Activation Functions) และกระบวนการเทรนโมเดล |
| 😴 **NN Test** | Application | ระบบคัดกรองความเสี่ยงโรคการนอนหลับ พร้อมแสดงค่าความน่าจะเป็น (Confidence Probability) |
""")