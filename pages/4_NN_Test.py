import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

st.set_page_config(page_title="NN Prediction Test", layout="wide")
st.title("😴 ระบบคัดกรองและวิเคราะห์ความเสี่ยงโรคการนอนหลับ")
st.markdown("ระบบวิเคราะห์ความเสี่ยงจากข้อมูลทางสรีรวิทยาและพฤติกรรม พร้อมตรวจจับตัวแปรต้นเหตุ (Root Cause) และให้คำแนะนำเบื้องต้น")
st.markdown("---")

try:
    nn_model = joblib.load("NN_Project/models/nn_mlp_model.pkl")
    nn_scaler = joblib.load("NN_Project/models/nn_scaler.pkl")
    nn_features = joblib.load("NN_Project/models/nn_features.pkl")
    le = joblib.load("NN_Project/models/nn_label_encoder.pkl")
except Exception as e:
    st.error(f"❌ ระบบไม่สามารถโหลดสถาปัตยกรรมโมเดลได้ (Error: {e})")
    st.stop()

with st.form("nn_form"):
    st.markdown("#### 🧬 ส่วนที่ 1: ข้อมูลทางกายภาพพื้นฐาน")
    c1, c2, c3 = st.columns(3)
    with c1: age = st.number_input("อายุ (Age)", 10, 100, 30)
    with c2: gender = st.selectbox("เพศ (Gender)", ["Male", "Female"])
    with c3: bmi = st.selectbox("ดัชนีมวลกาย (BMI Category)", ["Normal", "Overweight", "Obese"])

    st.markdown("#### 🏃‍♂️ ส่วนที่ 2: ดัชนีพฤติกรรมและสถิติการนอน")
    col1, col2, col3, col4 = st.columns(4)
    with col1: physical_act = st.number_input("กิจกรรมทางกาย/วัน (นาที)", 0, 300, 50)
    with col2: stress_level = st.slider("ความเครียดสะสม (1-10)", 1, 10, 5)
    with col3: sleep_dur = st.slider("เวลานอนเฉลี่ย (ชม.)", 3.0, 12.0, 7.0, step=0.1)
    with col4: sleep_qual = st.slider("คุณภาพการนอน (1-10)", 1, 10, 7)

    st.markdown("#### 🩺 ส่วนที่ 3: สัญญาณชีพทางคลินิก (Vital Signs)")
    med1, med2, med3, med4 = st.columns(4)
    with med1: daily_steps = st.number_input("จำนวนก้าวสะสม (ก้าว/วัน)", 0, 30000, 5000)
    with med2: heart_rate = st.number_input("อัตราเต้นหัวใจขณะพัก (bpm)", 40, 140, 70)
    with med3: systo = st.number_input("ความดันตัวบน (Systolic BP)", 80, 220, 120)
    with med4: diasto = st.number_input("ความดันตัวล่าง (Diastolic BP)", 40, 140, 80)

    submit = st.form_submit_button("🧠 วิเคราะห์สัญญาณชีพและประเมินความเสี่ยง", type="primary", use_container_width=True)

if submit:
    with st.spinner("🤖 สัญญาณกำลังส่งผ่าน Forward Propagation ใน Neural Network..."):
        time.sleep(1.5)
        
        input_df = pd.DataFrame(columns=nn_features)
        input_df.loc[0] = 0 
        
        input_df['Age'] = age
        input_df['Sleep Duration'] = sleep_dur
        input_df['Quality of Sleep'] = sleep_qual
        input_df['Physical Activity Level'] = physical_act
        input_df['Stress Level'] = stress_level
        input_df['Heart Rate'] = heart_rate
        input_df['Daily Steps'] = daily_steps
        input_df['Systolic_BP'] = systo
        input_df['Diastolic_BP'] = diasto
        
        if f"Gender_{gender}" in input_df.columns: input_df[f"Gender_{gender}"] = 1
        if f"BMI Category_{bmi}" in input_df.columns: input_df[f"BMI Category_{bmi}"] = 1
        
        input_scaled = nn_scaler.transform(input_df)
        pred_idx = nn_model.predict(input_scaled)[0]
        prediction = le.inverse_transform([pred_idx])[0]
        probabilities = nn_model.predict_proba(input_scaled)[0]
        confidence = probabilities[pred_idx] * 100
        
        st.markdown("---")
        
        # ---------------------------------------------------------
        # แสดงผลลัพธ์หลัก
        # ---------------------------------------------------------
        if prediction == "None":
            st.success(f"### 🟢 ผลคัดกรอง: ปกติ (Normal) — ความเชื่อมั่น: {confidence:.2f}%")
        elif prediction == "Insomnia":
            st.warning(f"### 🟠 ผลคัดกรอง: ภาวะนอนไม่หลับ (Insomnia) — ความเชื่อมั่น: {confidence:.2f}%")
        else:
            st.error(f"### 🔴 ผลคัดกรอง: ภาวะหยุดหายใจขณะหลับ (Sleep Apnea) — ความเชื่อมั่น: {confidence:.2f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # ---------------------------------------------------------
        # ส่วนที่เพิ่มใหม่: ระบบตรวจจับสาเหตุและข้อเสนอแนะทางการแพทย์
        # ---------------------------------------------------------
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("#### 🚨 สัญญาณอันตรายที่ตรวจพบ (Identified Risk Factors)")
            risks_found = False
            
            if stress_level >= 7:
                st.warning(f"**ความเครียดสูง (Stress Level = {stress_level}):** เป็นสาเหตุหลักของการตื่นกลางดึกและส่งผลโดยตรงต่อการเกิดภาวะ Insomnia")
                risks_found = True
            if sleep_dur < 6:
                st.warning(f"**ชั่วโมงการนอนต่ำ (Duration = {sleep_dur} ชม.):** การนอนไม่ถึง 6 ชั่วโมงต่อวันเพิ่มความเสี่ยงต่อโรคหลอดเลือดและหัวใจ")
                risks_found = True
            if bmi in ["Overweight", "Obese"]:
                st.error(f"**ดัชนีมวลกายเกินเกณฑ์ ({bmi}):** ภาวะน้ำหนักเกิน ทำให้ทางเดินหายใจแคบลงขณะหลับ เป็นปัจจัยเสี่ยงอันดับ 1 ของภาวะ Sleep Apnea")
                risks_found = True
            if systo >= 130 or diasto >= 85:
                st.error(f"**ความดันโลหิตสูง ({systo}/{diasto}):** ความดันที่สูงผิดปกติมีความสัมพันธ์ใกล้ชิดกับปัญหาทางเดินหายใจขณะหลับ")
                risks_found = True
            if physical_act < 30 or daily_steps < 4000:
                st.info("**กิจกรรมทางกายน้อย:** ขาดการออกกำลังกาย ซึ่งมีส่วนช่วยให้คุณภาพการนอนลึก (Deep Sleep) ลดลง")
                risks_found = True
                
            if not risks_found:
                st.success("**ไม่พบความเสี่ยงที่เด่นชัด:** สัญญาณชีพและพฤติกรรมของคุณอยู่ในเกณฑ์มาตรฐาน")

        with col_b:
            st.markdown("#### 💡 แนวทางการดูแลรักษา (Medical & Lifestyle Recommendations)")
            
            if prediction == "None":
                st.markdown("- **Preventive Care:** รักษาวินัยการนอนให้ตรงเวลา และควบคุมค่า BMI รวมถึงความดันโลหิตให้อยู่ในเกณฑ์ปกติต่อไป")
            
            elif prediction == "Insomnia":
                st.markdown("**การปรับพฤติกรรม (Cognitive Behavioral Therapy for Insomnia - CBT-I):**")
                st.markdown("- **Sleep Hygiene:** งดการใช้หน้าจอสมาร์ทโฟน หรือดื่มเครื่องดื่มที่มีคาเฟอีนอย่างน้อย 2 ชั่วโมงก่อนเข้านอน")
                if stress_level >= 7:
                    st.markdown("- **Stress Management:** แนะนำกิจกรรมคลายเครียด เช่น การทำสมาธิ (Meditation) หรือกำหนดลมหายใจก่อนนอน")
                if physical_act < 30:
                    st.markdown("- **Exercise:** เพิ่มกิจกรรมทางกายในช่วงเช้าหรือเย็น (แต่หลีกเลี่ยงการออกกำลังกายหนักก่อนนอน)")
                    
            elif prediction == "Sleep Apnea":
                st.markdown("**การจัดการความเสี่ยงสูง (High-Risk Management):**")
                st.markdown("- 🚨 **Medical Checkup:** ควรไปพบแพทย์อายุรกรรมเฉพาะทางโรคการนอนหลับ เพื่อทำการทดสอบการนอนหลับ (Sleep Test / Polysomnography)")
                if bmi in ["Overweight", "Obese"]:
                    st.markdown("- **Weight Loss:** การลดน้ำหนักเพียง 5-10% สามารถลดความรุนแรงของภาวะหยุดหายใจขณะหลับได้อย่างมีนัยสำคัญ")
                st.markdown("- **Sleep Position:** พยายามปรับท่านอนเป็นท่านอนตะแคง เพื่อลดการอุดกั้นของทางเดินหายใจ")

        # แสดงหลอดความน่าจะเป็น
        st.markdown("---")
        st.markdown("<b>รายละเอียดค่าความน่าจะเป็นจาก Neural Network (Softmax Output):</b>", unsafe_allow_html=True)
        classes = le.classes_
        for i, class_name in enumerate(classes):
            st.progress(float(probabilities[i]), text=f"{class_name}: {probabilities[i]*100:.1f}%")