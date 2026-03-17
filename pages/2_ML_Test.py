import streamlit as st
import pandas as pd
import joblib
import time

st.set_page_config(page_title="ML Prediction Test", layout="wide")
st.title("💰 ระบบประเมินและวิเคราะห์สุขภาพทางการเงิน (Financial Advisory AI)")
st.markdown("ระบบนี้รับตัวแปรอิสระ (Independent Variables) เพื่อคำนวณระดับรายได้ พร้อมวิเคราะห์ปัญหาและเสนอแนวทางการจัดการเงิน")
st.markdown("---")

try:
    model = joblib.load("ML_Project/models/ensemble_model.pkl")
    scaler = joblib.load("ML_Project/models/ml_scaler.pkl")
    features = joblib.load("ML_Project/models/ml_features.pkl")
except Exception as e:
    st.error(f"❌ ระบบไม่สามารถโหลดน้ำหนักของแบบจำลองได้ (Error: {e})")
    st.stop()

with st.form("ml_form"):
    st.markdown("#### 👤 ส่วนที่ 1: ข้อมูลประชากรศาสตร์ (Demographics)")
    c1, c2, c3 = st.columns(3)
    with c1: age = st.number_input("อายุ (Age)", 15, 80, 25)
    with c2: gender = st.selectbox("เพศ (Gender)", ["Male", "Female", "Non-binary"])
    with c3: year = st.selectbox("ระดับประสบการณ์ (Experience Level)", ["Freshman", "Sophomore", "Junior", "Senior"])

    st.markdown("#### 💸 ส่วนที่ 2: โครงสร้างการจัดสรรรายจ่ายประจำเดือน (Expenditure in USD)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**หมวดความต้องการพื้นฐาน (Needs)**")
        housing = st.number_input("ค่าที่พักอาศัย (Housing)", 0, 10000, 800, step=100)
        food = st.number_input("ค่าอาหารและเครื่องดื่ม (Food)", 0, 5000, 400, step=50)
        transport = st.number_input("ค่าคมนาคม/เดินทาง (Transportation)", 0, 2000, 150, step=10)
        personal_care = st.number_input("ของใช้ส่วนตัว (Personal Care)", 0, 2000, 100, step=10)
    with col2:
        st.markdown("**หมวดพัฒนาและเครื่องมือ (Development)**")
        tuition = st.number_input("ค่าการศึกษา/พัฒนาตนเอง (Tuition/Education)", 0, 15000, 500, step=100)
        books = st.number_input("อุปกรณ์การเรียน/ทำงาน (Books)", 0, 2000, 100, step=10)
        tech = st.number_input("ค่าเทคโนโลยี/ซอฟต์แวร์ (Technology)", 0, 5000, 150, step=50)
    with col3:
        st.markdown("**หมวดไลฟ์สไตล์ (Wants)**")
        entertainment = st.number_input("ความบันเทิงและสันทนาการ (Entertainment)", 0, 5000, 200, step=50)
        health = st.number_input("ค่าดูแลสุขภาพ/ฟิตเนส (Health/Wellness)", 0, 3000, 100, step=10)
        misc = st.number_input("ค่าใช้จ่ายจิปาถะอื่นๆ (Misc)", 0, 3000, 100, step=10)

    st.markdown("#### 💳 ส่วนที่ 3: พฤติกรรมทางธุรกรรม")
    payment = st.selectbox("ช่องทางชำระเงินหลัก (Preferred Payment)", ["Credit Card", "Mobile Payment App", "Cash"])

    submit = st.form_submit_button("🧠 ประมวลผลและวิเคราะห์สุขภาพการเงิน", type="primary", use_container_width=True)

if submit:
    with st.spinner("🤖 AI กำลังประมวลผลผ่านสมการ Ensemble และวิเคราะห์สัดส่วน..."):
        time.sleep(1.5)
        
        # คำนวณรายจ่ายรวมจาก Input
        total_expenses = housing + food + transport + personal_care + tuition + books + tech + entertainment + health + misc
        
        # ประมวลผล ML
        input_data = pd.DataFrame(columns=features)
        input_data.loc[0] = 0 
        input_data['age'] = age
        input_data['tuition'] = tuition
        input_data['housing'] = housing
        input_data['food'] = food
        input_data['transportation'] = transport
        input_data['books_supplies'] = books
        input_data['entertainment'] = entertainment
        input_data['personal_care'] = personal_care
        input_data['technology'] = tech
        input_data['health_wellness'] = health
        input_data['miscellaneous'] = misc

        if f"gender_{gender}" in input_data.columns: input_data[f"gender_{gender}"] = 1
        if f"year_in_school_{year}" in input_data.columns: input_data[f"year_in_school_{year}"] = 1
        if f"preferred_payment_method_{payment}" in input_data.columns: input_data[f"preferred_payment_method_{payment}"] = 1

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        # แสดงผลลัพธ์รายได้ประเมิน
        st.success("✅ การคำนวณเสร็จสมบูรณ์")
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #0ba360 0%, #3cba92 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1);'>
            <h4 style='margin: 0; opacity: 0.9;'>ระดับรายได้หรือกำลังซื้อที่ประเมินได้ (Estimated Monthly Income)</h4>
            <h1 style='margin: 10px 0; font-size: 3.5rem;'>${prediction:,.2f}</h1>
            <p style='margin: 0; opacity: 0.8;'>รายจ่ายรวมของคุณคือ ${total_expenses:,.2f} ต่อเดือน</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ---------------------------------------------------------
        # ส่วนที่เพิ่มใหม่: ระบบวิเคราะห์ปัญหาและให้คำแนะนำ
        # ---------------------------------------------------------
        st.markdown("### 📊 การวิเคราะห์และข้อเสนอแนะ (Financial Advisory)")
        
        balance = prediction - total_expenses
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("#### 🚨 การวิเคราะห์ปัญหา (Problem Analysis)")
            problems_found = False
            
            if balance < 0:
                st.error(f"**ภาวะขาดดุล:** รายจ่ายของคุณ (${total_expenses:,.2f}) สูงกว่ารายได้ประเมิน (${prediction:,.2f}) เสี่ยงต่อการเกิดหนี้สินสะสม")
                problems_found = True
            elif balance < (prediction * 0.1):
                st.warning(f"**สภาพคล่องต่ำ:** คุณมีเงินเหลือเก็บน้อยกว่า 10% ของรายได้ (${balance:,.2f}) ซึ่งไม่เพียงพอต่อการเป็นเงินสำรองฉุกเฉิน")
                problems_found = True
                
            if housing > (prediction * 0.4):
                st.warning("**ภาระที่พักอาศัยสูงเกินไป:** คุณจ่ายค่าที่พักเกิน 40% ของรายได้ประเมิน ซึ่งเป็นภาระที่หนักเกินมาตรฐานการเงินที่ดี")
                problems_found = True
                
            wants_total = entertainment + misc + tech
            if wants_total > (prediction * 0.3):
                st.warning(f"**พฤติกรรมบริโภคนิยม:** คุณใช้จ่ายกับหมวดไลฟ์สไตล์ (บันเทิง/จิปาถะ/ไอที) สูงถึง ${wants_total:,.2f} ซึ่งส่งผลกระทบต่อเงินออม")
                problems_found = True
                
            if tuition < (prediction * 0.05) and age < 30:
                st.info("**ขาดการลงทุนในตัวเอง:** คุณอยู่ในวัยพัฒนาศักยภาพ แต่จัดสรรงบเพื่อการศึกษาน้อยกว่า 5% ของรายได้")
                problems_found = True
                
            if not problems_found:
                st.success("**สัดส่วนการเงินสมดุล:** โครงสร้างรายจ่ายของคุณสัมพันธ์กับระดับรายได้ประเมินได้เป็นอย่างดี")

        with col_b:
            st.markdown("#### 💡 แนวทางการจัดการ (Actionable Recommendations)")
            st.markdown("""
            **โครงสร้างการเงินที่ AI แนะนำ (Budgeting Strategy):**
            * **กฏ 50/30/20:** พยายามจัดสรรรายได้เป็น Needs 50%, Wants 30%, และ Savings/Investments 20%
            """)
            
            if balance < 0 or housing > (prediction * 0.4) or wants_total > (prediction * 0.3):
                st.markdown("**การปรับลดรายจ่าย (Cost Reduction):**")
                if wants_total > (prediction * 0.3):
                    st.markdown("- ตัดลดรายจ่ายหมวดบันเทิงและจิปาถะลงอย่างน้อย 15-20%")
                if housing > (prediction * 0.4):
                    st.markdown("- พิจารณาหาเพื่อนร่วมห้อง (Roommate) หรือย้ายที่พักเพื่อลดภาระค่าเช่าระยะยาว")
            
            if tuition < (prediction * 0.05) and age < 30:
                st.markdown("**การเพิ่มศักยภาพ (Upskilling):**")
                st.markdown("- นำเงินออมส่วนหนึ่งไปลงทุนในคอร์สเรียนหรือเครื่องมือพัฒนาทักษะ เพื่อขยายขอบเขตกำลังซื้อ (Purchasing Power) ในอนาคต")
            
            st.markdown("**การชำระเงิน:**")
            if payment == "Credit Card" and balance < 0:
                st.markdown("- ⚠️ **คำเตือน:** คุณใช้บัตรเครดิตเป็นหลักในขณะที่กระแสเงินสดติดลบ ระวังการเกิดหนี้บัตรเครดิตดอกเบี้ยสูง ควรเปลี่ยนมาใช้ Cash หรือ Debit ชั่วคราว")