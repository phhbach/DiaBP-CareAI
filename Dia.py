# ==========================================================
# DiaBP-CareAI PRO MAX FINAL STABLE
# Offline | Competition Ready | Bilingual EN-VI
# Diabetes & Hypertension Focus
# ==========================================================

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import datetime
import io
from dataclasses import dataclass
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="DiaBP-CareAI PRO MAX", layout="wide")

st.markdown("""
<style>
.main-title {font-size:34px;font-weight:bold;color:#0b3d91;}
.section {background-color:#f4f8ff;padding:15px;border-radius:10px;margin-bottom:15px;}
.chat-box {background:#eef3ff;padding:10px;border-radius:8px;margin-bottom:5px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">DiaBP-CareAI PRO MAX 🧠</div>', unsafe_allow_html=True)
st.write("AI Clinical System – Diabetes & Hypertension (Hệ thống AI – ĐTĐ & THA)")

# =========================
# SIDEBAR
# =========================
menu = st.sidebar.radio(
    "Navigation (Điều hướng)",
    [
        "Dashboard (Bảng điều khiển)",
        "Stroke AI Camera (Camera AI Đột quỵ)",
        "Virtual Doctor Chat (Trợ lý y khoa)",
        "Digital Prescription (Đơn thuốc số)",
        "28-Day Analytics (Biểu đồ 28 ngày)",
        "PDF Report (Báo cáo PDF)"
    ]
)

# =========================
# DATA MODEL
# =========================
@dataclass
class Patient:
    name: str
    age: int
    sys: int
    dia: int
    glucose: int
    hba1c: float
    bmi: float

# =========================
# MEDICAL LOGIC
# =========================
def diabetes_logic(p):
    risk = 0
    meds = []
    advice = []

    if p.hba1c >= 8:
        risk += 40
    elif p.hba1c >= 6.5:
        risk += 25

    if p.glucose >= 180:
        risk += 30

    meds.append("Metformin 500mg twice daily (2 lần/ngày)")
    meds.append("Vitamin B12 supplementation (Bổ sung B12)")
    meds.append("Annual retina & kidney screening (Tầm soát mắt & thận hàng năm)")
    advice.append("Low carb diet (Chế độ ăn giảm tinh bột)")
    advice.append("Exercise 30 min/day (Tập 30 phút mỗi ngày)")

    return min(risk,100), meds, advice


def hypertension_logic(p):
    risk = 0
    meds = []
    advice = []

    if p.sys >= 180:
        risk += 60
        advice.append("Hypertensive crisis – Emergency (Cơn THA – Cấp cứu)")
    elif p.sys >= 160:
        risk += 40
    elif p.sys >= 140:
        risk += 25

    meds.append("Amlodipine 5mg daily (1 lần/ngày)")
    meds.append("Monitor potassium & creatinine (Theo dõi Kali & Creatinine)")
    advice.append("Low salt diet (Chế độ ăn giảm muối)")
    advice.append("Home BP monitoring (Theo dõi HA tại nhà)")

    return min(risk,100), meds, advice

# =========================
# DASHBOARD
# =========================
if "Dashboard" in menu:

    st.subheader("Patient Information (Thông tin bệnh nhân)")

    name = st.text_input("Patient Name (Tên bệnh nhân)")
    age = st.number_input("Age (Tuổi)", 1, 100, 55)
    sys = st.number_input("Systolic BP (HATT)", 80, 220, 150)
    dia = st.number_input("Diastolic BP (HATTr)", 40, 150, 90)
    glucose = st.number_input("Glucose mg/dL (Đường huyết)", 50, 400, 180)
    hba1c = st.number_input("HbA1c (%)", 4.0, 15.0, 8.0)
    bmi = st.number_input("BMI", 15.0, 40.0, 27.0)

    patient = Patient(name, age, sys, dia, glucose, hba1c, bmi)

    d_risk, d_meds, d_advice = diabetes_logic(patient)
    h_risk, h_meds, h_advice = hypertension_logic(patient)

    col1, col2 = st.columns(2)
    col1.metric("Diabetes Risk (%) (Nguy cơ ĐTĐ)", d_risk)
    col2.metric("Hypertension Risk (%) (Nguy cơ THA)", h_risk)

# =========================
# STROKE AI CAMERA
# =========================
if "Stroke AI Camera" in menu:

    st.warning("FAST+ Screening – Not a diagnosis (Chỉ tầm soát không thay thế chẩn đoán)")

    mp_face = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    face_mesh = mp_face.FaceMesh()
    pose = mp_pose.Pose()

    class StrokeAI(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            face_res = face_mesh.process(rgb)
            pose_res = pose.process(rgb)

            alert = False

            if face_res.multi_face_landmarks:
                for lm in face_res.multi_face_landmarks:
                    left = lm.landmark[61]
                    right = lm.landmark[291]
                    if abs(left.y - right.y) > 0.03:
                        alert = True
                    mp_draw.draw_landmarks(img, lm)

            if pose_res.pose_landmarks:
                lm = pose_res.pose_landmarks.landmark
                lw = lm[15]
                rw = lm[16]
                if abs(lw.y - rw.y) > 0.25:
                    alert = True

            if alert:
                h, w, _ = img.shape
                cv2.rectangle(img,(0,0),(w,h),(0,0,255),8)
                cv2.putText(img,"STROKE WARNING (CANH BAO DOT QUY)",
                            (30,h-40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,(0,0,255),3)

            return img

    webrtc_streamer(key="stroke", video_transformer_factory=StrokeAI)

# =========================
# VIRTUAL DOCTOR CHAT
# =========================
if "Virtual Doctor Chat" in menu:

    st.subheader("Medical Virtual Assistant (Trợ lý y khoa chuyên ĐTĐ & THA)")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    user_input = st.text_input("Describe your symptoms (Mô tả triệu chứng)")

    if st.button("Send (Gửi)"):

        text = user_input.lower()
        response = ""

        if "đau đầu" in text or "headache" in text:
            response = "Headache may relate to high blood pressure. Check BP immediately. (Đau đầu có thể liên quan THA. Hãy đo huyết áp ngay.)"

        elif "chóng mặt" in text or "dizziness" in text:
            response = "Dizziness may indicate BP fluctuation or glucose imbalance. Check both. (Chóng mặt có thể do dao động huyết áp hoặc đường huyết.)"

        elif "yếu tay" in text or "liệt" in text:
            response = "Possible stroke sign. Use FAST test and go to emergency if sudden onset. (Có thể dấu hiệu đột quỵ. Hãy đến cấp cứu ngay.)"

        elif "đường" in text or "glucose" in text:
            response = "Monitor HbA1c and kidney function. Maintain diet control. (Theo dõi HbA1c và chức năng thận.)"

        elif "huyết áp" in text or "blood pressure" in text:
            response = "Control salt intake and take medication regularly. (Giảm muối và uống thuốc đều đặn.)"

        else:
            response = "Please provide more details about symptoms. (Vui lòng mô tả chi tiết hơn.)"

        st.session_state.chat.append(("You", user_input))
        st.session_state.chat.append(("AI Doctor", response))

    for role, msg in st.session_state.chat:
        st.markdown(f'<div class="chat-box"><b>{role}:</b> {msg}</div>', unsafe_allow_html=True)

# =========================
# DIGITAL PRESCRIPTION
# =========================
if "Digital Prescription" in menu:

    st.subheader("Smart Digital Prescription (Đơn thuốc số thông minh)")

    if st.button("Generate Prescription (Tạo đơn thuốc)"):

        prescription = """
- Metformin 500mg twice daily (2 lần/ngày)
- Vitamin B12 supplementation (Bổ sung B12)
- Amlodipine 5mg daily (1 lần/ngày)
- Monitor kidney function (Theo dõi chức năng thận)
- Retina screening annually (Tầm soát võng mạc hàng năm)
"""

        st.success(prescription)

# =========================
# 28 DAY ANALYTICS
# =========================
if "28-Day Analytics" in menu:

    dates = pd.date_range(datetime.date.today(), periods=28)
    bp = np.random.normal(140, 10, 28)
    sugar = np.random.normal(150, 20, 28)

    df = pd.DataFrame({"Date":dates,"Blood Pressure":bp,"Glucose":sugar})
    st.line_chart(df.set_index("Date"))

# =========================
# PDF REPORT
# =========================
if "PDF Report" in menu:

    if st.button("Generate PDF Report (Tạo báo cáo PDF)"):

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        c.drawString(100,800,"DiaBP-CareAI Clinical Report")
        c.drawString(100,780,"Diabetes & Hypertension Screening")
        c.drawString(100,760,"This system supports clinical monitoring.")
        c.save()
        buffer.seek(0)

        st.download_button("Download PDF", buffer, "DiaBP_Report.pdf","application/pdf")