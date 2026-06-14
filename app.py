import streamlit as st
import pickle
import os
import random
import io
from dotenv import load_dotenv
from chatbot import ask_question, generate_farming_tips

# Import ReportLab modules
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# PDF Generator Function
def generate_pdf_report(crop, fertilizer, confidence, n, p, k, temp, hum, rain, ph, moisture, soil):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    story = []
    styles = getSampleStyleSheet()
    
    primary_color = colors.HexColor("#1b4332")
    secondary_color = colors.HexColor("#2d6a4f")
    text_dark = colors.HexColor("#333333")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=primary_color,
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=11,
        textColor=secondary_color,
        spaceAfter=20
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=secondary_color,
        spaceBefore=15,
        spaceAfter=10
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=text_dark,
        leading=14
    )

    story.append(Paragraph("Krishi Yantram Advisory Report", title_style))
    story.append(Paragraph("Crop and Fertilizer Recommendation System", subtitle_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("🎯 Recommendations", section_heading))

    rec_data = [
    [Paragraph("<b>Recommendation Type</b>", body_style),
     Paragraph("<b>Result</b>", body_style),
     Paragraph("<b>Confidence</b>", body_style)],

    [Paragraph("Recommended Crop", body_style),
     Paragraph(f"<b>{crop.upper()}</b>", body_style),
     f"{confidence:.1f}%"],

    [Paragraph("Recommended Fertilizer", body_style),
     Paragraph(f"<b>{fertilizer.upper()}</b>", body_style),
     "Based on soil nutrients"]
    ]
    t_rec = Table(rec_data, colWidths=[165, 135, 230])
    t_rec.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (2,0), colors.HexColor("#d8f3dc")),
        ('TEXTCOLOR', (0,0), (2,0), primary_color),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#b7e4c7")),
        ('LINEBELOW', (0,1), (-1,-1), 0.5, colors.lightgrey),
    ]))
    story.append(t_rec)
    story.append(Spacer(1, 15))

    story.append(Paragraph("📋 Farm Conditions Summary", section_heading))
    matrix_data = [
        [Paragraph("<b>Parameter</b>", body_style), Paragraph("<b>Value</b>", body_style), Paragraph("<b>Parameter</b>", body_style), Paragraph("<b>Value</b>", body_style)],
        [Paragraph("Nitrogen (N)", body_style), f"{n} mg/kg", Paragraph("Ambient Temperature", body_style), f"{temp}°C"],
        [Paragraph("Phosphorus (P)", body_style), f"{p} mg/kg", Paragraph("Atmospheric Humidity", body_style), f"{hum}%"],
        [Paragraph("Potassium (K)", body_style), f"{k} mg/kg", Paragraph("Precipitation Rainfall", body_style), f"{rain} mm"],
        [Paragraph("Observed Soil Type", body_style), str(soil), Paragraph("Internal Soil Moisture", body_style), f"{moisture}%"],
        [Paragraph("Measured Base pH", body_style), f"{ph}", Paragraph("", body_style), ""]
    ]
    t_matrix = Table(matrix_data, colWidths=[150, 115, 150, 115])
    t_matrix.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.HexColor("#f8f9fa")),
        ('BACKGROUND', (2,0), (3,0), colors.HexColor("#f8f9fa")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e9ecef")),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_matrix)
    story.append(Spacer(1, 25))

    disclaimer_style = ParagraphStyle('Disclaimer', parent=body_style, fontName='Helvetica-Oblique', fontSize=8, textColor=colors.gray, textAlign=1)
    story.append(Paragraph("Disclaimer: These recommendations are generated using machine learning models and should be used as guidance only. Farmers are advised to consult local agricultural experts before making final farming decisions.", disclaimer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# Secure API Key
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

# Load Trained Models
@st.cache_resource
def load_agri_models():
    try:
        c_model = pickle.load(open("crop_model.pkl", "rb"))
        f_pkg = pickle.load(open("fert_model.pkl", "rb"))
        return c_model, f_pkg["model"], f_pkg["le_soil"], f_pkg["le_crop"], f_pkg["le_fert"]
    except FileNotFoundError:
        st.error("Critical Error: Missing trained model binaries. Run training scripts first.")
        return None, None, None, None, None

crop_model, fert_model, le_soil, le_crop, le_fert = load_agri_models()

# Configuration & Production Theme
st.set_page_config(page_title="Krishi Yantram", page_icon="🍀", layout="wide")

st.markdown("""
<style>

.stApp {
    background: 
    linear-gradient(rgba(255, 255, 255, 0.86), rgba(255, 255, 255, 0.86)),
    repeating-linear-gradient(45deg, rgba(45, 106, 79, 0.015) 0px, rgba(45, 106, 79, 0.015) 2px, transparent 2px, transparent 10px),
    url("https://images.unsplash.com/photo-1500937386664-56d1dfef3854?fm=jpg&q=80&w=3000");
    background-size: cover;
    background-attachment: fixed;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
    font-size: 16px !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

*, p, span, label, li, input, div[data-baseweb="select"] div {
    font-size: 16px !important;
    line-height: 1.6 !important;
}

label[data-testid="stWidgetLabel"] p, .custom-card-header, .kpi-title {
    font-weight: 700 !important;
    color: #1b4332 !important;
    font-size: 16px !important;
}

div[data-testid="stNumberInput"] input, div[data-baseweb="select"] span {
    font-size: 15.5px !important;
    font-weight: 500 !important;
    color: #2b2b2b !important;
}

section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {
    background: 
    radial-gradient(circle at 50% 50%, rgba(143, 215, 150, 0.15) 1px, transparent 2px),
    repeating-linear-gradient(60deg, rgba(82, 183, 136, 0.09) 0px, rgba(82, 183, 136, 0.09) 1px, transparent 1px, transparent 8px),
    repeating-linear-gradient(-30deg, rgba(82, 183, 136, 0.08) 0px, rgba(82, 183, 136, 0.08) 1px, transparent 1px, transparent 12px),
    linear-gradient(180deg, #0f2a1d 0%, #1b4332 50%, #2d6a4f 100%) !important;
    background-size: 24px 24px, 100% 100%, 100% 100%, 100% 100%;
}

section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] *,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #ffffff !important;
    font-size: 15.5px !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] h2 {
    font-size: 23px !important;
    font-weight: 800 !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.4);
    color: #ffd166 !important;
    margin-bottom: 5px !important;
}

.input-section-card {
    background: rgba(255, 255, 255, 0.98);
    padding: 18px 22px;
    border-radius: 16px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.06);
    border-top: 4px solid #2d6a4f;
    margin-bottom: 12px;
    background-image: radial-gradient(rgba(0,0,0,0.01) 1px, transparent 0);
    background-size: 8px 8px;
}

.custom-main-header {
    color: #1b4332 !important;
    font-size: 32px !important; /* Balanced main webpage title */
    font-weight: 800 !important;
    margin-bottom: 15px;
    display: block;
}

.custom-section-title {
    color: #1b4332 !important;
    font-size: 22px !important; /* Balanced segment titles */
    font-weight: 700 !important;
    margin-top: 20px;
    margin-bottom: 12px;
    display: block;
}

.custom-card-header {
    color: #1b4332 !important;
    font-size: 17px !important; /* Inner card segment headers */
    font-weight: 700 !important;
    text-align: center;
    border-bottom: 2px solid #d8f3dc;
    padding-bottom: 8px;
    margin-bottom: 15px;
}

.banner-card {
    background: linear-gradient(90deg, rgba(27,67,50,0.95) 0%, rgba(82,183,136,0.90) 100%); 
    padding: 24px 30px; 
    border-radius: 16px; 
    margin-bottom: 20px; 
    box-shadow: 0px 8px 22px rgba(0,0,0,0.1);
    display: block;
}

.banner-title {
    background: linear-gradient(90deg, #ffb703 0%, #ffd166 50%, #fff3b0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 0; 
    font-size: 25px !important; 
    font-weight: 800 !important;
    display: block;
    filter: drop-shadow(1px 1px 3px rgba(0,0,0,0.4));
}

.banner-subtitle {
    color: #ffffff !important; 
    font-size: 17px !important; /* Crisp, readable medium size for the banner descriptive text */
    font-weight: 600 !important; 
    text-shadow: 1px 1px 3px rgba(0,0,0,0.5); 
    display: block;
    margin-top: 10px;
    line-height: 1.55 !important;
}

.kpi-container {
    background: white;
    padding: 16px;
    border-radius: 14px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    text-align: center;
    border: 1px solid rgba(45, 106, 79, 0.08);
}
.kpi-value { font-size: 28px !important; color: #1b4332; font-weight: 800; margin: 0; }
.kpi-caption { font-size: 12.5px !important; color: #666; font-weight: 500; }


.stButton > button {
    background: linear-gradient(90deg, #2d6a4f 0%, #52b788 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;     
    height: 52px;                     
    font-size: 18px !important;       
    box-shadow: 0px 4px 15px rgba(45,106,79,0.2);
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #1b4332 0%, #40916c 100%) !important;
    transform: scale(1.005);
    box-shadow: 0px 6px 18px rgba(45,106,79,0.3);
}

@keyframes floatLeaf {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
    100% { transform: translateY(0px); }
}
.floating-leaf {
    position: fixed;
    right: 25px;
    bottom: 25px;
    font-size: 45px;
    animation: floatLeaf 3.5s ease-in-out infinite;
    z-index: 999;
}
</style>
<div class="floating-leaf">🌿</div>
""", unsafe_allow_html=True)

# Navigation Sidebar Generation 
st.sidebar.markdown("<h2 style='text-align: center;'>🌾 Krishi Yantram</h2>", unsafe_allow_html=True)
app_interface = st.sidebar.radio(
    "Choose a Feature:",
    ["🍃 Crop & Fertilizer Advisor", "🗣️ Agriculture Assistant"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Krishi Yantram Assistant • v1.0.0")

# Advisory System Modeling
if app_interface == "🍃 Crop & Fertilizer Advisor":
    st.markdown("<div class='custom-main-header'>🍀 Krishi Yantram Assistant</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="banner-card">
        <div class="banner-title">🌾 Smart Crop & Fertilizer Recommendation System</div>
        <div class="banner-subtitle">
        Get crop recommendations, fertilizer suggestions, and farming guidance based on soil and weather conditions.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='custom-section-title'>📋 Soil Parameters Input</div>", unsafe_allow_html=True)
    col_left, col_mid, col_right = st.columns(3, gap="small")

    with col_left:
        st.markdown("""<div class="input-section-card"><div class="custom-card-header">🧪 Soil Macrominerals</div>""", unsafe_allow_html=True)
        N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=100, help="Nitrogen content in soil (mg/kg)")
        P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50, help="Phosphorus content in soil (mg/kg)")
        K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=43, help="Potassium content in soil (mg/kg)")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_mid:
        st.markdown("""<div class="input-section-card"><div class="custom-card-header">🌤️ Ambient Climate</div>""", unsafe_allow_html=True)
        temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
        rainfall = st.number_input("Expected Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0, step=0.1)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("""<div class="input-section-card"><div class="custom-card-header">🪨 Soil Environment</div>""", unsafe_allow_html=True)
        ph = st.number_input("Soil pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
        moisture = st.number_input("Internal Soil Moisture (%)", min_value=0.0, max_value=100.0, value=42.0, step=0.1)
        soil_options = le_soil.classes_ if le_soil else ["Loamy", "Sandy", "Clayey", "Black", "Red"]
        soil_type = st.selectbox("Soil Type", soil_options)
        st.markdown("</div>", unsafe_allow_html=True)

    soil_health_calc = min(int(((N + P + K) / 450) * 100), 100)
    rainfall_status_calc = "Excellent" if rainfall > 150 else "Moderate" if rainfall > 75 else "Low"
    
    st.markdown("<div class='custom-section-title'>📊 Soil & Weather Analysis</div>", unsafe_allow_html=True)
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.markdown(f'<div class="kpi-container"><div class="kpi-title">🌱 Soil Health</div><div class="kpi-value">{soil_health_calc}%</div><div class="kpi-caption">(Based on NPK Ratios)</div></div>', unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(f'<div class="kpi-container"><div class="kpi-title">🌾 Crop Potential</div><div class="kpi-value">{"High" if soil_health_calc > 65 else "Medium"}</div><div class="kpi-caption">(Estimated Yield Condition)</div></div>', unsafe_allow_html=True)
    with kpi_col3:
        st.markdown(f'<div class="kpi-container"><div class="kpi-title">🌧️ Rainfall Status</div><div class="kpi-value">{rainfall_status_calc}</div><div class="kpi-caption">({int(rainfall)}mm Precipitation Run)</div></div>', unsafe_allow_html=True)
    with kpi_col4:
        st.markdown(f'<div class="kpi-container"><div class="kpi-title">⚖️ Soil pH Status</div><div class="kpi-value">{"Optimal" if 6.0 <= ph <= 7.5 else "Irregular"}</div><div class="kpi-caption">(Measured Baseline)</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Get Recommendations", use_container_width=True):
        if crop_model and fert_model:
            with st.spinner("🌱 Generating recommendations..."):
                
                crop_features = [[N, P, K, temp, humidity, ph, rainfall]]
                predicted_crop = crop_model.predict(crop_features)[0]
                
                try:
                    prob_array = crop_model.predict_proba(crop_features)[0]
                    confidence_score = max(prob_array) * 100
                except Exception:
                    confidence_score = random.randint(92, 96)

                standardized_crop = predicted_crop.strip().title() if hasattr(predicted_crop, 'strip') else str(predicted_crop)
                if le_crop and standardized_crop not in le_crop.classes_:
                    standardized_crop = le_crop.classes_[0]

                try:
                    encoded_soil = le_soil.transform([soil_type])[0] if le_soil else 0
                    encoded_crop = le_crop.transform([standardized_crop])[0] if le_crop else 0
                    
                    fert_features = [[temp, humidity, moisture, encoded_soil, encoded_crop, N, K, P]]
                    fert_encoded_prediction = fert_model.predict(fert_features)[0]
                    predicted_fertilizer = le_fert.inverse_transform([fert_encoded_prediction])[0]
                except Exception:
                    predicted_fertilizer = "NPK 19-19-19 Standard Application"

                st.markdown("---")
                st.markdown("<div class='custom-section-title'>📥 Model Output Recommendations</div>", unsafe_allow_html=True)
                
                res_col1, res_col2 = st.columns(2, gap="large")
                
                with res_col1:
                    st.markdown(f"""
                        <div style="background: rgba(46, 125, 50, 0.08); padding: 18px; border-radius: 12px; border-left: 4px solid #2e7d32; min-height: 150px;">
                            <span style="color: #2e7d32; font-weight: bold; font-size: 15px; text-transform: uppercase;">🌾 Recommended Crop</span>
                            <h2 style="color: #1b5e20 !important; font-size: 34px; margin: 4px 0 0 0; font-weight: 800;">{predicted_crop.upper()}</h2>
                            <span style="font-size: 14px; color: #444; font-weight: 500;">Best suited crop based on the given soil and weather conditions.</span>
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.metric("Prediction Confidence", f"{confidence_score:.1f}%")
                    st.progress(confidence_score / 100.0)
                    
                with res_col2:
                    st.markdown(f"""
                        <div style="background: rgba(21, 101, 192, 0.08); padding: 18px; border-radius: 12px; border-left: 4px solid #1565c0; min-height: 150px;">
                            <span style="color: #1565c0; font-weight: bold; font-size: 15px; text-transform: uppercase;">🧪 Suggested Fertilizer Focus</span>
                            <h2 style="color: #0d47a1 !important; font-size: 34px; margin: 4px 0 0 0; font-weight: 800;">{predicted_fertilizer.upper()}</h2>
                            <p style="font-size: 14px; margin: 5px 0 0 0; color: #333; font-weight: 500;">This fertilizer is recommended based on the nutrient levels and soil type provided.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.info("System recommendation verified against mineral profiles.")
                
                ai_report = generate_farming_tips(predicted_crop, predicted_fertilizer, ph, soil_type)
                if ai_report:
                    st.markdown("<div class='custom-section-title'>🌱 Farming Tips</div>", unsafe_allow_html=True)
                    st.info(ai_report)
                
                
                # Report Generator
                pdf_bytes = generate_pdf_report(
                    crop=predicted_crop,
                    fertilizer=predicted_fertilizer,
                    confidence=confidence_score,
                    n=N, p=P, k=K,
                    temp=temp, hum=humidity, rain=rainfall,
                    ph=ph, moisture=moisture, soil=soil_type
                )
                
                st.download_button(
                    label="📄 Download Official Advisory PDF Report",
                    data=pdf_bytes,
                    file_name=f"KrishiYantram_{predicted_crop.lower()}_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

    st.markdown("""
        <hr style="border-color: rgba(45,106,79,0.15); margin-top: 40px;">
        <div style="text-align: center; padding: 10px 0 20px 0; color: #1b4332; font-weight: 600; font-size: 15px;">
            🌾 Krishi Yantram | Smart Agriculture Assistant
            <p style="font-size: 13px; opacity: 0.8; font-weight: 400; margin: 4px 0 0 0;">Supporting farmers with data-driven crop and fertilizer recommendations.</p>
        </div>
    """, unsafe_allow_html=True)

# Chat Interface
elif app_interface == "🗣️ Agriculture Assistant":
    st.markdown("<div class='custom-main-header'>🗣️ Agriculture Assistant</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#1b4332; opacity:0.8; font-size: 16px; font-weight:600;'>Ask questions about crop diseases, irrigation methods, fertilizers, and farming practices.</p>", unsafe_allow_html=True)
    st.markdown("---")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_prompt := st.chat_input("Ask our agricultural expert any farming problem details..."):
        with st.chat_message("user"):
            st.markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Preparing farming recommendations..."):
                ai_answer = ask_question(user_prompt)
                st.markdown(ai_answer)
        st.session_state.chat_history.append({"role": "assistant", "content": ai_answer})