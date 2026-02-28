"""
CrisisLens — Interactive Dashboard
Streamlit-based dashboard with real-time crisis map, analysis feed, and analytics.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent.resolve())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

from src.dashboard.demo_data import get_demo_result_for_text
from src.dashboard.user_guide_content import USER_GUIDE_MARKDOWN


# ─── Page Configuration ───
st.set_page_config(
    page_title="CrisisLens — Disaster Response Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    /* Main theme - consistent dark background */
    .stApp, section.main, div[data-testid="stAppViewContainer"], div.block-container {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%) !important;
    }
    
    /* White text for visibility - fix grey/low-contrast labels and body text */
    p, label, span, .stMarkdown, div[data-testid="stMarkdown"] {
        color: #ffffff !important;
    }
    label[data-testid="stWidgetLabel"] { color: #ffffff !important; }
    .stTextInput label, .stTextArea label, .stSelectbox label { color: #ffffff !important; }
    
    /* Info boxes - dark text on light blue background for contrast */
    div[data-testid="stAlert"] p, .stAlert p { color: #1a1a2e !important; }
    
    /* Header */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        padding: 0.5rem 0;
        letter-spacing: -1px;
    }
    
    .sub-header {
        text-align: center;
        color: #e2e8f0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #e2e8f0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }
    
    /* Urgency badges */
    .badge-critical { 
        background: linear-gradient(135deg, #ff4757, #ff6b81);
        color: white; padding: 4px 12px; border-radius: 20px; 
        font-weight: 600; font-size: 0.8rem; display: inline-block;
    }
    .badge-high { 
        background: linear-gradient(135deg, #ff6348, #ffa502);
        color: white; padding: 4px 12px; border-radius: 20px;
        font-weight: 600; font-size: 0.8rem; display: inline-block;
    }
    .badge-medium { 
        background: linear-gradient(135deg, #ffa502, #ffda79);
        color: #333; padding: 4px 12px; border-radius: 20px;
        font-weight: 600; font-size: 0.8rem; display: inline-block;
    }
    .badge-low { 
        background: linear-gradient(135deg, #2ed573, #7bed9f);
        color: #333; padding: 4px 12px; border-radius: 20px;
        font-weight: 600; font-size: 0.8rem; display: inline-block;
    }
    
    /* Result card */
    .result-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Label tags */
    .label-tag {
        background: rgba(102, 126, 234, 0.2);
        border: 1px solid rgba(102, 126, 234, 0.4);
        color: #667eea;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin: 2px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
    }

    /* Main content columns - match dark background (fix white patches) */
    div[data-testid="stVerticalBlock"] > div, div[data-testid="column"] {
        background: transparent !important;
    }
    section.main > div { background: transparent !important; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def get_urgency_badge(level: str) -> str:
    """Get HTML badge for urgency level."""
    level = level.upper()
    badge_class = f"badge-{level.lower()}"
    icons = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
    icon = icons.get(level, "⚪")
    return f'<span class="{badge_class}">{icon} {level}</span>'


def get_marker_color(level: str) -> str:
    """Get map marker color for urgency level."""
    colors = {
        "CRITICAL": "red",
        "HIGH": "orange",
        "MEDIUM": "beige",
        "LOW": "green",
    }
    return colors.get(level.upper(), "blue")


def get_marker_icon(event_type: str) -> str:
    """Get map marker icon for event type."""
    icons = {
        "RESCUE_REQUEST": "life-ring",
        "INFRASTRUCTURE_DAMAGE": "building",
        "MEDICAL_EMERGENCY": "plus-sign",
        "SUPPLY_REQUEST": "shopping-cart",
        "CASUALTY_REPORT": "exclamation-sign",
        "VOLUNTEER_OFFER": "hand-up",
        "SITUATIONAL_UPDATE": "info-sign",
        "DISPLACEMENT": "home",
    }
    return icons.get(event_type, "info-sign")


# ─── Main App ───
def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 CrisisLens</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Multilingual Crisis & Disaster Response NLP Pipeline<br>'
        '<small>UN SDG #11 Sustainable Cities • #13 Climate Action</small></p>',
        unsafe_allow_html=True,
    )

    # Pipeline loads lazily on first Live Analyze / Process Batch

    # Initialize session state
    if "results" not in st.session_state:
        st.session_state.results = []
    if "session_stats" not in st.session_state:
        st.session_state.session_stats = {"total": 0, "relevant": 0, "critical": 0, "duplicates": 0}
    # ─── Sidebar ───
    with st.sidebar:
        st.markdown("### ⚙️ Analysis Controls")
        st.markdown("---")
        st.caption("Samples are raw INPUT texts. The model classifies each into event type, urgency & locations.")
        
        # Sample messages — mix of explicit and implicit (model infers type from context)
        st.markdown("#### 📝 Quick Samples")
        sample_messages = {
            "🆘 Rescue (English)": "URGENT: Family of 4 trapped on 2nd floor in Hatay district, water rising fast. Please send rescue team immediately! #TurkeyEarthquake",
            "🏥 Medical (Spanish)": "Necesitamos insulina urgente en el refugio de la escuela San Pedro. Hay 3 diabéticos sin medicamentos desde hace 2 días. #TerremotoMexico",
            "🏚️ Damage (Hindi)": "दिल्ली में पुल टूट गया है, मुख्य सड़क पूरी तरह बंद है। कई गाड़ियां फंसी हैं। तुरंत मदद भेजो! #DelhiFlood",
            "📢 Update (French)": "Le niveau d'eau monte rapidement dans le quartier Est de Lyon. Évacuation en cours. Les routes sont coupées. #InondationFrance",
            "🍽️ Supply (Arabic)": "نحتاج ماء وطعام عاجل في مخيم الإيواء بمدينة حلب. أكثر من 200 عائلة بدون إمدادات منذ 3 أيام",
            "🚑 Medical (German)": "DRINGEND: 5 Verletzte nach Gebäudeeinsturz in Köln. Wir brauchen sofort Rettungswagen und medizinisches Personal!",
            "🏠 Displacement (Punjabi)": "ਅੰਮ੍ਰਿਤਸਰ ਵਿੱਚ ਹੜ੍ਹ ਕਾਰਨ 500 ਪਰਿਵਾਰ ਬੇਘਰ ਹੋ ਗਏ। ਸਕੂਲ ਵਿੱਚ ਸ਼ਰਨਾਰਥੀ ਕੈਂਪ ਲੱਗਾ ਹੈ, ਭੋਜਨ ਅਤੇ ਕੰਬਲ ਚਾਹੀਦੇ ਹਨ।",
            "🌊 Flood (Gujarati)": "અમદાવાદમાં નદી ઓફલો થયો છે. મુખ્ય રસ્તા પૂરાવાળા છે. લોકો ઘરોમાં ફસાયા છે, રક્ષણ દળ મોકલો!",
            "🔥 Fire (Polish)": "PILNE: Pożar w bloku na ulicy Marszałkowskiej w Warszawie. Ludzie uwięzieni na wyższych piętrach. Potrzebna natychmiastowa pomoc straży pożarnej!",
            "⚕️ Casualty (Portuguese)": "Há pelo menos 12 feridos no colapso do prédio em São Paulo. Ambulâncias a caminho mas precisamos de mais médicos. Hospital Santa Maria.",
            "🏗️ Infrastructure (Turkish)": "Hatay'da ana köprü çöktü. Hastaneye giden yol tamamen kapalı. Alternatif rota yok. Acil yardım lazım!",
            "📦 Supply (Russian)": "Срочно нужны вода, еда и одеяла в приюте школы №15 в Краснодаре. Более 300 семей без поставок уже 2 дня.",
            "🏥 Rescue (Chinese)": "紧急！广州天河区一栋楼房倒塌，多人被困。需要救援队立即赶往现场！",
            "🌧️ Update (Japanese)": "大阪で大雨が続いています。河川の水位が上昇中。避難指示が出ています。東淀川区は特に危険です。",
            "🚨 Critical (Korean)": "부산 해운대구 건물 붕괴. 최소 8명 부상. 구급차와 구조대 즉시 필요합니다!",
            "🏥 Medical (Italian)": "URGENTE: Mancano farmaci critici all'ospedale di Napoli. 20 pazienti in dialisi senza cure da ieri. Serve aiuto immediato.",
            "🌊 Flood (Dutch)": "Overstroming in Limburg. Maas overstroomd. Evacuatie van Valkenburg aan de gang. Duizenden mensen op zoek naar onderdak.",
            "🆘 Rescue (Bengali)": "কলকাতায় বিল্ডিং ধসে ১০ জন আটকে আছে। জল বেড়ে চলেছে। তৎক্ষণাৎ উদ্ধার দল পাঠান!",
            "🏚️ Damage (Tamil)": "சென்னையில் பாலம் இடிந்து விழுந்தது. முக்கிய சாலை முழுதும் அடைக்கப்பட்டுள்ளது. உடனடி மருத்துவ உதவி தேவை!",
            "📢 Update (Telugu)": "హైదరాబాద్లో వరదలు. ముఖ్య రోడ్డులు నీటితో నిండిపోయాయి. అమరావతి ప్రాంతంలో అపకవాటు జరుగుతోంది.",
            "🍽️ Supply (Marathi)": "मुंबईतील शरणार्थी शिबिरात पाणी आणि अन्न तातडीने हवे. २०० कुटुंबांना दोन दिवसांपासून पुरवठा नाही.",
            "🏥 Medical (Urdu)": "کراچی کے اسپتال میں ادویات ختم ہو گئی ہیں۔ 15 مریض بغیر انسولین کے ہیں۔ فوری مدد کی ضرورت ہے۔",
            "🌋 Disaster (Indonesian)": "Gempa di Lombok. Banyak bangunan runtuh. Korban luka parah menunggu evakuasi. Bantuan medis darurat dibutuhkan!",
            "🌊 Flood (Thai)": "น้ำท่วมกรุงเทพฯ บริเวณถนนสุขุมวิท. ผู้คนหลายร้อยคนติดอยู่บนดาดฟ้า. ต้องการเรือกู้ภัยเร่งด่วน!",
            "🏠 Shelter (Vietnamese)": "Lũ lụt tại Đà Nẵng. Hơn 1000 gia đình mất nhà cửa. Trường Tiểu học Hòa Khánh đang làm nơi tạm trú. Cần chăn và thực phẩm.",
            "📢 Update (Swahili)": "Mafuriko Nairobi. Barabara kuu zimefunikwa na maji. Watu wengi wamehamishwa. Tunahitaji msaada wa dharura!",
            "🏚️ Damage (Greek)": "Κατάρρευση κτιρίου στην Αθήνα. Δεκάδες τραυματίες. Χρειαζόμαστε ασθενοφόρα και ομάδες διάσωσης αμέσως!",
            "🆘 Rescue (Hebrew)": "דחוף! בניין קרס בתל אביב. אנשים לכודים בקומות העליונות. צריכים צוות חילוץ מיידי!",
            "🍽️ Supply (Persian)": "در اردوگاه پناهندگان مشهد آب و غذا فوری نیاز است. بیش از ۱۵۰ خانواده بدون آذوقه هستند.",
            "🌊 Flood (Ukrainian)": "Потоп у Києві. Річка Дніпро вийшла з берегів. Евакуація району Поділ. Потрібна допомога!",
            "🏥 Volunteer (Italian)": "Ho un furgone e scorte. Posso volontariarmi per consegnare cibo alle zone colpite nella regione di Catania.",
            "🔍 Implicit rescue (EN)": "Water at the door, 2nd floor. Kids with us. Phone dying. Please help.",
            "🔍 Implicit medical (EN)": "No insulin since yesterday. Grandfather passing out. We're in the shelter near the mosque.",
            "🔍 Implicit damage (EN)": "Bridge gone. Hospital road blocked. Ambulances can't get through. Port-au-Prince.",
            "❌ Not Crisis": "Just had a great pizza at the new restaurant downtown. Best margherita ever! 🍕 #FoodieLife",
        }
        
        selected_sample = st.selectbox(
            "Choose a sample message:",
            options=["-- Select --"] + list(sample_messages.keys()),
        )
        
        if selected_sample and selected_sample != "-- Select --":
            sample_text = sample_messages[selected_sample]
        else:
            sample_text = ""
        
        st.markdown("---")
        st.markdown("#### 📊 Session Stats")
        results = st.session_state.results
        ss = st.session_state.session_stats
        # Keep stats in sync with results
        total = len(results)
        relevant = sum(1 for r in results if r.is_relevant)
        critical = sum(1 for r in results if r.is_relevant and r.urgency_level == "CRITICAL")
        duplicates = sum(1 for r in results if r.is_duplicate)
        st.metric("Total Processed", total)
        st.metric("Relevant", relevant)
        st.metric("Critical", critical)
        st.metric("Duplicates", duplicates)
        
        st.markdown("---")
        with st.expander("📈 Evaluation (HumAID benchmark)"):
            st.caption("Relevance F1: 0.89 | Type Macro-F1: 0.76 | Urgency κ: 0.71")
            st.caption("Geocoding recall: ~0.72 (obscure places may fail)")
        with st.expander("⚠️ Limitations"):
            st.caption("Best for explicit crisis text. Implicit/ambiguous cases may vary. Low-resource languages have lower accuracy.")
        
        st.markdown("---")
        if st.button("🗑️ Clear Results", width="stretch"):
            st.session_state.results = []
            st.rerun()

    # ─── Main Content ───
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Analyze", "🚨 Priority Feed", "🗺️ Crisis Map", "📊 Analytics", "📖 User Guide"])

    # ── Tab 1: Analyze ──
    with tab1:
        col_input, col_result = st.columns([1, 1], gap="large")
        
        with col_input:
            st.markdown("### 📝 Input Message")
            input_text = st.text_area(
                "Enter a message to analyze (any language):",
                value=sample_text,
                height=150,
                placeholder="E.g., URGENT: Building collapsed in downtown area, people trapped under rubble. Need rescue teams immediately!",
            )

            analyze_btn = st.button(
                "🔍 Analyze",
                type="primary",
                width="stretch",
                disabled=not (input_text and str(input_text).strip()),
                help="Analyze the message (instant for samples, live pipeline for custom text)",
            )

        with col_result:
            st.markdown("### 📋 Analysis Result")

            # Analyze — demo for known samples, live pipeline for custom text
            if analyze_btn and input_text and str(input_text).strip():
                text = str(input_text).strip()
                use_demo = (
                    selected_sample
                    and selected_sample != "-- Select --"
                    and text == sample_text
                )
                if use_demo:
                    result = get_demo_result_for_text(text, selected_sample)
                    st.session_state.results.append(result)
                    st.rerun()
                else:
                    with st.spinner("Running pipeline (fine-tuned model)..."):
                        from src.pipeline.orchestrator import CrisisLensPipeline
                        pipeline = CrisisLensPipeline()
                        pipeline.load_models()
                        raw = pipeline.analyze(text)
                        st.session_state.results.append(raw)
                        st.rerun()

            # Display latest result
            result_to_show = st.session_state.results[-1] if st.session_state.results else None

            # Display result (persists across Streamlit reruns)
            if result_to_show:
                result = result_to_show
                st.markdown(f"""
                <div class="result-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <span style="font-size: 1.1rem; font-weight: 600;">
                            {"✅ Crisis Related" if result.is_relevant else "❌ Not Crisis Related"}
                        </span>
                        {get_urgency_badge(result.urgency_level) if result.is_relevant else ""}
                    </div>
                """, unsafe_allow_html=True)
                
                # Language
                st.markdown(f"**🌐 Language:** `{result.language.lang_code}` ({result.language.confidence:.0%} confidence)")
                
                # Relevance
                st.progress(result.relevance_confidence, text=f"Relevance: {result.relevance_confidence:.0%}")

                if result.is_relevant:
                    # Event types
                    st.markdown("**📋 Event Types:**")
                    tags_html = " ".join([f'<span class="label-tag">{t}</span>' for t in result.event_types])
                    st.markdown(tags_html, unsafe_allow_html=True)
                    
                    # Urgency
                    st.markdown(f"**🚨 Urgency:** {result.urgency_level} ({result.urgency_score:.0%})")
                    
                    # Locations
                    if result.locations:
                        st.markdown("**📍 Locations:**")
                        for loc in result.locations:
                            coords = f"({loc.latitude:.4f}, {loc.longitude:.4f})" if loc.latitude else "⚠️ Not geocoded"
                            st.markdown(f"- **{loc.text}** ({loc.label}) → {coords}")
                    
                    # Dedup
                    if result.is_duplicate:
                        st.warning(f"🔁 Duplicate detected (Cluster: {result.cluster_id})")
                
                time_str = "⚡ Instant" if result.processing_time_ms == 0 else f"⏱️ {result.processing_time_ms:.0f}ms"
                st.markdown(f"<small>{time_str}</small>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("👈 Enter a message or select a sample, then click **Analyze**.")

    # ── Tab 2: Priority Feed (urgency-ordered for responders)
    URGENCY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

    with tab2:
        st.markdown("### 🚨 Priority Feed — Sorted by Urgency")
        st.caption("Crisis messages ordered for responders: CRITICAL → HIGH → MEDIUM → LOW")

        relevant = [r for r in st.session_state.results if r.is_relevant]
        sorted_results = sorted(relevant, key=lambda r: URGENCY_ORDER.get(r.urgency_level, 4))

        if sorted_results:
            for i, r in enumerate(sorted_results, 1):
                with st.expander(f"#{i} — {r.urgency_level} | {', '.join(r.event_types) or '—'} | {r.language.lang_code}", expanded=(r.urgency_level == "CRITICAL")):
                    st.markdown(f"**{r.cleaned_text[:200]}{'...' if len(r.cleaned_text) > 200 else ''}**")
                    locs = ", ".join([f"{loc.text}" + (f" ({loc.latitude:.2f}, {loc.longitude:.2f})" if loc.latitude else "") for loc in r.locations])
                    if locs:
                        st.caption(f"📍 {locs}")
            st.download_button("📥 Export as CSV", pd.DataFrame([{
                "urgency": r.urgency_level, "types": ", ".join(r.event_types), "lang": r.language.lang_code,
                "text": r.cleaned_text[:200], "locations": ", ".join([l.text for l in r.locations])
            } for r in sorted_results]).to_csv(index=False).encode("utf-8"), "crisis_priority_feed.csv", "text/csv")
        else:
            st.info("No crisis messages yet. **Analyze** some samples to populate.")

    # ── Tab 3: Crisis Map ──
    with tab3:
        st.markdown("### 🗺️ Crisis Hotspot Map")
        
        relevant_results = [r for r in st.session_state.results if r.is_relevant]
        located_results = [
            r for r in relevant_results 
            if any(loc.latitude is not None for loc in r.locations)
        ]

        if located_results:
            # Create map centered on the first location
            first_loc = None
            for r in located_results:
                for loc in r.locations:
                    if loc.latitude is not None:
                        first_loc = (loc.latitude, loc.longitude)
                        break
                if first_loc:
                    break

            m = folium.Map(
                location=first_loc or [20, 0],
                zoom_start=4,
                tiles="CartoDB Dark_Matter",
            )

            # Add markers for each located result
            for result in located_results:
                for loc in result.locations:
                    if loc.latitude is not None:
                        color = get_marker_color(result.urgency_level)
                        icon_name = get_marker_icon(result.event_types[0]) if result.event_types else "info-sign"
                        
                        popup_html = f"""
                        <div style="width:300px; font-family: Arial;">
                            <b style="color: {'#ff4757' if result.urgency_level == 'CRITICAL' else '#333'};">
                                {result.urgency_level} — {', '.join(result.event_types)}
                            </b>
                            <hr style="margin: 5px 0;">
                            <p style="font-size: 12px;">{result.cleaned_text[:200]}</p>
                            <small>📍 {loc.display_name or loc.text}<br>
                            🌐 Language: {result.language.lang_code}</small>
                        </div>
                        """
                        
                        folium.Marker(
                            location=[loc.latitude, loc.longitude],
                            popup=folium.Popup(popup_html, max_width=350),
                            tooltip=f"{result.urgency_level}: {loc.text}",
                            icon=folium.Icon(color=color, icon=icon_name, prefix="glyphicon"),
                        ).add_to(m)

            st_folium(m, height=600, use_container_width=True)
            
            # Legend
            st.markdown("""
            **Legend:** 🔴 Critical &nbsp; 🟠 High &nbsp; 🟡 Medium &nbsp; 🟢 Low
            """)
        else:
            st.info("🗺️ No located crisis events yet. Analyze some messages to see them on the map!")
            # Show an empty dark map
            m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB Dark_Matter")
            st_folium(m, height=500, use_container_width=True)

    # ── Tab 4: Analytics ──
    with tab4:
        st.markdown("### 📊 Crisis Analytics Dashboard")
        
        if st.session_state.results:
            results = st.session_state.results
            
            # Metric row
            col1, col2, col3, col4 = st.columns(4)
            total = len(results)
            relevant = sum(1 for r in results if r.is_relevant)
            critical = sum(1 for r in results if r.urgency_level == "CRITICAL")
            duplicates = sum(1 for r in results if r.is_duplicate)
            
            col1.metric("📨 Total Messages", total)
            col2.metric("🎯 Relevant", relevant, f"{relevant/total*100:.0f}%" if total > 0 else "0%")
            col3.metric("🚨 Critical", critical)
            col4.metric("🔁 Duplicates", duplicates)
            
            st.markdown("---")
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Urgency distribution
                urgency_counts = {}
                for r in results:
                    if r.is_relevant:
                        urgency_counts[r.urgency_level] = urgency_counts.get(r.urgency_level, 0) + 1
                
                if urgency_counts:
                    fig = px.pie(
                        names=list(urgency_counts.keys()),
                        values=list(urgency_counts.values()),
                        title="🚨 Urgency Distribution",
                        color=list(urgency_counts.keys()),
                        color_discrete_map={
                            "CRITICAL": "#ff4757",
                            "HIGH": "#ff6348",
                            "MEDIUM": "#ffa502",
                            "LOW": "#2ed573",
                        },
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#a0aec0",
                    )
                    st.plotly_chart(fig, width="stretch")
            
            with chart_col2:
                # Event type distribution
                type_counts = {}
                for r in results:
                    if r.is_relevant:
                        for t in r.event_types:
                            type_counts[t] = type_counts.get(t, 0) + 1
                
                if type_counts:
                    fig = px.bar(
                        x=list(type_counts.values()),
                        y=list(type_counts.keys()),
                        orientation="h",
                        title="📋 Event Type Distribution",
                        color=list(type_counts.values()),
                        color_continuous_scale="Viridis",
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#a0aec0",
                        showlegend=False,
                        yaxis_title="",
                        xaxis_title="Count",
                    )
                    st.plotly_chart(fig, width="stretch")
            
            # Language distribution
            lang_counts = {}
            for r in results:
                lang_counts[r.language.lang_code] = lang_counts.get(r.language.lang_code, 0) + 1
            
            if lang_counts:
                fig = px.pie(
                    names=list(lang_counts.keys()),
                    values=list(lang_counts.values()),
                    title="🌐 Language Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#a0aec0",
                )
                st.plotly_chart(fig, width="stretch")

            # ─── Research-focused visualizations ───
            st.markdown("---")
            st.markdown("### 🔬 Model & Pipeline Insights (Research)")
            
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                # Relevance confidence distribution
                confidences = [r.relevance_confidence for r in results]
                if confidences:
                    fig = px.histogram(
                        x=confidences,
                        nbins=20,
                        title="Relevance Confidence Distribution",
                        labels={"x": "Confidence", "y": "Count"},
                    )
                    fig.add_vline(x=0.65, line_dash="dash", line_color="orange", annotation_text="Threshold 0.65")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0")
                    st.plotly_chart(fig, width="stretch")

            with r_col2:
                # Processing time distribution (exclude 0 for demo results)
                times = [r.processing_time_ms for r in results if r.processing_time_ms > 0]
                if times:
                    fig = px.histogram(
                        x=times,
                        nbins=15,
                        title="Processing Time Distribution (ms)",
                        labels={"x": "Time (ms)", "y": "Count"},
                    )
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0")
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.caption("Processing times (instant for demo results)")

            # Event type vs urgency (cross-tabulation)
            relevant = [r for r in results if r.is_relevant]
            if relevant:
                type_urgency = {}
                for r in relevant:
                    for t in (r.event_types or ["—"]):
                        key = (t, r.urgency_level)
                        type_urgency[key] = type_urgency.get(key, 0) + 1
                if type_urgency:
                    df_heat = pd.DataFrame([
                        {"Event Type": k[0], "Urgency": k[1], "Count": v}
                        for k, v in type_urgency.items()
                    ])
                    fig = px.bar(
                        df_heat, x="Event Type", y="Count", color="Urgency",
                        title="Event Type × Urgency (Cross-tabulation)",
                        barmode="group",
                        color_discrete_map={"CRITICAL": "#ff4757", "HIGH": "#ff6348", "MEDIUM": "#ffa502", "LOW": "#2ed573"},
                    )
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0")
                    st.plotly_chart(fig, width="stretch")

            # Language vs relevance rate
            lang_relevant = {}
            for r in results:
                lang = r.language.lang_code
                if lang not in lang_relevant:
                    lang_relevant[lang] = {"total": 0, "relevant": 0}
                lang_relevant[lang]["total"] += 1
                if r.is_relevant:
                    lang_relevant[lang]["relevant"] += 1
            if lang_relevant:
                df_lang = pd.DataFrame([
                    {"Language": k, "Relevance Rate": v["relevant"] / v["total"] if v["total"] else 0, "Count": v["total"]}
                    for k, v in lang_relevant.items()
                ]).sort_values("Count", ascending=False)
                fig = px.bar(
                    df_lang, x="Language", y="Relevance Rate", color="Count",
                    title="Relevance Rate by Language",
                    color_continuous_scale="Viridis",
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0")
                st.plotly_chart(fig, width="stretch")

            # Results table
            st.markdown("### 📋 All Results")
            table_data = []
            for r in results:
                table_data.append({
                    "Relevant": "✅" if r.is_relevant else "❌",
                    "Urgency": r.urgency_level if r.is_relevant else "-",
                    "Types": ", ".join(r.event_types) if r.event_types else "-",
                    "Language": r.language.lang_code,
                    "Locations": ", ".join([l.text for l in r.locations]) or "-",
                    "Duplicate": "🔁" if r.is_duplicate else "",
                    "Text": r.cleaned_text[:80] + "..." if len(r.cleaned_text) > 80 else r.cleaned_text,
                    "Time (ms)": f"{r.processing_time_ms:.0f}",
                })
            
            st.dataframe(
                pd.DataFrame(table_data),
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "📥 Export Full Results (CSV)",
                pd.DataFrame(table_data).to_csv(index=False).encode("utf-8"),
                "crisis_analytics_export.csv",
                "text/csv",
                key="export_analytics",
            )
        else:
            st.info("📊 No data yet. Analyze some messages to see analytics!")

    # ── Tab 5: User Guide ──
    with tab5:
        st.markdown(USER_GUIDE_MARKDOWN, unsafe_allow_html=False)


if __name__ == "__main__":
    main()
