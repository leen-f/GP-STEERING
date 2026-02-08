import streamlit as st
import os
import base64
import integrated as backend

st.set_page_config(
    page_title="Autonomous Driving Test",
    layout="wide"
)

def load_logo(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

LOGO_BASE64 = load_logo("logo.png")


st.markdown("""
<style>
     

body {
    background-color: #ffffff;
}

.header {
    background-color: #c4dcff;
    padding: 25px;
    text-align: center;
    margin-bottom: 25px;
}

.header img {
    height: 80px;
    display: block;
    margin: 0 auto;
}

.header h1 {
    color: #004aad;
    margin-bottom: 0;
    font-size: 32px;
}

.header h3 {
    color: #004aad;
    font-weight: 400;
    margin-top: 5px;
    font-size: 20px;
}

.sidebar-nav {
    background-color: #c4dcff;
    padding: 25px 20px;
    border-radius: 15px;
    margin-bottom: 25px;
    border-right: 5px solid #004aad;
    height: fit-content;
}

.sidebar-section {
    margin-bottom: 10px;
    padding-bottom: 5px;
}

.sidebar-section:not(:last-child) {
    border-bottom: 3px solid #004aad;
}

.sidebar-title {
    font-weight: bold;
    color: #004aad;
    font-size: 22px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.sidebar-title::before {
    content: "•";
    font-size: 30px;
    color: #004aad;
}

.sidebar-input {
    margin-bottom: 15px;
}

.sidebar-input label {
    font-size: 16px;
    color: #004aad;
    font-weight: 500;
    margin-bottom: 5px;
    display: block;
}

.sidebar-button {
    margin-top: 10px;
}

.main-card {
    background-color: #c4dcff;
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 25px;
    border-left: 5px solid #004aad;
    font-weight: bold
}

.main-title {
    color: #004aad;
    font-size: 24px;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 2px solid #004aad;
    
}

.stExpander {
    border: 2px solid #004aad !important;
    border-radius: 10px !important;
    margin-bottom: 15px !important;
    background-color: #e8f0ff !important;
}

.streamlit-expanderHeader {
    font-size: 20px !important;
    font-weight: bold !important;
    color: #004aad !important;
    background-color: #d4e2ff !important;
}

.stExpanderContent {
    font-size: 18px !important;
    background-color: #f5f8ff !important;
}

.stButton > button {
    background-color: #004aad !important;
    color: white !important;
    border-radius: 8px !important;
    width: 100% !important;
    height: 45px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    border: none !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    background-color: #003580 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(0, 74, 173, 0.3) !important;
}

.stTextInput > div > div > input {
    background-color: white !important;
    border: 2px solid #004aad !important;
    border-radius: 8px !important;
    font-size: 16px !important;
    color: #004aad !important;
}

.stTextInput label {
    font-size: 16px !important;
    color: #004aad !important;
    font-weight: 500 !important;
}

.status-card {
    background-color: #e8f0ff;
    padding: 20px;
    border-radius: 10px;
    margin: 15px 0;
    border-left: 5px solid #004aad;
}

.status-running {
    background-color: #d4ffd4 !important;
    border-left: 5px solid #28a745 !important;
}

.status-idle {
    background-color: #fff3cd !important;
    border-left: 5px solid #ffc107 !important;
}

.video-container {
    background-color: #e8f0ff;
    padding: 20px;
    border-radius: 10px;
    margin-top: 15px;
}

.results-header {
    font-size: 20px !important;
    font-weight: bold !important;
    color: #004aad !important;
    margin-bottom: 15px !important;
    padding-bottom: 8px !important;
    border-bottom: 2px solid #004aad !important;
}

.stText textarea {
    font-size: 16px !important;
    line-height: 1.6 !important;
    background-color: white !important;
    border: 2px solid #004aad !important;
    border-radius: 8px !important;
    padding: 15px !important;
}

.sidebar-container {
    position: relative;
    padding-right: 20px;
}

.sidebar-container::after {
    content: '';
    position: absolute;
    right: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    background: linear-gradient(to bottom, #004aad, #3a8cff, #004aad);
    border-radius: 2px;
}

.sidebar-nav {
    border-right: 4px solid #004aad !important;
    padding-right: 20px !important;
    margin-right: 15px !important;
}
            
.stAlert {
    font-size: 16px !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="header">
    <img src="data:image/png;base64,{LOGO_BASE64}">
    <h3>Autonomous Driving Test</h3>
</div>
""", unsafe_allow_html=True)


if "system" not in st.session_state:
    st.session_state.system = None

left, right = st.columns([1, 3])

with left:
    #st.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)
    #st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Personal Information</div>', unsafe_allow_html=True)
    
    full_name = st.text_input("Full Name", key="full_name")
    national_id = st.text_input("National ID or Passport No", key="national_id")
    
    if st.button("Submit Information", key="submit_info"):
        if national_id.strip():
            backend.NATIONAL_ID = national_id
            backend.BASE_DIR = os.path.join(os.getcwd(), national_id)
            os.makedirs(backend.BASE_DIR, exist_ok=True)
            backend.init_loggers()
            st.session_state.system = backend.IntegratedMonitoringSystem()
            st.success("Information saved successfully! \n READY TO START EVALUATION")
        else:
            st.error("National ID is required.")
    
    st.markdown('</div>', unsafe_allow_html=True)  
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Evaluation Controls</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start", key="start_eval"):
            if st.session_state.system and not st.session_state.system.running:
                st.session_state.system.run_combined_system()
                st.success("Evaluation started!")
            else:
                st.warning("System already running or not initialized.")
    
    with col2:
        if st.button("Stop", key="stop_eval"):
            if st.session_state.system:
                st.session_state.system.stop_combined_system()
                st.success("Evaluation stopped!")
            else:
                st.warning("System not initialized.")
    st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)


    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Generate Results", key="generate_results", use_container_width=True):
        if st.session_state.system:
            with st.spinner("Generating reports..."):
                reports = st.session_state.system.generate_mini_reports()
                driver_report = reports[0] if len(reports) > 0 else "No driver report generated."
                crossing_report = reports[1] if len(reports) > 1 else "No crossing report generated."

                final_report, final_result = st.session_state.system.generate_final_report()

                st.session_state.final_report = final_report
                st.session_state.final_result = final_result  
                st.session_state.final_ready = True

                st.session_state.driver_report = driver_report
                st.session_state.crossing_report = crossing_report
                st.session_state.final_ready = True
                st.success("Reports generated successfully!")
        else:
            st.error("Please initialize the system first.")
    
    st.markdown('</div>', unsafe_allow_html=True) 
    
    st.markdown('</div>', unsafe_allow_html=True) 

with right:
    #st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="main-title" style="font-weight:700; ">Playback Video</div>', unsafe_allow_html=True)

    if backend.BASE_DIR and os.path.exists(backend.BASE_DIR):
        annotated_videos = sorted(
            [
                f for f in os.listdir(backend.BASE_DIR)
                if f.endswith(".mp4") and "annotated" in f
            ],
            reverse=True
        )

        if annotated_videos:
            latest_video = annotated_videos[0]
            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            st.video(os.path.join(backend.BASE_DIR, latest_video))
            
        else:
            st.markdown("""
            <div class="status-card status-idle">
                <div style="font-size: 16px; color: #856404;">
                    No annotated video available yet. Start evaluation to record video.
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card status-idle">
            <div style="font-size: 16px; color: #856404;">
                No evaluation directory found. Please submit your information first.
            </div>
        </div>
        """, unsafe_allow_html=True)

    
    st.markdown('</div>', unsafe_allow_html=True)
    
    title_col, result_col = st.columns([4, 1])

    with title_col:
        st.markdown(
            '<div class="main-title"  style="font-weight:700; ">Results</div>',
            unsafe_allow_html=True
    )

    with result_col:
        if st.session_state.get("final_ready", False):
            result = st.session_state.final_result

            if result >= 35:
                st.markdown(
                    """
                    <div style="
                        background-color:#e7f8ef;
                        padding:12px;
                        border-radius:10px;
                        text-align:center;
                        border:2px solid #28a745;
                        font-size:24px;
                        font-weight:bold;
                        color:#155724;
                        margin-top:10px;">
                        PASS
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div style="
                        background-color:#ffd6d6;
                        padding:12px;
                        border-radius:10px;
                        text-align:center;
                        border:2px solid #dc3545;
                        font-size:24px;
                        font-weight:bold;
                        color:#721c24;
                        margin-top:10px;">
                        FAIL    
                    </div>
                    """,
                    unsafe_allow_html=True
                )


    with st.expander("FINAL COMPREHENSIVE REPORT", expanded=False):
        if st.session_state.get("final_ready", False):

            col1, col2 = st.columns([3, 1])

            st.markdown(
                    "<div class='results-header'>Complete Report</div>",
                    unsafe_allow_html=True
                )
            st.text(st.session_state.final_report)

            st.download_button(
                    label="Download Full Report",
                    data=st.session_state.final_report,
                    file_name=f"complete_report_{backend.NATIONAL_ID}.txt",
                    mime="text/plain",
                    use_container_width=True)

        else:
            st.info("Generate results to view the final comprehensive report.")

    
    with st.expander("Driver Behavior Analysis", expanded=False):
        if "driver_report" in st.session_state:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='results-header'>Analysis Summary</div>", unsafe_allow_html=True)
                st.text(st.session_state.driver_report)
            
        else:
            st.info("No driver behavior report available. Please generate results.")
    
    with st.expander("Crossing Events Analysis", expanded=False):
        if "crossing_report" in st.session_state:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='results-header'>Risk Summary</div>", unsafe_allow_html=True)
                st.text(st.session_state.crossing_report)
                report_lower = st.session_state.crossing_report.lower()
                if "collision" in report_lower or "risk" in report_lower or "high" in report_lower:
                    st.text("""
                            SAFETY RECOMMENDATIONS:
                            1. Increase following distance
                            2. Practice defensive driving techniques
                            3. Attend a hazard perception course
                            4. Slow down at intersections
                            """)
                elif "pedestrian" in report_lower:
                    st.text("""
                            SAFETY RECOMMENDATIONS:
                            1. Always scan for pedestrians
                            2. Reduce speed in crowded areas
                            3. Make eye contact when possible
                            4. Be prepared to stop suddenly
                            """)
                else:
                    st.text("""
                            SAFETY RECOMMENDATIONS:
                            1. Maintain current safe practices
                            2. Continue scanning intersections
                            3. Regular hazard awareness training
                            """)
        else:
            st.info("No collision report available. Please generate results.")
    
    st.markdown('</div>', unsafe_allow_html=True)
