import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime

# PAGE CONFIGURATION
st.set_page_config(
    page_title="InspectAI | Quality Inspection",
    page_icon="🔍",
    layout="wide"
)

# CUSTOM DESIGN
st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
[data-testid="stMetric"] {
    background: #17243a;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #263b59;
}
.stButton button {
    background: #2563eb;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 25px;
}
</style>
""", unsafe_allow_html=True)

# SESSION STORAGE
if "history" not in st.session_state:
    st.session_state.history = []

# SIDEBAR
st.sidebar.title("🔍 InspectAI")
st.sidebar.caption("Smart Quality Inspection System")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "New Inspection", "Inspection History", "About"]
)

# DASHBOARD
if page == "Dashboard":

    st.title("🔍 InspectAI Dashboard")
    st.write("AI-assisted manufacturing quality inspection")

    total = len(st.session_state.history)
    passed = sum(x["Result"] == "PASS" for x in st.session_state.history)
    failed = total - passed

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Inspections", total)
    col2.metric("Passed Products", passed)
    col3.metric("Defective Products", failed)

    st.divider()

    st.subheader("Welcome to InspectAI")

    st.write("""
    InspectAI helps quality inspection teams identify
    suspicious visual differences in manufactured products.

    Upload a reference image and a test image to begin
    your inspection.
    """)

    st.info("Select 'New Inspection' from the sidebar to start.")

# NEW INSPECTION
elif page == "New Inspection":

    st.title("Product Quality Inspection")
    st.write("Compare a reference product with a test product.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Reference Image")
        reference_file = st.file_uploader(
            "Upload defect-free reference image",
            type=["jpg", "jpeg", "png"],
            key="reference"
        )

    with col2:
        st.subheader("Test Product Image")
        test_file = st.file_uploader(
            "Upload product to inspect",
            type=["jpg", "jpeg", "png"],
            key="test"
        )

    sensitivity = st.slider(
        "Inspection Sensitivity",
        5, 100, 30
    )

    if reference_file and test_file:

        reference = Image.open(reference_file).convert("RGB")
        test = Image.open(test_file).convert("RGB")

        st.subheader("Uploaded Images")

        img1, img2 = st.columns(2)

        img1.image(reference, caption="Reference Product",
                   use_container_width=True)

        img2.image(test, caption="Test Product",
                   use_container_width=True)

        if st.button("🔍 Analyze Product", use_container_width=True):

            ref = cv2.cvtColor(np.array(reference), cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(np.array(test), cv2.COLOR_RGB2GRAY)

            img = cv2.resize(img, (ref.shape[1], ref.shape[0]))

            ref_blur = cv2.GaussianBlur(ref, (5, 5), 0)
            img_blur = cv2.GaussianBlur(img, (5, 5), 0)

            difference = cv2.absdiff(ref_blur, img_blur)

            _, threshold = cv2.threshold(
                difference,
                sensitivity,
                255,
                cv2.THRESH_BINARY
            )

            kernel = np.ones((5, 5), np.uint8)
            threshold = cv2.morphologyEx(
                threshold, cv2.MORPH_OPEN, kernel
            )

            contours, _ = cv2.findContours(
                threshold,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            highlighted = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            defect_count = 0

            for contour in contours:
                area = cv2.contourArea(contour)

                if area > 20:
                    defect_count += 1
                    x, y, w, h = cv2.boundingRect(contour)

                    cv2.rectangle(
                        highlighted,
                        (x, y),
                        (x + w, y + h),
                        (0, 0, 255),
                        2
                    )

            changed_pixels = np.count_nonzero(threshold)
            difference_percentage = (
                changed_pixels / threshold.size
            ) * 100

            result = (
                "DEFECT DETECTED"
                if defect_count > 0
                else "PASS"
            )

            st.session_state.history.append({
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Result": result,
                "Defect Regions": defect_count,
                "Difference (%)": round(difference_percentage, 2)
            })

            st.divider()
            st.subheader("Inspection Results")

            if defect_count > 0:
                st.error("⚠️ DEFECT DETECTED")
            else:
                st.success("✅ PRODUCT PASSED")

            c1, c2, c3 = st.columns(3)

            c1.metric("Defect Regions", defect_count)
            c2.metric("Difference", f"{difference_percentage:.2f}%")
            c3.metric("Result", result)

            st.subheader("Defect Highlighting")

            st.image(
                cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB),
                caption="Detected suspicious differences",
                use_container_width=True
            )

            st.info(
                "This prototype uses image processing to identify "
                "visual differences. Lighting, positioning and "
                "image alignment can affect results."
            )

    else:
        st.info("Upload both images to begin inspection.")

# HISTORY
elif page == "Inspection History":

    st.title("Inspection History")

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history)

        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Inspection Report",
            csv,
            "inspection_report.csv",
            "text/csv"
        )

    else:
        st.info("No inspections performed yet.")

# ABOUT
elif page == "About":

    st.title("About InspectAI")

    st.write("""
    InspectAI is a computer vision quality inspection
    application designed for manufacturing environments.

    Technologies Used:

    - Python
    - Streamlit
    - OpenCV
    - NumPy
    - Pandas
    - Pillow

    Main Objective:

    To assist manufacturers in identifying visual
    differences in products and improving quality inspection.
    """)

st.sidebar.divider()
st.sidebar.caption("InspectAI | Computer Vision Quality Inspection")
