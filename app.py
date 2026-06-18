import streamlit as st
import sqlite3
from datetime import datetime
import pandas as pd
import plotly.express as px
import os
import tempfile
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Gridlock Shield — Bengaluru Traffic AI",
    page_icon="🚦",
    layout="wide"
)

@st.cache_resource
def load_helmet_model():
    return YOLO("model/best.pt")

@st.cache_resource
def load_plate_model():
    return YOLO("model/plate_best.pt")

@st.cache_resource
def load_coco_model():
    return YOLO("yolov8n.pt")

model = load_helmet_model()
plate_model = load_plate_model()
coco_model = load_coco_model()

def init_db():
    conn = sqlite3.connect("violations.db", check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            violation_type TEXT,
            confidence REAL,
            location TEXT,
            plate_number TEXT
        )
    """)
    conn.commit()
    return conn

db = init_db()

try:
    db.execute("SELECT plate_number FROM violations LIMIT 1")
except sqlite3.OperationalError:
    db.execute("ALTER TABLE violations ADD COLUMN plate_number TEXT")
    db.commit()

def save_violation(label, conf, location, plate_number="N/A"):
    db.execute(
        "INSERT INTO violations (timestamp, violation_type, confidence, location, plate_number) VALUES (?,?,?,?,?)",
        (datetime.now().isoformat(), label, conf, location, plate_number)
    )
    db.commit()

def read_plate(image_array):
    """Detect plate in image, then read with EasyOCR. Crash-proof."""
    try:
        plate_results = plate_model(image_array, conf=0.25)
        boxes = plate_results[0].boxes

        if len(boxes) == 0:
            return "Not detected"

        box = boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_crop = image_array[max(0, y1):y2, max(0, x1):x2]

        if plate_crop.size == 0 or plate_crop.shape[0] < 10 or plate_crop.shape[1] < 10:
            return "Not detected"

        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        result = reader.readtext(plate_crop, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

        if result:
            text = "".join([r[1] for r in result]).upper().replace(" ", "")
            if len(text) >= 3:
                return text

        return "Unreadable"
    except Exception:
        return "OCR error"

def compute_iou(boxA, boxB):
    """Compute Intersection over Union between two boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(areaA + areaB - interArea)

def detect_triple_riding(image_array):
    """
    Detect triple riding using COCO pretrained model.
    Logic: For each motorcycle, count persons whose bottom-center
    falls within the motorcycle bounding box (strict overlap).
    Only flag if 3+ distinct persons are on the SAME motorcycle.
    """
    try:
        results = coco_model(image_array, conf=0.35, verbose=False)
        boxes = results[0].boxes

        motorcycles = []
        persons = []

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_id == 3:  # motorcycle in COCO
                motorcycles.append((x1, y1, x2, y2, conf))
            elif cls_id == 0:  # person in COCO
                persons.append((x1, y1, x2, y2, conf))

        violations = []

        for mx1, my1, mx2, my2, mconf in motorcycles:
            mw = mx2 - mx1
            mh = my2 - my1

            # Only consider persons whose bottom-center is inside motorcycle box
            # AND whose person box overlaps significantly with motorcycle box
            persons_on_bike = 0
            for px1, py1, px2, py2, pconf in persons:
                # Bottom center of person
                p_bottom_cx = (px1 + px2) // 2
                p_bottom_cy = py2

                # Check: bottom center within motorcycle box (with small vertical extension upward)
                in_x = mx1 <= p_bottom_cx <= mx2
                in_y = (my1 - int(mh * 1.2)) <= p_bottom_cy <= my2

                # Also check IoU overlap between person and motorcycle box
                iou = compute_iou((mx1, my1, mx2, my2), (px1, py1, px2, py2))

                if in_x and in_y and iou > 0.05:
                    persons_on_bike += 1

            if persons_on_bike >= 3:
                violations.append({
                    "type": "Triple Riding",
                    "confidence": round(mconf * 100, 1),
                    "box": (mx1, my1, mx2, my2),
                    "persons_count": persons_on_bike
                })

        return violations

    except Exception:
        return []

CLASS_LABELS = {0: "With Helmet", 1: "Without Helmet"}
CLASS_ICONS = {0: "🟢", 1: "🔴"}

# ---- SIDEBAR ----
st.sidebar.title("🚦 Gridlock Shield")
st.sidebar.markdown("AI-powered Traffic Enforcement · Bengaluru")
st.sidebar.success("✅ Models Loaded")

location = st.sidebar.selectbox("📍 Camera Location", [
    "Silk Board Junction", "MG Road", "Koramangala Signal",
    "Whitefield Main Road", "Hebbal Flyover"
])
page = st.sidebar.radio("Navigate", ["Live Detection", "Analytics", "Heatmap", "Impact Simulator", "Model Performance"])

# ==================== PAGE 1: LIVE DETECTION ====================
if page == "Live Detection":
    st.title("🔍 Live Violation Detection")
    st.markdown(f"**Active Camera:** {location}")

    col1, col2 = st.columns([1.3, 1])

    with col1:
        upload_type = st.radio("Input type", ["Image", "Video"], horizontal=True)

        uploaded = None
        uploaded_video = None
        results = None
        detected_plate = None
        triple_violations = []

        if upload_type == "Image":
            uploaded = st.file_uploader("Upload traffic image", type=["jpg", "jpeg", "png"])
        else:
            uploaded_video = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov"])

        # ---- IMAGE HANDLING ----
        if uploaded:
            img = Image.open(uploaded)
            img = img.convert("RGB")
            img_array = np.array(img)

            with st.spinner("Analyzing image..."):
                results = model(img_array, conf=0.3)

            annotated = img_array.copy()
            boxes = results[0].boxes

            # Helmet detection boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = CLASS_LABELS.get(cls_id, f"Class {cls_id}")
                color = (0, 200, 0) if cls_id == 0 else (255, 0, 0)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{label} {conf:.0%}", (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if cls_id == 1:
                    with st.spinner("Reading number plate..."):
                        detected_plate = read_plate(img_array)
                    save_violation(label, conf, location, detected_plate)

            # Triple riding detection
            triple_violations = detect_triple_riding(img_array)
            for tv in triple_violations:
                tx1, ty1, tx2, ty2 = tv['box']
                cv2.rectangle(annotated, (tx1, ty1), (tx2, ty2), (255, 165, 0), 3)
                cv2.putText(annotated, f"TRIPLE RIDING {tv['confidence']}% ({tv['persons_count']} persons)",
                            (tx1, max(ty1 - 15, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                save_violation("Triple Riding", tv['confidence'] / 100, location, "N/A")

            # Show final annotated image ONCE
            st.image(annotated, caption="Detection Result", use_container_width=True)

        # ---- VIDEO HANDLING ----
        if upload_type == "Video" and uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            video_path = tfile.name

            st.video(video_path)

            if st.button("▶ Analyze Video"):
                cap = cv2.VideoCapture(video_path)
                frame_placeholder = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_count = 0
                video_violations = []
                process_every_n = 5

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % process_every_n == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        vid_results = model(frame_rgb, conf=0.3, verbose=False)
                        vid_boxes = vid_results[0].boxes

                        annotated_frame = frame_rgb.copy()

                        # Helmet boxes
                        for box in vid_boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            label = CLASS_LABELS.get(cls_id, f"Class {cls_id}")
                            color = (0, 200, 0) if cls_id == 0 else (255, 0, 0)

                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(annotated_frame, f"{label} {conf:.0%}", (x1, max(y1 - 10, 20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                            if cls_id == 1:
                                plate = read_plate(frame_rgb)
                                save_violation(label, conf, location, plate)
                                video_violations.append({"type": label, "conf": conf, "plate": plate})

                        # Triple riding in video
                        vid_triple = detect_triple_riding(frame_rgb)
                        for tv in vid_triple:
                            tx1, ty1, tx2, ty2 = tv['box']
                            cv2.rectangle(annotated_frame, (tx1, ty1), (tx2, ty2), (255, 165, 0), 3)
                            cv2.putText(annotated_frame, f"TRIPLE RIDING {tv['confidence']}%",
                                        (tx1, max(ty1 - 15, 20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                            save_violation("Triple Riding", tv['confidence'] / 100, location, "N/A")
                            video_violations.append({"type": "Triple Riding", "conf": tv['confidence'] / 100, "plate": "N/A"})

                        frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

                    frame_count += 1
                    if total_frames > 0:
                        progress_bar.progress(min(frame_count / total_frames, 1.0))
                    status_text.text(f"Processing frame {frame_count}/{total_frames}")

                cap.release()
                st.success(f"✅ Video analysis complete! {len(video_violations)} violations detected.")

                for v in video_violations:
                    st.error(f"🔴 {v['type']} — Confidence: {v['conf']*100:.1f}% — Plate: {v['plate']}")

    with col2:
        st.subheader("⚠️ Detections")
        if uploaded and results:
            boxes = results[0].boxes
            has_detections = len(boxes) > 0 or len(triple_violations) > 0

            if not has_detections:
                st.info("No detections found")
            else:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = CLASS_LABELS.get(cls_id, f"Class {cls_id}")
                    icon = CLASS_ICONS.get(cls_id, "⚪")

                    if cls_id == 1:
                        st.error(f"{icon} **{label}**")
                        st.write(f"🔢 Plate: `{detected_plate}`")
                    else:
                        st.success(f"{icon} **{label}**")
                    st.write(f"Confidence: `{conf*100:.1f}%`")
                    st.divider()

                for tv in triple_violations:
                    st.error(f"🟠 **Triple Riding** ({tv['persons_count']} persons detected)")
                    st.write(f"Confidence: `{tv['confidence']}%`")
                    st.divider()
        else:
            st.info("Upload an image or video to see detections")

# ==================== PAGE 2: ANALYTICS ====================
elif page == "Analytics":
    st.title("📊 Violation Analytics")
    st.markdown("Insights from all detected violations")

    df = pd.read_sql("SELECT * FROM violations", db)

    if df.empty:
        st.warning("No violations recorded yet. Go to **Live Detection** and upload some images first!")
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['date'] = df['timestamp'].dt.date

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Violations", len(df))
        c2.metric("Locations Covered", df['location'].nunique())
        c3.metric("Avg Confidence", f"{df['confidence'].mean()*100:.1f}%")
        c4.metric("Most Common", df['violation_type'].mode()[0])

        col1, col2 = st.columns(2)

        with col1:
            loc_counts = df['location'].value_counts().reset_index()
            loc_counts.columns = ['location', 'count']
            fig1 = px.bar(loc_counts, x='location', y='count',
                          title="Violations by Location",
                          color='count', color_continuous_scale='Reds')
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            hourly = df.groupby('hour').size().reset_index(name='count')
            fig2 = px.line(hourly, x='hour', y='count',
                           title="Violations by Hour of Day", markers=True)
            st.plotly_chart(fig2, use_container_width=True)

        vtype = df['violation_type'].value_counts().reset_index()
        vtype.columns = ['violation_type', 'count']
        fig3 = px.pie(vtype, names='violation_type', values='count',
                      title="Violation Type Breakdown",
                      color_discrete_sequence=px.colors.sequential.Reds_r)
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("📋 Recent Violations")
        st.dataframe(
            df[['timestamp', 'violation_type', 'confidence', 'location', 'plate_number']].sort_values('timestamp', ascending=False),
            use_container_width=True
        )

        if st.button("🗑️ Clear All Records"):
            db.execute("DELETE FROM violations")
            db.commit()
            st.rerun()

# ==================== PAGE 3: HEATMAP ====================
elif page == "Heatmap":
    st.title("🗺️ Enforcement Priority Heatmap")
    st.markdown("AI-recommended zones for targeted traffic police deployment")

    junction_coords = {
        "Silk Board Junction": (12.9170, 77.6229),
        "MG Road": (12.9757, 77.6077),
        "Koramangala Signal": (12.9352, 77.6245),
        "Whitefield Main Road": (12.9698, 77.7499),
        "Hebbal Flyover": (13.0348, 77.5970),
    }

    df = pd.read_sql("SELECT location, COUNT(*) as count, AVG(confidence) as avg_conf FROM violations GROUP BY location", db)

    if df.empty:
        st.warning("No violation data yet. Showing demo data.")
        df = pd.DataFrame({
            "location": list(junction_coords.keys()),
            "count": [12, 7, 9, 4, 6],
            "avg_conf": [0.71, 0.65, 0.68, 0.60, 0.69]
        })

    max_count = df['count'].max()
    df['priority_score'] = ((df['count'] / max_count) * 10).round(1)

    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12, tiles="CartoDB dark_matter")

    for _, row in df.iterrows():
        loc_name = row['location']
        if loc_name not in junction_coords:
            continue
        lat, lon = junction_coords[loc_name]
        score = row['priority_score']
        count = row['count']
        color = "red" if score > 7 else "orange" if score > 4 else "green"

        folium.CircleMarker(
            location=[lat, lon],
            radius=10 + count,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{loc_name}</b><br>Violations: {count}<br>Priority Score: {score}/10",
                max_width=200
            ),
            tooltip=f"{loc_name} — Priority: {score}/10"
        ).add_to(m)

    st_folium(m, width=None, height=480, use_container_width=True)

    st.subheader("🎯 Enforcement Recommendations")
    priority_sorted = df.sort_values('priority_score', ascending=False)
    for _, row in priority_sorted.iterrows():
        score = row['priority_score']
        badge = "🔴 URGENT" if score > 7 else "🟡 HIGH" if score > 4 else "🟢 NORMAL"
        st.write(f"{badge} **{row['location']}** — Priority Score: {score}/10 · {int(row['count'])} violations detected")

# ==================== PAGE 4: IMPACT SIMULATOR ====================
elif page == "Impact Simulator":
    st.title("🔮 Congestion Impact Simulator")
    st.markdown("*Quantify how AI enforcement reduces Bengaluru's traffic congestion*")

    col1, col2 = st.columns(2)

    with col1:
        junction = st.selectbox("Select Junction", [
            "Silk Board Junction", "MG Road", "Koramangala Signal",
            "Whitefield Main Road", "Hebbal Flyover"
        ])
        cameras = st.slider("Number of AI Cameras Deployed", 1, 20, 5)
        months = st.slider("Deployment Period (months)", 1, 12, 6)
        compliance_improvement = st.slider("Expected Compliance Improvement (%)", 10, 60, 30)

    with col2:
        violations_per_camera_per_hour = 15
        active_hours_per_day = 8
        total_violations_caught = cameras * violations_per_camera_per_hour * active_hours_per_day * 30 * months
        congestion_reduction = round(compliance_improvement * 0.4, 1)
        time_saved_per_commuter = round(congestion_reduction * 0.5, 1)
        avg_fine = 500
        fine_revenue = total_violations_caught * avg_fine

        st.metric("Violations Detected (Projected)", f"{total_violations_caught:,}")
        st.metric("Estimated Congestion Reduction", f"{congestion_reduction}%")
        st.metric("Time Saved Per Commuter Daily", f"{time_saved_per_commuter} mins")
        st.metric("Potential Fine Revenue", f"₹{fine_revenue:,.0f}")

    months_range = list(range(1, months + 1))
    congestion_curve = [100 - (congestion_reduction / months) * m for m in months_range]

    fig = px.line(
        x=months_range, y=congestion_curve,
        labels={'x': 'Month', 'y': 'Congestion Index (100 = current level)'},
        title=f"Projected Congestion Reduction at {junction}",
        markers=True
    )
    fig.add_hline(y=100, line_dash="dash", annotation_text="Current Congestion Level")
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"""
    **📢 Pitch-ready statement:**

    Deploying **{cameras} AI cameras** at **{junction}** for **{months} months** will detect approximately
    **{total_violations_caught:,} violations**, reduce congestion by **{congestion_reduction}%**,
    and save each commuter **{time_saved_per_commuter} minutes** daily — while generating
    **₹{fine_revenue:,.0f}** in potential fine revenue for civic infrastructure improvement.
    """)

    st.caption("⚠️ Projections based on simulated enforcement model assumptions for demonstration purposes.")

# ==================== PAGE 5: MODEL PERFORMANCE ====================
elif page == "Model Performance":
    st.title("📈 Model Performance Metrics")
    st.markdown("Technical evaluation of trained AI models")

    tab1, tab2 = st.tabs(["🪖 Helmet Detection Model", "🔢 Number Plate Detection Model"])

    with tab1:
        st.subheader("Helmet Violation Detection — YOLOv11n")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("mAP50", "0.744")
        c2.metric("Precision", "0.824")
        c3.metric("Recall", "0.669")
        c4.metric("mAP50-95", "0.482")

        st.markdown("""
        **Training Details:**
        - Architecture: YOLOv11n (2.59M parameters)
        - Dataset: 8,791 training images, 1,955 validation images
        - Epochs: 50 | Image size: 640×640
        - Hardware: NVIDIA Tesla T4 GPU
        - Training time: ~85 minutes
        """)

    with tab2:
        st.subheader("Number Plate Detection — YOLOv11n")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("mAP50", "0.993")
        c2.metric("Precision", "0.990")
        c3.metric("Recall", "0.976")
        c4.metric("mAP50-95", "0.783")

        st.markdown("""
        **Training Details:**
        - Architecture: YOLOv11n (2.58M parameters)
        - Dataset: 2,069 training images, 336 validation images
        - Epochs: 50 | Image size: 640×640
        - Hardware: NVIDIA Tesla T4 GPU
        - Training time: ~20 minutes
        """)

    st.divider()
    st.subheader("🔧 System Pipeline")
    st.markdown("""
    1. **Image/Video Input** → traffic camera feed or uploaded file
    2. **Helmet Detection** (YOLOv11n, mAP50: 0.744) → identifies helmet violations
    3. **Triple Riding Detection** (YOLOv8n COCO + IoU overlap logic) → counts persons per motorcycle
    4. **Plate Localization** (YOLOv11n, mAP50: 0.993) → finds number plate region
    5. **OCR Extraction** (EasyOCR) → reads alphanumeric plate text
    6. **Database Logging** → timestamp, location, confidence, plate stored in SQLite
    7. **Analytics & Enforcement Priority** → actionable insights for traffic police
    """)