import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import json
import base64
from datetime import datetime
import tempfile
import os

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LeafDustLab · Area Measurement",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0f1117; }

    .hero-header {
        background: linear-gradient(135deg, #1a2f1a 0%, #0d2b1a 50%, #0a1f2e 100%);
        border: 1px solid #2d5a2d;
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 28px;
        display: flex;
        align-items: center;
        gap: 18px;
    }
    .hero-title { font-size: 2rem; font-weight: 700; color: #7fff7f; margin: 0; }
    .hero-sub { font-size: 0.95rem; color: #88aa88; margin: 4px 0 0 0; }

    .metric-card {
        background: #161b22;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #4a9a4a; }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #7fff7f; }
    .metric-label { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

    .result-box {
        background: #0d2b1a;
        border: 1px solid #2d5a2d;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 16px 0;
    }
    .result-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #1a3a1a; }
    .result-row:last-child { border-bottom: none; }
    .result-key { color: #88aa88; font-size: 0.9rem; }
    .result-val { color: #7fff7f; font-weight: 600; font-size: 1rem; }

    .log-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
    }
    .log-table th {
        background: #1a2f1a;
        color: #7fff7f;
        padding: 10px 14px;
        text-align: left;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.75rem;
    }
    .log-table td {
        padding: 10px 14px;
        color: #ccc;
        border-bottom: 1px solid #1e2a1e;
    }
    .log-table tr:hover td { background: #161b22; }

    .stButton > button {
        background: linear-gradient(135deg, #2d7a2d, #1a5a1a);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 28px;
        font-weight: 600;
        font-size: 0.95rem;
        cursor: pointer;
        transition: all 0.2s;
        width: 100%;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #3d9a3d, #2a7a2a); transform: translateY(-1px); }

    .warning-box {
        background: #2a1a00;
        border: 1px solid #7a4a00;
        border-radius: 10px;
        padding: 14px 18px;
        color: #ffaa44;
        font-size: 0.88rem;
        margin: 10px 0;
    }
    .success-box {
        background: #0a2a0a;
        border: 1px solid #2a6a2a;
        border-radius: 10px;
        padding: 14px 18px;
        color: #66dd66;
        font-size: 0.88rem;
        margin: 10px 0;
    }
    .section-header {
        color: #7fff7f;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 20px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #2d5a2d;
    }
    div[data-testid="stNumberInput"] label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stSelectbox"] label { color: #aaccaa !important; font-size: 0.88rem !important; }

    div[data-testid="stSlider"] label { color: #aaccaa !important; }

    .hsv-badge {
        display: inline-block;
        background: #1a3a1a;
        border: 1px solid #3a7a3a;
        border-radius: 6px;
        padding: 4px 10px;
        font-family: monospace;
        font-size: 0.85rem;
        color: #88ee88;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────
if "log" not in st.session_state:
    st.session_state.log = []  # list of result dicts
if "gdoc_creds" not in st.session_state:
    st.session_state.gdoc_creds = None
if "gdoc_url" not in st.session_state:
    st.session_state.gdoc_url = ""

# ─── CV Logic ───────────────────────────────────────────────────────────────
def find_card_scale(gray, thresh_val=40):
    """Detect matte black calibration card and return px_per_cm."""
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    # Pick largest dark region that is roughly square
    card_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(card_contour)
    return (w + h) / 2, (x, y, w, h)

def find_leaf_area(img_bgr, hsv_lower, hsv_upper):
    """Detect leaf using HSV colour mask."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, mask
    leaf_contour = max(contours, key=cv2.contourArea)
    leaf_px = cv2.contourArea(leaf_contour)
    return leaf_px, leaf_contour, mask

def annotate_image(img_bgr, card_rect, leaf_contour, px_per_cm, leaf_area_cm2):
    """Draw bounding boxes and labels on a copy of the image."""
    out = img_bgr.copy()
    # Card rectangle — cyan
    if card_rect:
        x, y, w, h = card_rect
        cv2.rectangle(out, (x, y), (x+w, y+h), (255, 220, 0), 3)
        cv2.putText(out, f"Cal card  {px_per_cm:.1f}px/cm", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2)
    # Leaf contour — green
    if leaf_contour is not None:
        cv2.drawContours(out, [leaf_contour], -1, (60, 255, 60), 3)
        M = cv2.moments(leaf_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(out, f"{leaf_area_cm2:.2f} cm2", (cx-60, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 255, 60), 3)
    return out

def live_preview(img_bytes, hsv_lower, hsv_upper, card_thresh):
    """
    Colour-overlay preview — updates live as sliders move.
      CYAN   = calibration card detected region
      LIME   = leaf HSV mask
    Returns (jpeg_bytes, card_found, leaf_found).
    """
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    overlay = img.copy()

    # ── Card overlay (cyan) ──────────────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh_card = cv2.threshold(gray, card_thresh, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh_card = cv2.morphologyEx(thresh_card, cv2.MORPH_CLOSE, kernel)
    contours_card, _ = cv2.findContours(thresh_card, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_found = False
    cx_card = cy_card = cw_card = ch_card = 0
    if contours_card:
        c = max(contours_card, key=cv2.contourArea)
        cx_card, cy_card, cw_card, ch_card = cv2.boundingRect(c)
        cv2.rectangle(overlay, (cx_card, cy_card),
                      (cx_card+cw_card, cy_card+ch_card), (255, 230, 50), -1)
        card_found = True

    # ── Leaf overlay (lime green) ────────────────────────────────────────
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    leaf_mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, k2)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, k2)
    overlay[leaf_mask > 0] = (50, 255, 80)
    leaf_found = bool(np.any(leaf_mask))

    # Blend
    result_img = cv2.addWeighted(overlay, 0.42, img, 0.58, 0)

    # Hard outlines
    if card_found:
        cv2.rectangle(result_img,
                      (cx_card, cy_card), (cx_card+cw_card, cy_card+ch_card),
                      (0, 220, 255), 3)
        cv2.putText(result_img, "CAL CARD", (cx_card+6, max(cy_card-8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

    contours_leaf, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_leaf:
        lc = max(contours_leaf, key=cv2.contourArea)
        cv2.drawContours(result_img, [lc], -1, (50, 255, 80), 3)
        M = cv2.moments(lc)
        if M["m00"] != 0:
            lcx = int(M["m10"] / M["m00"])
            lcy = int(M["m01"] / M["m00"])
            cv2.putText(result_img, "LEAF", (lcx-30, lcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 255, 80), 2)

    _, buf = cv2.imencode(".jpg", result_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes(), card_found, leaf_found

def process_image(img_bytes, card_size_cm, hsv_lower, hsv_upper, card_thresh):
    """Full pipeline. Returns dict with results + annotated image bytes."""
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    px_avg, card_rect = find_card_scale(gray, card_thresh)
    if px_avg is None:
        return {"error": "Could not detect calibration card. Check card threshold or image contrast."}

    px_per_cm = px_avg / card_size_cm
    px_per_cm2 = px_per_cm ** 2

    leaf_px, leaf_contour, mask = find_leaf_area(img, hsv_lower, hsv_upper)
    if leaf_px is None:
        return {"error": "Could not detect leaf. Adjust the HSV range for this species."}

    leaf_area_cm2 = round(leaf_px / px_per_cm2, 2)
    annotated = annotate_image(img, card_rect, leaf_contour, px_per_cm, leaf_area_cm2)
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    annotated_bytes = buf.tobytes()

    return {
        "leaf_area_cm2": leaf_area_cm2,
        "px_per_cm": round(px_per_cm, 2),
        "leaf_px": int(leaf_px),
        "annotated_bytes": annotated_bytes,
        "mask_bytes": _mask_bytes(mask),
    }

def _mask_bytes(mask):
    _, buf = cv2.imencode(".jpg", mask)
    return buf.tobytes()

# ─── Google Docs Export ──────────────────────────────────────────────────────
def export_to_gdoc(records, service_account_json: str, doc_id: str):
    """Append leaf data rows (with images) to a Google Doc."""
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload

    creds_dict = json.loads(service_account_json)
    SCOPES = [
        "https://www.googleapis.com/auth/documents",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    docs = build("docs", "v1", credentials=creds)
    drive = build("drive", "v3", credentials=creds)

    # Get current doc length
    doc = docs.documents().get(documentId=doc_id).execute()
    end_idx = doc["body"]["content"][-1]["endIndex"] - 1

    requests = []

    def txt(text, bold=False, heading=None):
        """Helper to build an insertText + style request pair."""
        nonlocal end_idx
        req = [{"insertText": {"location": {"index": end_idx}, "text": text}}]
        if bold or heading:
            style = {}
            if bold:
                style["bold"] = True
            pstyle = {}
            if heading:
                pstyle["namedStyleType"] = heading
            req.append({
                "updateTextStyle": {
                    "range": {"startIndex": end_idx, "endIndex": end_idx + len(text)},
                    "textStyle": style,
                    "fields": "bold",
                }
            })
        end_idx += len(text)
        return req

    # Title block
    for r in txt(f"\nLeafDustLab — Export  {datetime.now().strftime('%Y-%m-%d %H:%M')}\n", bold=True):
        requests.append(r)

    for rec in records:
        for r in txt(f"\n{'─'*60}\n"):
            requests.append(r)
        lines = [
            f"Sample No:    {rec.get('sample_no', '—')}\n",
            f"Leaf Name:    {rec.get('leaf_name', '—')}\n",
            f"Species:      {rec.get('species', '—')}\n",
            f"Leaf Area:    {rec.get('leaf_area_cm2', '—')} cm²\n",
            f"W_pre:        {rec.get('w_pre', '—')} g\n",
            f"W_post:       {rec.get('w_post', '—')} g\n",
            f"Dust Density: {rec.get('dust_density', '—')} mg/cm²\n",
            f"Timestamp:    {rec.get('timestamp', '—')}\n",
            f"Notes:        {rec.get('notes', '')}\n",
        ]
        for line in lines:
            for r in txt(line):
                requests.append(r)

        # Upload annotated image to Drive then embed
        if rec.get("annotated_bytes"):
            media = MediaIoBaseUpload(
                io.BytesIO(rec["annotated_bytes"]),
                mimetype="image/jpeg",
            )
            f = drive.files().create(
                body={"name": f"{rec.get('sample_no','leaf')}_annotated.jpg"},
                media_body=media,
                fields="id",
            ).execute()
            fid = f["id"]
            drive.permissions().create(
                fileId=fid,
                body={"role": "reader", "type": "anyone"},
            ).execute()

            requests.append({
                "insertInlineImage": {
                    "location": {"index": end_idx},
                    "uri": f"https://drive.google.com/uc?id={fid}",
                    "objectSize": {
                        "height": {"magnitude": 200, "unit": "PT"},
                        "width":  {"magnitude": 300, "unit": "PT"},
                    },
                }
            })
            end_idx += 1
            for r in txt("\n"):
                requests.append(r)

    docs.documents().batchUpdate(
        documentId=doc_id,
        body={"requests": requests},
    ).execute()
    return f"https://docs.google.com/document/d/{doc_id}/edit"

# ─── Sidebar ─────────────────────────────────────────────────────────────────
PRESETS = {
    "Standard Green (default)":           ([25, 40, 40], [95, 255, 255]),
    "Grey-green (Balfour Aralia)":         ([20, 20, 30], [110, 255, 255]),
    "Variegated / Yellow-green (Pothos)":  ([15, 40, 40], [95, 255, 255]),
    "Custom":                               ([25, 40, 40], [95, 255, 255]),
}

with st.sidebar:
    st.markdown('<div class="section-header">⚙️ CV Parameters</div>', unsafe_allow_html=True)
    card_size = st.number_input("Calibration card size (cm)", value=5.0, min_value=1.0, max_value=50.0, step=0.5)
    card_thresh = st.slider(
        "🟦 Card darkness threshold",
        min_value=10, max_value=100, value=40,
        help="Lower = only very dark pixels count as the calibration card. Raise if the card isn't being found.",
    )

    st.markdown('<div class="section-header">🎨 Leaf HSV Thresholds</div>', unsafe_allow_html=True)

    prev_preset = st.session_state.get("_prev_preset", None)
    preset = st.selectbox("Species preset", list(PRESETS.keys()))
    if preset != prev_preset:
        st.session_state["_prev_preset"] = preset
        lo, hi = PRESETS[preset]
        st.session_state["hsv_h"] = (lo[0], hi[0])
        st.session_state["hsv_s"] = (lo[1], hi[1])
        st.session_state["hsv_v"] = (lo[2], hi[2])

    if "hsv_h" not in st.session_state:
        lo, hi = PRESETS[preset]
        st.session_state["hsv_h"] = (lo[0], hi[0])
        st.session_state["hsv_s"] = (lo[1], hi[1])
        st.session_state["hsv_v"] = (lo[2], hi[2])

    h_range = st.slider("🟢 Hue (H) range", 0, 179,
                         st.session_state["hsv_h"], key="hsv_h",
                         help="Hue selects the colour family. 25-95 covers most greens.")
    s_range = st.slider("🟡 Saturation (S) range", 0, 255,
                         st.session_state["hsv_s"], key="hsv_s",
                         help="Raise lower bound to exclude pale/white background areas.")
    v_range = st.slider("🔵 Value (V) range", 0, 255,
                         st.session_state["hsv_v"], key="hsv_v",
                         help="Raise lower bound to exclude dark shadows.")

    hsv_lower = [h_range[0], s_range[0], v_range[0]]
    hsv_upper = [h_range[1], s_range[1], v_range[1]]

    st.markdown(
        f'<span class="hsv-badge">Low: {hsv_lower}</span>'
        f'<span class="hsv-badge">High: {hsv_upper}</span>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">📄 Google Docs Export</div>', unsafe_allow_html=True)
    gdoc_id = st.text_input("Google Doc ID", placeholder="Paste doc ID from URL")
    sa_json = st.text_area("Service Account JSON", placeholder='{"type":"service_account",...}', height=100)
    st.caption("Share your Google Doc with the service account email (Editor role).")

# ─── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <span style="font-size:2.6rem">🌿</span>
  <div>
    <div class="hero-title">LeafDustLab · Area & Dust Density</div>
    <div class="hero-sub">Upload a baseline leaf photo · measure area automatically · log dust density · export to Google Docs</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Main Input Panel ────────────────────────────────────────────────────────
col_form, col_preview = st.columns([1, 1.4], gap="large")

with col_form:
    st.markdown('<div class="section-header">📋 Sample Details</div>', unsafe_allow_html=True)
    sample_no  = st.text_input("Sample No.", placeholder="e.g. 001")
    leaf_name  = st.text_input("Leaf / Specimen Name", placeholder="e.g. Leaf-A")
    species    = st.text_input("Species (scientific / common)", placeholder="e.g. Ficus benghalensis")
    notes      = st.text_area("Notes", placeholder="Collection site, height, condition…", height=80)

    st.markdown('<div class="section-header">⚖️ Weighing Data</div>', unsafe_allow_html=True)
    w_col1, w_col2 = st.columns(2)
    with w_col1:
        w_pre  = st.number_input("W_pre (g)", value=0.0, format="%.4f", help="Clean leaf weight")
    with w_col2:
        w_post = st.number_input("W_post (g)", value=0.0, format="%.4f", help="Dusty leaf weight")

    st.markdown('<div class="section-header">📷 Baseline Photo</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload leaf photo (JPG/PNG)", type=["jpg", "jpeg", "png"])

    run_btn = st.button("🔬 Measure Leaf Area", use_container_width=True)

# ─── Processing & Results ────────────────────────────────────────────────────
result = None
with col_preview:
    if uploaded:
        img_bytes = uploaded.read()
        st.markdown('<div class="section-header">👁️ Live Threshold Preview</div>', unsafe_allow_html=True)
        st.caption("Cyan = calibration card  ·  Lime = leaf HSV mask  ·  Updates as you drag the sliders")
        preview_bytes, card_ok, leaf_ok = live_preview(img_bytes, hsv_lower, hsv_upper, card_thresh)
        st.image(preview_bytes, use_container_width=True)
        status_cols = st.columns(2)
        with status_cols[0]:
            if card_ok:
                st.success("✅ Calibration card detected")
            else:
                st.error("❌ Card not found — adjust threshold")
        with status_cols[1]:
            if leaf_ok:
                st.success("✅ Leaf region detected")
            else:
                st.warning("⚠️ No leaf — adjust HSV sliders")

    if run_btn and uploaded:
        with st.spinner("Running CV pipeline…"):
            result = process_image(img_bytes, card_size, hsv_lower, hsv_upper, card_thresh)

        if "error" in result:
            st.markdown(f'<div class="warning-box">⚠️ {result["error"]}</div>', unsafe_allow_html=True)
        else:
            # Dust density
            dust_g = w_post - w_pre
            dust_density = round((dust_g * 1000) / result["leaf_area_cm2"], 4) if result["leaf_area_cm2"] > 0 else 0.0

            st.markdown('<div class="section-header">✅ Results</div>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{result["leaf_area_cm2"]}</div><div class="metric-label">cm² leaf area</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{result["px_per_cm"]}</div><div class="metric-label">px / cm</div></div>', unsafe_allow_html=True)
            with m3:
                dd_disp = f"{dust_density:.4f}" if (w_post > 0 and w_pre > 0) else "—"
                st.markdown(f'<div class="metric-card"><div class="metric-value">{dd_disp}</div><div class="metric-label">mg/cm² dust</div></div>', unsafe_allow_html=True)

            st.markdown('<div class="section-header">🖼️ Annotated Output</div>', unsafe_allow_html=True)
            st.caption("🟡 Yellow box = calibration card  ·  🟢 Green contour = measured leaf area")
            st.image(result["annotated_bytes"], use_container_width=True)

            tab_orig, tab_mask, tab_details = st.tabs(["Original photo", "Leaf mask", "Full details"])
            with tab_orig:
                st.image(img_bytes, caption="Stage A baseline photo (unmodified)", use_container_width=True)
            with tab_mask:
                st.image(result["mask_bytes"], caption="Binary leaf HSV mask (white = selected pixels)", use_container_width=True)
            with tab_details:
                st.json({
                    "sample_no": sample_no, "leaf_name": leaf_name, "species": species,
                    "leaf_area_cm2": result["leaf_area_cm2"],
                    "px_per_cm": result["px_per_cm"], "leaf_px": result["leaf_px"],
                    "w_pre_g": w_pre, "w_post_g": w_post,
                    "dust_density_mg_cm2": dust_density if w_post > 0 else None,
                })

            # Save to log
            log_entry = {
                "sample_no": sample_no, "leaf_name": leaf_name, "species": species,
                "leaf_area_cm2": result["leaf_area_cm2"],
                "w_pre": w_pre, "w_post": w_post,
                "dust_density": dust_density if w_post > 0 else None,
                "notes": notes,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "annotated_bytes": result["annotated_bytes"],
            }
            # Avoid duplicate sample_no entries
            st.session_state.log = [r for r in st.session_state.log if r["sample_no"] != sample_no]
            st.session_state.log.append(log_entry)
            st.markdown('<div class="success-box">✅ Logged! Scroll down to view the session log.</div>', unsafe_allow_html=True)

    elif run_btn and not uploaded:
        st.markdown('<div class="warning-box">⚠️ Please upload a leaf photo first.</div>', unsafe_allow_html=True)

# ─── Session Log ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">📊 Session Log</div>', unsafe_allow_html=True)

if not st.session_state.log:
    st.info("No measurements yet. Process a leaf photo to start logging.")
else:
    rows_html = ""
    for r in st.session_state.log:
        dd = f"{r['dust_density']:.4f}" if r["dust_density"] is not None else "—"
        rows_html += f"""
        <tr>
          <td>{r['sample_no']}</td>
          <td>{r['leaf_name']}</td>
          <td>{r['species']}</td>
          <td>{r['leaf_area_cm2']} cm²</td>
          <td>{r['w_pre']} g</td>
          <td>{r['w_post']} g</td>
          <td>{dd} mg/cm²</td>
          <td>{r['timestamp']}</td>
        </tr>"""

    st.markdown(f"""
    <table class="log-table">
      <thead><tr>
        <th>Sample</th><th>Name</th><th>Species</th>
        <th>Area</th><th>W pre</th><th>W post</th><th>Dust density</th><th>Timestamp</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("")
    gcol1, gcol2 = st.columns([1, 2])
    with gcol1:
        if st.button("🗑️ Clear Log", use_container_width=True):
            st.session_state.log = []
            st.rerun()
    with gcol2:
        if st.button("📤 Export All to Google Docs", use_container_width=True):
            if not gdoc_id or not sa_json:
                st.error("Fill in the Google Doc ID and Service Account JSON in the sidebar first.")
            else:
                with st.spinner("Uploading to Google Docs…"):
                    try:
                        url = export_to_gdoc(st.session_state.log, sa_json, gdoc_id)
                        st.markdown(f'<div class="success-box">✅ Exported! <a href="{url}" target="_blank">Open Google Doc →</a></div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Export failed: {e}")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p style="text-align:center;color:#444;font-size:0.8rem;">LeafDustLab · ISEF 2025 · AJ · Spectral Leaf Dust Analysis</p>', unsafe_allow_html=True)
