import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import io
import json
from datetime import datetime

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
        border: 1px solid #2d5a2d; border-radius: 16px;
        padding: 28px 36px; margin-bottom: 28px;
        display: flex; align-items: center; gap: 18px;
    }
    .hero-title { font-size: 2rem; font-weight: 700; color: #7fff7f; margin: 0; }
    .hero-sub   { font-size: 0.95rem; color: #88aa88; margin: 4px 0 0 0; }
    .metric-card {
        background: #161b22; border: 1px solid #2d3748;
        border-radius: 12px; padding: 20px 24px; text-align: center;
    }
    .metric-card:hover { border-color: #4a9a4a; }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #7fff7f; }
    .metric-label { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
    .log-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    .log-table th {
        background: #1a2f1a; color: #7fff7f; padding: 10px 14px;
        text-align: left; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.5px; font-size: 0.75rem;
    }
    .log-table td { padding: 10px 14px; color: #ccc; border-bottom: 1px solid #1e2a1e; }
    .log-table tr:hover td { background: #161b22; }
    .stButton > button {
        background: linear-gradient(135deg, #2d7a2d, #1a5a1a);
        color: white; border: none; border-radius: 10px;
        padding: 10px 28px; font-weight: 600; font-size: 0.95rem;
        cursor: pointer; transition: all 0.2s; width: 100%;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #3d9a3d, #2a7a2a); }
    .warning-box {
        background: #2a1a00; border: 1px solid #7a4a00;
        border-radius: 10px; padding: 14px 18px; color: #ffaa44; font-size: 0.88rem; margin: 10px 0;
    }
    .success-box {
        background: #0a2a0a; border: 1px solid #2a6a2a;
        border-radius: 10px; padding: 14px 18px; color: #66dd66; font-size: 0.88rem; margin: 10px 0;
    }
    .section-header {
        color: #7fff7f; font-size: 1.1rem; font-weight: 600;
        margin: 20px 0 12px 0; padding-bottom: 6px; border-bottom: 1px solid #2d5a2d;
    }
    div[data-testid="stNumberInput"] label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stSelectbox"] label { color: #aaccaa !important; font-size: 0.88rem !important; }
    div[data-testid="stSlider"] label { color: #aaccaa !important; }
    .hsv-badge {
        display: inline-block; background: #1a3a1a; border: 1px solid #3a7a3a;
        border-radius: 6px; padding: 4px 10px; font-family: monospace;
        font-size: 0.85rem; color: #88ee88; margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────
if "log" not in st.session_state:
    st.session_state.log = []
if "gdoc_url" not in st.session_state:
    st.session_state.gdoc_url = ""

# ═══════════════════════════════════════════════════════════════════════════════
#  Pure numpy + Pillow + scipy CV helpers  ── zero dependency on opencv
# ═══════════════════════════════════════════════════════════════════════════════

def _load_rgb(img_bytes: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB (uint8) → HSV with OpenCV convention: H 0-179, S 0-255, V 0-255."""
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    m_r = (cmax == r) & (delta > 0)
    m_g = (cmax == g) & (delta > 0)
    m_b = (cmax == b) & (delta > 0)
    h[m_r] = (60.0 * ((g[m_r] - b[m_r]) / delta[m_r])) % 360.0
    h[m_g] = 60.0 * ((b[m_g] - r[m_g]) / delta[m_g]) + 120.0
    h[m_b] = 60.0 * ((r[m_b] - g[m_b]) / delta[m_b]) + 240.0
    h = (h / 2.0).astype(np.uint8)

    s = np.where(cmax > 0, delta / cmax, 0.0)
    s = (s * 255).astype(np.uint8)
    v = (cmax * 255).astype(np.uint8)
    return np.stack([h, s, v], axis=-1)

def _hsv_mask(hsv: np.ndarray, lo: list, hi: list) -> np.ndarray:
    return ((hsv[..., 0] >= lo[0]) & (hsv[..., 0] <= hi[0]) &
            (hsv[..., 1] >= lo[1]) & (hsv[..., 1] <= hi[1]) &
            (hsv[..., 2] >= lo[2]) & (hsv[..., 2] <= hi[2]))

def _morph(mask: np.ndarray, radius: int = 7) -> np.ndarray:
    """Morphological open + close using scipy if available, else raw mask."""
    try:
        from scipy.ndimage import binary_closing, binary_opening
        struct = np.ones((radius, radius), dtype=bool)
        mask = binary_closing(mask, structure=struct)
        mask = binary_opening(mask, structure=struct)
    except ImportError:
        pass
    return mask.astype(bool)

def _largest_dark_rect(gray: np.ndarray, thresh: int):
    """
    Find the calibration card among all dark blobs.

    Strategy: label every dark blob, then score each one on three criteria
    and pick the best scorer — not just the biggest blob.

    Scoring per blob:
      squareness  — aspect ratio penalty: 1.0 when w==h, drops toward 0 as it gets elongated
      solidity    — filled_pixels / bounding_box_pixels: a solid square card scores ~1.0;
                    a shadow/border/thin strip scores much lower
      size_fit    — prefer blobs that are 1%–25% of the image area (card is never the
                    whole frame, never a tiny speck)

    Combined score = squareness * solidity * size_fit
    The blob with the highest combined score is taken as the card.
    """
    from scipy.ndimage import label as sp_label

    dark   = gray < thresh
    img_px = gray.size        # total pixel count for relative-size check

    try:
        labeled, n = sp_label(dark)
    except Exception:
        # scipy unavailable — crude whole-mask bbox fallback
        rows = np.where(np.any(dark, axis=1))[0]
        cols = np.where(np.any(dark, axis=0))[0]
        if len(rows) == 0:
            return None, None
        y0, y1, x0, x1 = rows[0], rows[-1], cols[0], cols[-1]
        w, h = int(x1-x0), int(y1-y0)
        return (w+h)/2.0, (int(x0), int(y0), w, h)

    if n == 0:
        return None, None

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0   # ignore background label

    best_score = -1.0
    best_rect  = None

    for lbl in range(1, n + 1):
        blob_area = int(sizes[lbl])
        if blob_area < 30:          # skip tiny noise specks
            continue

        region = labeled == lbl
        rows   = np.where(np.any(region, axis=1))[0]
        cols   = np.where(np.any(region, axis=0))[0]
        if len(rows) < 2 or len(cols) < 2:
            continue

        y0, y1 = rows[0], rows[-1]
        x0, x1 = cols[0], cols[-1]
        bw = int(x1 - x0) or 1
        bh = int(y1 - y0) or 1
        bbox_area = bw * bh

        # ── squareness: 1.0 = perfect square, 0 = very elongated ────────
        aspect     = min(bw, bh) / max(bw, bh)       # 0..1
        squareness = aspect ** 2                       # penalise harder

        # ── solidity: filled pixels / bounding box pixels ───────────────
        solidity = blob_area / bbox_area               # 0..1

        # ── size_fit: card should be 0.5%–30% of frame ──────────────────
        rel = blob_area / img_px
        if rel < 0.005 or rel > 0.30:
            size_fit = 0.0                             # disqualify extremes
        else:
            # Peak around 5% of frame; gentle penalty away from peak
            size_fit = 1.0 - abs(rel - 0.05) / 0.25
            size_fit = max(0.0, size_fit)

        score = squareness * solidity * size_fit

        if score > best_score:
            best_score = score
            best_rect  = (int(x0), int(y0), bw, bh)

    if best_rect is None or best_score < 0.05:
        return None, None

    x, y, w, h = best_rect
    return (w + h) / 2.0, best_rect


def _largest_blob(mask: np.ndarray):
    """Area, bounding rect, centroid of largest True blob. Returns (area, rect, centroid) or Nones."""
    try:
        from scipy.ndimage import label as sp_label
        labeled, n = sp_label(mask)
        if n == 0:
            return None, None, None
        sizes = np.bincount(labeled.ravel()); sizes[0] = 0
        lbl = sizes.argmax()
        region = labeled == lbl
        rows = np.where(np.any(region, axis=1))[0]
        cols = np.where(np.any(region, axis=0))[0]
        y0, y1, x0, x1 = rows[0], rows[-1], cols[0], cols[-1]
        area = int(sizes[lbl])
        ys, xs = np.where(region)
        return area, (int(x0), int(y0), int(x1-x0), int(y1-y0)), (int(xs.mean()), int(ys.mean()))
    except ImportError:
        rows = np.where(np.any(mask, axis=1))[0]
        cols = np.where(np.any(mask, axis=0))[0]
        if len(rows) == 0:
            return None, None, None
        y0, y1, x0, x1 = rows[0], rows[-1], cols[0], cols[-1]
        return int(mask.sum()), (int(x0), int(y0), int(x1-x0), int(y1-y0)), \
               (int((x0+x1)//2), int((y0+y1)//2))

def _blend(base: np.ndarray, overlay_mask: np.ndarray,
           color: tuple, alpha: float = 0.45) -> np.ndarray:
    out = base.copy().astype(np.float32)
    col = np.array(color, dtype=np.float32)
    out[overlay_mask] = (1 - alpha) * out[overlay_mask] + alpha * col
    return out.clip(0, 255).astype(np.uint8)

def _draw_rect_pil(pil_img: Image.Image, rect, color, width: int = 4) -> Image.Image:
    x, y, w, h = rect
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle([x, y, x+w, y+h], outline=color, width=width)
    return pil_img

def _draw_mask_border(rgb: np.ndarray, mask: np.ndarray, color: tuple, thick: int = 3) -> np.ndarray:
    """Draw the outline of a boolean mask in the given RGB color."""
    pad = np.pad(mask, 1, constant_values=False)
    edge = mask & (
        ~pad[:-2, 1:-1] | ~pad[2:, 1:-1] |
        ~pad[1:-1, :-2] | ~pad[1:-1, 2:]
    )
    try:
        from scipy.ndimage import binary_dilation
        edge = binary_dilation(edge, iterations=thick)
    except ImportError:
        pass
    out = rgb.copy()
    out[edge] = color
    return out

def _label(pil_img: Image.Image, text: str, xy: tuple, color) -> Image.Image:
    draw = ImageDraw.Draw(pil_img)
    # Shadow for readability
    draw.text((xy[0]+1, xy[1]+1), text, fill=(0, 0, 0))
    draw.text(xy, text, fill=color)
    return pil_img

def _to_jpeg(rgb: np.ndarray, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

# ═══════════════════════════════════════════════════════════════════════════════
#  Core pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def find_card_scale(rgb: np.ndarray, thresh: int):
    gray = (0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]).astype(np.uint8)
    return _largest_dark_rect(gray, thresh)


# ─── Leaf mask helpers ────────────────────────────────────────────────────────

def _gradient_edges(gray: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    Sobel-like gradient magnitude using scipy.  Returns a 0-255 uint8 edge map.
    Falls back to simple numpy diff if scipy unavailable.
    """
    try:
        from scipy.ndimage import gaussian_filter, sobel
        blurred = gaussian_filter(gray.astype(np.float32), sigma=sigma)
        gx = sobel(blurred, axis=1)
        gy = sobel(blurred, axis=0)
        mag = np.hypot(gx, gy)
        mag = (mag / mag.max() * 255).astype(np.uint8) if mag.max() > 0 else mag.astype(np.uint8)
        return mag
    except ImportError:
        gx = np.abs(np.diff(gray.astype(np.int16), axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray.astype(np.int16), axis=0, prepend=gray[:1, :]))
        mag = np.hypot(gx, gy).clip(0, 255).astype(np.uint8)
        return mag

def _flood_fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill internal holes in a boolean mask by flood-filling from every border pixel
    that is False, then inverting: anything NOT reachable from the border = inside the leaf.
    Works entirely with scipy label / numpy — no opencv needed.
    """
    try:
        from scipy.ndimage import label as sp_label, binary_fill_holes
        # binary_fill_holes directly fills holes inside each connected component
        return binary_fill_holes(mask).astype(bool)
    except ImportError:
        return mask.astype(bool)

def _contour_mask(gray: np.ndarray, seed_mask: np.ndarray,
                  edge_thresh: int = 30, dilate_iters: int = 4) -> np.ndarray:
    """
    Grow a contour-based mask starting from `seed_mask` (the HSV hits).

    Steps:
      1. Compute gradient edge map of the grayscale image.
      2. Build a 'barrier' from strong edges — these are the leaf boundaries.
      3. Dilate the seed mask outward, stopping at barriers.
      4. Flood-fill holes so white spots / venation gaps inside are included.
      5. Return the union of the expanded mask + original seed.
    """
    try:
        from scipy.ndimage import binary_dilation, binary_fill_holes, label as sp_label
    except ImportError:
        return _flood_fill_holes(seed_mask)

    # ── Step 1-2: edge barrier ───────────────────────────────────────────
    edges = _gradient_edges(gray)
    barrier = edges > edge_thresh          # strong edges block expansion

    # ── Step 3: iteratively dilate seed, blocked by barrier ─────────────
    struct = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)  # 4-connected
    grown = seed_mask.copy().astype(bool)
    for _ in range(dilate_iters):
        expanded = binary_dilation(grown, structure=struct)
        # Allow expansion only into non-barrier pixels
        grown = expanded & ~barrier

    # ── Step 4: fill internal holes (white spots, veins, pale patches) ──
    filled = binary_fill_holes(grown).astype(bool)

    # ── Step 5: union with original seed ────────────────────────────────
    combined = filled | seed_mask.astype(bool)

    # Keep only the largest blob to avoid merging distant noise
    labeled, n = sp_label(combined)
    if n == 0:
        return seed_mask.astype(bool)
    sizes = np.bincount(labeled.ravel()); sizes[0] = 0
    combined = (labeled == sizes.argmax()).astype(bool)

    return combined


def find_leaf_mask(rgb: np.ndarray, hsv_lo: list, hsv_hi: list,
                   use_contour: bool = True,
                   edge_thresh: int = 30,
                   contour_expand: int = 4) -> np.ndarray:
    """
    Hybrid leaf segmentation:
      1. HSV colour mask  → reliable for clearly green areas
      2. Contour / flood-fill expansion → fills white spots, pale edges, venation

    `use_contour=False` gives pure HSV (fast, for species with clean colour).
    `use_contour=True`  gives HSV + contour fill (for leaves with white spots / pale edges).
    """
    hsv = _rgb_to_hsv(rgb)
    raw = _hsv_mask(hsv, hsv_lo, hsv_hi)
    hsv_clean = _morph(raw, radius=7)

    if not use_contour or not hsv_clean.any():
        return hsv_clean

    gray = (0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]).astype(np.uint8)
    return _contour_mask(gray, hsv_clean,
                         edge_thresh=edge_thresh,
                         dilate_iters=contour_expand)



def live_preview(img_bytes: bytes, hsv_lo: list, hsv_hi: list, card_thresh: int,
                 use_contour: bool = True, edge_thresh: int = 30, contour_expand: int = 4):
    """Returns (jpeg_bytes, card_found, leaf_found)."""
    rgb = _load_rgb(img_bytes)

    # Card: cyan fill
    gray = (0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]).astype(np.uint8)
    _, card_rect = _largest_dark_rect(gray, card_thresh)
    card_found = card_rect is not None

    canvas = rgb.copy()
    if card_found:
        cx, cy, cw, ch = card_rect
        card_region = np.zeros(rgb.shape[:2], dtype=bool)
        card_region[cy:cy+ch, cx:cx+cw] = True
        canvas = _blend(canvas, card_region, (0, 210, 255), alpha=0.45)

    # Leaf: lime fill (hybrid HSV + contour)
    leaf_mask = find_leaf_mask(rgb, hsv_lo, hsv_hi,
                               use_contour=use_contour,
                               edge_thresh=edge_thresh,
                               contour_expand=contour_expand)
    leaf_found = bool(leaf_mask.any())
    if leaf_found:
        canvas = _blend(canvas, leaf_mask, (50, 255, 80), alpha=0.45)

    # Outlines + labels
    pil = Image.fromarray(canvas)
    if card_found:
        pil = _draw_rect_pil(pil, card_rect, color=(0, 210, 255), width=3)
        pil = _label(pil, "CAL CARD", (card_rect[0]+6, max(card_rect[1]-22, 2)), (0, 210, 255))

    if leaf_found:
        canvas2 = _draw_mask_border(np.array(pil), leaf_mask, (50, 255, 80), thick=3)
        pil = Image.fromarray(canvas2)
        _, _, centroid = _largest_blob(leaf_mask)
        if centroid:
            pil = _label(pil, "LEAF", (centroid[0]-20, centroid[1]), (50, 255, 80))

    return _to_jpeg(np.array(pil), 90), card_found, leaf_found

def process_image(img_bytes: bytes, card_size_cm: float,
                  hsv_lo: list, hsv_hi: list, card_thresh: int,
                  use_contour: bool = True, edge_thresh: int = 30,
                  contour_expand: int = 4) -> dict:
    rgb  = _load_rgb(img_bytes)
    gray = (0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]).astype(np.uint8)

    avg_side, card_rect = _largest_dark_rect(gray, card_thresh)
    if avg_side is None:
        return {"error": "Could not detect calibration card. Try lowering the card threshold."}

    px_per_cm  = avg_side / card_size_cm
    px_per_cm2 = px_per_cm ** 2

    leaf_mask = find_leaf_mask(rgb, hsv_lo, hsv_hi,
                               use_contour=use_contour,
                               edge_thresh=edge_thresh,
                               contour_expand=contour_expand)
    if not leaf_mask.any():
        return {"error": "Could not detect leaf. Adjust the HSV sliders for this species."}

    leaf_px = int(leaf_mask.sum())
    leaf_area_cm2 = round(leaf_px / px_per_cm2, 2)

    # Annotate
    pil = Image.fromarray(rgb)
    if card_rect:
        pil = _draw_rect_pil(pil, card_rect, (255, 220, 0), width=4)
        pil = _label(pil, f"CAL CARD  {px_per_cm:.1f}px/cm",
                     (card_rect[0]+4, max(card_rect[1]-22, 2)), (255, 220, 0))
    ann = _draw_mask_border(np.array(pil), leaf_mask, (50, 255, 80), thick=4)
    pil2 = Image.fromarray(ann)
    _, _, centroid = _largest_blob(leaf_mask)
    if centroid:
        pil2 = _label(pil2, f"{leaf_area_cm2:.2f} cm2",
                      (centroid[0]-40, centroid[1]), (50, 255, 80))

    mask_rgb = np.where(leaf_mask[..., None], np.uint8(255), np.uint8(0)) * np.ones(3, dtype=np.uint8)

    return {
        "leaf_area_cm2": leaf_area_cm2,
        "px_per_cm":     round(px_per_cm, 2),
        "leaf_px":       leaf_px,
        "annotated_bytes": _to_jpeg(np.array(pil2), 92),
        "mask_bytes":      _to_jpeg(mask_rgb, 90),
    }

# ─── Google Docs Export ───────────────────────────────────────────────────────
def export_to_gdoc(records, service_account_json: str, doc_id: str):
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload

    creds = service_account.Credentials.from_service_account_info(
        json.loads(service_account_json),
        scopes=["https://www.googleapis.com/auth/documents",
                "https://www.googleapis.com/auth/drive"],
    )
    docs  = build("docs",  "v1", credentials=creds)
    drive = build("drive", "v3", credentials=creds)

    doc     = docs.documents().get(documentId=doc_id).execute()
    end_idx = doc["body"]["content"][-1]["endIndex"] - 1
    reqs    = []

    def _txt(text, bold=False):
        nonlocal end_idx
        r = [{"insertText": {"location": {"index": end_idx}, "text": text}}]
        if bold:
            r.append({"updateTextStyle": {
                "range": {"startIndex": end_idx, "endIndex": end_idx+len(text)},
                "textStyle": {"bold": True}, "fields": "bold",
            }})
        end_idx += len(text)
        return r

    for r in _txt(f"\nLeafDustLab — Export  {datetime.now().strftime('%Y-%m-%d %H:%M')}\n", bold=True):
        reqs.append(r)
    for rec in records:
        for r in _txt(f"\n{'─'*60}\n"): reqs.append(r)
        for line in [
            f"Sample No:    {rec.get('sample_no','—')}\n",
            f"Leaf Name:    {rec.get('leaf_name','—')}\n",
            f"Species:      {rec.get('species','—')}\n",
            f"Leaf Area:    {rec.get('leaf_area_cm2','—')} cm²\n",
            f"W_pre:        {rec.get('w_pre','—')} g\n",
            f"W_post:       {rec.get('w_post','—')} g\n",
            f"Dust Density: {rec.get('dust_density','—')} mg/cm²\n",
            f"Timestamp:    {rec.get('timestamp','—')}\n",
            f"Notes:        {rec.get('notes','')}\n",
        ]:
            for r in _txt(line): reqs.append(r)

        if rec.get("annotated_bytes"):
            fid = drive.files().create(
                body={"name": f"{rec.get('sample_no','leaf')}_annotated.jpg"},
                media_body=MediaIoBaseUpload(io.BytesIO(rec["annotated_bytes"]), mimetype="image/jpeg"),
                fields="id",
            ).execute()["id"]
            drive.permissions().create(fileId=fid, body={"role":"reader","type":"anyone"}).execute()
            reqs.append({"insertInlineImage": {
                "location": {"index": end_idx},
                "uri": f"https://drive.google.com/uc?id={fid}",
                "objectSize": {"height": {"magnitude": 200, "unit": "PT"},
                               "width":  {"magnitude": 300, "unit": "PT"}},
            }})
            end_idx += 1
            for r in _txt("\n"): reqs.append(r)

    docs.documents().batchUpdate(documentId=doc_id, body={"requests": reqs}).execute()
    return f"https://docs.google.com/document/d/{doc_id}/edit"

# ═══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
PRESETS = {
    "Standard Green (default)":           ([25, 40, 40],  [95, 255, 255]),
    "Grey-green (Balfour Aralia)":         ([20, 20, 30],  [110, 255, 255]),
    "Variegated / Yellow-green (Pothos)":  ([15, 40, 40],  [95, 255, 255]),
    "Custom":                              ([25, 40, 40],  [95, 255, 255]),
}

with st.sidebar:
    st.markdown('<div class="section-header">⚙️ CV Parameters</div>', unsafe_allow_html=True)
    card_size   = st.number_input("Calibration card size (cm)", value=2.0, min_value=0.5, max_value=50.0, step=0.5)
    card_thresh = st.slider("🟦 Card darkness threshold", 10, 120, 40,
                            help="Lower = only very dark pixels count as the card.")
    st.caption("💡 **Card tip:** The detector picks the darkest blob that is most square "
               "and solid. Avoid other dark square objects in frame. A white background "
               "works best — the card will stand out clearly.")

    st.markdown('<div class="section-header">🎨 Leaf HSV Thresholds</div>', unsafe_allow_html=True)
    prev_preset = st.session_state.get("_prev_preset")
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

    h_range = st.slider("🟢 Hue (H) range",       0, 179, st.session_state["hsv_h"], key="hsv_h",
                        help="Colour family. 25–95 covers most greens.")
    s_range = st.slider("🟡 Saturation (S) range", 0, 255, st.session_state["hsv_s"], key="hsv_s",
                        help="Raise lower bound to exclude pale/white background.")
    v_range = st.slider("🔵 Value (V) range",       0, 255, st.session_state["hsv_v"], key="hsv_v",
                        help="Raise lower bound to exclude deep shadows.")

    hsv_lo = [h_range[0], s_range[0], v_range[0]]
    hsv_hi = [h_range[1], s_range[1], v_range[1]]
    st.markdown(
        f'<span class="hsv-badge">Low: {hsv_lo}</span>'
        f'<span class="hsv-badge">High: {hsv_hi}</span>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">🔲 Contour / Hole-Fill</div>', unsafe_allow_html=True)
    use_contour = st.toggle("Enable contour expansion", value=True,
                            help="Expands the HSV mask outward along leaf edges and fills "
                                 "holes (white spots, veins, pale patches). "
                                 "Turn off for clean solid-colour leaves.")
    if use_contour:
        edge_thresh = st.slider("Edge sensitivity", 10, 80, 30,
                                help="Higher = only very sharp edges act as barriers. "
                                     "Lower = gentler edges also block expansion.")
        contour_expand = st.slider("Contour expansion (px)", 1, 20, 4,
                                   help="How many pixels outward the contour grows "
                                        "from the HSV seed before being stopped by edges.")
    else:
        edge_thresh    = 30
        contour_expand = 4
    st.caption("💡 **For white-spotted leaves** (e.g. Aralia): keep contour ON, "
               "raise expansion to 8–12 px.")
    gdoc_id = st.text_input("Google Doc ID", placeholder="Paste doc ID from URL")
    sa_json = st.text_area("Service Account JSON", placeholder='{"type":"service_account",...}', height=100)
    st.caption("Share your Google Doc with the service account email (Editor role).")

# ═══════════════════════════════════════════════════════════════════════════════
#  Hero
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-header">
  <span style="font-size:2.6rem">🌿</span>
  <div>
    <div class="hero-title">LeafDustLab · Area & Dust Density</div>
    <div class="hero-sub">Use live camera or upload a photo · measure leaf area automatically · log dust density · export to Google Docs</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  Top-level tabs
# ═══════════════════════════════════════════════════════════════════════════════
tab_tune, tab_measure, tab_log = st.tabs(["🎛️ Live Tune", "🔬 Measure", "📊 Session Log"])

# ───────────────────────────────────────────────────────────────────────────────
#  TAB 1 ─ Live Tune
#  Workflow:
#    1. Point camera at leaf
#    2. Click "Freeze frame" → bytes saved to session_state["tune_frame"]
#    3. Drag sidebar sliders → Streamlit reruns → overlay re-renders from frozen frame
#    4. When happy, click "➡ Send to Measure tab"
# ───────────────────────────────────────────────────────────────────────────────
with tab_tune:
    st.markdown(
        "<p style='color:#88aa88;margin:0 0 18px 0'>"
        "<b>How to use:</b> point the camera at your leaf + calibration card, "
        "click <b>Freeze frame ❄️</b>, then drag the sidebar sliders — "
        "the overlay updates instantly on every drag so you can dial in the "
        "perfect thresholds before measuring.</p>",
        unsafe_allow_html=True,
    )

    tune_cam_col, tune_overlay_col = st.columns([1, 1.55], gap="large")

    # ── Left: camera + freeze controls ───────────────────────────────────
    with tune_cam_col:
        st.markdown('<div class="section-header">📸 Camera</div>', unsafe_allow_html=True)

        tune_cam = st.camera_input("Live camera feed", key="tune_cam",
                                   label_visibility="collapsed")

        # Always persist the latest capture so slider drags don't reset it
        if tune_cam is not None:
            st.session_state["tune_frame"] = tune_cam.getvalue()

        has_frame = "tune_frame" in st.session_state

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            freeze_clicked = st.button("❄️ Freeze frame", use_container_width=True,
                                       disabled=(tune_cam is None),
                                       help="Lock this capture so slider drags update the overlay without resetting the camera")
        with btn_col2:
            clear_clicked = st.button("🗑️ Clear frame", use_container_width=True,
                                      disabled=not has_frame)

        if freeze_clicked and tune_cam is not None:
            st.session_state["tune_frame"] = tune_cam.getvalue()
            st.success("❄️ Frame frozen — now drag any sidebar slider!")

        if clear_clicked and has_frame:
            del st.session_state["tune_frame"]
            st.rerun()

        # Status
        if has_frame:
            st.markdown(
                '<div class="success-box" style="margin-top:10px">'
                '✅ Frame locked. Move any sidebar slider → overlay updates in real time.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="warning-box" style="margin-top:10px">'
                '📷 Snap a photo above to begin tuning.</div>',
                unsafe_allow_html=True)

        # Current parameter badge strip
        st.markdown('<div class="section-header" style="margin-top:18px">🎚️ Active parameters</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<span class="hsv-badge">H {hsv_lo[0]}–{hsv_hi[0]}</span>'
            f'<span class="hsv-badge">S {hsv_lo[1]}–{hsv_hi[1]}</span>'
            f'<span class="hsv-badge">V {hsv_lo[2]}–{hsv_hi[2]}</span>'
            f'<span class="hsv-badge">Card thr {card_thresh}</span>'
            + (f'<span class="hsv-badge">Edge {edge_thresh}</span>'
               f'<span class="hsv-badge">Expand {contour_expand}px</span>' if use_contour else ''),
            unsafe_allow_html=True,
        )

        # Transfer to Measure tab
        st.markdown("")
        if has_frame:
            if st.button("➡️ Send to Measure tab", use_container_width=True,
                         help="Copies this frame so the Measure tab uses it directly"):
                st.session_state["measure_frame"] = st.session_state["tune_frame"]
                st.success("Sent! Switch to the 🔬 Measure tab.")

    # ── Right: live overlay + mask — re-renders on every slider drag ──────
    with tune_overlay_col:
        st.markdown('<div class="section-header">👁️ Live Overlay & Mask</div>',
                    unsafe_allow_html=True)
        st.caption("🟦 Cyan = card · 🟢 Lime fill = leaf  |  drag any sidebar slider to update")

        # Use the frozen frame if available, else fall back to the just-snapped frame
        tune_frame_bytes = st.session_state.get("tune_frame") or \
                           (tune_cam.getvalue() if tune_cam else None)

        if tune_frame_bytes:
            with st.spinner("Rendering…"):
                tune_prev_bytes, t_card_ok, t_leaf_ok = live_preview(
                    tune_frame_bytes, hsv_lo, hsv_hi, card_thresh,
                    use_contour=use_contour,
                    edge_thresh=edge_thresh,
                    contour_expand=contour_expand,
                )
                # Build binary mask image
                rgb_t  = _load_rgb(tune_frame_bytes)
                mask_t = find_leaf_mask(rgb_t, hsv_lo, hsv_hi,
                                        use_contour=use_contour,
                                        edge_thresh=edge_thresh,
                                        contour_expand=contour_expand)
                mask_vis_t = np.where(mask_t[..., None],
                                      np.array([80, 255, 80],  dtype=np.uint8),
                                      np.array([15, 15, 15], dtype=np.uint8))
                mbuf = io.BytesIO()
                Image.fromarray(mask_vis_t.astype(np.uint8)).save(mbuf, "JPEG", quality=85)

            ov_col, mk_col = st.columns(2)
            with ov_col:
                st.image(tune_prev_bytes, caption="Colour overlay", use_container_width=True)
            with mk_col:
                st.image(mbuf.getvalue(), caption="Leaf mask (green = selected)",
                         use_container_width=True)

            # Detection badges
            db1, db2 = st.columns(2)
            with db1:
                if t_card_ok: st.success("✅ Card detected")
                else:          st.error("❌ Card not found")
            with db2:
                if t_leaf_ok: st.success("✅ Leaf detected")
                else:          st.warning("⚠️ No leaf — adjust HSV")

            # Live coverage % — useful sanity check
            if t_leaf_ok:
                h_t, w_t = rgb_t.shape[:2]
                cov = mask_t.sum() / (h_t * w_t) * 100
                st.markdown(
                    f'<div class="metric-card" style="margin-top:12px">'
                    f'<div class="metric-value">{cov:.1f}%</div>'
                    f'<div class="metric-label">leaf pixel coverage of frame</div></div>',
                    unsafe_allow_html=True)
        else:
            st.info("📷 Snap a photo on the left — the overlay will appear here.")


# ───────────────────────────────────────────────────────────────────────────────
#  TAB 2 ─ Measure
# ───────────────────────────────────────────────────────────────────────────────
with tab_measure:
    col_form, col_preview = st.columns([1, 1.4], gap="large")

    with col_form:
        st.markdown('<div class="section-header">📋 Sample Details</div>', unsafe_allow_html=True)
        sample_no = st.text_input("Sample No.",                    placeholder="e.g. 001")
        leaf_name = st.text_input("Leaf / Specimen Name",          placeholder="e.g. Leaf-A")
        species   = st.text_input("Species (scientific / common)", placeholder="e.g. Ficus benghalensis")
        notes     = st.text_area("Notes",                          placeholder="Collection site, height, condition…", height=80)

        st.markdown('<div class="section-header">⚖️ Weighing Data</div>', unsafe_allow_html=True)
        wc1, wc2 = st.columns(2)
        with wc1:
            w_pre  = st.number_input("W_pre (g)",  value=0.0, format="%.4f", help="Clean leaf weight")
        with wc2:
            w_post = st.number_input("W_post (g)", value=0.0, format="%.4f", help="Dusty leaf weight")

        st.markdown('<div class="section-header">📷 Photo Source</div>', unsafe_allow_html=True)

        # Show if a frame was transferred from Live Tune
        transferred = "measure_frame" in st.session_state
        if transferred:
            st.markdown(
                '<div class="success-box">✅ Using frame transferred from 🎛️ Live Tune tab.</div>',
                unsafe_allow_html=True)
            if st.button("✖ Discard & use new photo", use_container_width=False):
                del st.session_state["measure_frame"]
                st.rerun()

        src_tab_cam, src_tab_upload = st.tabs(["📸 Camera", "🗂️ Upload File"])
        camera_bytes  = None
        uploaded_file = None

        with src_tab_cam:
            st.caption("Snap a photo, or use a frame already transferred from Live Tune.")
            meas_cam = st.camera_input("Take a photo", key="meas_cam",
                                       label_visibility="collapsed")
            if meas_cam is not None:
                camera_bytes = meas_cam.getvalue()

        with src_tab_upload:
            uploaded_file = st.file_uploader("Upload leaf photo (JPG/PNG)",
                                             type=["jpg", "jpeg", "png"],
                                             label_visibility="collapsed")

        # Priority: fresh camera snap > transferred tune frame > upload
        img_bytes = None
        if camera_bytes:
            img_bytes = camera_bytes
        elif transferred:
            img_bytes = st.session_state["measure_frame"]
        elif uploaded_file:
            img_bytes = uploaded_file.read()

        run_btn = st.button("🔬 Measure Leaf Area", use_container_width=True,
                            disabled=(img_bytes is None))

    # ── Right: preview + results ──────────────────────────────────────────
    result = None
    with col_preview:
        if img_bytes:
            st.markdown('<div class="section-header">👁️ Preview</div>', unsafe_allow_html=True)
            st.caption("🟦 Cyan = card · 🟢 Lime = leaf — updates with sidebar sliders")

            with st.spinner("Rendering preview…"):
                preview_bytes, card_ok, leaf_ok = live_preview(
                    img_bytes, hsv_lo, hsv_hi, card_thresh,
                    use_contour=use_contour, edge_thresh=edge_thresh,
                    contour_expand=contour_expand)

            pv1, pv2 = st.columns(2)
            with pv1:
                st.image(preview_bytes, caption="Overlay", use_container_width=True)
            with pv2:
                rgb_prev  = _load_rgb(img_bytes)
                leaf_mask = find_leaf_mask(rgb_prev, hsv_lo, hsv_hi,
                                           use_contour=use_contour,
                                           edge_thresh=edge_thresh,
                                           contour_expand=contour_expand)
                mask_vis  = np.where(leaf_mask[..., None],
                                     np.array([80, 255, 80], dtype=np.uint8),
                                     np.array([20, 20, 20],  dtype=np.uint8))
                mask_buf  = io.BytesIO()
                Image.fromarray(mask_vis.astype(np.uint8)).save(mask_buf, "JPEG", quality=85)
                st.image(mask_buf.getvalue(), caption="Leaf mask", use_container_width=True)

            sc1, sc2 = st.columns(2)
            with sc1:
                if card_ok: st.success("✅ Card detected")
                else:        st.error("❌ Card not found — lower threshold")
            with sc2:
                if leaf_ok: st.success("✅ Leaf detected")
                else:        st.warning("⚠️ No leaf — adjust HSV")
        else:
            st.info("👈 Snap a photo, transfer from Live Tune, or upload a file.")

        if run_btn and img_bytes:
            with st.spinner("Running CV pipeline…"):
                result = process_image(img_bytes, card_size, hsv_lo, hsv_hi, card_thresh,
                                       use_contour=use_contour, edge_thresh=edge_thresh,
                                       contour_expand=contour_expand)

            if "error" in result:
                st.markdown(f'<div class="warning-box">⚠️ {result["error"]}</div>',
                            unsafe_allow_html=True)
            else:
                dust_g       = w_post - w_pre
                dust_density = round((dust_g * 1000) / result["leaf_area_cm2"], 4) \
                               if result["leaf_area_cm2"] > 0 else 0.0

                st.markdown('<div class="section-header">✅ Results</div>', unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{result["leaf_area_cm2"]}</div>'
                                f'<div class="metric-label">cm² leaf area</div></div>',
                                unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{result["px_per_cm"]}</div>'
                                f'<div class="metric-label">px / cm</div></div>',
                                unsafe_allow_html=True)
                with m3:
                    dd_disp = f"{dust_density:.4f}" if (w_post > 0 and w_pre > 0) else "—"
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{dd_disp}</div>'
                                f'<div class="metric-label">mg/cm² dust</div></div>',
                                unsafe_allow_html=True)

                st.markdown('<div class="section-header">🖼️ Annotated Output</div>',
                            unsafe_allow_html=True)
                st.caption("🟡 Yellow box = calibration card  ·  🟢 Green outline = measured leaf")
                st.image(result["annotated_bytes"], use_container_width=True)

                tab_orig, tab_mask, tab_json = st.tabs(["Original photo", "Leaf mask", "Full details"])
                with tab_orig:
                    st.image(img_bytes, caption="Captured photo (unmodified)",
                             use_container_width=True)
                with tab_mask:
                    st.image(result["mask_bytes"],
                             caption="Binary HSV mask (white = selected pixels)",
                             use_container_width=True)
                with tab_json:
                    st.json({
                        "sample_no": sample_no, "leaf_name": leaf_name, "species": species,
                        "leaf_area_cm2": result["leaf_area_cm2"],
                        "px_per_cm": result["px_per_cm"], "leaf_px": result["leaf_px"],
                        "w_pre_g": w_pre, "w_post_g": w_post,
                        "dust_density_mg_cm2": dust_density if w_post > 0 else None,
                    })

                log_entry = {
                    "sample_no": sample_no, "leaf_name": leaf_name, "species": species,
                    "leaf_area_cm2": result["leaf_area_cm2"],
                    "w_pre": w_pre, "w_post": w_post,
                    "dust_density": dust_density if w_post > 0 else None,
                    "notes": notes,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "annotated_bytes": result["annotated_bytes"],
                }
                st.session_state.log = [r for r in st.session_state.log
                                        if r["sample_no"] != sample_no]
                st.session_state.log.append(log_entry)
                st.markdown(
                    '<div class="success-box">✅ Logged! Switch to the 📊 Session Log tab.</div>',
                    unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────────
#  TAB 3 ─ Session Log
# ───────────────────────────────────────────────────────────────────────────────
with tab_log:
    if not st.session_state.log:
        st.info("No measurements yet. Run a measurement in the 🔬 Measure tab to start logging.")
    else:
        rows_html = ""
        for r in st.session_state.log:
            dd = f"{r['dust_density']:.4f}" if r["dust_density"] is not None else "—"
            rows_html += (
                f"<tr><td>{r['sample_no']}</td><td>{r['leaf_name']}</td><td>{r['species']}</td>"
                f"<td>{r['leaf_area_cm2']} cm²</td><td>{r['w_pre']} g</td><td>{r['w_post']} g</td>"
                f"<td>{dd} mg/cm²</td><td>{r['timestamp']}</td></tr>"
            )
        st.markdown(f"""
        <table class="log-table">
          <thead><tr>
            <th>Sample</th><th>Name</th><th>Species</th>
            <th>Area</th><th>W pre</th><th>W post</th><th>Dust density</th><th>Timestamp</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)

        st.markdown("")
        gc1, gc2 = st.columns([1, 2])
        with gc1:
            if st.button("🗑️ Clear Log", use_container_width=True):
                st.session_state.log = []
                st.rerun()
        with gc2:
            if st.button("📤 Export All to Google Docs", use_container_width=True):
                if not gdoc_id or not sa_json:
                    st.error("Fill in the Google Doc ID and Service Account JSON in the sidebar first.")
                else:
                    with st.spinner("Uploading to Google Docs…"):
                        try:
                            url = export_to_gdoc(st.session_state.log, sa_json, gdoc_id)
                            st.markdown(
                                f'<div class="success-box">✅ Exported! '
                                f'<a href="{url}" target="_blank">Open Google Doc →</a></div>',
                                unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Export failed: {e}")

st.markdown("---")
st.markdown('<p style="text-align:center;color:#444;font-size:0.8rem;">'
            'LeafDustLab · ISEF 2025 · AJ · Spectral Leaf Dust Analysis</p>',
            unsafe_allow_html=True)
