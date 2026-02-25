import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Beban Capsa", page_icon="🃏", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;600&display=swap');

* { font-family: 'DM Sans', sans-serif; }

.main-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 64px;
    letter-spacing: 4px;
    background: linear-gradient(135deg, #FFD700 0%, #FF6B35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    margin: 0;
}
.tagline {
    font-size: 14px;
    color: #888;
    font-style: italic;
    margin: 4px 0 0 2px;
}
.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 8px;
}
.card-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 6px 10px;
    border-radius: 8px;
    margin: 3px;
    font-size: 15px;
    font-weight: 600;
    cursor: default;
    border: 1px solid;
}
.sample-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 8px;
    cursor: pointer;
    transition: border-color 0.2s;
    text-align: center;
}
.sample-card:hover { border-color: #FFD700; }
.combo-bomb {
    background: #1a0808;
    border-left: 3px solid #FF4444;
    padding: 10px 14px;
    border-radius: 0 6px 6px 0;
    margin: 4px 0;
}
.combo-regular {
    background: #0d1117;
    border-left: 3px solid #444;
    padding: 10px 14px;
    border-radius: 0 6px 6px 0;
    margin: 4px 0;
}
.author-line {
    text-align: center;
    color: #444;
    font-size: 12px;
    margin-top: 48px;
    padding-top: 16px;
    border-top: 1px solid #1e1e1e;
}
div[data-testid="stButton"] button {
    border-radius: 8px;
}
.editor-card {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 55px !important;
    height: 36px !important;
    border-radius: 8px;
    border: 1px solid;
    font-size: 15px;
    font-weight: 600;
    margin: 0 auto 4px auto !important; /* Tengahkan dan beri jarak bawah ke tombol */
    box-sizing: border-box;
}
div[data-testid="stHorizontalBlock"]:has(.editor-card) {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important; /* Jarak antar kartu */
    justify-content: flex-start !important;
}
div[data-testid="stHorizontalBlock"]:has(.editor-card) > div[data-testid="column"] {
    width: 55px !important;
    min-width: 55px !important;
    max-width: 55px !important;
    flex: 0 0 55px !important;
    padding: 0 !important;
    margin-bottom: 8px !important; /* Jarak jika kartu terlalu banyak dan turun baris */
}

div[data-testid="stHorizontalBlock"]:has(.editor-card) div[data-testid="stButton"] button {
    width: 55px !important;
    min-width: 55px !important;
    height: 36px !important;
    min-height: 36px !important;
    padding: 0 !important;
    margin: 0 auto !important;
}
div[data-testid="stHorizontalBlock"]:has(.editor-card) div[data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
    padding: 0 !important;
}        
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# KONSTANTA
# ══════════════════════════════════════════════════════════════════════════════

VALUE_RANK = {
    '3':1,  '4':2,  '5':3,  '6':4,  '7':5,  '8':6,  '9':7,
    '10':8, 'J':9,  'Q':10, 'K':11, 'A':12, '2':13
}
SUIT_RANK  = {'D':1, 'C':2, 'H':3, 'S':4}
SUIT_EMOJI = {'S':'♠', 'H':'♥', 'C':'♣', 'D':'♦'}
SUIT_NAME  = {'S':'Spade', 'H':'Heart', 'C':'Club', 'D':'Diamond'}
SUIT_COLOR = {'S':'#E0E0E0', 'H':'#FF5555', 'C':'#AAAAAA', 'D':'#FF8C00'}

ALL_VALUES = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
ALL_SUITS  = ['S','H','C','D']
ALL_CARDS  = [f"{v}{s}" for v in ALL_VALUES for s in ALL_SUITS]


# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════

def parse_card(cls):
    cls = str(cls).strip().upper()
    if cls.startswith('10'):
        return '10', cls[2] if len(cls) > 2 else '?'
    return (cls[0] if cls else '?'), (cls[1] if len(cls) >= 2 else '?')

def card_sort_key(card):
    v, s = parse_card(card)
    return (VALUE_RANK.get(v, 0), SUIT_RANK.get(s, 0))

def deduplicate(cards):
    return list(set(cards))

def card_label(card):
    v, s = parse_card(card)
    return f"{v}{SUIT_EMOJI.get(s,'?')}"

def render_cards_html(cards):
    html = ""
    for card in sorted(cards, key=card_sort_key, reverse=True):
        v, s  = parse_card(card)
        color = SUIT_COLOR.get(s, '#fff')
        emoji = SUIT_EMOJI.get(s, '?')
        html += (
            f'<span class="card-chip" style="color:{color};border-color:{color}33;'
            f'background:{color}11">{v}{emoji}</span>'
        )
    return html

def ensure_model():
    model_path = Path("models/best_capsa.pt")
    if model_path.exists():
        return

    try:
        file_id = st.secrets["MODEL_FILE_ID"]
    except KeyError:
        st.error("MODEL_FILE_ID tidak ditemukan di Streamlit Secrets. Hubungi admin.")
        st.stop()

    import gdown
    model_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"

    with st.spinner("Menyiapkan model... (hanya saat pertama)"):
        gdown.download(url, str(model_path), quiet=False)

ensure_model()

@st.cache_resource
def load_model():
    p = Path("models/best_capsa.pt")
    return YOLO(str(p)) if p.exists() else None


# ══════════════════════════════════════════════════════════════════════════════
# KOMBINASI
# ══════════════════════════════════════════════════════════════════════════════

def detect_combinations(cards):
    combos       = []
    value_groups = {}
    suit_groups  = {}

    for card in cards:
        v, s = parse_card(card)
        value_groups.setdefault(v, []).append(card)
        suit_groups.setdefault(s,  []).append(card)

    # Four of a Kind
    for val, grp in value_groups.items():
        if len(grp) >= 4:
            combos.append({
                'type':'Four of a Kind', 'is_bomb':True,
                'cards':sorted(grp, key=card_sort_key, reverse=True)[:4],
                'rank':8, 'desc':f'4x {val}'
            })

    # Straight Flush / Royal Flush
    for suit, sc in suit_groups.items():
        if len(sc) >= 5:
            uranks = sorted(set(VALUE_RANK.get(parse_card(c)[0], 0) for c in sc))
            for i in range(len(uranks) - 4):
                wr = uranks[i:i+5]
                if wr[-1] - wr[0] == 4 and len(wr) == 5:
                    wc = []
                    for r in wr:
                        cands = [c for c in sc if VALUE_RANK.get(parse_card(c)[0],0)==r]
                        wc.append(sorted(cands, key=card_sort_key, reverse=True)[0])
                    vals     = [parse_card(c)[0] for c in wc]
                    is_royal = set(vals) == {'10','J','Q','K','A'}
                    combos.append({
                        'type'   : 'Royal Flush' if is_royal else 'Straight Flush',
                        'is_bomb': True,
                        'cards'  : sorted(wc, key=card_sort_key, reverse=True),
                        'rank'   : 9 if is_royal else 8,
                        'desc'   : 'Tertinggi!' if is_royal else f'Straight Flush {SUIT_NAME.get(suit,suit)}'
                    })

    # Full House
    for tv in [v for v, g in value_groups.items() if len(g) >= 3]:
        b3 = sorted(value_groups[tv], key=card_sort_key, reverse=True)[:3]
        for pv in [v for v, g in value_groups.items() if len(g) >= 2 and v != tv]:
            b2 = sorted(value_groups[pv], key=card_sort_key, reverse=True)[:2]
            combos.append({
                'type':'Full House', 'is_bomb':False,
                'cards':b3+b2, 'rank':6,
                'desc':f'Three {tv} + Pair {pv}'
            })

    # Flush
    for suit, sc in suit_groups.items():
        if len(sc) >= 5:
            combos.append({
                'type':f'Flush {SUIT_NAME.get(suit,suit)}', 'is_bomb':False,
                'cards':sorted(sc, key=card_sort_key, reverse=True)[:5],
                'rank':5, 'desc':f'5 kartu {SUIT_NAME.get(suit,suit)}'
            })

    # Straight
    uranks = sorted(set(VALUE_RANK.get(parse_card(c)[0], 0) for c in cards))
    for i in range(len(uranks) - 4):
        wr = uranks[i:i+5]
        if wr[-1] - wr[0] == 4 and len(set(wr)) == 5:
            sc = []
            for r in wr:
                cands = [c for c in cards if VALUE_RANK.get(parse_card(c)[0],0)==r]
                sc.append(sorted(cands, key=card_sort_key, reverse=True)[0])
            combos.append({
                'type':'Straight', 'is_bomb':False,
                'cards':sorted(sc, key=card_sort_key, reverse=True),
                'rank':4, 'desc':'5 kartu berurutan'
            })

    # Tris
    for val, grp in value_groups.items():
        if len(grp) >= 3:
            combos.append({
                'type':'Three of a Kind', 'is_bomb':False,
                'cards':sorted(grp, key=card_sort_key, reverse=True)[:3],
                'rank':3, 'desc':f'3x {val}'
            })

    # Pair
    for val, grp in value_groups.items():
        if len(grp) >= 2:
            combos.append({
                'type':'Pair', 'is_bomb':False,
                'cards':sorted(grp, key=card_sort_key, reverse=True)[:2],
                'rank':2, 'desc':f'Sepasang {val}'
            })

    combos.sort(
        key=lambda x: (x['is_bomb'], x['rank'],
                       card_sort_key(x['cards'][0]) if x['cards'] else (0,0)),
        reverse=True
    )
    return combos


# ══════════════════════════════════════════════════════════════════════════════
# RENDER ANALISIS
# ══════════════════════════════════════════════════════════════════════════════

def render_analysis(cards):
    if not cards:
        st.warning("Tidak ada kartu untuk dianalisis.")
        return

    sorted_cards = sorted(cards, key=card_sort_key, reverse=True)

    st.markdown('<p class="section-label">Kartu — Terkuat ke Terlemah</p>',
                unsafe_allow_html=True)
    st.markdown(render_cards_html(sorted_cards), unsafe_allow_html=True)

    combos    = detect_combinations(cards)
    bombs     = [c for c in combos if c['is_bomb']]
    non_bombs = [c for c in combos if not c['is_bomb']]

    st.markdown("---")
    st.markdown('<p class="section-label">Kombinasi yang Bisa Dimainkan</p>',
                unsafe_allow_html=True)

    if not combos:
        st.caption("Tidak ada kombinasi — mainkan High Card.")
    else:
        if bombs:
            for c in bombs:
                cards_str = "  ".join(card_label(x) for x in c['cards'])
                st.markdown(f"""
                <div class="combo-bomb">
                    <span style="color:#FF4444;font-weight:600">{c['type']} — BOMB</span>
                    &nbsp;&nbsp;
                    <span style="font-size:16px">{cards_str}</span>
                    <br><span style="color:#666;font-size:12px">{c['desc']}</span>
                </div>""", unsafe_allow_html=True)

        shown = set()
        for c in non_bombs:
            key = c['type']
            if key in shown and c['rank'] >= 4:
                continue
            shown.add(key)
            cards_str = "  ".join(card_label(x) for x in c['cards'])
            st.markdown(f"""
            <div class="combo-regular">
                <span style="color:#ccc;font-weight:600">{c['type']}</span>
                &nbsp;&nbsp;
                <span style="font-size:16px">{cards_str}</span>
                <br><span style="color:#555;font-size:12px">{c['desc']}</span>
            </div>""", unsafe_allow_html=True)

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Kartu di Tangan", len(cards))
    col2.metric("Kombinasi Tersedia", len(combos))
    col3.metric("Bomb", f"{len(bombs)}" if bombs else "—")

    # Rekomendasi singkat
    if bombs:
        st.success("Simpan bomb. Gunakan hanya untuk menutup permainan atau mengalahkan kombinasi besar.")
    best = next((c for c in non_bombs if c['rank'] >= 4), None)
    if best:
        st.info(f"Kombinasi terkuat: {best['type']} — {'  '.join(card_label(x) for x in best['cards'])}")


# ══════════════════════════════════════════════════════════════════════════════
# CARD EDITOR
# ══════════════════════════════════════════════════════════════════════════════

def render_card_editor(cards, key_prefix):

    current = list(cards)

    st.markdown('<p class="section-label">Kartu Terdeteksi</p>',
                unsafe_allow_html=True)

    if not current:
        st.caption("Belum ada kartu.")
    else:
        cols = st.columns(len(current)) 

        for col, card in zip(cols, current):
            v, s  = parse_card(card)
            color = SUIT_COLOR.get(s, '#fff')
            emoji = SUIT_EMOJI.get(s, '?')
            with col:
                st.markdown(
                    f'<div class="editor-card" style="color:{color};'
                    f'border-color:{color}55;background:{color}11">'
                    f'{v}{emoji}</div>',
                    unsafe_allow_html=True
                )
                if st.button("−", key=f"{key_prefix}_del_{card}",
                             use_container_width=True, help=f"Hapus {v} {SUIT_NAME.get(s,s)}"):
                    current.remove(card)
                    st.session_state['detected_cards'] = current 
                    
                    st.rerun()

    # Tambah kartu
    remaining = [c for c in ALL_CARDS if c not in current]

    if st.button("+ Tambah Kartu", key=f"{key_prefix}_add_toggle",
                 use_container_width=False):
        st.session_state[f"{key_prefix}_show_add"] = \
            not st.session_state.get(f"{key_prefix}_show_add", False)

    if st.session_state.get(f"{key_prefix}_show_add", False):
        with st.container():
            st.markdown('<p class="section-label" style="margin-top:12px">Pilih kartu untuk ditambahkan</p>',
                        unsafe_allow_html=True)

            if not remaining:
                st.caption("Semua kartu sudah ada di tangan.")
            else:
                for val in reversed(ALL_VALUES):
                    val_cards = [c for c in remaining if parse_card(c)[0] == val]
                    if not val_cards:
                        continue

                    cols = st.columns(len(val_cards))
                    for col, card in zip(cols, val_cards):
                        v, s  = parse_card(card)
                        color = SUIT_COLOR.get(s, '#fff')
                        emoji = SUIT_EMOJI.get(s, '?')
                        with col:
                            if st.button(
                                f"{v}{emoji}",
                                key=f"{key_prefix}_pick_{card}",
                                use_container_width=True,
                                help=f"Tambah {v} of {SUIT_NAME.get(s,s)}"
                            ):
                                current.append(card)
                                st.session_state[f"{key_prefix}_show_add"] = False
                                st.session_state['detected_cards'] = current
                                st.rerun()

    return current


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE IMAGES
# ══════════════════════════════════════════════════════════════════════════════

def get_sample_images():
    sample_dir = Path("sample")
    if not sample_dir.exists():
        return []
    imgs = sorted(sample_dir.glob("*.jpg"))[:3]
    return imgs


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown('<div class="main-title">BEBAN CAPSA</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="tagline">'
    '"Jangan biarkan tongkrongan asyikmu terganggu karena kamu jadi BEBAN"'
    '</div>',
    unsafe_allow_html=True
)

# Info expanders
col_info1, col_info2 = st.columns(2)

with col_info1:
    with st.expander("Tentang Aplikasi"):
        st.markdown("""
        Aplikasi ini menggunakan **YOLOv12** untuk mendeteksi kartu capsa
        dari foto, lalu menganalisis kombinasi yang bisa dimainkan.

        Upload foto kartu yang kamu pegang — aplikasi akan mendeteksi
        kartu, mengurutkan dari terkuat ke terlemah, dan menampilkan
        semua kombinasi yang tersedia beserta rekomendasi strategi.

        Gunakan fitur **tambah / hapus kartu** jika ada yang tidak
        terdeteksi dengan benar.
        """)

with col_info2:
    with st.expander("Aturan Capsa"):
        st.markdown("""
        **Urutan nilai** (tertinggi → terendah):
        
        2 > A > K > Q > J > 10 > 9 > 8 > 7 > 6 > 5 > 4 > 3

                    
        **Urutan simbol** pada nilai yang sama:
                    
        Spade > Heart > Club > Diamond
                    

        **Kombinasi** (terkuat → terlemah):
                    
        Royal Flush · Straight Flush · Four of a Kind *(ketiganya Bomb)*
        Full House · Flush · Straight · Tris · Pair · High Card

                    
        Pair dilawan Pair, Tris dilawan Tris.
        **Bomb** mengalahkan kombinasi apapun.
        """)

st.divider()

# ── Sample Images ─────────────────────────────────────────────────────────────
sample_imgs = get_sample_images()

if sample_imgs:
    st.markdown('<p class="section-label">Sample Image</p>', unsafe_allow_html=True)
    s_cols = st.columns(len(sample_imgs))
    for col, img_path in zip(s_cols, sample_imgs):
        with col:
            thumb = Image.open(img_path)
            st.image(thumb, use_container_width=True)
            if st.button("Gunakan foto ini",
                         key=f"sample_{img_path.stem}",
                         use_container_width=True):
                st.session_state['active_image']  = Image.open(img_path).convert('RGB')
                st.session_state['active_source'] = img_path.stem
                st.session_state.pop('detected_cards', None)
                st.session_state.pop('detect_results', None)
                st.rerun()

    st.markdown("")

# ── Upload & Deteksi ──────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<p class="section-label">Upload Foto</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Pilih foto kartu",
        type=['jpg','jpeg','png','webp'],
        label_visibility="collapsed"
    )

    if uploaded:
        st.session_state['active_image']  = Image.open(uploaded).convert('RGB')
        st.session_state['active_source'] = uploaded.name
        st.session_state.pop('detected_cards', None)
        st.session_state.pop('detect_results', None)

    if 'active_image' in st.session_state:
        st.image(
            st.session_state['active_image'],
            caption=st.session_state.get('active_source', ''),
            use_container_width=True
        )

        conf_thresh = st.slider("Confidence", 0.05, 0.9, 0.10, 0.05,
                                help="Turunkan jika kartu tidak terdeteksi")

        if st.button("Deteksi Kartu", type="primary", use_container_width=True):
            model = load_model()
            if model is None:
                st.error("Model tidak ditemukan di models/best_capsa.pt")
            else:
                with st.spinner("Mendeteksi..."):
                    img_bgr = cv2.cvtColor(
                        np.array(st.session_state['active_image']),
                        cv2.COLOR_RGB2BGR
                    )
                    results = model.predict(
                        source=img_bgr,
                        conf=conf_thresh,
                        iou=0.2,
                        agnostic_nms=True,
                        verbose=False
                    )
                    boxes = results[0].boxes
                    det_items = []
                    for i in range(len(boxes)):
                        x_center = boxes.xywh[i][0].item() 
                        cls_idx  = int(boxes.cls[i].item())
                        cls_name = model.names[cls_idx]
                        det_items.append((x_center, cls_name))

                    det_items.sort(key=lambda x: x[0])

                    detected = []
                    seen = set()
                    for _, cls_name in det_items:
                        if cls_name not in seen:
                            seen.add(cls_name)
                            detected.append(cls_name)

                    st.session_state['detected_cards'] = detected
                    st.session_state['detect_results'] = results

with col_right:
    if 'detect_results' in st.session_state:
        st.markdown('<p class="section-label">Hasil Deteksi</p>',
                    unsafe_allow_html=True)
        annotated = cv2.cvtColor(
            st.session_state['detect_results'][0].plot(),
            cv2.COLOR_BGR2RGB
        )
        st.image(annotated, use_container_width=True)

# ── Card Editor & Analisis ────────────────────────────────────────────────────
if 'detected_cards' in st.session_state:
    st.divider()

    edited_cards = render_card_editor(
        st.session_state['detected_cards'],
        key_prefix='main'
    )
    st.session_state['detected_cards'] = edited_cards

    if edited_cards:
        st.divider()
        render_analysis(edited_cards)
    else:
        st.caption("Tidak ada kartu. Tambahkan kartu secara manual atau upload foto ulang.")

# Footer
st.markdown("""
<div class="author-line">
    Beban Capsa &nbsp;·&nbsp; Built by <strong>Uqi Shafi</strong>
    &nbsp;·&nbsp; Powered by YOLOv12 & Streamlit
</div>
""", unsafe_allow_html=True)