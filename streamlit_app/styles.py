"""Navy–gold–ivory theme tokens and shared CSS for the regulatory Streamlit UI."""

# Palette tokens
NAVY_900 = "#0B1F33"
NAVY_800 = "#102A44"
NAVY_700 = "#16324F"
GOLD_500 = "#C4A35A"
GOLD_600 = "#A8883F"
IVORY_50 = "#F6F3EC"
IVORY_100 = "#EDE8DC"
IVORY_200 = "#E2DCCE"
WHITE = "#FFFFFF"
INK = "#0E1A28"
SLATE = "#4A5A6A"
FOREST = "#1F6B4A"
BURGUNDY = "#8B2E2E"
AMBER = "#9A6B2F"


def inject_css() -> None:
    """Inject shared finance-sector styling used by all Streamlit pages."""
    import streamlit as st

    st.markdown(
        f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    .stApp {{
        background:
            radial-gradient(ellipse 80% 50% at 100% -10%, rgba(196,163,90,0.12), transparent 50%),
            linear-gradient(180deg, {IVORY_50} 0%, #F0EBE0 100%);
        color: {INK};
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    }}

    /* Hide default Streamlit chrome noise */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{
        background: transparent;
    }}

    /* —— Sidebar —— */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {NAVY_900} 0%, {NAVY_800} 100%);
        border-right: 1px solid rgba(196,163,90,0.35);
    }}
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 1.25rem;
    }}
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] small {{
        color: {IVORY_100} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] span {{
        color: {IVORY_100} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-selected="true"] {{
        background: rgba(196,163,90,0.18) !important;
        border-left: 3px solid {GOLD_500} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] {{
        border-left: 3px solid transparent;
        border-radius: 0 6px 6px 0;
        margin: 0.15rem 0.4rem;
    }}
    section[data-testid="stSidebar"] .stButton > button {{
        background: transparent !important;
        color: {IVORY_50} !important;
        border: 1px solid rgba(196,163,90,0.55) !important;
        width: 100%;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        background: rgba(196,163,90,0.2) !important;
        border-color: {GOLD_500} !important;
    }}
    section[data-testid="stSidebar"] .stSuccess {{
        background: rgba(31,107,74,0.25);
        color: {IVORY_50};
    }}

    /* —— Main content typography (force dark ink — do not inherit sidebar ivory) —— */
    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }}
    .main h1, .main h2, .main h3, .main h4 {{
        font-family: "Source Serif 4", Georgia, serif !important;
        color: {NAVY_900} !important;
        letter-spacing: -0.01em;
    }}
    .main p, .main li, .main label, .main span,
    .main [data-testid="stMarkdownContainer"] p,
    .main [data-testid="stMarkdownContainer"] li {{
        color: {INK} !important;
    }}
    .main [data-testid="stCaptionContainer"],
    .main .stCaption {{
        color: {SLATE} !important;
    }}

    /* Hero brand strip */
    .desk-hero {{
        background: linear-gradient(135deg, {NAVY_900} 0%, {NAVY_700} 70%, #1a3d5c 100%);
        border-radius: 10px;
        padding: 1.6rem 1.85rem 1.35rem;
        margin-bottom: 1.35rem;
        border: 1px solid rgba(196,163,90,0.4);
        box-shadow: 0 8px 28px rgba(11,31,51,0.18);
        position: relative;
        overflow: hidden;
    }}
    .desk-hero::after {{
        content: "";
        position: absolute;
        left: 0; right: 0; bottom: 0;
        height: 3px;
        background: linear-gradient(90deg, {GOLD_500}, {GOLD_600}, transparent);
    }}
    .desk-hero .eyebrow {{
        color: {GOLD_500};
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin: 0 0 0.45rem 0;
        font-family: "IBM Plex Sans", sans-serif;
    }}
    .desk-hero h1 {{
        color: {IVORY_50} !important;
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        margin: 0 0 0.4rem 0 !important;
        border: none !important;
        padding: 0 !important;
        font-family: "Source Serif 4", Georgia, serif !important;
    }}
    .desk-hero .subtitle {{
        color: rgba(246,243,236,0.82);
        font-size: 0.98rem;
        margin: 0;
        max-width: 42rem;
        line-height: 1.45;
        font-family: "IBM Plex Sans", sans-serif;
    }}
    .desk-hero .meta {{
        margin-top: 0.85rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
    }}
    .desk-hero .meta span {{
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: {NAVY_900};
        background: {GOLD_500};
        padding: 0.22rem 0.55rem;
        border-radius: 3px;
    }}

    /* Layer architecture */
    .layer-row {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.85rem;
        margin: 0.85rem 0 1.4rem 0;
    }}
    .layer-card {{
        background: {WHITE};
        border: 1px solid {IVORY_200};
        border-radius: 8px;
        padding: 1.05rem 1.15rem;
        box-shadow: 0 2px 8px rgba(11,31,51,0.05);
        position: relative;
    }}
    .layer-card::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 8px 8px 0 0;
        background: {GOLD_500};
    }}
    .layer-card .layer-tag {{
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: {GOLD_600};
        margin-bottom: 0.4rem;
    }}
    .layer-card h4 {{
        margin: 0 0 0.35rem 0;
        color: {NAVY_900} !important;
        font-size: 1.05rem;
        font-family: "Source Serif 4", Georgia, serif;
    }}
    .layer-card p {{
        margin: 0;
        color: {SLATE} !important;
        font-size: 0.88rem;
        line-height: 1.45;
    }}
    .layer-card.featured {{
        background: linear-gradient(160deg, {WHITE} 60%, #FBF7EE);
        border-color: rgba(196,163,90,0.55);
        box-shadow: 0 4px 14px rgba(196,163,90,0.12);
    }}

    /* Module grid */
    .module-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 0.65rem;
        margin: 0.6rem 0 1.1rem 0;
    }}
    .module-card {{
        background: {WHITE};
        border: 1px solid {IVORY_200};
        border-left: 3px solid {NAVY_700};
        border-radius: 6px;
        padding: 0.8rem 0.9rem;
    }}
    .module-card strong {{
        color: {NAVY_900} !important;
        display: block;
        font-size: 0.88rem;
        margin-bottom: 0.25rem;
    }}
    .module-card span {{
        color: {SLATE} !important;
        font-size: 0.8rem;
        line-height: 1.35;
    }}

    /* Section panel */
    .section-panel {{
        background: {WHITE};
        border: 1px solid {IVORY_200};
        border-radius: 8px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 1rem;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.15rem;
        background: {IVORY_100};
        padding: 0.35rem;
        border-radius: 8px;
        border: 1px solid {IVORY_200};
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {SLATE} !important;
        background: transparent !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        padding: 0.45rem 0.9rem !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {NAVY_900} !important;
        background: {WHITE} !important;
        font-weight: 650 !important;
        box-shadow: 0 1px 3px rgba(11,31,51,0.08);
        border-bottom: 2px solid {GOLD_500} !important;
    }}

    /* Buttons — main area */
    .main .stButton > button {{
        background-color: {NAVY_900} !important;
        color: {IVORY_50} !important;
        border: 1px solid {NAVY_900} !important;
        border-radius: 5px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
        padding: 0.4rem 1.1rem !important;
    }}
    .main .stButton > button:hover {{
        background-color: {NAVY_700} !important;
        border-color: {GOLD_500} !important;
        color: {IVORY_50} !important;
    }}
    .main .stButton > button:focus {{
        box-shadow: 0 0 0 2px {GOLD_500}66 !important;
    }}

    /* Metrics */
    [data-testid="stMetric"] {{
        background: {WHITE};
        border: 1px solid {IVORY_200};
        border-top: 3px solid {GOLD_500};
        border-radius: 8px;
        padding: 0.85rem 1rem;
        box-shadow: 0 2px 6px rgba(11,31,51,0.04);
    }}
    [data-testid="stMetricLabel"] {{
        color: {SLATE} !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {NAVY_900} !important;
        font-family: "Source Serif 4", Georgia, serif;
    }}

    /* Cards */
    .reg-card {{
        background: {WHITE};
        border: 1px solid {IVORY_200};
        border-left: 4px solid {GOLD_500};
        border-radius: 8px;
        padding: 1.15rem 1.25rem;
        margin: 0.7rem 0;
        box-shadow: 0 2px 10px rgba(11,31,51,0.06);
    }}
    .reg-card h4 {{
        margin: 0 0 0.5rem 0;
        color: {NAVY_900} !important;
        font-size: 1rem;
        font-family: "Source Serif 4", Georgia, serif;
    }}
    .reg-card p {{
        margin: 0;
        color: {INK} !important;
        line-height: 1.55;
    }}
    .reg-muted {{
        color: {SLATE} !important;
        font-size: 0.86rem;
    }}

    /* Status chips */
    .status-chip {{
        display: inline-block;
        padding: 0.28rem 0.7rem;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 0.2rem 0.35rem 0.2rem 0;
        border: 1px solid {IVORY_200};
        background: {WHITE};
        color: {INK} !important;
    }}
    .status-ok {{
        color: {FOREST} !important;
        border-color: rgba(31,107,74,0.35);
        background: #eef6f1;
    }}
    .status-bad {{
        color: {BURGUNDY} !important;
        border-color: rgba(139,46,46,0.35);
        background: #f8eeee;
    }}
    .status-warn {{
        color: {AMBER} !important;
        border-color: rgba(154,107,47,0.4);
        background: #f7f1e6;
    }}

    .claim-badge {{
        display: inline-block;
        padding: 0.18rem 0.55rem;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        background: #f7f1e6;
        color: {AMBER} !important;
        border: 1px solid rgba(154,107,47,0.4);
        vertical-align: middle;
    }}

    thead tr th {{
        background-color: {IVORY_100} !important;
        color: {NAVY_900} !important;
        font-weight: 600 !important;
    }}
    tbody tr td {{
        color: {INK} !important;
    }}

    [data-testid="stExpander"] {{
        background: {WHITE};
        border: 1px solid {IVORY_200};
        border-radius: 8px;
    }}

    .stTextInput input, .stTextArea textarea {{
        border-radius: 5px !important;
        border-color: {IVORY_200} !important;
        color: {INK} !important;
        background: {WHITE} !important;
    }}
    .stTextInput input:focus, .stTextArea textarea:focus {{
        border-color: {GOLD_500} !important;
        box-shadow: 0 0 0 1px {GOLD_500} !important;
    }}
    .stSelectbox [data-baseweb="select"] > div {{
        background: {WHITE} !important;
        color: {INK} !important;
    }}

    /* Sidebar brand block */
    .sidebar-brand {{
        padding: 0.25rem 0.5rem 1rem;
        border-bottom: 1px solid rgba(196,163,90,0.3);
        margin-bottom: 0.85rem;
    }}
    .sidebar-brand .brand-name {{
        color: {IVORY_50} !important;
        font-family: "Source Serif 4", Georgia, serif;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.25;
    }}
    .sidebar-brand .brand-sub {{
        color: {GOLD_500} !important;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 0.35rem 0 0 0;
    }}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand(subtitle: str = "Compliance intelligence desk") -> None:
    import streamlit as st

    st.sidebar.markdown(
        f"""
<div class="sidebar-brand">
  <p class="brand-name">RBI &amp; SEBI Regulatory</p>
  <p class="brand-sub">{subtitle}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(
    title: str = "RBI & SEBI Regulatory",
    subtitle: str = (
        "Grounded answers from Master Directions, Circulars, and Guidelines — "
        "with temporal context and a multi-stage retrieval layer."
    ),
    tags=None,
) -> None:
    import streamlit as st

    if tags is None:
        tags = ["Temporal Layer", "Retrieval Layer", "Grounded Generation"]
    tags_html = "".join(f"<span>{t}</span>" for t in tags)
    st.markdown(
        f"""
<div class="desk-hero">
  <p class="eyebrow">Financial Regulatory Desk</p>
  <h1>{title}</h1>
  <p class="subtitle">{subtitle}</p>
  <div class="meta">{tags_html}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def status_chip(label: str, ok=None, warn: bool = False) -> str:
    if warn:
        cls = "status-chip status-warn"
    elif ok is True:
        cls = "status-chip status-ok"
    elif ok is False:
        cls = "status-chip status-bad"
    else:
        cls = "status-chip"
    return f'<span class="{cls}">{label}</span>'


def answer_card(title: str, body: str, muted: str = "") -> str:
    muted_html = f'<p class="reg-muted" style="margin-top:0.65rem;">{muted}</p>' if muted else ""
    return f"""
<div class="reg-card">
  <h4>{title}</h4>
  <p>{body}</p>
  {muted_html}
</div>
"""


def claim_card_html(
    number: int,
    claim: str,
    query: str,
    citations: list,
    snippet: str,
    status: str = "Pending SLM verification",
) -> str:
    cites = "<br>".join(f"• {c}" for c in citations) if citations else "• —"
    snippet_safe = (snippet or "—")[:400]
    return f"""
<div class="reg-card">
  <h4>Claim [{number}] <span class="claim-badge">{status}</span></h4>
  <p><strong>{claim}</strong></p>
  <p class="reg-muted" style="margin-top:0.55rem;"><strong>Query:</strong> {query}</p>
  <p class="reg-muted" style="margin-top:0.35rem;"><strong>Citations:</strong><br>{cites}</p>
  <p class="reg-muted" style="margin-top:0.35rem;"><strong>Supporting snippet:</strong> {snippet_safe}</p>
</div>
"""
