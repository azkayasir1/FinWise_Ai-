"""Theme CSS and small HTML helpers (all user-supplied text is HTML-escaped)."""
from __future__ import annotations

import html
from typing import Optional

import pandas as pd

LIGHT = {"bg": "#F5F7FB", "card": "#FFFFFF", "text": "#172033", "muted": "#64748B", "border": "#E2E8F0",
         "primary": "#2563EB", "on_primary": "#FFFFFF", "side": "#FFFFFF", "input": "#FFFFFF",
         "ok_bg": "#DCFCE7", "ok": "#166534", "warn_bg": "#FEF3C7", "warn": "#92400E",
         "err_bg": "#FEE2E2", "err": "#991B1B", "info_bg": "#DBEAFE", "info": "#1E3A8A", "shadow": "rgba(15,23,42,.08)"}
DARK = {"bg": "#0B1220", "card": "#111C30", "text": "#F1F5F9", "muted": "#CBD5E1", "border": "#2A3A57",
        "primary": "#60A5FA", "on_primary": "#0B1220", "side": "#0E1729", "input": "#0B1220",
        "ok_bg": "#0F3D26", "ok": "#BBF7D0", "warn_bg": "#4A3410", "warn": "#FDE68A",
        "err_bg": "#4C1616", "err": "#FECACA", "info_bg": "#17305F", "info": "#BFDBFE", "shadow": "rgba(0,0,0,.35)"}


def get_css(theme: str) -> str:
    c = DARK if theme == "dark" else LIGHT
    return f"""
<style>
:root {{ --bg:{c['bg']}; --card:{c['card']}; --text:{c['text']}; --muted:{c['muted']}; --border:{c['border']};
         --primary:{c['primary']}; --on-primary:{c['on_primary']}; --teal:#14B8A6; }}
html, body, [data-testid="stAppViewContainer"], .stApp {{ background:var(--bg); color:var(--text);
  font-family: "Inter","Segoe UI",system-ui,-apple-system,Roboto,Helvetica,Arial,sans-serif; }}
[data-testid="stHeader"] {{ background:var(--bg); }}
[data-testid="stSidebar"] {{ background:{c['side']}; border-right:1px solid var(--border); }}
[data-testid="stSidebar"] * {{ color:var(--text); }}
.block-container {{ padding-top:1.5rem; max-width:1250px; animation: fw-fade .35s ease-out; }}
@keyframes fw-fade {{ from {{opacity:0; transform:translateY(6px);}} to {{opacity:1; transform:none;}} }}
h1,h2,h3,h4,h5,h6, label, p, li, span, div[data-testid="stMarkdownContainer"] {{ color:var(--text); }}
[data-testid="stCaptionContainer"], .fw-muted {{ color:var(--muted) !important; }}
/* inputs */
.stTextInput input, .stNumberInput input, .stTextArea textarea, .stDateInput input {{
  background:{c['input']} !important; color:var(--text) !important; border:1px solid var(--border) !important; }}
[data-baseweb="select"] > div {{ background:{c['input']} !important; color:var(--text) !important; border-color:var(--border) !important; }}
[data-baseweb="select"] * {{ color:var(--text) !important; }}
[data-baseweb="popover"] ul, [data-baseweb="popover"] li, [data-baseweb="menu"] {{ background:var(--card) !important; color:var(--text) !important; }}
[data-testid="stNumberInput"] button {{ background:{c['input']}; color:var(--text); border-color:var(--border); }}
/* buttons */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
  background:var(--card); color:var(--text); border:1px solid var(--border); border-radius:10px;
  transition:all .18s ease; font-weight:600; }}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {{
  border-color:var(--primary); color:var(--primary); transform:translateY(-1px); box-shadow:0 4px 12px {c['shadow']}; }}
button[kind="primary"], button[kind="primaryFormSubmit"] {{ background:var(--primary) !important; color:var(--on-primary) !important; border-color:var(--primary) !important; }}
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover {{ color:var(--on-primary) !important; filter:brightness(1.08); }}
/* tabs, expanders, forms */
[data-baseweb="tab-list"] {{ gap:6px; border-bottom:1px solid var(--border); }}
button[data-baseweb="tab"] {{ color:var(--muted); }}
button[data-baseweb="tab"][aria-selected="true"] {{ color:var(--primary); }}
[data-testid="stExpander"], [data-testid="stForm"] {{ background:var(--card); border:1px solid var(--border); border-radius:14px; }}
[data-testid="stExpander"] summary {{ color:var(--text); }}
[data-testid="stMetric"] {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:12px 16px; }}
[data-testid="stMetricLabel"] *, [data-testid="stMetricValue"] * {{ color:var(--text) !important; }}
[data-testid="stChatMessage"] {{ background:var(--card); border:1px solid var(--border); border-radius:14px; }}
[data-testid="stAlert"] {{ border-radius:12px; }}
.stProgress > div > div > div {{ background-color:var(--primary); }}
/* components */
.fw-header {{ display:flex; align-items:center; gap:14px; padding:18px 22px; border-radius:16px; margin-bottom:12px;
  background:linear-gradient(135deg,#0B1220 0%,#111C30 55%,#1E3A8A 100%); box-shadow:0 6px 18px {c['shadow']}; }}
.fw-header * {{ color:#F1F5F9 !important; margin:0; }}
.fw-logo {{ font-size:2rem; background:#2563EB; border-radius:12px; width:52px; height:52px; display:flex; align-items:center; justify-content:center; }}
.fw-header h1 {{ font-size:1.7rem; line-height:1.2; }}
.fw-header .sub {{ font-size:1rem; color:#BFDBFE !important; font-weight:600; }}
.fw-header .desc {{ font-size:.9rem; color:#CBD5E1 !important; }}
.fw-card {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:16px 18px;
  transition:transform .18s ease, box-shadow .18s ease; height:100%; }}
.fw-card:hover {{ transform:translateY(-2px); box-shadow:0 8px 20px {c['shadow']}; }}
.fw-label {{ color:var(--muted) !important; font-size:.9rem; font-weight:600; }}
.fw-value {{ color:var(--text) !important; font-size:1.65rem; font-weight:700; line-height:1.3; }}
.fw-sub {{ color:var(--muted) !important; font-size:.85rem; }}
.fw-accent-blue {{ border-top:3px solid #2563EB; }} .fw-accent-teal {{ border-top:3px solid #14B8A6; }}
.fw-accent-green {{ border-top:3px solid #16A34A; }} .fw-accent-red {{ border-top:3px solid #DC2626; }}
.fw-pos {{ color:{'#86EFAC' if theme=='dark' else '#16A34A'} !important; }} .fw-neg {{ color:{'#FCA5A5' if theme=='dark' else '#DC2626'} !important; }}
.fw-badge {{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:.85rem; font-weight:700; }}
.fw-ok {{ background:{c['ok_bg']}; color:{c['ok']} !important; }} .fw-warn {{ background:{c['warn_bg']}; color:{c['warn']} !important; }}
.fw-err {{ background:{c['err_bg']}; color:{c['err']} !important; }} .fw-info {{ background:{c['info_bg']}; color:{c['info']} !important; }}
.fw-disclaimer {{ background:{c['info_bg']}; color:{c['info']} !important; border-radius:12px; padding:10px 14px; font-size:.85rem; line-height:1.45; margin:8px 0; }}
.fw-disclaimer * {{ color:{c['info']} !important; }}
.fw-tag {{ font-size:.78rem; font-weight:700; letter-spacing:.04em; text-transform:uppercase; color:var(--primary) !important; }}
table.fw-table {{ width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--border); border-radius:12px; overflow:hidden; font-size:.92rem; }}
table.fw-table th {{ background:var(--primary); color:var(--on-primary) !important; text-align:left; padding:8px 12px; }}
table.fw-table td {{ padding:7px 12px; border-top:1px solid var(--border); color:var(--text) !important; }}
@media (max-width:760px) {{ .fw-value {{ font-size:1.3rem; }} .fw-header h1 {{ font-size:1.3rem; }} }}
</style>
"""


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def metric_card(label: str, value: str, icon: str, accent: str = "blue", sub: str = "",
                sub_class: str = "") -> str:
    sub_html = f'<div class="fw-sub {sub_class}">{esc(sub)}</div>' if sub else ""
    return (f'<div class="fw-card fw-accent-{accent}"><div class="fw-label">{icon} {esc(label)}</div>'
            f'<div class="fw-value">{esc(value)}</div>{sub_html}</div>')


def badge(text: str, kind: str = "info") -> str:
    return f'<span class="fw-badge fw-{kind}">{esc(text)}</span>'


def disclaimer_html(text: str) -> str:
    return f'<div class="fw-disclaimer">ℹ️ {esc(text)}</div>'


def header_html() -> str:
    return ('<div class="fw-header"><div class="fw-logo">📈</div><div>'
            '<h1>FinWise AI</h1><div class="sub">Your Personal Financial Intelligence Dashboard</div>'
            '<div class="desc">Analyse your monthly finances, track goals and get educational budgeting insights. '
            'For educational use only. No bank connection, no transactions.</div></div></div>')


def html_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    """Theme-aware table (st.dataframe renders on a canvas that ignores our theme)."""
    if df is None or df.empty:
        return '<div class="fw-sub">No rows to display.</div>'
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>"
                   for row in df.head(max_rows).itertuples(index=False))
    return f'<table class="fw-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
