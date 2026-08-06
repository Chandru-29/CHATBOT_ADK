"""
page_styles.py — CSS theme for the Streamlit chatbot UI.

The full <style> block is stored here as a constant string so app.py
stays clean. To change colours, fonts, or layout, edit this file only.
"""

CSS: str = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Apply clean typography globally */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #f9f9fb !important;
    color: #18181b !important;
}

/* Force headings and text to be dark grey/black in main content */
h1, h2, h3, h4, h5, h6, p, span, li {
    color: #18181b !important;
}

/* Fix top header white bar */
header[data-testid="stHeader"], [data-testid="stHeader"] {
    background-color: #f9f9fb !important;
    background: #f9f9fb !important;
}

/* Fix bottom chat input container white bar */
[data-testid="stBottom"], [data-testid="stBottom"] > div {
    background-color: #f9f9fb !important;
    background: #f9f9fb !important;
}

/* Align user chat message container to the right */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse !important;
}

/* Fix avatar margins when row is reversed for user */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageAvatarUser"] {
    margin-left: 12px !important;
    margin-right: 0 !important;
}

/* Remove default message borders and backgrounds */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 8px 0 !important;
}

/* User chat bubble styling - aligned right, bottom-right sharp */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background-color: #f4f4f5 !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 12px 18px !important;
    border: 1px solid #e4e4e7 !important;
    margin-left: auto !important;
    margin-right: 0 !important;
    color: #18181b !important;
    width: fit-content !important;
    max-width: 75% !important;
    flex-grow: 0 !important;
}

/* Bot chat bubble styling - aligned left, bottom-left sharp */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
    background-color: #ffffff !important;
    border-radius: 16px 16px 16px 4px !important;
    padding: 14px 20px !important;
    border: 1px solid #e4e4e7 !important;
    color: #18181b !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    margin-left: 0 !important;
    margin-right: auto !important;
    width: fit-content !important;
    max-width: 75% !important;
    flex-grow: 0 !important;
}

/* Remove Streamlit sidebar overlay background dimming and blur */
[data-testid="stSidebarOverlay"], div[data-testid="stSidebarOverlay"] {
    background-color: transparent !important;
    background: transparent !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
}

/* Floating Light Glassmorphism Sidebar Container */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.88) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border-right: 1px solid rgba(228, 228, 231, 0.8) !important;
    box-shadow: 10px 0 30px rgba(0, 0, 0, 0.08) !important;
    z-index: 999999 !important;
}

/* Sidebar elements text and headings contrast for light glassmorphism */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption {
    color: #18181b !important;
}

/* Ensure text inputs and buttons in sidebar have good borders/colors */
[data-testid="stSidebar"] input {
    background-color: #ffffff !important;
    color: #18181b !important;
    border: 1px solid #d4d4d8 !important;
    border-radius: 6px !important;
}



/* Custom width for content column */
.block-container {
    max-width: 100% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-top: 1.5rem !important;
    padding-bottom: 7rem !important;
}

/* Scrollbars */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #e4e4e7; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #d4d4d8; }

/* Clean buttons - flat styling */
.stButton > button {
    border-radius: 6px !important;
    font-weight: 500 !important;
    border: 1px solid #e4e4e7 !important;
    background: #ffffff !important;
    color: #18181b !important;
    transition: background 0.15s ease;
}
.stButton > button:hover {
    background: #f4f4f5 !important;
    border-color: #d4d4d8 !important;
}
.stButton > button[kind="primary"] {
    background: #10a37f !important;
    border: none !important;
    color: white !important;
}
.stButton > button[kind="primary"]:hover {
    background: #1a7f64 !important;
}

/* Minimal Table styling for markdown outputs */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 14px;
    background: #ffffff;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #e4e4e7;
}
th {
    background: #f4f4f5;
    color: #18181b;
    font-weight: 600;
    text-align: left;
    padding: 10px 14px;
    border-bottom: 1.5px solid #e4e4e7;
}
td {
    padding: 8px 14px;
    border-bottom: 1px solid #f4f4f5;
    color: #27272a;
}
tr:hover { background-color: #fafafa; }

/* Hide standard Streamlit decorations */
footer { visibility: hidden; }
[data-testid="stElementToolbar"] { display: none !important; }

/* Status pill row styling */
.status-pill {
    background: #ffffff;
    border: 1px solid #e4e4e7;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    color: #71717a;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}
</style>
"""
