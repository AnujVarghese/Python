import os
import runpy
import sys
from contextlib import contextmanager
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent

PROJECTS = [
    {
        "id": "ml",
        "number": "01",
        "title": "Solar Power ML Suite",
        "domain": "Machine Learning",
        "summary": "Predict plant power output and efficiency tier from operating conditions.",
        "tech": "Scikit-Learn, Pandas, Joblib",
        "folder": "01_Machine_Learning",
        "accent": "#f59e0b",
    },
    {
        "id": "dl",
        "number": "02",
        "title": "Industrial AI Diagnostics",
        "domain": "Deep Learning",
        "summary": "Run ANN sensor-vitals diagnostics and CNN visual defect inspection.",
        "tech": "PyTorch, Torchvision, Seaborn",
        "folder": "02_Deep_Learning",
        "accent": "#2563eb",
    },
    {
        "id": "llm",
        "number": "03",
        "title": "CodeIQ & DocSense",
        "domain": "Simple LLM",
        "summary": "Analyze code, refactor snippets, and summarize business documents.",
        "tech": "Transformers, FLAN-T5, APIs",
        "folder": "03_Simple_LLM",
        "accent": "#059669",
    },
    {
        "id": "vision",
        "number": "04",
        "title": "SmartVision Studio",
        "domain": "OpenCV Vision",
        "summary": "Scan documents, segment colors, blur faces, and apply image filters.",
        "tech": "OpenCV, NumPy, Pillow",
        "folder": "04_OpenCV_Vision",
        "accent": "#dc2626",
    },
    {
        "id": "genai",
        "number": "05",
        "title": "AuraGen Canvas",
        "domain": "Generative AI",
        "summary": "Create narrative storyboards and procedural concept artwork.",
        "tech": "Transformers, PIL, NumPy",
        "folder": "05_Generative_AI",
        "accent": "#7c3aed",
    },
    {
        "id": "chatbot",
        "number": "06",
        "title": "NexusChat AI",
        "domain": "AI Chatbot",
        "summary": "Chat with selectable personas, metrics, memory, and export tools.",
        "tech": "Transformers, Session State",
        "folder": "06_AI_Chatbot",
        "accent": "#0891b2",
    },
]


def rerun_app():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


@contextmanager
def project_runtime(project_dir):
    project_path = str(project_dir)
    previous_cwd = os.getcwd()
    added_path = False

    if project_path not in sys.path:
        sys.path.insert(0, project_path)
        added_path = True

    os.chdir(project_path)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        if added_path:
            try:
                sys.path.remove(project_path)
            except ValueError:
                pass


def run_project(project):
    app_path = ROOT_DIR / project["folder"] / "app.py"
    if not app_path.exists():
        st.error(f"Cannot find {app_path}")
        return

    st.sidebar.markdown("---")
    if st.sidebar.button("Back to Main Page", use_container_width=True):
        st.session_state.selected_project = "Home"
        rerun_app()

    original_set_page_config = st.set_page_config
    st.set_page_config = lambda *args, **kwargs: None
    try:
        with project_runtime(app_path.parent):
            runpy.run_path(str(app_path), run_name=f"__{project['id']}_streamlit_app__")
    finally:
        st.set_page_config = original_set_page_config


def inject_styles():
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 2rem;
            }

            .hub-header {
                border-bottom: 1px solid rgba(148, 163, 184, 0.24);
                margin-bottom: 1.2rem;
                padding-bottom: 1rem;
            }

            .hub-title {
                color: #111827;
                font-size: 2.4rem;
                font-weight: 800;
                letter-spacing: 0;
                line-height: 1.1;
                margin: 0;
            }

            .hub-subtitle {
                color: #475569;
                font-size: 1rem;
                margin-top: 0.55rem;
                max-width: 780px;
            }

            .project-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                min-height: 228px;
                padding: 1.05rem 1.05rem 0.9rem;
                margin-bottom: 0.65rem;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
            }

            .project-number {
                align-items: center;
                border-radius: 999px;
                color: #ffffff;
                display: inline-flex;
                font-size: 0.76rem;
                font-weight: 800;
                height: 1.65rem;
                justify-content: center;
                margin-bottom: 0.8rem;
                width: 1.65rem;
            }

            .project-domain {
                color: #64748b;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                margin-bottom: 0.35rem;
                text-transform: uppercase;
            }

            .project-title {
                color: #0f172a;
                font-size: 1.1rem;
                font-weight: 800;
                line-height: 1.25;
                margin-bottom: 0.45rem;
            }

            .project-summary {
                color: #334155;
                font-size: 0.92rem;
                line-height: 1.45;
                min-height: 4rem;
            }

            .project-tech {
                color: #475569;
                font-size: 0.8rem;
                font-weight: 650;
                margin-top: 0.75rem;
            }

            .suite-metric {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 0.85rem 1rem;
            }

            .suite-metric-label {
                color: #64748b;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.06em;
                text-transform: uppercase;
            }

            .suite-metric-value {
                color: #0f172a;
                font-size: 1.45rem;
                font-weight: 800;
                margin-top: 0.15rem;
            }

            div.stButton > button {
                border-radius: 7px;
                font-weight: 700;
                min-height: 2.45rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric(label, value):
    st.markdown(
        f"""
        <div class="suite-metric">
            <div class="suite-metric-label">{label}</div>
            <div class="suite-metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home():
    st.markdown(
        """
        <div class="hub-header">
            <h1 class="hub-title">AI & ML Project Hub</h1>
            <div class="hub-subtitle">
                A single Streamlit main page for launching all six portfolio projects from one app.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(3)
    with metric_cols[0]:
        render_metric("Projects", len(PROJECTS))
    with metric_cols[1]:
        render_metric("Domains", "ML, DL, LLM, CV, GenAI, Chat")
    with metric_cols[2]:
        render_metric("Run Command", "streamlit run app.py")

    st.markdown("### Project Launcher")

    for row_start in range(0, len(PROJECTS), 3):
        cols = st.columns(3)
        for col, project in zip(cols, PROJECTS[row_start : row_start + 3]):
            with col:
                st.markdown(
                    f"""
                    <div class="project-card">
                        <div class="project-number" style="background: {project['accent']};">
                            {project['number']}
                        </div>
                        <div class="project-domain">{project['domain']}</div>
                        <div class="project-title">{project['title']}</div>
                        <div class="project-summary">{project['summary']}</div>
                        <div class="project-tech">{project['tech']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(
                    f"Open {project['number']}",
                    key=f"open_{project['id']}",
                    use_container_width=True,
                ):
                    st.session_state.selected_project = project["title"]
                    rerun_app()

    st.markdown("### Direct Project Files")
    st.dataframe(
        [
            {
                "Project": project["title"],
                "Folder": project["folder"],
                "Streamlit file": f"{project['folder']}/app.py",
            }
            for project in PROJECTS
        ],
        use_container_width=True,
    )


def main():
    st.set_page_config(
        page_title="AI & ML Project Hub",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_styles()

    labels = ["Home", *[project["title"] for project in PROJECTS]]
    selected = st.session_state.get("selected_project", "Home")
    if selected not in labels:
        selected = "Home"

    st.sidebar.title("AI Project Suite")
    st.sidebar.caption("Choose a project to run inside this app.")
    selected = st.sidebar.radio(
        "Navigation",
        labels,
        index=labels.index(selected),
        label_visibility="collapsed",
    )
    st.session_state.selected_project = selected

    if selected == "Home":
        render_home()
        return

    project = next(item for item in PROJECTS if item["title"] == selected)
    run_project(project)


if __name__ == "__main__":
    main()
