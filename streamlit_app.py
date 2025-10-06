# ---------------- Streamlit Wrapper ---------------- #
import streamlit as st
import tempfile
import pandas as pd
import os
import time
from dotenv import load_dotenv
from prompt import prompt_v2
from azure_seal_classifier import ClassifierApp  

load_dotenv()

# ---------------- Config ---------------- #
class Config:
    ROOT_DIR = st.secrets["ROOT_DIR"]
    INPUT_PATH = st.secrets["INPUT_PATH"]

    OUTPUT_YES = os.path.join(ROOT_DIR, "yes")
    OUTPUT_NO = os.path.join(ROOT_DIR, "no")
    LOG_FILE = os.path.join(ROOT_DIR, "classification_log.csv")
    ENHANCED_DIR = os.path.join(ROOT_DIR, "enhanced_images")

    API_KEY = st.secrets["AZURE_API_KEY"]
    ENDPOINT = st.secrets["AZURE_ENDPOINT"]
    API_VERSION = st.secrets["AZURE_API_VERSION"]
    MODEL_NAME = st.secrets["AZURE_DEPLOYMENT_NAME"]
    
    PROMPT = prompt_v2
    
    @staticmethod
    def init_dirs():
        os.makedirs(Config.OUTPUT_YES, exist_ok=True)
        os.makedirs(Config.OUTPUT_NO, exist_ok=True)
        os.makedirs(Config.ENHANCED_DIR, exist_ok=True)

# ---------------- Streamlit Page ---------------- #
st.set_page_config(page_title="Seal/Stamp/Classify App", layout="wide")
st.title("Seal / Stamp / Handwritten Signature Classifier")

# File uploader
uploaded_files = st.file_uploader(
    "Upload images or PDFs", 
    type=["jpg", "jpeg", "png", "pdf"], 
    accept_multiple_files=True
)

if uploaded_files:
    start_time = time.time()
    upload_progress = st.progress(0)
    upload_status = st.empty()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_paths = []
        for idx, uploaded_file in enumerate(uploaded_files):
            temp_path = os.path.join(tmpdir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            input_paths.append(temp_path)
            upload_progress.progress((idx + 1) / len(uploaded_files))
            upload_status.text(f"Uploaded {idx + 1} of {len(uploaded_files)} files...")

        upload_progress.empty()
        upload_status.empty()
        st.success(f"Uploaded {len(uploaded_files)} files successfully!")

        Config.INPUT_PATH = tmpdir
        Config.init_dirs()
        app = ClassifierApp(Config)

        # Calculate total pages for progress bar
        total_files = 0
        for uploaded_file in uploaded_files:
            if uploaded_file.name.lower().endswith(".pdf"):
                from pdf2image import convert_from_path
                pages = convert_from_path(os.path.join(tmpdir, uploaded_file.name), dpi=220)
                total_files += len(pages)
            else:
                total_files += 1

        process_progress = st.progress(0)
        processed_count = 0
        results_container = st.container()

        import csv
        with open(Config.LOG_FILE, "w", newline="", encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([
                "filename", "filetype", "page", "decision", "step_used",
                "response_text", "time_taken_sec", "tokens_used", "specification"
            ])

            # ---------------- Processing Loop ---------------- #
            for uploaded_file in uploaded_files:
                file_path = os.path.join(tmpdir, uploaded_file.name)

                def update_progress():
                    global processed_count
                    processed_count += 1
                    process_progress.progress(processed_count / total_files)

                with st.spinner(f"Processing {uploaded_file.name} ..."):
                    # Process file and get results as list of dicts
                    file_results = app._process_file(file_path, writer, progress_callback=update_progress)

                    # Display each result immediately
                    for row in file_results:
                        with results_container.expander(f"{row['filename']} → Seal? {str(row['decision']).upper()}", expanded=False):
                            st.markdown(f"**Step Used:** {row['step_used']}")
                            st.markdown(f"**Specification:** {row['specification']}")

                            if row['filetype'] == 'image':
                                img_path = os.path.join(Config.INPUT_PATH, row['filename'])
                            else:
                                page_img_name = f"{os.path.splitext(row['filename'])[0]}_page{row['page']}.jpg"
                                yes_path = os.path.join(Config.OUTPUT_YES, page_img_name)
                                no_path = os.path.join(Config.OUTPUT_NO, page_img_name)
                                img_path = yes_path if os.path.exists(yes_path) else no_path

                            if os.path.exists(img_path):
                                st.image(img_path, use_container_width=True)
                            else:
                                st.warning("Image not found for display.")

        process_progress.empty()
        elapsed = time.time() - start_time
        st.success(f"Processing completed in {elapsed:.2f} seconds!")

        # ---------------- CSV Download ---------------- #
        log_df = pd.read_csv(Config.LOG_FILE)
        simplified_df = log_df[['filename', 'page', 'decision']].copy()
        simplified_df.rename(columns={'decision': 'is_seal_detected'}, inplace=True)

        csv_data = simplified_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Classification CSV",
            data=csv_data,
            file_name="classification_log.csv",
            mime='text/csv'
        )
