# pylint: disable=all

# ---------------- Imports ---------------- #
import os
import cv2
import csv
import base64
import shutil
import numpy as np
import time
from io import BytesIO
from tqdm import tqdm
from pdf2image import convert_from_path, convert_from_bytes
from openai import AzureOpenAI
from prompt import prompt_v2
from dotenv import load_dotenv
import streamlit as st
import pandas as pd

load_dotenv()

# ---------------- Config ---------------- #
class Config:
    """Holds configuration constants for the pipeline."""

    # ---- Root Settings ----
    ROOT_DIR = st.secrets["ROOT_DIR"]

    # ---- Derived Paths ----
    OUTPUT_YES = os.path.join(ROOT_DIR, "yes")
    OUTPUT_NO = os.path.join(ROOT_DIR, "no")
    LOG_FILE = os.path.join(ROOT_DIR, f"classification_log.csv")
    ENHANCED_DIR = os.path.join(ROOT_DIR, "enhanced_images")

    # ---- Azure OpenAI configuration ----
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


# ---------------- Analyzer ---------------- #
class GPTImageAnalyzer:
    """Wrapper for Azure OpenAI GPT Vision model."""

    def __init__(self, config: Config):
        self.client = AzureOpenAI(
            api_key=config.API_KEY,
            api_version=config.API_VERSION,
            azure_endpoint=config.ENDPOINT,
        )
        self.prompt = config.PROMPT
        self.model = config.MODEL_NAME

    def analyze(self, image_bytes: bytes) -> dict:
        """Send image bytes to Azure OpenAI and return the response and tokens."""
        encoded_image = base64.b64encode(image_bytes).decode("ascii")

        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": self.prompt},
                    ],
                },
            ],
        )
        end_time = time.time()

        tokens_used = response.usage.total_tokens if response.usage else 0

        return {
            "text": response.choices[0].message.content.strip(),
            "tokens": tokens_used,
            "time": end_time - start_time
        }


# ---------------- Enhancer (not used for display) ---------------- #
class Enhancer:
    """Applies image enhancement techniques."""

    @staticmethod
    def coloured(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] *= 0.7
        hsv[:, :, 1] *= 1.5
        hsv[:, :, 2] *= 0.5
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def gaussian(img):
        return cv2.GaussianBlur(img, (7, 7), 0)

    @staticmethod
    def sharpen(img):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)

    @staticmethod
    def equalize(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)


# ---------------- Processor Base ---------------- #
class Processor:
    """Base processor with cascade decision logic."""

    def __init__(self, analyzer: GPTImageAnalyzer, config: Config):
        self.analyzer = analyzer
        self.config = config

    def _classify(self, image, base_name):
        """Run analysis and return decision and metrics."""
        _, buffer = cv2.imencode(".jpg", image)
        image_bytes = buffer.tobytes()

        result = self.analyzer.analyze(image_bytes)
        response_text = result["text"]
        tokens_used = result["tokens"]
        elapsed_time = result["time"]

        decision = "yes" if "yes" in response_text.lower() else "no"

        return decision, response_text, tokens_used, elapsed_time


# ---------------- Streamlit UI ---------------- #
def main():
    Config.init_dirs()
    analyzer = GPTImageAnalyzer(Config)
    processor = Processor(analyzer, Config)

    st.title("📄 Seal / Stamp / Signature Detection")
    uploaded_files = st.file_uploader(
        "Upload Images or PDFs", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        results = []

        with open(Config.LOG_FILE, "w", newline="", encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(["filename", "filetype", "detection", "time_sec"])

            for uploaded in uploaded_files:
                file_bytes = uploaded.read()
                filename = uploaded.name

                if filename.lower().endswith(".pdf"):
                    pages = convert_from_bytes(file_bytes)
                    for i, page in enumerate(pages, start=1):
                        buf = BytesIO()
                        page.save(buf, format="JPEG")
                        np_img = np.frombuffer(buf.getvalue(), np.uint8)
                        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

                        decision, _, _, elapsed_time = processor._classify(image, f"{filename}_page{i}")

                        # Print decision before image
                        if decision.lower() == "yes":
                            st.markdown(f"### ✅ Is Seal (Page {i}): Yes")
                        else:
                            st.markdown(f"### ❌ Is Seal (Page {i}): No")

                        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"{filename} (Page {i})")

                        results.append([filename, "pdf", decision, round(elapsed_time, 2)])
                        writer.writerow([filename, "pdf", decision, round(elapsed_time, 2)])

                else:
                    np_img = np.frombuffer(file_bytes, np.uint8)
                    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

                    decision, response, tokens, elapsed_time = processor._classify(image, filename)

                    # Print decision before image
                    if decision.lower() == "yes":
                        st.markdown(f"### ✅ Is Seal: Yes")
                    else:
                        st.markdown(f"### ❌ Is Seal: No")

                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"{filename}")

                    results.append([filename, "image", decision, round(elapsed_time, 2)])
                    writer.writerow([filename, "image", decision, round(elapsed_time, 2)])

        st.success("Classification complete. Log file generated.")

        df = pd.DataFrame(results, columns=["filename", "filetype", "detection", "time_sec"])
        st.dataframe(df)

        with open(Config.LOG_FILE, "r", encoding="utf-8") as f:
            st.download_button("Download Log CSV", f, file_name="classification_log.csv")


if __name__ == "__main__":
    main()
