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
from pdf2image import convert_from_path
from openai import AzureOpenAI
from prompt import prompt_v2
from dotenv import load_dotenv
import streamlit as st

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

    def extract_specification(self, image_bytes: bytes) -> dict:
        encoded_image = base64.b64encode(image_bytes).decode("ascii")
        prompt = (
            "The previous classification detected a seal, stamp, or handwritten signature.\n"
            "Please provide:\n"
            "1. Type: seal / stamp / handwritten signature\n"
            "2. Brief description of its appearance\n"
            "3. Approximate location in the image (e.g., top-right corner)\n"
        )
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=500,
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
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        )
        end_time = time.time()
        tokens_used = response.usage.total_tokens if response.usage else 0
        return {
            "specification": response.choices[0].message.content.strip(),
            "tokens": tokens_used,
            "time": end_time - start_time
        }


# ---------------- Enhancer ---------------- #
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

    def _classify(self, image, base_name, filetype, page="-"):
        steps = [
            ("original", lambda x: x),
            ("coloured", Enhancer.coloured),
            ("sharpened", Enhancer.sharpen),
            ("gaussian", Enhancer.gaussian),
        ]

        final_decision = "no"
        response_text = ""
        step_used = "none"
        total_tokens = 0
        total_time = 0.0
        current_image = image

        for step_name, func in steps:
            enhanced_image = func(current_image)
            _, buffer = cv2.imencode(".jpg", enhanced_image)
            image_bytes = buffer.tobytes()
            result = self.analyzer.analyze(image_bytes)
            response_text = result["text"]
            total_tokens += result["tokens"]
            total_time += result["time"]
            if response_text and "yes" in response_text.lower():
                final_decision = "yes"
                step_used = step_name
                out_name = f"{base_name}_{step_name}.jpg"
                cv2.imwrite(os.path.join(self.config.ENHANCED_DIR, out_name), enhanced_image)
                current_image = enhanced_image
                break
            current_image = enhanced_image

        if final_decision == "no":
            equalized_image = Enhancer.equalize(current_image)
            _, buffer = cv2.imencode(".jpg", equalized_image)
            result = self.analyzer.analyze(buffer.tobytes())
            response_text = result["text"]
            total_tokens += result["tokens"]
            total_time += result["time"]
            if response_text and "yes" in response_text.lower():
                final_decision = "yes"
            step_used = "equalized"
            out_name = f"{base_name}_equalized.jpg"
            cv2.imwrite(os.path.join(self.config.ENHANCED_DIR, out_name), equalized_image)
            current_image = equalized_image

        if final_decision == "yes":
            _, buffer = cv2.imencode(".jpg", current_image)
            spec_result = self.analyzer.extract_specification(buffer.tobytes())
            spec_text = spec_result["specification"]
            total_tokens += spec_result["tokens"]
            total_time += spec_result["time"]
        else:
            spec_text = "There was no detection of seal, stamp, or handwritten signature."

        return final_decision, response_text, step_used, total_time, total_tokens, spec_text


# ---------------- Streamlit Application ---------------- #
st.set_page_config(page_title="Seal/Stamp/Signature Detection", layout="wide")
st.title("📄 Seal/Stamp/Signature Detection")

Config.init_dirs()
analyzer = GPTImageAnalyzer(Config)
processor = Processor(analyzer, Config)

uploaded_file = st.file_uploader("Upload an image or PDF", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name
    file_ext = filename.split(".")[-1].lower()

    csv_buffer = BytesIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["filename", "filetype", "page", "decision", "step_used", "response_text", "time_taken_sec", "tokens_used", "specification"])

    if file_ext in ["jpg", "jpeg", "png"]:
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        decision, response, step_used, elapsed_time, tokens_used, spec_text, enhanced_image = processor._classify(image, os.path.splitext(filename)[0])

        # Print decision
        if decision.lower() == "yes":
            st.markdown(f"### ✅ Is Seal: Yes")
        else:
            st.markdown(f"### ❌ Is Seal: No")

        # Resize image for display
        h, w = enhanced_image.shape[:2]
        max_width = 600
        if w > max_width:
            scale = max_width / w
            enhanced_image = cv2.resize(enhanced_image, (int(w * scale), int(h * scale)))

        st.image(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB), caption=f"{filename}")

        writer.writerow([filename, "image", "-", decision, step_used, response, round(elapsed_time, 2), tokens_used, spec_text])

    elif file_ext == "pdf":
        from pdf2image import convert_from_bytes
        pages = convert_from_bytes(file_bytes)
        for i, page in enumerate(pages, start=1):
            buffer = BytesIO()
            page.save(buffer, format="JPEG")
            np_img = np.frombuffer(buffer.getvalue(), np.uint8)
            image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            decision, response, step_used, elapsed_time, tokens_used, spec_text, enhanced_image = processor._classify(image, f"{os.path.splitext(filename)[0]}_page{i}")

            if decision.lower() == "yes":
                st.markdown(f"### ✅ Page {i} - Is Seal: Yes")
            else:
                st.markdown(f"### ❌ Page {i} - Is Seal: No")

            h, w = enhanced_image.shape[:2]
            max_width = 600
            if w > max_width:
                scale = max_width / w
                enhanced_image = cv2.resize(enhanced_image, (int(w * scale), int(h * scale)))
            st.image(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB), caption=f"Page {i}")

            writer.writerow([filename, "pdf", i, decision, step_used, response, round(elapsed_time, 2), tokens_used, spec_text])

    # Download CSV
    csv_buffer.seek(0)
    st.download_button(
        label="📥 Download Classification CSV",
        data=csv_buffer.getvalue(),
        file_name="classification_log.csv",
        mime="text/csv"
    )
