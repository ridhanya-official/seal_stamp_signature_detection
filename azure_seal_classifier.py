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
from pdf2image import convert_from_path
from openai import AzureOpenAI
from prompt import prompt_v2
from dotenv import load_dotenv
load_dotenv()

# ---------------- Config ---------------- #
class Config:
    """Holds configuration constants for the pipeline."""

    # ---- Root Settings ----
    ROOT_DIR = os.getenv("ROOT_DIR", "temp_output")
    INPUT_PATH = os.getenv("INPUT_PATH", "image.jpg")

    # ---- Derived Paths ----
    OUTPUT_YES = os.path.join(ROOT_DIR, "yes")
    OUTPUT_NO = os.path.join(ROOT_DIR, "no")
    LOG_FILE = os.path.join(ROOT_DIR, f"classification_log.csv")
    ENHANCED_DIR = os.path.join(ROOT_DIR, "enhanced_images")

    # ---- Azure OpenAI configuration ----
    API_KEY = os.getenv("AZURE_API_KEY")
    ENDPOINT = os.getenv("AZURE_ENDPOINT")
    API_VERSION = os.getenv("AZURE_API_VERSION")
    MODEL_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")

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

    def extract_specification(self, image_bytes: bytes) -> dict:
        """Get detailed specification if a seal/stamp/signature is detected."""
        encoded_image = base64.b64encode(image_bytes).decode("ascii")

        prompt = (
            "The previous classification detected a seal, stamp, or handwritten signature.\n"
            "Please provide:\n"
            "1. Type: seal / stamp / handwritten signature\n"
            "2. Brief description of its appearance\n"
            "3. Describe the location in the image\n"
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
                        {"type": "text", "text": prompt_v2 + prompt},
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
        """Run cascade of enhancements until 'yes' is found, log time and tokens."""
        steps = [
            ("original", lambda x: x),
            ("coloured", Enhancer.coloured),
            ("sharpened", Enhancer.sharpen),
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

        # fallback equalized if none detected
        if final_decision == "no":
            gaussian_image = Enhancer.gaussian(current_image)
            equalized_image = Enhancer.equalize(gaussian_image)
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

        # Second-level specification extraction if "yes"
        if final_decision == "yes":
            _, buffer = cv2.imencode(".jpg", current_image)
            spec_result = self.analyzer.extract_specification(buffer.tobytes())
            spec_text = spec_result["specification"]
            total_tokens += spec_result["tokens"]
            total_time += spec_result["time"]
        else:
            spec_text = "No seals, stamps, or handwritten signatures on the document were detected."

        return final_decision, response_text, step_used, total_time, total_tokens, spec_text


# ---------------- Image Processor ---------------- #
class ImageProcessor(Processor):
    """Processes single image files."""

    def process(self, file_path, writer):
        image = cv2.imread(file_path)
        if image is None:
            print(f"Cannot read image: {file_path}")
            return

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        decision, response, step_used, elapsed_time, tokens_used, spec_text = self._classify(
            image, base_name, "image"
        )

        target = self.config.OUTPUT_YES if decision == "yes" else self.config.OUTPUT_NO
        shutil.copy(file_path, os.path.join(target, os.path.basename(file_path)))

        writer.writerow([
            os.path.basename(file_path),
            "image",
            "-",
            decision,
            step_used,
            response,
            round(elapsed_time, 2),
            tokens_used,
            spec_text
        ])
        print(f"{file_path} → {decision.upper()} ({step_used}) | Time: {elapsed_time:.2f}s | Tokens: {tokens_used}")


# ---------------- PDF Processor ---------------- #
class PDFProcessor(Processor):
    """Processes PDF files page by page."""

    def process(self, file_path, writer, progress_callback=None):
        pages = convert_from_path(file_path, dpi=220)

        for i, page in enumerate(pages, start=1):
            buffer = BytesIO()
            page.save(buffer, format="JPEG")
            np_img = np.frombuffer(buffer.getvalue(), np.uint8)
            image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            if image is None:
                print(f"Cannot read page {i} of {file_path}")
                continue

            base_name = f"{os.path.splitext(os.path.basename(file_path))[0]}_page{i}"
            decision, response, step_used, elapsed_time, tokens_used, spec_text = self._classify(
                image, base_name, "pdf", page=i
            )

            target = self.config.OUTPUT_YES if decision == "yes" else self.config.OUTPUT_NO
            out_name = f"{base_name}.jpg"
            cv2.imwrite(os.path.join(target, out_name), image)

            writer.writerow([
                os.path.basename(file_path),
                "pdf",
                i,
                decision,
                step_used,
                response,
                round(elapsed_time, 2),
                tokens_used,
                spec_text
            ])
            print(f"{file_path} (Page {i}) → {decision.upper()} ({step_used}) | Time: {elapsed_time:.2f}s | Tokens: {tokens_used}")

            if progress_callback:
                progress_callback()

# ---------------- Main Application ---------------- #
class ClassifierApp:
    """Main driver for classification workflow."""

    def __init__(self, config: Config):
        Config.init_dirs()
        self.analyzer = GPTImageAnalyzer(config)
        self.img_processor = ImageProcessor(self.analyzer, config)
        self.pdf_processor = PDFProcessor(self.analyzer, config)
        self.config = config

    def run(self):
        with open(self.config.LOG_FILE, "w", newline="", encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([
                "filename", "filetype", "page", "decision", "step_used",
                "response_text", "time_taken_sec", "tokens_used", "specification"
            ])

            if os.path.isdir(self.config.INPUT_PATH):
                for filename in tqdm(os.listdir(self.config.INPUT_PATH)):
                    path = os.path.join(self.config.INPUT_PATH, filename)
                    self._process_file(path, writer)
            elif os.path.isfile(self.config.INPUT_PATH):
                self._process_file(self.config.INPUT_PATH, writer)
            else:
                print("Invalid INPUT_PATH. Must be a folder or a file.")

    def _process_file(self, file_path, writer, progress_callback=None):
        if file_path.lower().endswith((".jpg", ".jpeg", ".png")):
            self.img_processor.process(file_path, writer)
            if progress_callback:
                progress_callback()
        elif file_path.lower().endswith(".pdf"):
            self.pdf_processor.process(file_path, writer, progress_callback)
        else:
            print(f"Skipping unsupported file: {file_path}")



# ---------------- Run ---------------- #
if __name__ == "__main__":
    app = ClassifierApp(Config)
    app.run()
