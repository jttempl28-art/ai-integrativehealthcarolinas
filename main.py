from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import openai
import os
import tempfile
import shutil
import re
import unicodedata
from datetime import datetime
import pdfplumber
import json
import numpy as np

# Your redaction imports

# ----------------- Config -----------------
openai.api_key = os.environ.get("OPENAI_API_KEY")
EMBEDDINGS_FILE = "embeddings.npy"
CHUNKS_FILE = "chart_chunks.npy"
SYSTEM_PROMPT_FILE = "system_prompt.txt"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 5


counter = 1

street_suffixes = [
    "Street", "St", "Avenue", "Ave", "Road", "Rd", "Lane", "Ln",
    "Drive", "Dr", "Court", "Ct", "Circle", "Cir", "Boulevard", "Blvd",
    "Place", "Pl", "Terrace", "Ter", "Parkway", "Pkwy", "Way", "Trail", "Trl"
]
# ---------------- Your existing functions ----------------
# Place all your normalize_and_redact and helper functions here
# (generate_safe_filename, normalize_text, chunk_text, extract_text_columns, normalize_and_redact, etc.)
# Make sure you also define `counter`, `street_suffixes`, and `log` function if used
def generate_safe_filename():
    return f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.txt"

def normalize_text(text):
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    text = re.sub(r'[_–—]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, max_chars=100_000):
    chunk = []
    size = 0
    for line in text.splitlines(keepends=True):
        chunk.append(line)
        size += len(line)
        if size >= max_chars:
            yield "".join(chunk)
            chunk = []
            size = 0
    if chunk:
        yield "".join(chunk)

def extract_text_columns(pdf_path, y_tolerance=1, gap_threshold=40, right_start_threshold=300):
    pages_out = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            chars = sorted(page.chars, key=lambda c: (c["top"], c["x0"]))
            if not chars:
                pages_out.append(f"--- Page {page_num} ---\n")
                continue

            # Group chars into lines
            lines = []
            current_line = [chars[0]]
            current_y = chars[0]["top"]
            for c in chars[1:]:
                if abs(c["top"] - current_y) <= y_tolerance:
                    current_line.append(c)
                else:
                    lines.append(current_line)
                    current_line = [c]
                    current_y = c["top"]
            lines.append(current_line)

            left_lines, right_lines = [], []

            for line in lines:
                left_part, right_part = [], []
                line = sorted(line, key=lambda c: c["x0"])
                i = 0
                while i < len(line):
                    c = line[i]
                    if c["x0"] > right_start_threshold and not left_part:
                        right_part.extend(line[i:])
                        break
                    if left_part:
                        prev_c = left_part[-1]
                        gap = c["x0"] - prev_c["x1"]
                        if gap <= gap_threshold:
                            left_part.append(c)
                        else:
                            right_part.extend(line[i:])
                            break
                    else:
                        left_part.append(c)
                    i += 1
                if left_part:
                    left_lines.append("".join(ch["text"] for ch in left_part))
                if right_part:
                    right_lines.append("".join(ch["text"] for ch in right_part))

            def clean(lines):
                return "\n".join(normalize_text(l) for l in lines if l.strip())

            page_text = clean(left_lines)
            if right_lines:
                page_text += "\n\n" + clean(right_lines)
            pages_out.append(f"--- Page {page_num} ---\n{page_text}\n")
    return pages_out

def normalize_and_redact(pdf_path):
    global counter

    # 1️⃣ Normalize text
    pages = extract_text_columns(pdf_path)
    text = "".join(pages)

    # 2️⃣ Seed line & name redaction
    lines = text.splitlines()
    if len(lines) >= 2:
        seed_line = lines[1].replace('"', '').strip()
    else:
        seed_line = lines[0].replace('"', '').strip()

    words = [re.sub(r'[^a-zA-Z-]', '', w) for w in seed_line.split()]
    words = [w for w in words if w]
    concat_parts = re.findall(r'[A-Z][a-z]+', seed_line)
    all_parts = list(set(words + concat_parts))

    redacted_text = text
    for part in all_parts:
        pattern = r'\b' + re.escape(part) + r'\b' if len(part) == 1 else re.escape(part)
        redacted_text = re.sub(pattern, "[NAME]", redacted_text, flags=re.IGNORECASE)

    # 3️⃣ Address redaction
    suffix_pattern = r'(?:' + '|'.join(street_suffixes) + r')\.?'
    address_pattern = re.compile(
        r'(?<!\w)\d{1,5}\s+(?:[A-Z][A-Za-z]+(?:\s+)){1,4}'
        + suffix_pattern
        + r'(?:[,\s]+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)?'
        r'(?:[,\s]+[A-Z]{2})?'
        r'(?:\s+\d{5}(?:-\d{4})?)?'
        r'(?!\w)',
        flags=re.MULTILINE
    )
    redacted_text = re.sub(address_pattern, "[ADDRESS]", redacted_text)

    # 4️⃣ Age redaction
    AGE_MASK = "__AGE_DURATION__"
    age_duration_pattern = re.compile(
        r'\b\d+\s*(?:year|years|yr|yrs|y/o|yo|old|month|months|mo|mos|week|weeks|wk|wks|day|days)\b',
        flags=re.IGNORECASE
    )
    protected_ages = {}
    def protect_ages(match):
        key = f"{AGE_MASK}{len(protected_ages)}__"
        protected_ages[key] = match.group(0)
        return key
    redacted_text = age_duration_pattern.sub(protect_ages, redacted_text)

    # 5️⃣ PII Redaction via Presidio
    phone_pattern = r"""
    \b                              # Word boundary
    (?:\+?\d{1,3}[\s.-]?)?          # Optional country code
    (?:\(?\d{3}\)?[\s.-]?)?         # Optional area code with or without parentheses
    \d{3}                            # First 3 digits
    [\s.-]?                          # Separator (space, dot, dash)
    \d{4}                            # Last 4 digits
    \b
    """

    redacted_text = re.sub(phone_pattern, "[REDACTED_PHONE]", redacted_text, flags=re.VERBOSE)

    # Redact email addresses
    email_pattern = r"\b[\w\.-]+@[\w\.-]+\.\w+\b"
    redacted_text = re.sub(email_pattern, "[REDACTED_EMAIL]", redacted_text)
    date_pattern = r"""
    \b(
        \d{1,2}[/-]\d{1,2}[/-]\d{2,4} |      # 01/15/2026 or 1-5-26
        \d{4}[/-]\d{1,2}[/-]\d{1,2} |        # 2026-01-15
        \d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[.,]?\s\d{2,4} |  # 15 Jan 2026
        (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2}[,]?\s\d{2,4}   # January 15, 2026
    )\b
    """
    redacted_text = re.sub(date_pattern, "[REDACTED_DATE]", redacted_text, flags=re.IGNORECASE | re.VERBOSE)
    # Redact full and abbreviated months

    months_pattern = r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b"
    redacted_text = re.sub(months_pattern, "[REDACTED_MONTH]", redacted_text, flags=re.IGNORECASE)

    for key, value in protected_ages.items():
        redacted_text = redacted_text.replace(key, value)

    
    # 6️⃣ Long number redaction
    redacted_text = re.sub(r'\d{5,}', "[LONG_NUMBER]", redacted_text)

    counter += 1
    return redacted_text


# ---------------- Flask App ----------------

app = Flask(__name__)
CORS(app)


# ----------------- Routes -----------------
@app.route("/upload", methods=["POST"])
def upload_file():
    global chart_chunks, embeddings

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDFs supported"}), 400

    try:
        # 1️⃣ Read PDF and extract text
        pdf_bytes = file.read()
        pdf_stream = io.BytesIO(pdf_bytes)
        text = extract_text_columns(pdf_stream)

        # 2️⃣ Chunk text
        new_chunks = chunk_text(text)
        if not new_chunks:
            return jsonify({"error": "No text extracted"}), 400

        # 3️⃣ Generate embeddings
        new_embeddings = []
        for chunk in new_chunks:
            emb = openai.Embeddings.create(model=EMBEDDING_MODEL, input=chunk)
            new_embeddings.append(emb["data"][0]["embedding"])
        new_embeddings = np.array(new_embeddings)

        # 4️⃣ Append to existing arrays
        chart_chunks += new_chunks
        embeddings = np.vstack([embeddings, new_embeddings])

        # 5️⃣ Save back to files
        np.save(CHUNKS_FILE, chart_chunks, allow_pickle=True)
        np.save(EMBEDDINGS_FILE, embeddings)

        return jsonify({"message": f"File processed and {len(new_chunks)} chunks added."})

    except Exception as e:
        return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    global chart_chunks, embeddings, system_prompt

    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = data["message"]

    try:
        # 1️⃣ Generate embedding for query
        query_embed = openai.Embeddings.create(
            model=EMBEDDING_MODEL,
            input=user_message
        )["data"][0]["embedding"]
        query_vec = np.array(query_embed)

        # 2️⃣ Compute cosine similarity with all embeddings
        sims = np.array([cosine_similarity(query_vec, e) for e in embeddings])
        top_idx = sims.argsort()[-TOP_K:][::-1]
        context_chunks = [chart_chunks[i] for i in top_idx]

        # 3️⃣ Build messages
        context_text = "\n\n".join(context_chunks)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_message}"}
        ]

        # 4️⃣ Ask GPT
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        reply = completion.choices[0].message.content

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/set_prompt", methods=["POST"])
def set_prompt():
    global system_prompt
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "No prompt provided"}), 400
    system_prompt = data["prompt"]
    with open(SYSTEM_PROMPT_FILE, "w", encoding="utf-8") as f:
        f.write(system_prompt)
    return jsonify({"message": "System prompt updated."})

# ----------------- Run -----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)


