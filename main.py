from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import openai
import os
import tempfile
import shutil

# Your redaction imports
import pdfplumber, re, unicodedata
from datetime import datetime

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

openai.api_key = os.environ.get("OPENAI_API_KEY")

# GPT Chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "No message provided"}), 400
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": message}]
        )
        return jsonify({"reply": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# File Upload & Redaction
@app.route("/upload", methods=["POST"])
def upload_file():
    uploaded_file = request.files.get("file")
    if not uploaded_file:
        return jsonify({"error": "No file uploaded"}), 400

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, uploaded_file.filename)
    uploaded_file.save(input_path)

    # Run your redaction code
    try:
        redacted_text = normalize_and_redact(input_path)
        redacted_filename = f"redacted_{uploaded_file.filename.rsplit('.', 1)[0]}.txt"
        redacted_path = os.path.join(temp_dir, redacted_filename)
        with open(redacted_path, "w", encoding="utf-8") as f:
            f.write(redacted_text)
    except Exception as e:
        shutil.rmtree(temp_dir)
        return jsonify({"error": str(e)}), 500

    # Return redacted file
    response = send_file(redacted_path, as_attachment=True)
    shutil.rmtree(temp_dir, ignore_errors=True)
    return response

# ---------------- Main ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
