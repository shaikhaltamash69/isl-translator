# Sign Language Live Translation

Sign Language Live Translation is a Flask-based web application that converts spoken audio or video content into Indian Sign Language (ISL) animations. The system uses a Whisper-based transcription engine, linguistic processing with Stanza, and a sign avatar that plays SiGML files generated from the processed text.

---

## Project Goals

- Enable real-time or offline conversion of audio and video into ISL to improve accessibility for Deaf and Hard of Hearing users
- Build a modular pipeline separating transcription, text-to-ISL conversion, and avatar rendering so each stage can be improved independently
- Continuously improve ISL grammar handling, model quality, and user experience while keeping the repository production-ready

---

## Tech Stack

- **Backend**: Python, Flask (`main.py`)
- **Speech-to-Text**: `faster-whisper` (small model)
- **NLP & Grammar Processing**:
  - Stanza (dependency parsing, POS tagging)
  - spaCy (tokenization)
  - Custom ISL grammar rules (reordering, stop-word filtering, suffix handling)
- **Frontend**: HTML templates with CSS and JavaScript
- **Assets**:
  - SiGML sign files for avatar animation
  - Temporary media uploads stored locally

---

## Core Features

- Upload support for common audio and video formats:
  `mp4`, `mov`, `mp3`, `wav`, `avi`, `mkv`, `m4a`, `aac`, `flac`
- Secure upload handling with unique filenames and server-side validation
- Automatic speech transcription using Whisper
- Text-to-ISL conversion engine that:
  - Cleans punctuation and normalizes words
  - Removes non-essential stop words while preserving ISL semantics
  - Reorders words into ISL-friendly grammar structure
  - Maps processed words to corresponding SiGML sign files

---

## Repository Structure

```text
isl-translator/
│
├── main.py
├── requirements.txt
├── stanza_resources/
│
├── templates/
│   ├── upload.html
│   └── index.html
│
├── static/
│   ├── css/
│   ├── js/
│   ├── uploads/
│   └── SignFiles/
```

* **main.py**: Flask app, transcription pipeline, ISL conversion logic
* **templates/**: Frontend HTML pages
* **static/**: CSS, JavaScript, uploads, and SiGML sign files
* **stanza_resources/**: Local Stanza language models
* **requirements.txt**: Python dependencies

---

## Local Development Setup

> Python 3.9 – 3.11 recommended

### 1. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # macOS / Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download required language resources

#### Stanza English models

```bash
python -c "import stanza; stanza.download('en')"
```

#### spaCy English model

```bash
python -m spacy download en_core_web_sm
```

> The first run may take longer as models are downloaded and cached locally.

### 4. Run the application

```bash
python main.py
```

The Flask app runs at:
`http://127.0.0.1:5000`

On first run, **faster-whisper** will automatically download the speech model from Hugging Face. This may take a few minutes.

---

## Dependency Compatibility Notes

* This project requires:

  ```
  numpy==1.25.x
  ```
* Newer NumPy `2.x` versions may cause runtime errors with **PyTorch** and **CTranslate2**
* If you encounter crashes, reinstall dependencies strictly using `requirements.txt`

---

## Performance Notes

* The application runs on **CPU by default**
* Transcription speed improves significantly with a **CUDA-enabled GPU**
* GPU support is optional and not required for correctness

---

## Roadmap

* Improve ISL grammar rules and phrase-level reordering
* Add phrase-level and contextual signing
* Experiment with larger or domain-specific Whisper models
* Add real-time streaming support for live events
* Integrate feedback and evaluation from the Deaf community