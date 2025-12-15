# Sign Language Live Translation

Sign Language Live Translation is a Flask-based web application that converts spoken audio or video content into Indian Sign Language (ISL) animations. The system uses a Whisper-based transcription engine, linguistic processing with Stanza, and a sign avatar that plays SiGML files generated from the processed text.

## Project goals

- Enable real-time or offline conversion of audio and video into ISL to improve accessibility for Deaf and Hard of Hearing users.
- Build a modular pipeline that separates transcription, text-to-ISL conversion, and avatar rendering so each stage can be improved independently.
- Iterate daily on model quality, ISL grammar handling, and user experience while keeping the repository production ready.

## Tech stack

- Backend: Python, Flask application in `main.py` exposing upload and processing routes.
- Speech-to-text: `faster-whisper` small model for efficient audio and video transcription.
- NLP and grammar: Stanza pipeline (with spaCy tokenizer) plus custom ISL reordering, stop-word filtering, and suffix handling.
- Frontend: HTML templates in `templates/` with a media player, transcription view, and ISL output panel styled via static assets under `static/`.
- Assets: Pre-generated SiGML sign files stored in `static/SignFiles`, with user uploads stored in `static/uploads`.

## Core features

- File upload endpoint that accepts common audio and video formats (`mp4`, `mov`, `mp3`, `wav`, `avi`, `mkv`, `m4a`, `aac`, `flac`).
- Robust upload handling with unique filenames, server-side validation, and optional cleanup endpoint for temporary media.
- Transcription pipeline that joins Whisper segments into a single text transcript for each uploaded file.
- Text-to-ISL engine that:
  - Cleans punctuation and normalizes word forms.
  - Removes non-essential stop words while preserving connectors important for sign language semantics.
  - Reorders tokens into an ISL-friendly structure (time, topic, subject, object, verb, questions/emotions).
  - Maps processed words to SiGML sign files used by the avatar.

## Repository structure

- `main.py`: Flask application, transcription pipeline, ISL conversion logic, and file management utilities.
- `templates/`:
  - `upload.html`: Landing page for uploading audio or video.
  - `index.html`: Results page showing the original media, transcript, ISL format, and synchronized avatar controls.
- `static/`:
  - `css/`, `js/`: Frontend styling and client-side scripts.
  - `SignFiles/`: SiGML files for individual signs and phrases.
  - `uploads/`: Temporary directory for user-uploaded media.
- `stanza_resources/`: Local Stanza model directory for English processing.
- `requirements.txt`: Python dependencies including Flask, faster-whisper, and Stanza.   

## Local development

1. Create and activate a virtual environment.
2. Install dependencies:  
   `pip install -r requirements.txt`   
3. Ensure `stanza_resources/` and `static/SignFiles` exist or are populated on first run.
4. Start the server:  
   `python main.py` (Flask app runs on `http://127.0.0.1:5000`).
5. Open `/` to upload a media file and view the generated transcript and ISL animation.

## Roadmap

- Improve ISL grammar rules and phrase-level reordering for more natural signing.
- Extend language support and experiment with larger or domain-specific Whisper models.
- Add real-time streaming mode for live events and integrate evaluation metrics with Deaf community feedback.

