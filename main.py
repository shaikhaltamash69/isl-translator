# =========================
# Fix for macOS OpenMP / Stanza pthread crash
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STANZA_USE_TOKENIZER"] = "spacy"
# =========================

# Now import the rest
from flask import Flask, request, jsonify, render_template, send_from_directory
from faster_whisper import WhisperModel
import stanza
import logging
from flask import url_for
import uuid
from datetime import datetime
import zipfile
import sys
import time
import ssl
import pprint
import re


ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Directory settings
BASE_PATH = os.path.dirname(os.path.realpath(__file__))
SIGN_FILES_DIR = os.path.join(BASE_PATH, 'static', 'SignFiles')
UPLOAD_DIR = os.path.join(BASE_PATH, 'static', 'uploads')

# Load Whisper model for transcription
model = WhisperModel("small")

# Set up Stanza
stanza.download('en', model_dir='stanza_resources')
en_nlp = stanza.Pipeline('en', processors={'tokenize': 'spacy'})

# Allowed extensions
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'mp3', 'wav', 'avi', 'mkv', 'm4a', 'aac', 'flac'}

# Enhanced ISL processing variables
stop_words = set(["am","are","is","was","were","be","being","been","have","has","had",
                  "does","did","could","should","would","can","shall","will","may","might","must","let"])

# Additional words to remove - be more selective for sign language
additional_stop_words = set(["a", "an", "the", "in", "on", "at", "for",
                           "of", "with", "by", "from", "up", "about", "into", "through", "during"])

# Keep some connecting words that are important for meaning in sign language
important_connectors = set(["to", "and", "or", "but", "where", "we", "our"])

# Combine stop words but exclude important connectors
all_stop_words = stop_words.union(additional_stop_words) - important_connectors

# Global processing lists
final_words = []
final_output_in_sent = []
sent_list = []
sent_list_detailed = []
word_list = []
word_list_detailed = []
final_words_detailed = []
final_words_dict = {}

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if allowed_file(file.filename):
        # Generate unique filename to avoid conflicts
        original_filename = file.filename
        file_extension = original_filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        file.save(file_path)
        logging.info(f"File uploaded: {original_filename} -> {unique_filename}")

        try:
            # Transcribe audio
            segments, info = model.transcribe(file_path)
            text = " ".join([seg.text for seg in segments]).strip()

            if not text:
                # Clean up file if transcription fails
                os.remove(file_path)
                return jsonify({'error': 'No transcription available.'}), 500

            logging.info(f"Transcription: {text}")

            # Convert text to ISL using enhanced logic
            isl_text, sigml_files = convert_to_isl(text)
            flat_sigml_files = [file for sentence_files in sigml_files for file in sentence_files]

            # Generate media URL for the uploaded file
            media_url = url_for('static', filename=f'uploads/{unique_filename}')

            # Render the results page with media information
            return render_template('index.html', 
                                 text=text, 
                                 isl_text=isl_text, 
                                 flat_sigml_files=flat_sigml_files,
                                 media_url=media_url,
                                 filename=original_filename,
                                 unique_filename=unique_filename)
                                 
        except Exception as e:
            logging.error(f"Error processing file: {e}")
            # Clean up file if processing fails
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': 'Error processing file.'}), 500
    else:
        return jsonify({'error': 'Invalid file format. Supported formats: mp4, mov, mp3, wav, avi, mkv, m4a, aac, flac'}), 400

@app.route('/cleanup/<filename>')
def cleanup_file(filename):
    """Optional: Endpoint to clean up uploaded files after use"""
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({'success': 'File cleaned up'}), 200
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        logging.error(f"Error cleaning up file: {e}")
        return jsonify({'error': 'Error cleaning up file'}), 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Enhanced word processing functions
def remove_suffixes(word):
    """Remove common English suffixes like -ed, -ing, -ly, -er, -est, -s"""
    word = word.lower().strip()
    
    # List of suffixes to remove (ordered by length to avoid issues)
    suffixes = ['ing', 'ed', 'ly', 'er', 'est', 's']
    
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:  # Keep minimum word length
            # Special handling for some cases
            if suffix == 'ed':
                if word.endswith('ied'):
                    return word[:-3] + 'y'  # tried -> try
                elif word.endswith('ed'):
                    return word[:-2]
            elif suffix == 'ing':
                if word.endswith('ing'):
                    base = word[:-3]
                    # Handle doubling cases (running -> run)
                    if len(base) >= 2 and base[-1] == base[-2] and base[-1] in 'bcdfghjklmnpqrstvwxyz':
                        return base[:-1]
                    return base
            elif suffix == 's' and not word.endswith('ss'):
                if word.endswith('ies'):
                    return word[:-3] + 'y'  # flies -> fly
                elif word.endswith('es'):
                    return word[:-2]
                elif word.endswith('s'):
                    return word[:-1]
            else:
                return word[:-len(suffix)]
    
    return word

def remove_punctuation(text):
    """Remove punctuation from text while preserving word structure"""
    # Remove common punctuation but keep apostrophes in contractions
    text = re.sub(r'[^\w\s\']', ' ', text)
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def filter_and_process_words(word_list_input):
    """Enhanced word filtering with stop word removal and suffix handling"""
    processed_sentences = []
    
    for sentence_words in word_list_input:
        processed_words = []
        
        for word in sentence_words:
            # Convert to lowercase and remove punctuation
            cleaned_word = remove_punctuation(word.lower())
            
            # Skip empty words
            if not cleaned_word:
                continue
                
            # Remove suffixes
            processed_word = remove_suffixes(cleaned_word)
            
            # Skip stop words
            if processed_word not in all_stop_words and len(processed_word) > 1:
                processed_words.append(processed_word)
        
        processed_sentences.append(processed_words)
    
    return processed_sentences

# NEW IMPROVED ISL REORDERING FUNCTIONS
def simple_isl_reorder(words):
    """
    Reorders English words to follow ISL (Indian Sign Language) grammar structure.
    ISL follows: TIME + TOPIC + SUBJECT + OBJECT + VERB + QUESTION/EMOTION
    """
    if not words or len(words) <= 2:
        return words
    
    # Initialize categories
    time_words = []
    topic_words = []
    subject_words = []
    object_words = []
    verb_words = []
    question_words = []
    greeting_words = []
    location_words = []
    remaining_words = []
    
    # Define word categories for ISL
    time_indicators = {'today', 'tomorrow', 'yesterday', 'now', 'then', 'when', 'time', 
                      'morning', 'evening', 'night', 'day', 'week', 'month', 'year'}
    
    greetings = {'hello', 'hi', 'welcome', 'good', 'bye', 'thanks', 'thank'}
    
    question_indicators = {'what', 'where', 'when', 'why', 'how', 'who', 'which', 'can', 'do'}
    
    location_indicators = {'here', 'there', 'where', 'place', 'home', 'school', 'office'}
    
    # Common verbs - action words typically come at the end in ISL
    common_verbs = {'help', 'go', 'come', 'eat', 'drink', 'work', 'play', 'learn', 
                   'teach', 'read', 'write', 'see', 'hear', 'speak', 'sign'}
    
    # Topic/subject indicators
    topic_indicators = {'we', 'our', 'sign', 'engine', 'system', 'app'}
    
    # Process each word
    for i, word in enumerate(words):
        word_lower = word.lower()
        
        # Categorize words based on ISL grammar
        if word_lower in greetings:
            greeting_words.append(word)
        elif word_lower in time_indicators:
            time_words.append(word)
        elif word_lower in question_indicators:
            question_words.append(word)
        elif word_lower in location_indicators:
            location_words.append(word)
        elif word_lower in common_verbs:
            verb_words.append(word)
        elif word_lower in topic_indicators or word_lower in {'deaf', 'people', 'person'}:
            if word_lower in {'we', 'our', 'i', 'you'}:
                subject_words.append(word)
            else:
                topic_words.append(word)
        else:
            # Determine context-based placement
            if i < len(words) // 2:  # First half - likely topic/subject
                if word_lower in {'deaf', 'people', 'person', 'student', 'teacher'}:
                    object_words.append(word)
                else:
                    topic_words.append(word)
            else:  # Second half - likely object/remaining
                remaining_words.append(word)
    
    # ISL sentence structure: GREETING + TIME + TOPIC + SUBJECT + OBJECT + LOCATION + VERB + QUESTION
    reordered = []
    
    # Add greetings first (very important in ISL)
    reordered.extend(greeting_words)
    
    # Add time references
    reordered.extend(time_words)
    
    # Add topic/theme of conversation
    reordered.extend(topic_words)
    
    # Add subject (who is doing)
    reordered.extend(subject_words)
    
    # Add object (who/what is being acted upon)
    reordered.extend(object_words)
    
    # Add location
    reordered.extend(location_words)
    
    # Add remaining words
    reordered.extend(remaining_words)
    
    # Add verbs at the end (ISL is typically verb-final)
    reordered.extend(verb_words)
    
    # Add questions/emotions at the very end
    reordered.extend(question_words)
    
    return reordered if reordered else words

def advanced_isl_reorder(words):
    """
    Advanced ISL reordering with better context understanding
    """
    if not words or len(words) <= 1:
        return words
    
    # Join words to analyze as a sentence for better context
    sentence = ' '.join(words).lower()
    
    # Special patterns for common ISL structures
    if 'hello' in sentence and 'welcome' in sentence:
        # Greeting pattern: Put greetings first
        greeting_first = []
        topic_middle = []
        action_end = []
        
        for word in words:
            if word.lower() in ['hello', 'hi', 'welcome']:
                greeting_first.append(word)
            elif word.lower() in ['help', 'go', 'come', 'work']:
                action_end.append(word)
            else:
                topic_middle.append(word)
        
        return greeting_first + topic_middle + action_end
    
    # For questions: Question word + Topic + Subject + Object + Verb
    elif any(q in sentence for q in ['what', 'where', 'who', 'how', 'why']):
        question_words = [w for w in words if w.lower() in ['what', 'where', 'who', 'how', 'why']]
        other_words = [w for w in words if w.lower() not in ['what', 'where', 'who', 'how', 'why']]
        return question_words + simple_isl_reorder(other_words)
    
    # Default to simple reordering
    return simple_isl_reorder(words)

# UPDATED: Replaced Stanford Parser with ISL-specific reordering
def reorder_eng_to_isl(input_string):
    """
    Improved ISL reordering function without Stanford Parser dependency
    """
    # Check if all words are single letters (alphabets)
    if isinstance(input_string, str):
        input_string = input_string.split()
    
    count = 0
    for word in input_string:
        if len(word) == 1:
            count += 1
    
    if count == len(input_string):
        return input_string
    
    # Use our ISL-specific reordering instead of Stanford Parser
    try:
        # First try advanced reordering
        reordered = advanced_isl_reorder(input_string)
        logging.info(f"Original: {input_string}")
        logging.info(f"ISL Reordered: {reordered}")
        return reordered
    except Exception as e:
        logging.error(f"Reordering error: {e}")
        # Fallback to simple reordering
        return simple_isl_reorder(input_string)

# REMOVED: All Stanford Parser related functions (no longer needed)
# - label_parse_subtrees
# - handle_noun_clause
# - handle_verb_prop_clause
# - modify_tree_structure
# - apply_basic_isl_rules (replaced with better functions above)

# Enhanced ISL conversion functions
def convert_to_isl(text):
    clear_all()
    take_input(text)
    sigml_files = map_to_sigml_files(final_output_in_sent)
    return final_output_in_sent, sigml_files

def take_input(text):
    # Clean and preprocess text
    test_input = remove_punctuation(text.strip()).replace("\n", "").replace("\t", "")
    test_input2 = ""
    
    if len(test_input) == 1:
        test_input2 = test_input
    else:
        # Split by periods and capitalize
        for word in test_input.split("."):
            if word.strip():
                test_input2 += word.capitalize() + " ."
    
    # Pass the text through stanza
    some_text = en_nlp(test_input2)
    convert(some_text)

def convert(some_text):
    convert_to_sentence_list(some_text)
    convert_to_word_list(sent_list_detailed)
    
    # Apply enhanced word filtering and processing
    processed_word_list = filter_and_process_words(word_list)
    
    # Reorders the words in input using improved ISL logic
    for i, words in enumerate(processed_word_list):
        if words:  # Only process non-empty word lists
            processed_word_list[i] = reorder_eng_to_isl(words)
    
    # Update the final processing
    final_words.extend(processed_word_list)
    convert_to_final()
    print_lists()

def convert_to_sentence_list(text):
    for sentence in text.sentences:
        sent_list.append(sentence.text)
        sent_list_detailed.append(sentence)

def convert_to_word_list(sentences):
    temp_list = []
    temp_list_detailed = []
    for sentence in sentences:
        for word in sentence.words:
            temp_list.append(word.text)
            temp_list_detailed.append(word)
        word_list.append(temp_list.copy())
        word_list_detailed.append(temp_list_detailed.copy())
        temp_list.clear()
        temp_list_detailed.clear()

def final_output(input_words):
    """Process final words and handle missing sigml files"""
    words_file_path = os.path.join(BASE_PATH, "words.txt")
    if os.path.exists(words_file_path):
        valid_words = open(words_file_path, 'r').read().split('\n')
    else:
        valid_words = []
    
    fin_words = []
    for word in input_words:
        word = word.lower().strip()
        if not word:
            continue
            
        # Check if sigml file exists for the word
        sigml_path = os.path.join(SIGN_FILES_DIR, f"{word}.sigml")
        if not os.path.exists(sigml_path) and word not in valid_words:
            # If no sigml file exists, use letters
            for letter in word:
                if letter.isalpha():
                    fin_words.append(letter.lower())
        else:
            fin_words.append(word)

    return fin_words

def convert_to_final():
    for words in final_words:
        if words:  # Only process non-empty word lists
            final_output_in_sent.append(final_output(words))

def print_lists():
    print("--------------------Word List------------------------")
    pprint.pprint(word_list)
    print("--------------------Final Words------------------------")
    pprint.pprint(final_words)
    print("---------------Final sentence with letters--------------")
    pprint.pprint(final_output_in_sent)

def map_to_sigml_files(isl_text_list):
    sigml_file_urls = []
    for sentence in isl_text_list:
        sentence_files = []
        for word in sentence:
            if not word:
                continue
                
            sigml_file = f"{word.lower()}.sigml"
            sigml_path = os.path.join(SIGN_FILES_DIR, sigml_file)
            
            if os.path.exists(sigml_path):
                sigml_url = url_for('static', filename=f'SignFiles/{sigml_file}')
                sentence_files.append(sigml_url)
            else:
                # Use character-by-character spelling
                spelling_files = []
                for char in word.lower():
                    if char.isalpha():
                        char_sigml = f"{char}.sigml"
                        char_sigml_path = os.path.join(SIGN_FILES_DIR, char_sigml)
                        if os.path.exists(char_sigml_path):
                            char_sigml_url = url_for('static', filename=f'SignFiles/{char_sigml}')
                            spelling_files.append(char_sigml_url)
                sentence_files.extend(spelling_files)
        sigml_file_urls.append(sentence_files)
    return sigml_file_urls

def clear_all():
    final_words.clear()
    final_output_in_sent.clear()
    sent_list.clear()
    sent_list_detailed.clear()
    word_list.clear()
    word_list_detailed.clear()
    final_words_detailed.clear()
    final_words_dict.clear()

# Serve static files from custom directory
@app.route('/jas/loc2021/cwa/<path:filename>')
def serve_jas_files(filename):
    jas_dir = os.path.join(BASE_PATH, 'jas', 'loc2021', 'cwa')
    return send_from_directory(jas_dir, filename)

@app.route('/static/<path:path>')
def serve_signfiles(path):
    return send_from_directory('static', path)

def cleanup_old_files(max_age_hours=24):
    """Clean up uploaded files older than max_age_hours"""
    import time
    current_time = time.time()
    
    for filename in os.listdir(UPLOAD_DIR):
        if filename == '.gitkeep':
            continue
            
        file_path = os.path.join(UPLOAD_DIR, filename)
        file_age_hours = (current_time - os.path.getctime(file_path)) / 3600
        
        if file_age_hours > max_age_hours:
            try:
                os.remove(file_path)
                logging.info(f"Cleaned up old file: {filename}")
            except Exception as e:
                logging.error(f"Error cleaning up {filename}: {e}")

# Test function for ISL reordering (optional - for debugging)
def test_isl_reordering():
    """Test function to demonstrate ISL reordering"""
    
    test_sentences = [
        ["Hello", "welcome", "our", "sign", "engine", "we", "help", "deaf", "people", "where"],
        ["What", "is", "your", "name"],
        ["I", "am", "learning", "sign", "language"],
        ["Good", "morning", "how", "are", "you"],
        ["We", "work", "together", "tomorrow"]
    ]
    
    print("=== ISL Reordering Test ===\n")
    
    for sentence in test_sentences:
        original = " ".join(sentence)
        reordered = advanced_isl_reorder(句子)
        reordered_text = " ".join(reordered)
        
        print(f"English: {original}")
        print(f"ISL:     {reordered_text}")
        print("-" * 50)

if __name__ == '__main__':
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(SIGN_FILES_DIR, exist_ok=True)
    
    # Optional: Run ISL reordering test
    # test_isl_reordering()
    
    app.run(debug=True)