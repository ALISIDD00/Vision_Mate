# app.py (enhanced) — keeps your vit-gpt2 service, adds translation, TTS fallback, feedback, SOS, optional SocketIO
import os
import time
import json
import logging
import threading
from uuid import uuid4
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from gtts import gTTS
from PIL import Image

# --- optional realtime (SocketIO). If package not installed, app still works ---
USE_SOCKETIO = False
try:
    from flask_socketio import SocketIO, emit
    USE_SOCKETIO = True
except Exception:
    USE_SOCKETIO = False

# --- optional translation libs ---
# Try googletrans first (simple), fallback to transformers if user installed (heavy)
TRANSLATOR = None
try:
    from googletrans import Translator
    TRANSLATOR = Translator()
except Exception:
    TRANSLATOR = None

# Try transformers translation fallback (optional & heavy)
HF_TRANSLATION_OK = False
try:
    from transformers import pipeline
    trans_hi = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
    trans_ar = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ar")
    HF_TRANSLATION_OK = True
except Exception:
    HF_TRANSLATION_OK = False

# --- optional offline TTS ---
PYTTSX3_OK = False
try:
    import pyttsx3
    PYTTSX3_OK = True
except Exception:
    PYTTSX3_OK = False

# --- Twilio placeholders for SOS (optional) ---
TWILIO_OK = False
try:
    from twilio.rest import Client
    TWILIO_OK = True
except Exception:
    TWILIO_OK = False

# --- load HF image-caption model (your original) ---
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model.to(device)

# --- Flask app ---
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
app.config['UPLOAD_FOLDER'] = 'static'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# SocketIO init if available
socketio = SocketIO(app) if USE_SOCKETIO else None

# in-memory history (small) and feedback queue file
history = []
FEEDBACK_FILE = os.path.join('artifacts', 'feedback_queue.jsonl')
os.makedirs('artifacts', exist_ok=True)

# quick phrases (saved, offline)
QUICK_PHRASES = [
    "Hello",
    "I need help",
    "Call my family",
    "Where is washroom"
]

# language map for gTTS and client hint
LANG_MAP = {
    'en': 'en',
    'urdu': 'ur',
    'hi': 'hi',
    'ar': 'ar'
}

# -------------------------
# Helper utilities
# -------------------------
def emit_socket(event, payload):
    """Emit via SocketIO if available (safe no-op otherwise)"""
    if USE_SOCKETIO and socketio:
        socketio.emit(event, payload)

def translate_text(text, target_lang):
    """Translate English text to target_lang ('urdu','hi','ar','en')"""
    if not text:
        return text
    target_lang = (target_lang or 'en').lower()
    if target_lang == 'en':
        return text
    # try googletrans (fast, may be unstable)
    try:
        if TRANSLATOR:
            dest = 'ur' if target_lang=='urdu' else target_lang
            res = TRANSLATOR.translate(text, dest=dest)
            return res.text
    except Exception as e:
        app.logger.info("googletrans failed: %s", e)
    # try HF pipeline if available
    try:
        if HF_TRANSLATION_OK:
            if target_lang == 'hi' and trans_hi:
                return trans_hi(text, max_length=200)[0]['translation_text']
            if target_lang == 'ar' and trans_ar:
                return trans_ar(text, max_length=200)[0]['translation_text']
    except Exception as e:
        app.logger.info("HF translation failed: %s", e)
    # fallback: return original text
    return text

def make_tts(text, lang='en'):
    """
    Make TTS audio file. Tries gTTS (online) then pyttsx3 (offline) as fallback.
    Returns relative path to saved file or None.
    """
    if not text or not text.strip():
        return None
    fname = f"sound_{int(time.time())}_{uuid4().hex[:6]}.mp3"
    out_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    # Try gTTS (online)
    try:
        gtts_lang = LANG_MAP.get(lang, 'en')
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        tts.save(out_path)
        return out_path
    except Exception as e:
        app.logger.warning("gTTS failed or offline: %s", e)
    # Fallback to pyttsx3 (offline) -> save as WAV using pyttsx3 (mp3 not guaranteed)
    try:
        if PYTTSX3_OK:
            engine = pyttsx3.init()
            # pyttsx3 saves to file via engine.save_to_file if supported
            engine.save_to_file(text, out_path)
            engine.runAndWait()
            return out_path
    except Exception as e:
        app.logger.warning("pyttsx3 failed: %s", e)
    return None

def queue_feedback(image_path, correct_caption, user_id=None):
    rec = {
        "id": str(uuid4()),
        "timestamp": int(time.time()),
        "image": image_path,
        "correct_caption": correct_caption,
        "user_id": user_id
    }
    with open(FEEDBACK_FILE, 'a', encoding='utf8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec

def send_sos(phone, message="Emergency! Please help.", image_path=None):
    """
    Try Twilio if configured in env vars, else just log.
    Set TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM in env to enable.
    """
    if TWILIO_OK and os.environ.get('TWILIO_SID') and os.environ.get('TWILIO_TOKEN') and os.environ.get('TWILIO_FROM'):
        try:
            client = Client(os.environ['TWILIO_SID'], os.environ['TWILIO_TOKEN'])
            msg = client.messages.create(body=message, from_=os.environ['TWILIO_FROM'], to=phone)
            # if image_path present, you could send MMS (depends on Twilio account)
            return {"status":"sent", "sid": msg.sid}
        except Exception as e:
            app.logger.exception("Twilio error")
            return {"status":"error", "error": str(e)}
    else:
        app.logger.info("SOS requested: phone=%s message=%s image=%s", phone, message, image_path)
        return {"status":"logged", "note":"twilio not configured"}

# -------------------------
# Routes
# -------------------------
@app.route('/')
def index():
    return render_template('index.html')

# predict route (keeps same HF captioning but supports language param + realtime emits)
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    img = request.files['file']
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'file.jpg')
    img.save(save_path)
    app.logger.info("Image saved to %s", save_path)
    emit_socket('progress', {'step':'saved', 'msg':'image saved'})

    # caption generation (same as original)
    try:
        max_length = int(request.form.get('max_length', 16))
        num_beams = int(request.form.get('num_beams', 4))
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

        i_image = Image.open(save_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        emit_socket('progress', {'step':'model_infer', 'msg':'running caption model'})
        output_ids = model.generate(pixel_values, **gen_kwargs)
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        caption_en = ' '.join(preds).capitalize()
        emit_socket('progress', {'step':'done', 'msg':'caption generated'})

    except Exception as e:
        app.logger.exception("Caption generation failed")
        return f"Caption generation failed: {e}", 500

    # multi-language handling
    lang = request.form.get('lang', 'en').lower()
    translated = translate_text(caption_en, lang) if lang != 'en' else caption_en

    # build TTS (server-side if possible)
    sound_path = None
    try:
        sound_path = make_tts(translated, lang=lang)
    except Exception as e:
        app.logger.warning("TTS generation failed: %s", e)
        sound_path = None

    # Save to in-memory history
    entry = {'image': save_path, 'caption_en': caption_en, 'caption_translated': translated, 'lang': lang, 'sound': sound_path, 'timestamp': int(time.time())}
    history.insert(0, entry)
    if len(history) > 200:
        history.pop()

    # return result page
    # If you prefer JSON (for SocketIO front-end), we can do that too. For now, render after.html (backward-compatible).
    return render_template('after.html', data=translated, sound=sound_path)

@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Expects JSON or form:
      - image: path (e.g., static/file.jpg)
      - correct_caption: full corrected caption (plain text)
      - user_id: optional
    """
    data = request.get_json() or request.form.to_dict()
    image = data.get('image')
    correct = data.get('correct_caption')
    user_id = data.get('user_id')
    if not image or not correct:
        return jsonify({'status':'error', 'msg':'image and correct_caption required'}), 400
    rec = queue_feedback(image, correct, user_id)
    emit_socket('feedback_queued', {'id': rec['id']})
    return jsonify({'status':'queued', 'id': rec['id']})

@app.route('/history')
def history_page():
    return render_template('history.html', history=history)

@app.route('/search')
def search_page():
    return render_template('search.html')

@app.route('/settings')
def settings_page():
    return render_template('settings.html')

@app.route('/newchat')
def newchat_page():
    history.clear()
    return redirect(url_for('index'))

@app.route('/sos', methods=['POST'])
def sos_route():
    """
    Expects JSON: {"phone":"+92300...", "message":"help", "image_path":"static/file.jpg"}
    """
    data = request.get_json() or {}
    phone = data.get('phone')
    message = data.get('message', 'Emergency! Please help.')
    image_path = data.get('image_path')
    if not phone:
        return jsonify({'status':'error','msg':'phone required'}), 400
    res = send_sos(phone, message, image_path)
    return jsonify(res)

# Run (use socketio if available so emits work)
if __name__ == "__main__":
    if USE_SOCKETIO:
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    else:
        app.run(host='0.0.0.0', port=5000, debug=True)


