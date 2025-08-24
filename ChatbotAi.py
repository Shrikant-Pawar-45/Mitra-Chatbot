import streamlit as st
import requests
import json
import re
import io
import base64
import tempfile

from typing import List, Tuple

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None
try:
    from PIL import Image
except Exception:
    Image = None
try:
    from gtts import gTTS
except Exception:
    gTTS = None
try:
    import pyttsx3  # Offline TTS (Windows SAPI5)
except Exception:
    pyttsx3 = None

st.set_page_config(page_title="Mitra Studio", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
    :root {
      --bg: #0f1115;
      --surface: #141823;
      --card: #1a2030;
      --muted: #9aa5b1;
      --text: #e6eaf2;
      --primary: #4f8cff;
      --primary-600: #3b6fe0;
      --radius: 14px;
      --radius-sm: 10px;
    }

    html, body { background-color: var(--bg) !important; }
    .block-container { padding-top: 1.25rem; padding-bottom: calc(2.5rem + 90px); max-width: 1100px; margin: 3rem auto 0; }

    /* Header */
    .header { 
      text-align: center; 
      font-size: clamp(20px, 3.2vw, 30px); 
      font-weight: 800; 
      letter-spacing: 0.2px; 
      margin: 2px 0 12px 0; 
      color: var(--text); 
      overflow-wrap: anywhere; 
      word-break: break-word; 
      padding: 0 8px;
    }
    .header:after { content: ""; display: block; width: 90px; height: 2px; background: var(--primary); opacity: 0.6; margin: 10px auto 0; border-radius: 2px; }

    /* Cards & surfaces */
    .main { background-color: var(--surface); padding: 20px; border-radius: var(--radius); min-height: 70vh; box-shadow: 0 1px 0 rgba(255,255,255,0.03), 0 8px 24px rgba(0,0,0,0.28); }

    /* Tabs: underline style */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #222b3a; padding-bottom: 6px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: var(--muted); border-radius: 8px; padding: 6px 10px; }
    .stTabs [aria-selected="true"] { color: var(--text); position: relative; }
    .stTabs [aria-selected="true"]:after { content: ""; position: absolute; left: 10px; right: 10px; bottom: -7px; height: 2px; background: var(--primary); border-radius: 2px; }

    /* Chat bubbles */
    .chat-message { max-width: 720px; padding: 12px 16px; border-radius: 14px; margin: 8px 0; font-size: 16px; line-height: 1.48; box-shadow: 0 2px 8px rgba(0,0,0,0.22); }
    .user-message { background: linear-gradient(135deg, #3b6fe0, #4f8cff); color: #fff; margin-left: auto; border-bottom-right-radius: 6px; text-align: right; }
    .assistant-message { background: var(--card); color: var(--text); margin-right: auto; border-bottom-left-radius: 6px; text-align: left; }

    /* Inputs & buttons */
    .stTextArea textarea, .stTextInput input, .stNumberInput input { background: var(--card) !important; color: var(--text) !important; border: 1px solid #2b3242 !important; border-radius: 12px !important; }
    .stTextArea textarea::placeholder, .stTextInput input::placeholder { color: #7d8793 !important; }
    .stSelectbox div[data-baseweb="select"] > div { background: var(--card); color: var(--text); border-radius: 10px; }
    .stSlider { padding-top: 8px; }
    .stButton button { background: var(--primary); border: none; color: #fff; border-radius: 12px; padding: 0.5rem 1rem; box-shadow: 0 6px 16px rgba(79,140,255,0.25); }
    .stButton button:hover { background: var(--primary-600); }

    /* Chat input bar */
    .stChatInputContainer { position: fixed; bottom: 0; left: 0; right: 0; background-color: var(--surface); padding: 12px 20px; border-top: 1px solid #20293a; z-index: 1000; }
    /* Ensure input spans centered width */
    .stChatInputContainer > div { max-width: 1100px; margin: 0 auto; }

    /* Responsive tweaks */
    @media (max-width: 900px) {
      .block-container { max-width: 96%; }
      .chat-message { max-width: 100%; }
    }

    /* Footer */
    .app-footer { color: var(--muted); text-align: center; font-size: 13px; margin-top: 18px; opacity: 0.9; }
    .app-footer a { color: var(--text); text-decoration: none; }
    .app-footer a:hover { text-decoration: underline; }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------- Header -----------
st.markdown("<div class='header'><span>Chat with Mitra</span></div>", unsafe_allow_html=True)

# ----------- Session State -----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Mitra, your AI assistant. How can I help you today?"}
    ]

# Load API keys once
API_KEY = None
STABILITY_API_KEY = None
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    API_KEY = None
try:
    STABILITY_API_KEY = st.secrets["STABILITY_API_KEY"]
except Exception:
    STABILITY_API_KEY = None

# ----------- Function: Call Gemini 2.0 Flash API -----------

def ask_gemini(prompt, api_key):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Safely extract text
        candidates = data.get("candidates", [])
        if not candidates:
            return " No response from Mitra."
        
        text_parts = []
        for candidate in candidates:
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            for part in parts:
                if "text" in part:
                    text_parts.append(part["text"])
        
        raw_text = "\n".join(text_parts)
        
        # Clean up Markdown, extra spaces
        plain_text = re.sub(r"\*\*(.*?)\*\*", r"\1", raw_text)  # remove bold
        plain_text = re.sub(r"\n{2,}", "\n\n", plain_text)      # normalize line breaks
        
        return plain_text.strip()
    
    except Exception as e:
        full_resp = ""
        try:
            full_resp = response.text
        except Exception:
            pass
        return f"Error: {e}\n{full_resp}"


# ----------- Helpers: PDF, Chunking, Context Q&A -----------
def _chunk_text(text: str, max_chars: int = 3000) -> List[str]:
    chunks = []
    buf = []
    total = 0
    for line in text.splitlines():
        if total + len(line) + 1 > max_chars:
            chunks.append("\n".join(buf))
            buf, total = [line], len(line) + 1
        else:
            buf.append(line)
            total += len(line) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def ask_gemini_with_context(question: str, context: str, api_key: str) -> str:
    system_prompt = (
        "You are a helpful assistant. Use the provided context to answer the user's question. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    return ask_gemini(system_prompt, api_key)


# ----------- Helpers: Voice (TTS) -----------
def _elevenlabs_secrets():
    api_key = None
    male_voice_id = None
    female_voice_id = None
    try:
        api_key = st.secrets.get("ELEVENLABS_API_KEY")
        male_voice_id = st.secrets.get("ELEVENLABS_VOICE_MALE")
        female_voice_id = st.secrets.get("ELEVENLABS_VOICE_FEMALE")
    except Exception:
        pass
    # Defaults provided by user if not set in secrets
    if not male_voice_id:
        male_voice_id = "nPczCjzI2devNBz1zQrb"
    if not female_voice_id:
        female_voice_id = "Xb7hH8MSUJpSbSDYk0k2"
    return api_key, male_voice_id, female_voice_id


def _synthesize_with_elevenlabs(text: str, gender: str) -> tuple[bytes, str, str] | None:
    """Return (audio_bytes, mime, filename) using ElevenLabs if API key is present. None if unavailable."""
    api_key, male_id, female_id = _elevenlabs_secrets()
    if not api_key:
        return None
    g = (gender or "male").lower()
    # Prefer explicit voice IDs from secrets
    voice_id = male_id if g.startswith("m") else female_id

    # Try official SDK by voice name if no voice_id provided
    try:
        if voice_id is None:
            try:
                from elevenlabs import generate as el_generate, set_api_key as el_set_key
                el_set_key(api_key)
                voice_name = "Adam" if g.startswith("m") else "Bella"
                audio = el_generate(text=text, voice=voice_name, model="eleven_turbo_v2")
                if audio:
                    return bytes(audio), "audio/mpeg", f"speech_{g}.mp3"
            except Exception:
                pass

        # REST fallback (requires voice_id)
        if voice_id:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "xi-api-key": api_key,
                "accept": "audio/mpeg",
                "content-type": "application/json",
            }
            payload = {
                "text": text,
                "model_id": "eleven_turbo_v2_5",
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.ok:
                return resp.content, "audio/mpeg", f"speech_{g}.mp3"
    except Exception:
        return None
    return None
def _synthesize_with_pyttsx3(text: str, gender: str) -> tuple[bytes, str, str] | None:
    """Return (audio_bytes, mime, filename) using pyttsx3 if a voice is found. None if unavailable."""
    if pyttsx3 is None:
        return None
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty("voices") or []
        g = (gender or "").lower()
        # Simple heuristic to pick a voice by gender
        female_hints = ["female", "zira", "aria", "salli", "ivy", "emma", "joanna", "linda"]
        male_hints = ["male", "david", "guy", "matthew", "brian", "john", "tom"]
        hints = female_hints if g.startswith("f") else male_hints

        def is_match(v):
            name = (getattr(v, "name", "") or getattr(v, "id", "")).lower()
            gender_attr = str(getattr(v, "gender", "")).lower()
            return any(h in name for h in hints) or (g and g in gender_attr)

        voice_id = None
        for v in voices:
            if is_match(v):
                voice_id = getattr(v, "id", None)
                break
        # Fallback to first available voice if none matched
        if voice_id is None and voices:
            voice_id = getattr(voices[0], "id", None)

        if voice_id is None:
            return None

        engine.setProperty("voice", voice_id)
        # Save to temp WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            tmp_path = tf.name
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        with open(tmp_path, "rb") as f:
            data = f.read()
        return data, "audio/wav", f"speech_{gender.lower() if gender else 'voice'}.wav"
    except Exception:
        return None


def synthesize_voice_gtts(text: str) -> bytes:
    if gTTS is None:
        raise RuntimeError("gTTS not installed. Run: pip install gTTS")
    tts = gTTS(text=text, lang="en")
    with io.BytesIO() as buf:
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()


def synthesize_voice(text: str, gender: str) -> tuple[bytes, str, str]:
    """Return (audio_bytes, mime, filename).
    Preference: ElevenLabs (cloud, true male/female) -> pyttsx3 (offline) -> gTTS.
    """
    # 1) ElevenLabs if available (works on Streamlit Cloud)
    res = _synthesize_with_elevenlabs(text, gender)
    if res is not None:
        return res
    # 2) Offline pyttsx3
    res = _synthesize_with_pyttsx3(text, gender)
    if res is not None:
        return res
    # 3) Fallback to gTTS (no gender control)
    audio = synthesize_voice_gtts(text)
    return audio, "audio/mp3", "speech.mp3"


# ----------- Helpers: Images (Text-to-Image / Image-to-Image) -----------
def _ensure_genai_client(api_key: str):
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        raise RuntimeError(
            "google-genai not available or failed to init"
        ) from e


def generate_images_text(prompt: str, size: str, num_images: int, seed: int | None, api_key: str, model: str = "imagen-4.0-ultra-generate-001") -> tuple[list[bytes], str | None]:
    """Return list of image bytes and optional info string. Fixes size enum issue."""
    try:
        client = _ensure_genai_client(api_key)
        images_bytes: list[bytes] = []
        info_msg = None

        try:
            from google.genai import types as genai_types
            size_mapping = {
                "512x512": genai_types.ImageSize.IMAGE_SIZE_512,
                "768x768": genai_types.ImageSize.IMAGE_SIZE_768,
                "1024x1024": genai_types.ImageSize.IMAGE_SIZE_1024,
            }

            cfg = genai_types.GenerateImagesConfig(
                number_of_images=num_images,
                size=size_mapping.get(size, genai_types.ImageSize.IMAGE_SIZE_512),
                seed=seed,
            )

            result = client.models.generate_images(model=model, prompt=prompt, config=cfg)

            for item in getattr(result, "images", []) or getattr(result, "data", []):
                b64 = (
                    getattr(item, "base64_data", None)
                    or (item.get("b64_data") if isinstance(item, dict) else None)
                )
                if b64:
                    images_bytes.append(base64.b64decode(b64))

        except Exception:
            # REST fallback for older SDKs
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateImages"
            headers = {"Content-Type": "application/json", "X-goog-api-key": api_key}
            payload = {
                "prompt": prompt,
                "config": {"number_of_images": num_images, "size": size},
            }
            if seed is not None:
                payload["config"]["seed"] = seed
            resp = requests.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("images") or data.get("data") or []
            for item in items:
                b64 = item.get("b64_data") or item.get("base64_data") or item.get("image_base64")
                if b64:
                    images_bytes.append(base64.b64decode(b64))
            if not images_bytes:
                info_msg = data.get("error", {}).get("message") or "No images returned from REST API."

        if not images_bytes:
            info_msg = info_msg or "No images returned. Ensure your API key has Imagen access enabled."
        return images_bytes, info_msg

    except Exception as e:
        return [], f"Image generation error: {e}"


def generate_images_text_stability(
    prompt: str,
    size: str,
    num_images: int,
    seed: int | None,
    api_key: str,
    engine: str = "stable-diffusion-xl-1024-v1-0",
) -> Tuple[List[bytes], str | None]:
    """Text-to-Image via Stability AI REST v1.
    Returns (list of PNG bytes, optional info message).
    """
    try:
        if not api_key:
            return [], "Missing STABILITY_API_KEY in secrets."

        # Map size to width/height; Stability expects multiples of 64.
        size_map = {
            "512x512": (512, 512),
            "768x768": (768, 768),
            "1024x1024": (1024, 1024),
        }
        width, height = size_map.get(size, (768, 768))

        info_msg: str | None = None
        # SDXL engines require a specific set of sizes; square is only 1024x1024.
        if "xl" in engine and (width, height) != (1024, 1024):
            width, height = (1024, 1024)
            info_msg = "Engine requires 1024x1024; size overridden automatically."

        url = f"https://api.stability.ai/v1/generation/{engine}/text-to-image"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "text_prompts": [{"text": prompt}],
            "cfg_scale": 7,
            "clip_guidance_preset": "NONE",
            "height": height,
            "width": width,
            "samples": max(1, min(4, int(num_images))),
        }
        if seed is not None:
            payload["seed"] = int(seed)

        resp = requests.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            try:
                err = resp.json()
            except Exception:
                err = {"message": resp.text}
            return [], f"Stability error {resp.status_code}: {err}"

        data = resp.json()
        artifacts = data.get("artifacts", [])
        images: List[bytes] = []
        for art in artifacts:
            b64 = art.get("base64") or art.get("b64")
            if b64:
                images.append(base64.b64decode(b64))
        if not images:
            return [], "No images returned from Stability."
        return images, info_msg
    except Exception as e:
        return [], f"Stability generation error: {e}"


def generate_images_image2image(base_image_bytes: bytes, prompt: str, strength: float, size: str, num_images: int, api_key: str, model: str = "imagen-4.0-ultra-generate-001") -> tuple[list[bytes], str | None]:
    """Image-to-Image generation with fixed size mapping."""
    try:
        client = _ensure_genai_client(api_key)
        images_bytes: list[bytes] = []
        info_msg = None

        try:
            from google.genai import types as genai_types
            size_mapping = {
                "512x512": genai_types.ImageSize.IMAGE_SIZE_512,
                "768x768": genai_types.ImageSize.IMAGE_SIZE_768,
                "1024x1024": genai_types.ImageSize.IMAGE_SIZE_1024,
            }

            try:
                edits = getattr(client.images, "edits")
                result = edits.create(
                    model=model,
                    image=base64.b64encode(base_image_bytes).decode("utf-8"),
                    prompt=prompt,
                    number_of_images=num_images,
                    strength=strength,
                    size=size_mapping.get(size, genai_types.ImageSize.IMAGE_SIZE_512),
                )
            except Exception:
                result = client.images.edit(
                    model=model,
                    image=base64.b64encode(base_image_bytes).decode("utf-8"),
                    prompt=prompt,
                    number_of_images=num_images,
                    strength=strength,
                    size=size_mapping.get(size, genai_types.ImageSize.IMAGE_SIZE_512),
                )

            for item in getattr(result, "images", []) or getattr(result, "data", []):
                b64 = item.get("b64_data") if isinstance(item, dict) else getattr(item, "b64_data", None)
                if b64:
                    images_bytes.append(base64.b64decode(b64))

        except AttributeError:
            info_msg = "Image-to-image not available in this SDK version."

        if not images_bytes:
            info_msg = info_msg or "No images returned. Ensure your API key has Imagen and img2img access."
        return images_bytes, info_msg

    except Exception as e:
        return [], f"Image editing error: {e}"



def generate_images_image2image(base_image_bytes: bytes, prompt: str, strength: float, size: str, num_images: int, api_key: str, model: str = "imagen-4.0-ultra-generate-001") -> Tuple[List[bytes], str | None]:
    try:
        client = _ensure_genai_client(api_key)
        images_bytes: List[bytes] = []
        info_msg = None
        try:
            # Some SDKs expose images.edits; others may use images.edit. Try edits first.
            try:
                edits = getattr(client.images, "edits")
                result = edits.create(
                    model=model,
                    image=base64.b64encode(base_image_bytes).decode("utf-8"),
                    prompt=prompt,
                    number_of_images=num_images,
                    strength=strength,
                    size=size,
                )
            except Exception:
                # Fallback to images.edit if available
                result = client.images.edit(
                    model=model,
                    image=base64.b64encode(base_image_bytes).decode("utf-8"),
                    prompt=prompt,
                    number_of_images=num_images,
                    strength=strength,
                    size=size,
                )
            for item in getattr(result, "images", []) or getattr(result, "data", []):
                b64 = item.get("b64_data") if isinstance(item, dict) else getattr(item, "b64_data", None)
                if b64:
                    images_bytes.append(base64.b64decode(b64))
        except AttributeError:
            info_msg = "Image-to-image not available in this SDK version."

        if not images_bytes:
            info_msg = info_msg or "No images returned."
        return images_bytes, info_msg
    except Exception as e:
        return [], f"Image editing error: {e}"



# ----------- Tabs -----------
tab_chat, tab_img, tab_pdf, tab_voice = st.tabs(["Chat", "Images", "PDF", "Voice"])

# ----------- Tab: Chat -----------
with tab_chat:
    if API_KEY is None:
        st.warning("Add GEMINI_API_KEY to your Streamlit secrets to use chat.")
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"<div class='chat-message user-message'>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-message assistant-message'>{msg['content']}</div>", unsafe_allow_html=True)

    if API_KEY:
        if prompt := st.chat_input("Type your message here..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner("🤖 Mitra is thinking..."):
                response = ask_gemini(prompt, API_KEY)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

# ----------- Tab: Image Lab -----------
with tab_img:
    st.subheader("Image Generation (Stability AI)")
    if STABILITY_API_KEY is None:
        st.info("Add STABILITY_API_KEY (Stability AI) in .streamlit/secrets.toml to enable image generation.")

    # Only Stability provider
    mode = st.radio("Mode", ["Text-to-Image", "Image-to-Image"], horizontal=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        prompt_text = st.text_area("Prompt", placeholder="A photorealistic portrait of a robot reading a book at sunset", height=100)
    with col2:
        # Popular Stability engines; adjust to what's enabled on the key
        model_name = st.selectbox("Engine", [
            "stable-diffusion-xl-1024-v1-0",
        ], index=0)
        size = st.selectbox("Size", ["512x512", "768x768", "1024x1024"], index=1)
        num_images = st.slider("Number of images", 1, 4, 2)
        seed = st.number_input("Seed (optional)", min_value=0, value=0, step=1)
        seed_val = int(seed) if seed else None

    base_img_bytes = None
    strength = None
    if mode == "Image-to-Image":
        upload = st.file_uploader("Upload base image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        strength = st.slider("Transformation strength", 0.0, 1.0, 0.5, 0.05)
        if upload is not None:
            base_img_bytes = upload.read()
            if Image is not None:
                st.image(Image.open(io.BytesIO(base_img_bytes)), caption="Base image", use_container_width=True)

    # Determine if we have a usable key for Stability
    has_key = STABILITY_API_KEY is not None
    disabled = not has_key or not prompt_text or (mode == "Image-to-Image" and base_img_bytes is None)
    btn_left, btn_right = st.columns([2, 1])
    with btn_right:
        do_generate = st.button("Generate Images", type="primary", disabled=disabled, help="Create images using Stability AI", use_container_width=True)
    if do_generate:
        with st.spinner("Generating images..."):
            if mode == "Text-to-Image":
                imgs, info = generate_images_text_stability(prompt_text, size, num_images, seed_val, STABILITY_API_KEY, engine=model_name)
            else:
                imgs, info = [], "Image-to-Image for Stability not implemented yet here. Use Text-to-Image."

        if info:
            st.info(info)
        if imgs:
            cols = st.columns(min(len(imgs), 4))
            for i, img_bytes in enumerate(imgs):
                with cols[i % len(cols)]:
                    if Image is not None:
                        st.image(Image.open(io.BytesIO(img_bytes)), caption=f"Result {i+1}", use_container_width=True)
                    st.download_button(
                        label=f"Download {i+1}.png",
                        data=img_bytes,
                        file_name=f"image_{i+1}.png",
                        mime="image/png",
                    )

# ----------- Tab: PDF Analysis -----------
with tab_pdf:
    st.subheader("Upload a PDF and ask questions")
    if PdfReader is None:
        st.warning("PyPDF2 not installed.")
    if API_KEY is None:
        st.warning("Add GEMINI_API_KEY to secrets.")

    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file and PdfReader is not None:
        with st.spinner("Reading PDF..."):
            reader = PdfReader(pdf_file)
            full_text = "\n".join([p.extract_text() or "" for p in reader.pages])
        st.success("PDF loaded.")
        st.text_area("Extracted text (preview)", full_text[:4000], height=200)

        question = st.text_input("Ask a question about this PDF")
        if st.button("Answer", disabled=not (API_KEY and question)):
            chunks = _chunk_text(full_text, 3000)
            # Use the first few chunks as context (simple baseline). For production, do retrieval.
            context = "\n\n".join(chunks[:3])
            with st.spinner("Thinking..."):
                ans = ask_gemini_with_context(question, context, API_KEY)
            st.write(ans)

# ----------- Tab: Voice -----------
with tab_voice:
    st.subheader("Text to Speech")
    eleven_key, _, _ = _elevenlabs_secrets()
    if not eleven_key and (pyttsx3 is None and gTTS is None):
        st.warning("No TTS engine available. Add ELEVENLABS_API_KEY in secrets, or install gTTS locally.")
    text = st.text_area("Text to speak", "Hello! This voice was generated by Mitra.")
    col_m, col_f = st.columns(2)
    disabled_btns = (not eleven_key and pyttsx3 is None and gTTS is None) or not text.strip()
    gen_male = col_m.button("Male Voice", disabled=disabled_btns, use_container_width=True)
    gen_female = col_f.button("Female Voice", disabled=disabled_btns, use_container_width=True)
    if gen_male or gen_female:
        with st.spinner("Synthesizing..."):
            gender = "male" if gen_male else "female"
            audio_bytes, mime, fname = synthesize_voice(text.strip(), gender)
        st.audio(audio_bytes, format=mime)
        st.download_button("Download", data=audio_bytes, file_name=fname, mime=mime)

# ----------- Footer -----------
st.markdown(
    """
    <div class='app-footer'>
      2025 Mitra. Built with Streamlit. 
      <span style='opacity:0.7'>Theme: Professional Dark</span>
    </div>
    """,
    unsafe_allow_html=True,
)
