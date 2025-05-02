import streamlit as st
from openai import OpenAI
import pandas as pd
import PyPDF2
import os
import base64
from datetime import datetime
from PIL import Image
import uuid

# --- Configuration ---
st.set_page_config(page_title="iRIS Training Assistant", layout="wide")

TRAINING_DIR = "training"
SUPPORTED_TYPES = [".pdf"]
QUESTIONS_PER_SESSION = 2
PASSING_SCORE = 1

client = OpenAI(api_key=st.secrets["API_key"])

# --- Utility Functions ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def extract_text_by_page(filepath):
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages.append({"page": i + 1, "text": text})
        return pages

def generate_questions_from_text(pages):
    import json, re
    combined_text = "\n".join([f"[Page {p['page']}]\n{p['text']}" for p in pages])
    prompt = f"""
You are a training assistant. Based on the training material below, generate {QUESTIONS_PER_SESSION} quiz questions.

**Exclude** any questions related to:
- Company address
- Website URLs
- Copyright notices
- Page headers or footers
- Metadata or branding content

Instead, focus on **concepts, insights, reasoning, and practical application** of the subject matter.
Each question must be meaningful and test true comprehension, not rote memorization.

Only use these two formats:
1. Multiple choice with 4 options (A, B, C, D)
2. True or False with options ["True", "False"]

Each item must include:
- "question": the question text
- "type": "multiple_choice" or "true_false"
- "options": list of answer choices
- "answer": the correct answer (must match one of the options)
- "page": the page number the question was derived from

Return the result as a **raw JSON array only** (no explanations, no markdown).

[
  {{
    "question": "What is the capital of France?",
    "type": "multiple_choice",
    "options": ["A. Berlin", "B. Madrid", "C. Paris", "D. Rome"],
    "answer": "C. Paris",
    "page": 2
  }},
  {{
    "question": "The Earth orbits the Sun.",
    "type": "true_false",
    "options": ["True", "False"],
    "answer": "True",
    "page": 1
  }}
]

Training Text:
{combined_text[:5000]}
"""
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"```$", "", raw.strip())
    raw = raw.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    raw = re.sub(r",(\s*[\]}])", r"\1", raw)
    try:
        return json.loads(raw)
    except:
        st.error("\u26a0\ufe0f GPT returned invalid JSON. Please try again.")
        st.text_area("Raw GPT Response:", raw, height=300)
        return []

def evaluate_answer(user_answer, correct_answer):
    return user_answer.strip().lower() == correct_answer.strip().lower()

def save_progress_global(training_dir, module_name, user_name, score):
    progress_file = os.path.join(training_dir, "progress.csv")
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "module": module_name,
        "user": user_name,
        "score": score,
        "session_id": st.session_state.get("session_id", "")
    }
    df = pd.DataFrame([entry])
    write_header = not os.path.exists(progress_file)
    df.to_csv(progress_file, mode="a", header=write_header, index=False)

# --- UI Layout ---
st.markdown("""
<style>
.block-container { padding-top: 5rem !important; }
.top-banner {
    position: absolute; top: 0; left: 0; width:  100%;
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.5rem 1rem; background-color: white; z-index: 9999;
}
.top-banner h1 { font-size: 3em; font-weight: bold; margin: 0; }
</style>
""", unsafe_allow_html=True)

# Initialize flags
if "name_warning" not in st.session_state:
    st.session_state["name_warning"] = False
if "name_ready" not in st.session_state:
    st.session_state["name_ready"] = False
if "summary_generated" not in st.session_state:
    st.session_state["summary_generated"] = False  # <-- add this line


# --- Module & User Selection ---
modules = sorted([m for m in os.listdir(TRAINING_DIR) if os.path.isdir(os.path.join(TRAINING_DIR, m))])
if "selected_module" not in st.session_state:
    st.session_state.selected_module = modules[0]

# --- User Name Input and Persistence ---
# --- User Name Input and Persistence ---
# --- User Name Input and Persistence ---
def set_user_name():
    name = st.session_state.get("name_input", "").strip()
    if name:
        st.session_state["user_name"] = name
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["name_ready"] = True  # trigger for rerun
    else:
        st.session_state["name_warning"] = True

# Initialize flags
if "name_warning" not in st.session_state:
    st.session_state["name_warning"] = False
if "name_ready" not in st.session_state:
    st.session_state["name_ready"] = False

# Show input field only if not set
if "user_name" not in st.session_state:
    st.sidebar.text_input(
        "Enter Your Name",
        key="name_input",
        on_change=set_user_name,
        placeholder="Type your name and press Enter"
    )

    if st.session_state.name_warning:
        st.sidebar.warning("⚠️ Please enter your name before submitting.")

# React to name submission and rerun
if st.session_state.get("name_ready", False):
    st.session_state.name_ready = False
    st.rerun()

# Show current name if set
if "user_name" in st.session_state:
    st.sidebar.markdown(f"👤 **User:** {st.session_state['user_name']}")


# --- Banner (always show) ---
image_path = "picture1.png"
image_html = ""
if os.path.exists(image_path):
    encoded_image = get_base64_image(image_path)
    image_html = f'<img src="data:image/png;base64,{encoded_image}" style="height:60px; margin-left: 20px;">'

st.markdown(f"""
<div class="top-banner">
    <h1>{'Welcome to iRIS Training! Enter your name to get started...' if 'user_name' not in st.session_state else st.session_state.selected_module}</h1>
    {image_html}
</div>
""", unsafe_allow_html=True)

st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)


if "user_name" not in st.session_state:
    st.stop()

selected_module = st.sidebar.selectbox("Select Training Module", modules, index=modules.index(st.session_state.selected_module), key="module_select")

# If the selected module is different from the current one in session, update and rerun
if selected_module != st.session_state.selected_module:
    st.session_state.selected_module = selected_module
    st.rerun()

current_module = st.session_state.selected_module

st.session_state.selected_module = selected_module
current_module = st.session_state.selected_module

# --- Initialize Quiz State ---
default_quiz_state = {
    "questions": [],
    "current_q": 0,
    "answers": [],
    "scores": [],
    "feedback_shown": False,
    "last_correct": None,
    "quiz_complete": False,
    "passed_quiz": False,
    "questions_loading": False
}
for key, value in default_quiz_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Load PDF ---
module_path = os.path.join(TRAINING_DIR, current_module)
pdf_files = [f for f in os.listdir(module_path) if f.endswith(".pdf")]
if not pdf_files:
    st.error("No training material found.")
    st.stop()

#selected_module = st.sidebar.selectbox("Select Training Module", modules, index=modules.index(st.session_state.selected_module))
#st.session_state.selected_module = selected_module
#current_module = st.session_state.selected_module
#module_path = os.path.join(TRAINING_DIR, current_module)

# PDF and optional video detection
pdf_files = [f for f in os.listdir(module_path) if f.endswith(".pdf")]
mp4_files = [f for f in os.listdir(module_path) if f.endswith(".mp4")]
if not pdf_files:
    st.error("No training material found.")
    st.stop()
pdf_path = os.path.join(module_path, pdf_files[0])
mp4_path = os.path.join(module_path, mp4_files[0]) if mp4_files else None

# --- Sidebar: Training Content Controls ---
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🎓 Training Materials")

    with open(pdf_path, "rb") as f:
        st.download_button("📄 Download Training Material", f, file_name=pdf_files[0], key="download_pdf")

    if mp4_path:
        if "show_video" not in st.session_state:
            st.session_state["show_video"] = False

        if not st.session_state["show_video"]:
            if st.button("▶️ Watch Video", key="watch_video_btn"):
                st.session_state["show_video"] = True
                st.rerun()

    # 📌 Always show Module Summary button
    if st.button("🧠 Module Summary", key="generate_summary_sidebar"):
        st.session_state["auto_generated_prompt"] = "Generate a training summary based on the material."
        st.session_state.summary_generated = True  # Mark as generated (optional)
        st.rerun()

    if not st.session_state.questions:
        if st.button("📋 Start Quiz", key="start_quiz_btn"):
            st.session_state.questions_loading = True
            with st.spinner("AI is generating questions..."):
                pages = extract_text_by_page(pdf_path)
                questions = generate_questions_from_text(pages)
                if questions:
                    st.session_state.questions = questions
                    st.session_state.current_q = 0
                    st.session_state.answers = []
                    st.session_state.scores = []
                    st.session_state.feedback_shown = False
                    st.session_state.last_correct = None
                    st.session_state.quiz_complete = False
                    st.session_state.passed_quiz = False
                    st.session_state.questions_loading = False
                    st.rerun()
                else:
                    st.session_state.questions_loading = False
                    st.error("❌ No questions were generated. Please try again.")

if mp4_path:
    if "show_video" not in st.session_state:
        st.session_state["show_video"] = True  # Trigger on load

    if st.session_state["show_video"]:
        st.markdown("### 🎬 Training Video")

        video_url = os.path.join(module_path, mp4_files[0]).replace("\\", "/")
        video_bytes = open(video_url, "rb").read()
        base64_video = base64.b64encode(video_bytes).decode()

        video_html = f"""
        <video width="100%" height="auto" autoplay muted controls>
            <source src="data:video/mp4;base64,{base64_video}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        """
        st.markdown(video_html, unsafe_allow_html=True)

        if st.button("❌ Close Video"):
            st.session_state["show_video"] = False
            st.rerun()

# --- Defensive Check ---
quiz_ready = st.session_state.questions and st.session_state.current_q < len(st.session_state.questions)

if quiz_ready:
    # ✅ Show live score before the question
    correct = sum(st.session_state.scores)
    attempted = len(st.session_state.scores)
    st.markdown(f"🧮 **Score: {correct} / {attempted} correct**")
    
    # --- Show Current Question ---
    question = st.session_state.questions[st.session_state.current_q]
    with st.chat_message("assistant"):
        st.markdown(f"**Question {st.session_state.current_q + 1} (Page {question.get('page', '?')}):** {question['question']}")

    if not st.session_state.feedback_shown:
        with st.chat_message("user"):
            st.markdown("**Select your answer:**")
            for option in question["options"]:
                if st.button(option):
                    st.session_state.answers.append(option)
                    correct = evaluate_answer(option, question["answer"])
                    st.session_state.scores.append(1 if correct else 0)
                    st.session_state.last_correct = correct
                    st.session_state.feedback_shown = True
                    st.rerun()

    if st.session_state.feedback_shown:
        with st.chat_message("assistant"):
            if st.session_state.last_correct:
                st.success("✅ Correct!")
            else:
                st.error(f"❌ Incorrect. The correct answer was: **{question['answer']}**")

        if st.session_state.current_q < QUESTIONS_PER_SESSION - 1:
            if st.button("Next Question"):
                st.session_state.current_q += 1
                st.session_state.feedback_shown = False
                st.session_state.last_correct = None
                st.rerun()
        else:
            total = sum(st.session_state.scores)
            st.session_state.quiz_complete = True
            st.session_state.passed_quiz = total >= PASSING_SCORE

# --- Completion ---
if st.session_state.quiz_complete:
    with st.chat_message("assistant"):
        total_score = sum(st.session_state.scores)

        # ✅ Save progress only once
        if not st.session_state.get("progress_saved", False):
            try:
                save_progress_global(TRAINING_DIR, current_module, st.session_state["user_name"], total_score)
                st.session_state["progress_saved"] = True
            except Exception as e:
                st.error(f"⚠️ Failed to save progress: {e}")

        # Separate outcomes
        if st.session_state.passed_quiz:
            st.markdown("🎉 **Congratulations! You passed the quiz!** 🎉")

            trophy_path = os.path.join(module_path, "trophy.png")
            if os.path.exists(trophy_path):
                with open(trophy_path, "rb") as img_file:
                    image = Image.open(img_file)
                    image = image.resize((300, int(300 * image.height / image.width)))
                    st.image(image, caption="Well done!")

            st.markdown("🎯 You may now proceed to the next training module.")

            # Review missed questions if any
            missed = [
                (i, q, user_answer)
                for i, (score, q, user_answer) in enumerate(
                    zip(st.session_state.scores, st.session_state.questions, st.session_state.answers)
                )
                if score == 0
            ]
            if missed:
                st.markdown("### ❌ Review Missed Questions")
                for i, q, user_answer in missed:
                    with st.expander(f"Question {i+1}: {q['question']}"):
                        st.markdown(f"- **Your answer:** {user_answer}")
                        st.markdown(f"- **Correct answer:** {q['answer']}")

            # Continue button
            if st.button("✅ Continue"):
                try:
                    current_index = modules.index(current_module)
                    if current_index < len(modules) - 1:
                        st.session_state.selected_module = modules[current_index + 1]
                    else:
                        st.info("✅ You’ve completed all modules.")
                except ValueError:
                    st.warning("⚠️ Could not determine next module.")

                for key in default_quiz_state:
                    st.session_state[key] = default_quiz_state[key]
                st.session_state["progress_saved"] = False
                st.session_state["chat_history"] = []
                st.rerun()

        else:
            st.markdown("🔁 You missed 1 or more questions. Please review below before retrying.")

            missed = [
                (i, q, user_answer)
                for i, (score, q, user_answer) in enumerate(
                    zip(st.session_state.scores, st.session_state.questions, st.session_state.answers)
                )
                if score == 0
            ]
            if missed:
                st.markdown("### ❌ Review Missed Questions")
                for i, q, user_answer in missed:
                    with st.expander(f"Question {i+1}: {q['question']}"):
                        st.markdown(f"- **Your answer:** {user_answer}")
                        st.markdown(f"- **Correct answer:** {q['answer']}")

            if st.button("🔁 Retry Quiz"):
                pdf_pages = extract_text_by_page(pdf_path)
                st.session_state.questions = generate_questions_from_text(pdf_pages)
                st.session_state.current_q = 0
                st.session_state.answers = []
                st.session_state.scores = []
                st.session_state.feedback_shown = False
                st.session_state.last_correct = None
                st.session_state.quiz_complete = False
                st.session_state.passed_quiz = False
                st.session_state.progress_saved = False
                st.rerun()

# Handle summary generation
if "auto_generated_prompt" in st.session_state:
    st.session_state.last_prompt = st.session_state.pop("auto_generated_prompt")
    st.session_state.chat_history.append(("user", st.session_state.last_prompt))
    st.session_state.awaiting_gpt = True
    st.rerun()

# --- Chat State Initialization ---
import uuid

# --- Init Chat State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = None
if "awaiting_gpt" not in st.session_state:
    st.session_state.awaiting_gpt = False
if "chat_input_key" not in st.session_state:
    st.session_state.chat_input_key = str(uuid.uuid4())

# --- Show chat messages (top-down) ---
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# --- GPT logic: generate reply if needed ---
if st.session_state.awaiting_gpt and st.session_state.last_prompt:
    with st.spinner("Thinking..."):
        pdf_pages = extract_text_by_page(pdf_path)
        context = "\n".join([f"[Page {p['page']}]\n{p['text']}" for p in pdf_pages])

        prompt = f"""
You are a strategic training advisor. Based on the training material below, answer the user's question.

Training Material:
{context[:5000]}

User's Question:
{st.session_state.last_prompt}

Provide a clear, strategic, and actionable response.
"""
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            reply = response.choices[0].message.content.strip()
        except Exception as e:
            reply = f"⚠️ GPT failed to respond: {e}"

    st.session_state.chat_history.append(("assistant", reply))
    st.session_state.awaiting_gpt = False
    st.session_state.last_prompt = None
    st.session_state.chat_input_key = str(uuid.uuid4())  # Resets input box
    st.rerun()

# --- Chat Input Always Anchored at Bottom ---
with st.container():
    st.markdown('<div class="chat-input-wrapper">', unsafe_allow_html=True)

    user_prompt = st.text_input(
        "Ask about the training material...",
        key=st.session_state.chat_input_key,
        label_visibility="collapsed",
        placeholder="e.g. What are the main points?"
    )

    if st.button("Send"):
        if user_prompt.strip():
            st.session_state.last_prompt = user_prompt.strip()
            st.session_state.chat_history.append(("user", user_prompt.strip()))
            st.session_state.awaiting_gpt = True
            st.session_state.chat_input_key = str(uuid.uuid4())
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# --- GPT Handling ---
if st.session_state.awaiting_gpt and st.session_state.last_prompt:
    with st.spinner("Thinking..."):
        try:
            pdf_pages = extract_text_by_page(pdf_path)
            context = "\n".join([f"[Page {p['page']}]\n{p['text']}" for p in pdf_pages])
            strategic_prompt = f"""
You are a strategic training advisor. Based on the training material below, answer the user's question.

Training Material:
{context[:5000]}

User's Question:
{st.session_state.last_prompt}

Provide a clear, strategic, and actionable response.
"""
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": strategic_prompt}]
            )
            reply = response.choices[0].message.content.strip()
        except Exception as e:
            reply = f"⚠️ GPT failed to respond: {e}"

    # Display response, then clear
    st.session_state.chat_history.append(("assistant", reply))
    st.session_state.awaiting_gpt = False
    st.session_state.last_prompt = None
    st.session_state.bottom_strategic_prompt = ""
    st.session_state.gpt_reply = reply

    # Show assistant message once before wiping history
    st.chat_message("assistant").markdown(reply)

    # Optionally clear history after 2 messages
    st.session_state.chat_history = []

