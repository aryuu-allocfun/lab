"""
Dental Bot v3.1 - OpenRouter Model Routing
Telegram <-> Dify with patient context and OpenRouter model selection.
"""

import logging
import os
import sys
import json
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    filters,
)

# =============================================================================
# Configuration
# =============================================================================

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DIFY_API_KEY = os.getenv("DIFY_API_KEY")
DIFY_BASE_URL = os.getenv("DIFY_BASE_URL", "http://localhost/v1").rstrip("/")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openrouter/auto")
ALLOWED_MODELS = [m.strip() for m in os.getenv("ALLOWED_MODELS", "").split(",") if m.strip()]

REQUEST_TIMEOUT = 60

# Paths
BASE_DIR = Path(__file__).parent
PATIENTS_DIR = BASE_DIR / "patients"
STATE_FILE = BASE_DIR / "state.json"

# Validate required config
if not TELEGRAM_TOKEN:
    sys.exit("Error: TELEGRAM_BOT_TOKEN must be set")
if not DIFY_API_KEY:
    sys.exit("Error: DIFY_API_KEY must be set")
if not ALLOWED_MODELS:
    sys.exit("Error: ALLOWED_MODELS must be set")

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Ensure patients directory exists
PATIENTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# State Management
# =============================================================================

def load_state() -> dict:
    """Load persisted state from JSON file."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("Failed to load state, starting fresh")
    return {"users": {}}


def save_state(state: dict) -> None:
    """Persist state to JSON file."""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except IOError as e:
        logger.error(f"Failed to save state: {e}")


def get_user_state(state: dict, user_id: int) -> dict:
    """Get or create user state."""
    user_key = str(user_id)
    if user_key not in state["users"]:
        state["users"][user_key] = {
            "current_patient": None,
            "current_model": None,  # None = use default (hidden)
            "pending_switch": None,
        }
    return state["users"][user_key]


# Global state
STATE = load_state()


# =============================================================================
# Patient File Management
# =============================================================================

def get_patient_dir(patient_name: str) -> Path:
    return PATIENTS_DIR / patient_name


def patient_exists(patient_name: str) -> bool:
    return get_patient_dir(patient_name).is_dir()


def create_patient(patient_name: str) -> bool:
    patient_dir = get_patient_dir(patient_name)
    if patient_dir.exists():
        return False
    
    patient_dir.mkdir(parents=True)
    (patient_dir / "images").mkdir()
    
    profile_content = f"""# Patient Profile: {patient_name}

Created: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Basic Information
- Name: {patient_name}
- Age: 
- Contact: 

## Medical History
- Allergies: 
- Medications: 
- Conditions: 

## Dental History
- Last Visit: 
- Ongoing Treatments: 

## Notes

"""
    (patient_dir / "profile.md").write_text(profile_content)
    
    log_content = f"""# Conversation Log: {patient_name}

---

"""
    (patient_dir / "conversation_log.md").write_text(log_content)
    return True


def list_patients() -> list:
    if not PATIENTS_DIR.exists():
        return []
    return [d.name for d in PATIENTS_DIR.iterdir() if d.is_dir()]


def append_to_conversation_log(patient_name: str, role: str, content: str, image_ref: str = None) -> None:
    log_file = get_patient_dir(patient_name) / "conversation_log.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    entry = f"\n**[{timestamp}] {role}:**\n{content}\n"
    if image_ref:
        entry += f"\n📷 Image: {image_ref}\n"
    entry += "\n---\n"
    
    with open(log_file, "a") as f:
        f.write(entry)


def get_conversation_history(patient_name: str, num_entries: int = 10) -> str:
    log_file = get_patient_dir(patient_name) / "conversation_log.md"
    if not log_file.exists():
        return "No conversation history found."
    
    content = log_file.read_text()
    entries = content.split("---")
    recent = entries[-num_entries-1:-1] if len(entries) > num_entries else entries[1:]
    return "---".join(recent) if recent else "No recent conversations."


def get_patient_context(patient_name: str) -> str:
    patient_dir = get_patient_dir(patient_name)
    
    profile = ""
    profile_file = patient_dir / "profile.md"
    if profile_file.exists():
        profile = profile_file.read_text()
    
    recent_history = get_conversation_history(patient_name, 5)
    
    return f"""=== ACTIVE PATIENT CONTEXT ===
{profile}

=== RECENT CONVERSATION ===
{recent_history}
=== END CONTEXT ===

"""


# =============================================================================
# Model Helpers
# =============================================================================

def get_effective_model(user_model: str | None) -> str:
    """Get the model to use - user selection or default."""
    if user_model and user_model in ALLOWED_MODELS:
        return user_model
    return DEFAULT_MODEL


def get_display_model(user_model: str | None) -> str:
    """Get model name for display (hide default)."""
    if user_model and user_model in ALLOWED_MODELS:
        return user_model
    return "auto"


def format_model_for_display(model: str) -> str:
    """Format model name for user display (remove provider prefix)."""
    if "/" in model:
        return model.split("/", 1)[1]
    return model


# =============================================================================
# Dify API
# =============================================================================

def call_dify_api(query: str, user_id: str, model: str, context: str = "") -> str:
    """Send message to Dify with model routing via inputs."""
    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type": "application/json",
    }
    
    full_query = f"{context}{query}" if context else query
    
    payload = {
        "inputs": {
            "model": model,  # Pass model to Dify for routing
        },
        "query": full_query,
        "response_mode": "blocking",
        "conversation_id": "",
        "user": user_id,
    }

    try:
        response = requests.post(
            f"{DIFY_BASE_URL}/chat-messages",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json().get("answer", "No response received.")
    
    except requests.exceptions.Timeout:
        logger.error("Dify API timeout")
        return "⚠️ Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to Dify")
        return "⚠️ Connection error. Please try again later."
    except requests.exceptions.HTTPError as e:
        logger.error(f"Dify API error: {e.response.status_code}")
        return "⚠️ Service error. Please try again later."
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return "⚠️ Something went wrong. Please try again."


# =============================================================================
# Command Handlers
# =============================================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🦷 Dental Assistant Bot v3.1\n\n"
        "Commands:\n"
        "/new [Name] - Create new patient\n"
        "/switch [Name] - Switch to patient\n"
        "/list - List all patients\n"
        "/history - View conversation history\n"
        "/model [name] - Set AI model\n"
        "/model reset - Use default model\n"
        "/summary - Generate patient summary\n"
        "/status - Current session status"
    )


async def cmd_new(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /new [PatientName]")
        return
    
    patient_name = " ".join(context.args)
    user_id = update.effective_user.id
    
    if patient_exists(patient_name):
        await update.message.reply_text(f"⚠️ Patient '{patient_name}' already exists. Use /switch to select.")
        return
    
    if create_patient(patient_name):
        user_state = get_user_state(STATE, user_id)
        user_state["current_patient"] = patient_name
        save_state(STATE)
        
        await update.message.reply_text(
            f"✅ Created new patient: {patient_name}\n"
            f"🔵 [{patient_name}] is now active"
        )
        logger.info(f"Created patient: {patient_name}")
    else:
        await update.message.reply_text("⚠️ Failed to create patient folder.")


async def cmd_switch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /switch [PatientName]")
        return
    
    patient_name = " ".join(context.args)
    user_id = update.effective_user.id
    user_state = get_user_state(STATE, user_id)
    
    if not patient_exists(patient_name):
        await update.message.reply_text(f"⚠️ Patient '{patient_name}' does not exist.")
        return
    
    current = user_state.get("current_patient")
    if current and current != patient_name:
        user_state["pending_switch"] = patient_name
        save_state(STATE)
        await update.message.reply_text(
            f"⚠️ You are currently on [{current}].\n"
            f"Send /confirm to switch to [{patient_name}] or /cancel to stay."
        )
    else:
        user_state["current_patient"] = patient_name
        user_state["pending_switch"] = None
        save_state(STATE)
        await update.message.reply_text(f"🔵 Active patient set to [{patient_name}].")


async def cmd_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_state = get_user_state(STATE, user_id)
    pending = user_state.get("pending_switch")
    if not pending:
        await update.message.reply_text("No pending switch.")
        return
    user_state["current_patient"] = pending
    user_state["pending_switch"] = None
    save_state(STATE)
    await update.message.reply_text(f"✅ Switched to [{pending}].")


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_state = get_user_state(STATE, user_id)
    user_state["pending_switch"] = None
    save_state(STATE)
    cur = user_state.get("current_patient") or "None"
    await update.message.reply_text(f"Cancelled. Staying with [{cur}].")


async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    patients = list_patients()
    if not patients:
        await update.message.reply_text("No patients yet. Use /new to create one.")
    else:
        await update.message.reply_text("📋 Patients:\n" + "\n".join(f"• {p}" for p in sorted(patients)))


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_state = get_user_state(STATE, user_id)
    if not context.args:
        cur = get_display_model(user_state.get("current_model"))
        opts = "\n".join(f"• {format_model_for_display(m)}" for m in ALLOWED_MODELS)
        await update.message.reply_text(
            f"🤖 Current model: {cur}\n\nAvailable models:{opts}\n\n"
            "Usage: /model [name]   – set model\n"
            "       /model reset   – use default"
        )
        return
    
    arg = context.args[0].lower()
    if arg == "reset":
        user_state["current_model"] = None
        save_state(STATE)
        await update.message.reply_text("✅ Model reset to default (auto).")
        return
    
    wanted = " ".join(context.args)
    match = None
    for m in ALLOWED_MODELS:
        short = format_model_for_display(m)
        if wanted.lower() in (m.lower(), short.lower()):
            match = m
            break
    if not match:
        await update.message.reply_text(
            f"⚠️ Model not found. Available: {', '.join(format_model_for_display(m) for m in ALLOWED_MODELS)}"
        )
    else:
        user_state["current_model"] = match
        save_state(STATE)
        await update.message.reply_text(f"✅ Model set to: {format_model_for_display(match)}")


async def cmd_summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_state = get_user_state(STATE, user_id)
    patient = user_state.get("current_patient")
    if not patient:
        await update.message.reply_text("⚠️ No active patient. Use /switch first.")
        return
    ctx = get_patient_context(patient)
    model = get_effective_model(user_state.get("current_model"))
    prompt = "Please provide a concise clinical summary of this patient based on the context above."
    answer = call_dify_api(prompt, f"tg_{user_id}", model, ctx)
    await update.message.reply_text(f"🔵 [{patient}] Summary:\n\n{answer}")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_state = get_user_state(STATE, user_id)
    patient = user_state.get("current_patient") or "None"
    model_disp = get_display_model(user_state.get("current_model"))
    await update.message.reply_text(f"📊 Session status\nPatient: [{patient}]\nModel: {model_disp}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_state = get_user_state(STATE, user_id)
    patient = user_state.get("current_patient")
    if not patient:
        await update.message.reply_text("⚠️ No active patient. Use /new or /switch first.")
        return
    
    # Forward to Dify
    ctx = get_patient_context(patient)
    model = get_effective_model(user_state.get("current_model"))
    answer = call_dify_api(update.message.text, f"tg_{user_id}", model, ctx)
    
    # Log
    append_to_conversation_log(patient, "User", update.message.text)
    append_to_conversation_log(patient, "Assistant", answer)
    
    await update.message.reply_text(f"🔵 [{patient}]:\n\n{answer}")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_state = get_user_state(STATE, user_id)
    patient = user_state.get("current_patient")
    if not patient:
        await update.message.reply_text("⚠️ No active patient. Use /new or /switch first.")
        return
    
    photo = update.message.photo[-1]
    file = await photo.get_file()
    data = await file.download_as_bytearray()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = get_patient_dir(patient) / "images" / f"{ts}.jpg"
    img_path.write_bytes(data)
    caption = update.message.caption or "Image uploaded"
    append_to_conversation_log(patient, "User", caption, image_ref=img_path.name)
    await update.message.reply_text(f"🔵 [{patient}]: Image saved as {img_path.name}. You can ask a question about it.")


def main() -> None:
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("new", cmd_new))
    app.add_handler(CommandHandler("switch", cmd_switch))
    app.add_handler(CommandHandler("confirm", cmd_confirm))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CommandHandler("list", cmd_list))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("summary", cmd_summary))
    app.add_handler(CommandHandler("status", cmd_status))
    
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    logger.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
