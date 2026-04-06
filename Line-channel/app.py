# app.py – Minimal Flask bridge for Dify ↔ LINE

"""A very small Flask server that can be extended to:

1. Receive webhook events from the LINE Messaging API.
2. Forward the user's message to a Dify agent (DentistBot).
3. Return the agent's response back to LINE.

You will need to:
- Set `LINE_CHANNEL_ACCESS_TOKEN` and `LINE_CHANNEL_SECRET` as environment variables.
- Set `DIFY_API_KEY` and `DIFY_AGENT_ID` (the DentistBot) as environment variables.
- Install the required packages (see requirements.txt).
"""

import os
from flask import Flask, request, abort, jsonify
import requests

app = Flask(__name__)

# ----- Configuration -------------------------------------------------------
LINE_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_SECRET = os.getenv("LINE_CHANNEL_SECRET")
DIFY_API_KEY = os.getenv("DIFY_API_KEY")
DIFY_AGENT_ID = os.getenv("DIFY_AGENT_ID")

if not all([LINE_ACCESS_TOKEN, LINE_SECRET, DIFY_API_KEY, DIFY_AGENT_ID]):
    raise RuntimeError("Missing required environment variables for LINE/Dify integration")

# ----- Helper functions ---------------------------------------------------
def reply_to_line(reply_token: str, messages: list):
    """Send a reply message back to LINE using the reply API."""
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_ACCESS_TOKEN}",
    }
    payload = {"replyToken": reply_token, "messages": messages}
    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()

def ask_dify(user_text: str) -> str:
    """Call the Dify agent (DentistBot) and return its response text."""
    url = f"https://api.dify.ai/v1/agents/{DIFY_AGENT_ID}/chat-messages"
    headers = {"Authorization": f"Bearer {DIFY_API_KEY}", "Content-Type": "application/json"}
    payload = {"query": user_text, "response_mode": "blocking"}
    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    # Dify returns a list of messages; we take the first text response.
    return data.get("answer", "")

# ----- Routes -------------------------------------------------------------
@app.route("/callback", methods=["POST"])
def callback():
    # Verify request signature (omitted for brevity – add in production)
    body = request.get_json()
    if not body:
        abort(400)
    events = body.get("events", [])
    for ev in events:
        if ev.get("type") != "message" or ev["message"].get("type") != "text":
            continue
        user_text = ev["message"]["text"]
        reply_token = ev["replyToken"]
        # Ask Dify (DentistBot) for a response
        answer = ask_dify(user_text)
        # Build LINE message format
        line_msg = {"type": "text", "text": answer}
        reply_to_line(reply_token, [line_msg])
    return jsonify({"status": "ok"})

# Simple health check
@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    # For local development use port 5000
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000))
