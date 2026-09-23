"""
classifier.py
--------------
Takes a support ticket (subject + body) and asks an LLM to classify its
priority as Critical / High / Medium / Low, returning strict JSON.

Why JSON output matters: if the model just replies in free text like
"This seems pretty urgent to me...", your code can't reliably act on that.
Forcing structured output is what lets an LLM be wired into a real system
instead of just being a chatbot.
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url=os.getenv("API_BASE_URL"),
)

MODEL_NAME = os.getenv("MODEL_NAME")

SYSTEM_PROMPT = """You are a support ticket triage assistant. Given a ticket's
subject and body, classify its priority and respond with ONLY valid JSON,
no other text, in exactly this shape:

{"priority": "Critical" | "High" | "Medium" | "Low", "reason": "<one short sentence>"}

Priority guide:
- Critical: outages, data loss, security issues, payment failures affecting many users
- High: a significant feature broken for some users, but not a full outage
- Medium: a real but non-blocking issue affecting usability
- Low: cosmetic issues, typos, minor feature requests

Examples:

Ticket: "Production database is down, all customers affected"
Response: {"priority": "Critical", "reason": "Full outage affecting all customers"}

Ticket: "Typo in the welcome email"
Response: {"priority": "Low", "reason": "Cosmetic issue with no functional impact"}
"""


def classify_ticket(subject: str, body: str, extra_context: str = "") -> dict:
    """
    Sends one ticket to the LLM and returns a parsed dict like:
    {"priority": "High", "reason": "..."}

    extra_context is optional — this is where retrieval.py will later inject
    similar past tickets (the RAG step). For now it's just an empty string.
    """
    user_message = f"Ticket subject: {subject}\nTicket body: {body}"

    if extra_context:
        user_message += f"\n\nSimilar past tickets for context:\n{extra_context}"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,         # low for better consistentcy
    )

    raw_text = response.choices[0].message.content.strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`").replace("json\n", "", 1).strip()

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        raise ValueError(f"Model did not return valid JSON: {raw_text}")

    return result


if __name__ == "__main__":
    test_result = classify_ticket(
        subject="Production database is down, all customers affected",
        body="Our main app has been throwing 500 errors for 15 minutes."
    )
    print(test_result)