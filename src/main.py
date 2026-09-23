import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from classifier import classify_ticket
from retrieval import find_similar_tickets
from tools import TOOLS, AVAILABLE_FUNCTIONS

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url=os.getenv("API_BASE_URL"),
)
MODEL_NAME = os.getenv("MODEL_NAME")


def maybe_escalate(ticket: dict, classification: dict):
    prompt = (
        f"Ticket ID: {ticket['id']}\n"
        f"Classification: {classification['priority']} — {classification['reason']}\n\n"
        "Decide whether this ticket needs escalation to a human on-call "
        "engineer right now. If yes, call the escalate_ticket tool. "
        "If no, just reply with the word: no."
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        tools=TOOLS,
        tool_choice="auto",  # the model decides here and we don't force it
        temperature=0.1,
    )

    message = response.choices[0].message

    # If the model chose to call a tool, tool_calls will be populated.
    if message.tool_calls:
        results = []
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # Look up the real Python function by name and run it
            function_to_call = AVAILABLE_FUNCTIONS[function_name]
            result = function_to_call(**function_args)
            results.append(result)
        return results

    return None  # model decided not to escalate


def run_pipeline(data_path: str = "data/tickets.json"):
    with open(data_path, "r") as f:
        data = json.load(f)

    for ticket in data["new_tickets"]:
        print(f"\n--- Processing {ticket['id']}: {ticket['subject']} ---")

        # RAG, we pull similar past tickets as context
        context = find_similar_tickets(ticket["subject"], ticket["body"])

        # Classify using that context
        classification = classify_ticket(
            subject=ticket["subject"],
            body=ticket["body"],
            extra_context=context,
        )
        print(f"Priority: {classification['priority']} — {classification['reason']}")

        # we only use tool/function calling on higher-priority tickets
        if classification["priority"] in ("Critical", "High"):
            escalation_result = maybe_escalate(ticket, classification)
            if escalation_result:
                print(f"Tool called: {escalation_result}")
            else:
                print("Model decided not to escalate.")


if __name__ == "__main__":
    run_pipeline()