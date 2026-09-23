def escalate_ticket(ticket_id: str, reason: str) -> dict:
    print(f" ESCALATING {ticket_id}: {reason}")
    return {"status": "escalated", "ticket_id": ticket_id, "reason": reason}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "escalate_ticket",   
            "description": (
                "Escalate a support ticket to a human on-call engineer. "
                "Call this ONLY when the ticket is Critical priority or "
                "you are not confident enough to handle it automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "The ID of the ticket being escalated",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this ticket needs human escalation",
                    },
                },
                "required": ["ticket_id", "reason"],
            },
        },
    }
]

AVAILABLE_FUNCTIONS = {
    "escalate_ticket": escalate_ticket,
}