import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_resolved_notes(path: str = "data/tickets.json") -> list:       # gets resolved notes from the json file
    with open(path, "r") as f:
        data = json.load(f)
    return data["resolved_notes"]


def find_similar_tickets(subject: str, body: str, top_k: int = 2) -> str:   
    notes = load_resolved_notes()

    corpus = [f"{n['subject']} {n['body']}" for n in notes]   # combines subject and body of each note into a string
    new_ticket_text = f"{subject} {body}"
    corpus.append(new_ticket_text)

    vectorizer = TfidfVectorizer(stop_words="english")          # ignores common words like the, is and stuff
    tfidf_matrix = vectorizer.fit_transform(corpus)

    new_ticket_vector = tfidf_matrix[-1]
    note_vectors = tfidf_matrix[:-1]

    similarities = cosine_similarity(new_ticket_vector, note_vectors).flatten()

    top_indices = similarities.argsort()[::-1][:top_k]   # argsort gives indices in ascending order

    context_lines = []
    for idx in top_indices:
        note = notes[idx]
        context_lines.append(
            f'- Similar past ticket: "{note["subject"]}"\n'
            f"  Resolution: {note['resolution']}\n"
            f"  Was classified as: {note['priority']}"
        )

    return "\n".join(context_lines)


if __name__ == "__main__":
    result = find_similar_tickets(
        subject="Payment processing failing for all EU customers",
        body="Checkout is failing for every EU customer since this morning.",
    )
    print(result)