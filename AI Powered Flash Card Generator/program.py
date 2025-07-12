from transformers import pipeline

# Load pipelines
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
question_answerer = pipeline("question-answering")
question_generator = pipeline("text2text-generation", model="valhalla/t5-small-qa-qg-hl")

# Function to summarize long text
def summarize_text(text, max_len=130):
    summary = summarizer(text, max_length=max_len, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Function to highlight key sentence and ask question
def generate_flashcards(text):
    # Prepare input for the model
    sentences = text.split('. ')
    flashcards = []

    for sentence in sentences:
        # Add highlighting tag for T5 format
        hl_text = sentence + " </hl> " + sentence
        input_text = f"generate question: {hl_text}"
        result = question_generator(input_text, max_length=64)[0]['generated_text']

        # Get answer from sentence using QA model
        answer = question_answerer(question=result, context=sentence)['answer']

        flashcards.append((result, answer))
    return flashcards

# Example long input
input_text = input("Enter the information for flashcard\n")

# Summarize input
summary = summarize_text(input_text)
print("\n🔍 Summary:\n", summary)

# Generate flashcards
cards = generate_flashcards(summary)

print("\n🧠 Generated Flashcards:")
for idx, (q, a) in enumerate(cards, 1):
    print(f"\nFlashcard {idx}:")
    print("Q:", q)
    print("A:", a)
