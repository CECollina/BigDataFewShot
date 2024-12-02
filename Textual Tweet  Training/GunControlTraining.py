import csv
from ollama import Client
from sklearn.metrics import confusion_matrix, classification_report

client = Client()

def handle_result(tweet_text, actual_stance, llm_answer):
    result = f"Actual: {str(actual_stance).ljust(7)} LLM: {llm_answer.ljust(7)} Tweet: {tweet_text[:100]}..."
    print(result)

def handle_error(tweet_text, ex):
    result = f"Error processing tweet: {tweet_text[:100]}...\nError: {str(ex)}"
    print(result)

def map_stance(value):
    if value.lower().strip() in ['support', 'oppose']:
        return value.lower().strip()
    return 'oppose'  # Default to 'oppose' for any unexpected values

def interpret_stance(llm_output):
    llm_output = llm_output.lower()
    if "support" in llm_output:
        return "support"
    return "oppose"

with open('gctweets_train.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)
    sentences = [(row[0], map_stance(row[1])) for row in reader if len(row) >= 2]
    print(f"Loaded {len(sentences)} tweets from CSV.")

prompt = """Classify the stance of the following tweet on gun control as either 'support' or 'oppose'. Respond with a single word: support or oppose.

Examples:
Tweet: This is a great new policy for gun control!
Stance: support

Tweet: I don't agree with this gun control decision at all.
Stance: oppose

Now classify the following tweet:
"""

llm_answers = []
model_name = "llama2"

for sentence, actual_stance in sentences:
    try:
        full_prompt = prompt + f"Tweet: {sentence}\nStance:"
        response = client.generate(model_name, full_prompt)
        llm_says = response.response.strip().lower()
        interpreted_stance = interpret_stance(llm_says)
        handle_result(sentence, actual_stance, interpreted_stance)
        llm_answers.append(interpreted_stance)
    except Exception as ex:
        handle_error(sentence, str(ex))
        llm_answers.append("oppose")

print("\nFew-shot learning results:")
for tweet, actual_stance, predicted_stance in zip(sentences, [s[1] for s in sentences], llm_answers):
    print(f"Tweet: {tweet[0][:100]}... - Actual Stance: {actual_stance} - Predicted Stance: {predicted_stance}")

y_true = [s[1] for s in sentences]
y_pred = llm_answers

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=["support", "oppose"]))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=["support", "oppose"], zero_division=0))
