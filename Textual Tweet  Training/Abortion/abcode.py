import csv
from ollama import Client
from sklearn.metrics import confusion_matrix, classification_report

client = Client()

def handle_result(tweet_text, actual_stance, llm_answer, raw_response):
    result = f"Actual: {str(actual_stance).ljust(7)} LLM: {llm_answer.ljust(7)} Tweet: {tweet_text[:100]}..."
    print(result)
    print(f"Raw response: {raw_response[:100]}...")

def handle_error(tweet_text, ex):
    result = f"Error processing tweet: {tweet_text[:100]}...\nError: {str(ex)}"
    print(result)

def map_stance(value):
    if value and isinstance(value, str):
        value = value.lower().strip()
        if value in ["oppose", "support"]:
            return value
    return "unknown"

def interpret_stance(llm_output):
    llm_output = llm_output.lower()
    if "support" in llm_output and "oppose" not in llm_output:
        return "support"
    elif "oppose" in llm_output or "against" in llm_output:
        return "oppose"
    return "unknown"

# Load tweets from CSV
with open('abtweets_train.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    sentences = [(row[0], map_stance(row[1])) for row in reader if len(row) >= 2]
    print(f"Loaded {len(sentences)} tweets from CSV.")

# Define prompt for model
prompt = """Classify the stance of the following tweet on abortion as either 'support' or 'oppose'. Respond with a single word: support or oppose.

Examples:
Tweet: Free, legal and safe access to abortion for all is essential to gender equality.
Stance: support

Tweet: Abortion ends the life of a child and it robs the mother of her future.
Stance: oppose

Tweet: We need to protect women's reproductive rights and ensure access to safe abortion services.
Stance: support

Tweet: Every life is precious from conception. We must defend the unborn.
Stance: oppose

Now classify the following tweet:
"""

llm_answers = []
model_name = "llama2"

# Process each tweet
for sentence, actual_stance in sentences:
    try:
        full_prompt = prompt + f"Tweet: {sentence}\nStance:"
        response = client.generate(model_name, full_prompt)
        llm_says = response.response.strip().lower()
        interpreted_stance = interpret_stance(llm_says)
        handle_result(sentence, actual_stance, interpreted_stance, response.response)
        llm_answers.append(interpreted_stance)
    except Exception as ex:
        handle_error(sentence, str(ex))
        llm_answers.append("unknown")

# Display results
print("\nFew-shot learning results:")
for tweet, actual_stance, predicted_stance in zip(sentences, [s[1] for s in sentences], llm_answers):
    print(f"Tweet: {tweet[0][:100]}... - Actual Stance: {actual_stance} - Predicted Stance: {predicted_stance}")

# Prepare for evaluation
y_true = [s[1] for s in sentences if s[1] != "unknown"]
y_pred = [p for p, s in zip(llm_answers, sentences) if s[1] != "unknown"]

# Print confusion matrix and classification report
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=["support", "oppose"]))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=["support", "oppose"], zero_division=0))
