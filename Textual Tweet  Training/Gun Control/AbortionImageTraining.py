import csv
import ollama
from ollama import Client
from sklearn.metrics import confusion_matrix, classification_report

client = Client()

imagePath = 'data/dataset/dataset/Abort/1120110594730930176.jpg'

#Use Ollama to analyze the image with Llama 3.2-Vision:
modelResponse = ollama.chat(
    model="llama3.2-vision",
    messages=[{
      "role": "user",
      "content": "Classify the stance of the following image on abortion as either 'support' or 'oppose'. Respond with a single word: support or oppose.",
      "images": [imagePath]
    }],
)

#Extract the model's response about the image:
cleanedText = modelResponse['message']['content'].strip()
print(f"Model Response: {cleanedText}")