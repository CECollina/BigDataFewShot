import csv
import ollama
from ollama import Client
from sklearn.metrics import confusion_matrix, classification_report

client = Client()

#Read all of the image paths from the CSV file, along with the image stance:
imagePaths=[]
imageStance=[]
with open('Textual Tweet Training/Abortion/AbortionImageTrainingPaths.csv', 'r') as file:
    csvReader = csv.reader(file)
    for tempRow in csvReader:
        imagePaths.append(tempRow[0])
        imageStance.append(tempRow[1])

#Use Ollama to analyze the image with Llama 3.2-Vision:
modelResponse = ollama.chat(
    model="llama3.2-vision",
    messages=[{
      "role": "user",
      "content": "Classify the stance of the following image on abortion as either 'support' or 'oppose'. Respond with a single word: support or oppose.",
      "images": [imagePaths[1]]
    }],
)

#Extract the model's response about the image:
cleanedText = modelResponse['message']['content'].strip()
print(f"Model Response: {cleanedText}")