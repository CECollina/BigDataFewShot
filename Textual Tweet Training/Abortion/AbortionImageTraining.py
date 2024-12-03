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

promptText="Here are examples of images that support abortion:\n"

tempCount=0
for imagePath in imagePaths:
    if tempCount<5:
        promptText+=imagePath+"\n"
    tempCount+=1

promptText+="Now, here are examples of images that oppose abortion:\n"
tempCount=0
for imagePath in imagePaths:
    if tempCount>4:
        promptText+=imagePath+"\n"
    tempCount+=1

promptText+="Please describe what is shown in the second example image. Be descriptive."

print(promptText)

#Use Ollama to analyze the image with Llama 3.2-Vision:
modelResponse = ollama.chat(
    model="llama3.2-vision",
    messages=[{
      "role": "user",
      "content": promptText,
      "images": [imagePaths[7]]
    }],
)

#Extract the model's response about the image:
cleanedText = modelResponse['message']['content'].strip()
print(f"Model Response: {cleanedText}")