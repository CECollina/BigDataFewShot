import csv
import ollama
from ollama import Client
from sklearn.metrics import confusion_matrix, classification_report

client = Client()

#Read all of the image paths from the CSV file, along with the image stance:
imagePaths=[]
imageStance=[]
with open('Textual Tweet Training/Abortion/AbortionImageTrainingPaths2.csv', 'r') as file:
    csvReader = csv.reader(file)
    for tempRow in csvReader:
        imagePaths.append(tempRow[0])
        imageStance.append(tempRow[1])

promptText="Here are some example images, as well as whether the image supports or opposes abortion:\n"

for itNum in range(len(imagePaths)):
    promptText+="Image #"+str(itNum+1)+": "+imagePaths[itNum]+" - "+imageStance[itNum]+"\n"

promptText+="Describe each image in the list. Be descriptive, and specify things are in the image. "

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