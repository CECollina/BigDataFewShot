import csv
import ollama
from ollama import Client
from sklearn.metrics import confusion_matrix, classification_report

client = Client()

"""
This program tests the effect of providing context to the LLM through images.
"""
#Test image:
promptImage="data/dataset/dataset/Abort/1079913760536571904.jpg"

#Without providing context, record the model's responses:
print("LLM Results Prior To Training:")
promptText="Determine if the following image supports, opposes, or is neutral to abortion."

#Iterate through each query image, and find the result:
modelResponse = ollama.chat(
    model="llama3.2-vision",
    messages=[{
        "role": "user",
        "content": promptText,
        "images": [promptImage]
    }],
)

#Extract the model's response about the image:
cleanedText = modelResponse['message']['content'].strip()
print(cleanedText)

print("---")
print("Trained (Few-Shot) LLM Responses: ")

#Read all of the example image paths from the CSV file, along with the image stance:
imagePaths=[]
imageStance=[]
with open('Textual Tweet Training/Abortion/Image Processing/AbortionImageTrainingPaths.csv', 'r') as file:
    csvReader = csv.reader(file)
    for tempRow in csvReader:
        imagePaths.append(tempRow[0])
        imageStance.append(tempRow[1])

#After providing context with few-shot learning, provide the model's response:
promptText="Here are some example image paths, as well as whether each image supports or opposes abortion:\n"

for itNum in range(len(imagePaths)):
    promptText+="Image #"+str(itNum+1)+": "+imagePaths[itNum]+" - "+imageStance[itNum]+"\n"

promptText+="Based on the above example images, determine if the following image supports or opposes abortion."

#Use Ollama to analyze the image with Llama 3.2-Vision:
modelResponse = ollama.chat(
  model="llama3.2-vision",
  messages=[{
        "role": "user",
        "content": promptText,
        "images": [promptImage]
    }],
)

#Extract the model's response about the image:
cleanedText = modelResponse['message']['content'].strip()
print(cleanedText)

