import csv
import ollama
from ollama import Client
from sklearn.metrics import confusion_matrix, classification_report

client = Client()

"""
This program uses prompt engineering coupled with the few-shot learning method. The
model is asked to determine whether several images support or oppose abortion. No context is provided
within the prompt. The accuracy is recorded. Then, the model is provided with context within the
prompt (i.e. example images and their stance), and then asked to determine the stance of the same initial
images. The accuracy is recorded, and compared to the initial results.

"""

#Collect the image paths, as well as the image stances, for the query set:
queryPaths=[]
queryStances=[]
with open('Textual Tweet Training/Abortion/AbortionImageQueryPaths.csv', 'r') as file:
    csvReader = csv.reader(file)
    for tempRow in csvReader:
        queryPaths.append(tempRow[0])
        queryStances.append(tempRow[1])

#Without providing context, record the model's responses:
print("Pre-Trained Model Results:")
promptText="Determine if the following image supports abortion, opposes abortion, or is neutral. For each image, only respond with a single word: support, oppose, or neutral"

#Iterate through each query image, and find the result:
responseAr=[]
for tempPath in queryPaths:
  modelResponse = ollama.chat(
      model="llama3.2-vision",
      messages=[{
        "role": "user",
        "content": promptText,
        "images": [tempPath]
      }],
  )
  #Extract the model's response about the image:
  cleanedText = modelResponse['message']['content'].strip()
  responseAr.append(cleanedText)
  print(cleanedText)

print("Actual Stances:\n")
print(queryStances)
print("Generated Stances:\n")
print(responseAr)

"""
print("---")
print("Trained (Few-Shot) Model Responses: ")

responseAr=[]

#Read all of the example image paths from the CSV file, along with the image stance:
imagePaths=[]
imageStance=[]
with open('Textual Tweet Training/Abortion/AbortionImageTrainingPaths.csv', 'r') as file:
    csvReader = csv.reader(file)
    for tempRow in csvReader:
        imagePaths.append(tempRow[0])
        imageStance.append(tempRow[1])

#After providing context with few-shot learning, provide the model's response:
promptText="Here are some example images, as well as whether the image supports or opposes abortion:\n"

for itNum in range(len(imagePaths)):
    promptText+="Image #"+str(itNum+1)+": "+imagePaths[itNum]+" - "+imageStance[itNum]+"\n"

promptText+="Determine if the following images support abortion, oppose abortion, or are neutral. For each image, respond with a single word: support, oppose, neutral."

#Use Ollama to analyze the image with Llama 3.2-Vision:
for tempPath in queryPaths:
  modelResponse = ollama.chat(
      model="llama3.2-vision",
      messages=[{
        "role": "user",
        "content": promptText,
        "images": [queryPaths]
      }],
  )
  #Extract the model's response about the image:
  cleanedText = modelResponse['message']['content'].strip()
  responseAr.append(cleanedText)

#print(f"Model Response: {cleanedText}")
print("Actual Stances:\n")
print(queryStances)
print("Generated Stances:\n")
print(responseAr)
"""