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

#Read all of the example image paths from the CSV file, along with the image stance:
imagePaths=[]
imageStance=[]
with open('Textual Tweet Training/Abortion/Image Processing/AbortionImageTrainingPaths.csv', 'r') as file:
    csvReader = csv.reader(file)
    for tempRow in csvReader:
        imagePaths.append(tempRow[0])
        imageStance.append(tempRow[1])

#After providing context with few-shot learning, provide the model's response:
promptText="Here are image paths, and the image's stance. Please convert the image-stance pairs to .JSON format:"

for itNum in range(len(imagePaths)):
    promptText+=imagePaths[itNum]+", "+imageStance[itNum]+"\n"

#Use Ollama to analyze the image with Llama 3.2-Vision:
modelResponse = ollama.chat(
      model="llama3.2-vision",
      messages=[{
        "role": "user",
        "content": promptText,
        "images": []
      }],
)

#Extract the model's response about the image:
cleanedText = modelResponse['message']['content'].strip()
print(cleanedText)
