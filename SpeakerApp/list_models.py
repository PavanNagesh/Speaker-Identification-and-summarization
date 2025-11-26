import google.generativeai as genai
import os

# Use your API key here (but don't save it in the file permanently!)
os.environ["GOOGLE_API_KEY"] = "AIzaSyCpdigOk69FwZG6v1Ipq1jhySmEazLwmas"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("Listing available models...")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)