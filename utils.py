from textblob import TextBlob
import pyttsx3

# map số -> chữ cái ASL
def map_sign_to_letter(index):
    return chr(65+index)

# auto-correct câu
def correct_sentence(sentence):
    return str(TextBlob(sentence).correct())

# text-to-speech
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
