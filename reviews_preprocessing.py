import re
from contractions import fix
from emoji import demojize
import string 
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Download Packages 
nltk.download("punkt_tab")
nltk.download("stopwords")

# Define Functions 
stop_word = set(stopwords.words("english"))
stemer  = SnowballStemmer(language="english")


# String Preprocessing 
def String_processing(text):
    text = text.lower()
    text = fix(text)
    # Remove links 
    text = re.sub(r"http\S+", "" , text)
    # Remove HTML Tags 
    text = re.sub(r"<.*?>","",text)
    # Remove Emojis 
    text = demojize(text, language= "en", delimiters=(" " , " "))
    text = text.replace(" _ " , " ")
    # Removing Special Character
    text = text.translate(str.maketrans("" ,"" ,string.punctuation))
    # text = re.sub(r"[^a-zA-Z\s]" ,"" ,text)
    # tokenize the string 
    text = word_tokenize(text )
    # Removing Stop words
    negative_words = {'not', 'no', 'nor', 'never',    'bad', 'worst', 'poor', 'terrible', 'awful','horrible', 'disappointed', 'disappointing', 'useless', 'waste', 'annoying', 'unhappy','hate', 'slow', 'weak', 'broken', 'defective',
    'faulty', 'damaged', 'noisy', 'expensive' ,  'hardly', 'barely', 'without'}
    stop_word = set(stopwords.words("english"))
    stop_word = stop_word - negative_words
    text = [word for word in text if word not in stop_word]
    text = [stemer.stem(word) for word in text]
    text = " ".join(text)
    return text



