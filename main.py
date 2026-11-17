import pandas as pd 
import matplotlib.pyplot as plt
import seaborn  as sns

import re
from contractions import fix
from emoji import demojize
import string 
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.svm import LinearSVC
# Download Packages 
nltk.download("punkt_tab")
nltk.download("stopwords")

# Define Functions 
stop_word = set(stopwords.words("english"))
stemer  = SnowballStemmer(language="english")


# Loading Dataset from Device 
read_xlsx = pd.read_excel("data_set/Virtual Assistant Devices Dataset _Final Version.xlsx",usecols= ["Comments","Sentiment"])
print(read_xlsx.head(20))

# Data_set EDA 
print(read_xlsx.info())
print(read_xlsx.describe())
find_missing_values = read_xlsx.isnull().sum()
print(find_missing_values)
find_duplicate_values = read_xlsx.duplicated().sum()
print(find_duplicate_values)

# remove Duplicate values 
read_xlsx.drop_duplicates(inplace = True)
print(read_xlsx.shape)

# Check Data Stability 
print(read_xlsx["Sentiment"].value_counts())
plt.figure(figsize= (14 , 8))
plt.pie(read_xlsx["Sentiment"].value_counts(),labels=read_xlsx["Sentiment"].value_counts().index,autopct = '%1.2f%%')
plt.title("Data Stability")
plt.show()


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



sample_string = r"I love it when professors draw a big question mark next to my answer on an exam because I’m always like yeah I don’t either ¯\_(ツ)_/¯ @VolphanCarol @littlewhitty @mysticalmanatee https://t.co/yZlafy0lsd <h1>hello</h1> 🤪 growthing referring teachings not no"
strings = String_processing(sample_string)
print(strings)
read_xlsx["Process_Reviews"] = read_xlsx["Comments"].apply(lambda x:String_processing(x))
print(read_xlsx["Process_Reviews"])


# Encode the Label Column
encoder = LabelEncoder()
read_xlsx["Sentiment"] = encoder.fit_transform(read_xlsx["Sentiment"])
print(dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))
print(read_xlsx["Sentiment"].head())
X = read_xlsx[["Process_Reviews"]]
y = read_xlsx["Sentiment"]
x_train,x_test,y_train,y_test  = train_test_split(X,y,random_state=42,test_size=0.2)
print(x_test.shape)

# Apply TF-IDF Vecterizor
vectorizor = TfidfVectorizer(ngram_range = (1,2) , max_features = 100000, sublinear_tf = True )
x_train_vectorize = vectorizor.fit_transform(x_train["Process_Reviews"])
x_test_vectorize = vectorizor.transform(x_test["Process_Reviews"])
print(x_train_vectorize.toarray())

# Linear SVC Model 
svc_model = LinearSVC()
svc_model.fit(x_train_vectorize,y_train)
svc_model_pred = svc_model.predict(x_test_vectorize)
print(f"Linear SVC Accuracy : {accuracy_score(y_test , svc_model_pred)}")
print(f"Linear SVC Confusion_matrix : {confusion_matrix(y_test , svc_model_pred)}")
print(f"Linear SVC Classification_report : {classification_report(y_test , svc_model_pred)}")

# RandomForestClassifier Model 
rfc_model = RandomForestClassifier()
rfc_model.fit(x_train_vectorize,y_train)
rfc_model_pred = rfc_model.predict(x_test_vectorize)
print(f"RandomForestClassifier Accuracy : {(accuracy_score(y_test , rfc_model_pred))}")
print(f"RandomForestClassifier Confusion_matrix : {(confusion_matrix(y_test , rfc_model_pred))}")
print(f"RandomForestClassifier Classification_report : {(classification_report(y_test , rfc_model_pred))}")

def user_input(user_input):
    input_process = String_processing(user_input)
    print(input_process)
    input_vectorize = vectorizor.transform([input_process])
    print(input_vectorize.toarray())
    model_pred = svc_model.predict(input_vectorize)
    return model_pred[0]

user_inp = input("Drop a Comment : ")
# {'Negative'(0), 'Neutral'(1), 'Positive'(2)}

result = user_input(user_inp)
if result  == 0 :
    result = "Negative"
elif result == 1:
    result = "Neutral"
elif result == 2:
    result = "Positive"
print(f"Model_pediction : {result}")
 
