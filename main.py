import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from reviews_preprocessing import String_processing

# Loading Dataset from Device 
read_xlsx = pd.read_excel("data_set/Virtual Assistant Devices Dataset _Final Version.xlsx",usecols= ["Comments","Sentiment","Aspect(s)"])
print(read_xlsx.head(20))

# Data_set EDA 
print(read_xlsx.info())
print(read_xlsx.describe())
find_missing_values = read_xlsx.isnull().sum()
print(find_missing_values)
find_duplicate_values = read_xlsx.duplicated().sum()
print(find_duplicate_values)

# Null Value Aspects (coparesion in sentiment column values)
print(read_xlsx.isnull().mean()*100)
nul_col = read_xlsx[read_xlsx["Aspect(s)"].isnull()]
print(nul_col["Sentiment"].value_counts())
plt.figure(figsize=(14,8))
sns.countplot(x = nul_col["Sentiment"])
plt.title("Null Value Aspects (coparesion in sentiment column values)")
plt.show()
# view catagorie in aspect column 
plt.figure(figsize = (20,8))
sns.countplot(read_xlsx["Aspect(s)"])
plt.title("Catagories in Aspect(s)")
plt.show()
# remove NUll values 
read_xlsx.dropna(inplace=True)
print(read_xlsx.isnull().sum())

# remove Duplicate values 
read_xlsx.drop_duplicates(inplace = True)
print(read_xlsx.duplicated().sum())

# Capping Catagories in Aspects 
aspect_common = read_xlsx["Aspect(s)"].value_counts() 
aspect_common = aspect_common[aspect_common > 45].index
print(aspect_common)
read_xlsx["Aspect(s)"] = np.where(read_xlsx["Aspect(s)"].isin(aspect_common) ,read_xlsx["Aspect(s)"] , "other" )
print(read_xlsx["Aspect(s)"].value_counts())

# Check Data Stability 
print(read_xlsx["Sentiment"].value_counts())
plt.figure(figsize= (14 , 8))
plt.pie(read_xlsx["Sentiment"].value_counts(),labels=read_xlsx["Sentiment"].value_counts().index,autopct = '%1.2f%%')
plt.title("Data Stability")
plt.show()


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

# Encode the Aspect Column
aspect_encoder = LabelEncoder()
read_xlsx["Aspect(s)"] = aspect_encoder.fit_transform(read_xlsx["Aspect(s)"])
print(dict(zip(aspect_encoder.classes_, aspect_encoder.transform(aspect_encoder.classes_))))



X_sentiment = read_xlsx[["Process_Reviews"]]
y_sentiment = read_xlsx["Sentiment"]
x_train_sentient,x_test_sentient,y_train_sentient,y_test_sentient  = train_test_split(X_sentiment,y_sentiment,random_state=42,test_size=0.2)
print(x_test_sentient.shape)

X_aspect = read_xlsx[["Process_Reviews"]]
y_aspect = read_xlsx["Aspect(s)"]
x_train_aspect,x_test_aspect,y_train_aspect,y_test_aspect  = train_test_split(X_aspect ,y_aspect,random_state=42,test_size=0.2)
print()

# Apply TF-IDF Vecterizor
vectorizor = TfidfVectorizer(ngram_range = (1,2) , max_features = 100000, sublinear_tf = True )
x_train_vectorize = vectorizor.fit_transform(x_train_sentient["Process_Reviews"])
x_test_vectorize = vectorizor.transform(x_test_sentient["Process_Reviews"])
print(x_train_vectorize.shape)

vector = TfidfVectorizer(ngram_range = (1,2) , max_features = 100000, sublinear_tf = True )
x_train_aspect_vec = vector.fit_transform(x_train_aspect["Process_Reviews"])
x_test_aspect_vec = vector.transform(x_test_aspect["Process_Reviews"])

# Linear SVC Model 
svc_model = LinearSVC()
svc_model.fit(x_train_vectorize,y_train_sentient)
svc_model_pred = svc_model.predict(x_test_vectorize)
print(f"Linear SVC Accuracy : {accuracy_score(y_test_sentient , svc_model_pred)}")
print(f"Linear SVC Confusion_matrix : {confusion_matrix(y_test_sentient , svc_model_pred)}")
print(f"Linear SVC Classification_report : {classification_report(y_test_sentient , svc_model_pred)}")

# Aspect_model Model 
aspect_model = LinearSVC()
aspect_model.fit(x_train_aspect_vec,y_train_aspect)
aspect_model_pred = aspect_model.predict(x_test_aspect_vec)
print(f"Aspect_Model Accuracy : {(accuracy_score(y_test_aspect, aspect_model_pred))}")
print(f"Aspect_Model Confusion_matrix : {(confusion_matrix(y_test_aspect , aspect_model_pred))}")
print(f"Aspect_Model Classification_report : {(classification_report(y_test_aspect , aspect_model_pred))}")

def user_input(user_input):
    input_process = String_processing(user_input)
    print(input_process)
    input_vectorize = vector.transform([input_process])
    print(input_vectorize.toarray())
    model_pred = aspect_model.predict(input_vectorize)
    return model_pred[0]

user_inp = input("Drop a Comment : ")
result = user_input(user_inp)
print(aspect_encoder.inverse_transform([result])[0])

# Cross Validate the Accuracy of Model
svc_pipline = make_pipeline(
 (vectorizor),
 (svc_model)
)
x_train_sentient = x_train_sentient.iloc[:, 0]
print(x_train_sentient.shape)
print(y_train_sentient.shape)
k_fold = KFold(n_splits = 5 , shuffle = True , random_state = 42)
cv_result = cross_val_score(svc_pipline , x_train_sentient , y_train_sentient , cv = k_fold , scoring= "accuracy")
print(cv_result.mean())

# Cross Validate the Accuracy of Aspect's_Model
aspect_pipline = make_pipeline(
 (vector),
 (aspect_model)
)
x_train_aspect = x_train_aspect.iloc[:, 0]
print(x_train_aspect.shape)
print(y_train_aspect.shape)
k_fold = KFold(n_splits = 5 , shuffle = True , random_state = 42)
cv_result = cross_val_score(aspect_pipline , x_train_aspect , y_train_aspect , cv = k_fold , scoring= "accuracy")
print(cv_result.mean())