#Librairies
import gradio as gr
import kagglehub
#Download the training dataset
kashishparmar02_social_media_sentiments_analysis_dataset_path = kagglehub.dataset_download('kashishparmar02/social-media-sentiments-analysis-dataset')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, init

import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from tqdm.notebook import tqdm
from collections import Counter
from wordcloud import WordCloud

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

import warnings
warnings.filterwarnings('ignore')

#Load the training dataset
df = pd.read_csv(f"{kashishparmar02_social_media_sentiments_analysis_dataset_path}/sentimentdataset.csv")

#count the number of null values in the dataframe
#def null_count():
#    return pd.DataFrame({'features': df.columns,
#                'dtypes': df.dtypes.values,
#                'NaN count': df.isnull().sum().values,
#                'NaN percentage': df.isnull().sum().values/df.shape[0]}).style.background_gradient(cmap='Set3',low=0.1,high=0.01)
#null_count()

# same mapping as the months to only keep 3 different sentiments
sentiments = {'Negative' : 'Negative','Positive' : 'Positive','Neutral': 'Neutral','Anger':'Negative','Fear':'Negative','Sadness':'Negative','Disgust':'Negative','Happiness':'Positive','Joy':'Positive','Love':'Positive','Amusement':'Positive','Enjoyment':'Positive','Admiration':'Positive','Affection':'Positive','Awe':'Positive','Disappointed':'Negative','Surprise':'Neutral','Acceptance':'Neutral','Adoration':'Positive','Anticipation':'Positive','Bitter':'Negative','Calmness':'Positive','Confusion':'Negative','Excitement':'Positive','Kind':'Positive','Pride':'Positive','Shame':'Negative','Elation':'Positive','Euphoria':'Positive','Contentment':'Neutral','Serenity':'Positive','Gratitude':'Positive','Hope':'Positive','Empowerment':'Positive','Compassion':'Positive','Tenderness':'Positive','Arousal':'Neutral','Enthusiasm':'Positive','Fulfillment':'Positive','Reverence':'Neutral','Despair':'Negative','Grief':'Negative','Loneliness':'Negative','Jealousy':'Negative','Resentment':'Negative','Frustration':'Negative','Boredom':'Negative','Anxiety':'Negative','Intimidation':'Negative','Helplessness':'Negative','Envy':'Negative','Regret':'Negative','Curiosity':'Neutral','Indifference':'Neutral','Numbness':'Negative','Melancholy':'Negative','Nostalgia':'Neutral','Ambivalence':'Neutral','Determination':'Positive','Zest':'Positive','Hopeful':'Positive','Proud':'Positive','Grateful':'Positive','Empathetic':'Positive','Compassionate':'Positive','Playful':'Positive','Free-spirited':'Positive','Inspired':'Positive','Confident':'Positive','Bitterness':'Negative','Yearning':'Positive','Fearful':'Negative','Apprehensive':'Negative','Overwhelmed':'Negative','Jealous':'Negative','Devastated':'Negative','Frustrated':'Negative','Envious':'Negative','Dismissive':'Negative','Thrill':'Positive','Bittersweet':'Neutral','Overjoyed':'Positive','Inspiration':'Positive','Motivation':'Positive','Contemplation':'Neutral','JoyfulReunion':'Positive','Satisfaction':'Positive','Blessed':'Positive','Reflection':'Neutral','Appreciation':'Positive','Confidence':'Positive','Accomplishment':'Positive','Wonderment':'Positive','Optimism':'Positive','Enchantment':'Positive','Intrigue':'Positive','PlayfulJoy':'Positive','Mindfulness':'Positive','DreamChaser':'Positive','Elegance':'Positive','Whimsy':'Neutral','Pensive':'Neutral','Harmony':'Positive','Creativity':'Positive','Radiance':'Positive','Wonder':'Positive','Rejuvenation':'Positive','Coziness':'Positive','Adventure':'Positive','Melodic':'Positive','FestiveJoy':'Positive','InnerJourney':'Positive','Freedom':'Positive','Dazzle':'Positive','Adrenaline':'Positive','ArtisticBurst':'Positive','CulinaryOdyssey':'Positive','Resilience':'Neutral','Immersion':'Positive','Spark':'Positive','Marvel':'Positive','Heartbreak':'Negative','Betrayal':'Negative','Suffering':'Negative','EmotionalStorm':'Negative','Isolation':'Negative','Disappointment':'Negative','LostLove':'Negative','Exhaustion':'Negative','Sorrow':'Negative','Darkness':'Negative','Desperation':'Negative','Ruins':'Negative','Desolation':'Negative','Loss':'Negative','Heartache':'Negative','Solitude':'Negative','Positivity':'Positive','Kindness':'Positive','Friendship':'Positive','Success':'Positive','Exploration':'Positive','Amazement':'Positive','Romance':'Positive','Captivation':'Positive','Tranquility':'Positive','Grandeur':'Positive','Emotion':'Positive','Energy':'Positive','Celebration':'Positive','Charm':'Positive','Ecstasy':'Positive','Colorful':'Positive','Hypnotic':'Neutral','Connection':'Positive','Iconic':'Neutral','Journey':'Neutral','Engagement':'Positive','Touched':'Positive','Suspense':'Neutral','Triumph':'Positive','Heartwarming':'Positive','Obstacle':'Negative','Sympathy':'Positive','Pressure':'Negative','Renewed Effort':'Positive','Miscalculation':'Negative','Challenge':'Neutral','Solace':'Positive','Breakthrough':'Positive','Joy in Baking':'Positive','Envisioning History':'Positive','Imagination':'Positive','Vibrancy':'Positive','Mesmerizing':'Positive','Culinary Adventure':'Positive','Winter Magic':'Positive','Thrilling Journey':'Positive','Nature s Beauty':'Positive','Celestial Wonder':'Positive','Creative Inspiration':'Positive','Runway Creativity':'Positive','Ocean s Freedom':'Positive','Whispers of the Past':'Positive','Relief':'Positive','Embarrassed':'Negative','Mischievous':'Negative','Sad':'Negative','Hate':'Negative','Bad':'Negative','Happy':'Positive'}
df['Sentiment'] = df['Sentiment'].str.strip()
df["Sentiment"] = df["Sentiment"].map(sentiments)
df['Sentiment'] = df['Sentiment'].astype('object')

#Splits the text to get the words themselves
df['temp_list'] = df['Text'].apply(lambda x: str(x).split())

#Model building
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

#cleaning the data from stopwords (words that are not useful to the model like : the, a an ...)
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
df['temp_list_cleaned'] = df['temp_list'].apply(lambda word_list: [w for w in word_list if w.lower() not in stops])

#Removes the rows with null values in the Sentiment column
df = df.dropna(subset = ["Sentiment"], axis = 0, inplace = False)

#Creates the X and y variables
X = df['temp_list_cleaned'].apply(lambda x: ' '.join(x)).values
y = df['Sentiment'].values

#dividing into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

# Create a bag of words for our dataset, using the CountVectorizer
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)
feature_names = vectorizer.get_feature_names_out()

#Model used : Passive Aggressive Classifier      (Accuracy: 0.84)
from sklearn.linear_model import PassiveAggressiveClassifier
PassiveAggressiveClassifier = PassiveAggressiveClassifier()
PassiveAggressiveClassifier.fit(X_train_bow, y_train)

def predict_sentiment(text):
    #preparing the data
    from nltk.tokenize import sent_tokenize, word_tokenize
    #Overwriting the previous train bow one final time to get the answer
    X_test_bow = vectorizer.transform([text])
    #Prediction
    y_pred_logistic = PassiveAggressiveClassifier.predict(X_test_bow)
    return("Your text : '" + text + "' is considered by the AI as : " + y_pred_logistic[0])


demo = gr.Interface(
    fn=predict_sentiment,
    inputs="text",
    outputs="text",
    title="Sentiment Analysis",
    description="Analyse a text's /sentence's sentiment"
)
demo.launch()