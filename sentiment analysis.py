import kagglehub
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

df = pd.read_csv(f"{kashishparmar02_social_media_sentiments_analysis_dataset_path}/sentimentdataset.csv")

#count the number of null values in the dataframe
#def null_count():
#    return pd.DataFrame({'features': df.columns,
#                'dtypes': df.dtypes.values,
#                'NaN count': df.isnull().sum().values,
#                'NaN percentage': df.isnull().sum().values/df.shape[0]}).style.background_gradient(cmap='Set3',low=0.1,high=0.01)
#null_count()

#Data cleaning
df['Country'] = df['Country'].str.strip()
df['Platform'] = df['Platform'].str.strip()

df['Timestamp'] = pd.to_datetime(df['Timestamp'])

df['Day_of_Week'] = df['Timestamp'].dt.day_name()

#Month mapping
month_mapping = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
}

df['Month'] = df['Month'].map(month_mapping)

df['Month'] = df['Month'].astype('object')

# same mapping as the months to only keep 3 different sentiments
sentiments = {'Negative' : 'Negative','Positive' : 'Positive','Neutral': 'Neutral','Anger':'Negative','Fear':'Negative','Sadness':'Negative','Disgust':'Negative','Happiness':'Positive','Joy':'Positive','Love':'Positive','Amusement':'Positive','Enjoyment':'Positive','Admiration':'Positive','Affection':'Positive','Awe':'Positive','Disappointed':'Negative','Surprise':'Neutral','Acceptance':'Neutral','Adoration':'Positive','Anticipation':'Positive','Bitter':'Negative','Calmness':'Positive','Confusion':'Negative','Excitement':'Positive','Kind':'Positive','Pride':'Positive','Shame':'Negative','Elation':'Positive','Euphoria':'Positive','Contentment':'Neutral','Serenity':'Positive','Gratitude':'Positive','Hope':'Positive','Empowerment':'Positive','Compassion':'Positive','Tenderness':'Positive','Arousal':'Neutral','Enthusiasm':'Positive','Fulfillment':'Positive','Reverence':'Neutral','Despair':'Negative','Grief':'Negative','Loneliness':'Negative','Jealousy':'Negative','Resentment':'Negative','Frustration':'Negative','Boredom':'Negative','Anxiety':'Negative','Intimidation':'Negative','Helplessness':'Negative','Envy':'Negative','Regret':'Negative','Curiosity':'Neutral','Indifference':'Neutral','Numbness':'Negative','Melancholy':'Negative','Nostalgia':'Neutral','Ambivalence':'Neutral','Determination':'Positive','Zest':'Positive','Hopeful':'Positive','Proud':'Positive','Grateful':'Positive','Empathetic':'Positive','Compassionate':'Positive','Playful':'Positive','Free-spirited':'Positive','Inspired':'Positive','Confident':'Positive','Bitterness':'Negative','Yearning':'Positive','Fearful':'Negative','Apprehensive':'Negative','Overwhelmed':'Negative','Jealous':'Negative','Devastated':'Negative','Frustrated':'Negative','Envious':'Negative','Dismissive':'Negative','Thrill':'Positive','Bittersweet':'Neutral','Overjoyed':'Positive','Inspiration':'Positive','Motivation':'Positive','Contemplation':'Neutral','JoyfulReunion':'Positive','Satisfaction':'Positive','Blessed':'Positive','Reflection':'Neutral','Appreciation':'Positive','Confidence':'Positive','Accomplishment':'Positive','Wonderment':'Positive','Optimism':'Positive','Enchantment':'Positive','Intrigue':'Positive','PlayfulJoy':'Positive','Mindfulness':'Positive','DreamChaser':'Positive','Elegance':'Positive','Whimsy':'Neutral','Pensive':'Neutral','Harmony':'Positive','Creativity':'Positive','Radiance':'Positive','Wonder':'Positive','Rejuvenation':'Positive','Coziness':'Positive','Adventure':'Positive','Melodic':'Positive','FestiveJoy':'Positive','InnerJourney':'Positive','Freedom':'Positive','Dazzle':'Positive','Adrenaline':'Positive','ArtisticBurst':'Positive','CulinaryOdyssey':'Positive','Resilience':'Neutral','Immersion':'Positive','Spark':'Positive','Marvel':'Positive','Heartbreak':'Negative','Betrayal':'Negative','Suffering':'Negative','EmotionalStorm':'Negative','Isolation':'Negative','Disappointment':'Negative','LostLove':'Negative','Exhaustion':'Negative','Sorrow':'Negative','Darkness':'Negative','Desperation':'Negative','Ruins':'Negative','Desolation':'Negative','Loss':'Negative','Heartache':'Negative','Solitude':'Negative','Positivity':'Positive','Kindness':'Positive','Friendship':'Positive','Success':'Positive','Exploration':'Positive','Amazement':'Positive','Romance':'Positive','Captivation':'Positive','Tranquility':'Positive','Grandeur':'Positive','Emotion':'Positive','Energy':'Positive','Celebration':'Positive','Charm':'Positive','Ecstasy':'Positive','Colorful':'Positive','Hypnotic':'Neutral','Connection':'Positive','Iconic':'Neutral','Journey':'Neutral','Engagement':'Positive','Touched':'Positive','Suspense':'Neutral','Triumph':'Positive','Heartwarming':'Positive','Obstacle':'Negative','Sympathy':'Positive','Pressure':'Negative','Renewed Effort':'Positive','Miscalculation':'Negative','Challenge':'Neutral','Solace':'Positive','Breakthrough':'Positive','Joy in Baking':'Positive','Envisioning History':'Positive','Imagination':'Positive','Vibrancy':'Positive','Mesmerizing':'Positive','Culinary Adventure':'Positive','Winter Magic':'Positive','Thrilling Journey':'Positive','Nature s Beauty':'Positive','Celestial Wonder':'Positive','Creative Inspiration':'Positive','Runway Creativity':'Positive','Ocean s Freedom':'Positive','Whispers of the Past':'Positive','Relief':'Positive','Embarrassed':'Negative','Mischievous':'Negative','Sad':'Negative','Hate':'Negative','Bad':'Negative','Happy':'Positive'}
df['Sentiment'] = df['Sentiment'].str.strip()
df["Sentiment"] = df["Sentiment"].map(sentiments)

df['Sentiment'] = df['Sentiment'].astype('object')

#Inspecting the data
"""
#Sentiment distribution
colors = ['#66b3ff', '#99ff99', '#ffcc99']

explode = (0.1, 0, 0)

sentiment_counts = df.groupby("Sentiment").size()

fig, ax = plt.subplots()

wedges, texts, autotexts = ax.pie(
    x=sentiment_counts,
    labels=sentiment_counts.index,
    autopct=lambda p: f'{p:.2f}%\n({int(p*sum(sentiment_counts)/100)})',
    wedgeprops=dict(width=0.7),
    textprops=dict(size=10, color="r"),
    pctdistance=0.7,
    colors=colors,
    explode=explode,
    shadow=True)

center_circle = plt.Circle((0, 0), 0.6, color='white', fc='white', linewidth=1.25)
fig.gca().add_artist(center_circle)

ax.text(0, 0, 'Sentiment\nDistribution', ha='center', va='center', fontsize=14, fontweight='bold', color='#333333')

ax.legend(sentiment_counts.index, title="Sentiment", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

ax.axis('equal')

plt.show()


#Relationship between Years and Sentiment
plt.figure(figsize=(12, 6))
sns.countplot(x='Year', hue='Sentiment', data=df, palette='Paired')
plt.title('Relationship between Years and Sentiment')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

#Relationship between Month and Sentiment
plt.figure(figsize=(12, 6))
sns.countplot(x='Month', hue='Sentiment', data=df, palette='Paired')
plt.title('Relationship between Month and Sentiment')
plt.xlabel('Month')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

#Relationship between Day of Week and Sentiment
plt.figure(figsize=(12, 6))
sns.countplot(x='Day_of_Week', hue='Sentiment', data=df, palette='Paired')
plt.title('Relationship between Day of Week and Sentiment')
plt.xlabel('Day of Week')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

#Relationship between Platform and Sentiment
plt.figure(figsize=(12, 6))
sns.countplot(x='Platform', hue='Sentiment', data=df, palette='Paired')
plt.title('Relationship between Platform and Sentiment')
plt.xlabel('Platform')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

#Relationship between Country and Sentiment (Top 10 Countries)
plt.figure(figsize=(12, 6))
top_10_countries = df['Country'].value_counts().head(10).index
df_top_10_countries = df[df['Country'].isin(top_10_countries)]
sns.countplot(x='Country', hue='Sentiment', data=df_top_10_countries, palette='Paired')
plt.title('Relationship between Country and Sentiment (Top 10 Countries)')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

#word data
#Most common words
df['temp_list'] = df['Text'].apply(lambda x: str(x).split())
top_words = Counter([item for sublist in df['temp_list'] for item in sublist])
top_words_df = pd.DataFrame(top_words.most_common(20), columns=['Common_words', 'count'])
fig = px.bar(top_words_df,
            x="count",
            y="Common_words",
            title='Common Words in Text Data',
            orientation='h',
            width=700,
            height=700,
            color='Common_words')
fig.show()

#positive words
df_positif = df[df['Sentiment'] == 'Positive']
colonne_temp_list = df_positif['temp_list']
top = Counter([item for sublist in colonne_temp_list for item in sublist])
temp_positive = pd.DataFrame(top.most_common(10), columns=['Common_words', 'count'])
temp_positive.style.background_gradient(cmap='Greens')

#negative words
df_negatif = df[df['Sentiment'] == 'Negative']
colonne_temp_list = df_negatif['temp_list']
top = Counter([item for sublist in colonne_temp_list for item in sublist])
temp_negative = pd.DataFrame(top.most_common(10), columns=['Common_words', 'count'])
temp_negative.style.background_gradient(cmap='Reds')

#Wordcloud view of sentiment words
Positive_sent = df[df['Sentiment'] == 'Positive']
Negative_sent = df[df['Sentiment'] == 'Negative']
Neutral_sent = df[df['Sentiment'] == 'Neutral']

#Positive word cloud
words = ' '.join([item for sublist in df[df['Sentiment'] == 'Positive']['temp_list'] for item in sublist])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#Negative word cloud
words = ' '.join([item for sublist in df[df['Sentiment'] == 'Negative']['temp_list'] for item in sublist])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


#Neutral word cloud
words = ' '.join([item for sublist in df[df['Sentiment'] == 'Neutral']['temp_list'] for item in sublist])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
"""
#Splits the text to get the words themselves
df['temp_list'] = df['Text'].apply(lambda x: str(x).split())

#Model building
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

#cleaning the data from stopwords (words that are not useful for the model like : the, a an ...)
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

#Model 1 : Logistic Regression      Accuracy: 0.81
logistic_classifier = LogisticRegression(max_iter=50, random_state=42)
logistic_classifier.fit(X_train_bow, y_train)
y_pred_logistic = logistic_classifier.predict(X_test_bow)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
classification_rep_logistic = classification_report(y_test, y_pred_logistic)
#Results
"""
print("Logistic Regression Results:")
print(f"Accuracy: {accuracy_logistic}")
print("Classification Report:\n", classification_rep_logistic)
"""

#Another approach     Accuracy: 0.76
#Training with another vectorizer
vectorizer2 = TfidfVectorizer(max_features=5000)
X_train_bow = vectorizer2.fit_transform(X_train)
X_test_bow = vectorizer2.transform(X_test)
feature_names = vectorizer2.get_feature_names_out()
#Training with the new vectorizer
logistic_classifier = LogisticRegression(max_iter=50, random_state=42)
logistic_classifier.fit(X_train_bow, y_train)
y_pred_logistic = logistic_classifier.predict(X_test_bow)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
classification_rep_logistic = classification_report(y_test, y_pred_logistic)

#results
"""
print("Logistic Regression Results:")
print(f"Accuracy: {accuracy_logistic}")
print("Classification Report:\n", classification_rep_logistic)
"""
# Back to the CountVectorizer, overwriting the previous train and test bow
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)
feature_names = vectorizer.get_feature_names_out()

#Model 2 : Passive Aggressive Classifier      Accuracy: 0.84
from sklearn.linear_model import PassiveAggressiveClassifier
PassiveAggressiveClassifier = PassiveAggressiveClassifier()
PassiveAggressiveClassifier.fit(X_train_bow, y_train)
y_pred_logistic = PassiveAggressiveClassifier.predict(X_test_bow)

accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
classification_rep_logistic = classification_report(y_test, y_pred_logistic)
#results
"""
print("Passive Agressive Classifier:")
print(f"Accuracy: {accuracy_logistic}")
print("Classification Report:\n", classification_rep_logistic)
"""

#Model 3 : Random Forest      Accuracy: 0.56
from sklearn.ensemble import RandomForestClassifier
RandomForestClassifier = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42)
RandomForestClassifier.fit(X_train_bow, y_train)
y_pred_logistic = RandomForestClassifier.predict(X_test_bow)

accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
classification_rep_logistic = classification_report(y_test, y_pred_logistic)
#results
"""
print("Random Forest :")
print(f"Accuracy: {accuracy_logistic}")
print("Classification Report:\n", classification_rep_logistic)
"""

#Model 4 : Support Vector Machine      Accuracy: 0.75
from sklearn.svm import SVC
SVC = SVC (random_state=42)
SVC.fit(X_train_bow, y_train)
y_pred_logistic =SVC.predict(X_test_bow)

accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
classification_rep_logistic = classification_report(y_test, y_pred_logistic)
#results
"""
print("SVC :")
print(f"Accuracy: {accuracy_logistic}")
print("Classification Report:\n", classification_rep_logistic)
"""

#Model 5 : Naive Bayes      Accuracy: 0.75
from sklearn.naive_bayes import MultinomialNB
MultinomialNB = MultinomialNB ()
MultinomialNB.fit(X_train_bow, y_train)
y_pred_logistic =MultinomialNB.predict(X_test_bow)

accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
classification_rep_logistic = classification_report(y_test, y_pred_logistic)
#results
"""
print("MultinomialNB :")
print(f"Accuracy: {accuracy_logistic}")
print("Classification Report:\n", classification_rep_logistic)
"""
#----------------------------------------------------
#With the CountVectorizer
#Model 1 : Logistic Regression                Accuracy: 0.81
#Model 2 : Passive Aggressive Classifier      Accuracy: 0.84
#Model 3 : Random Forest                      Accuracy: 0.56
#Model 4 : Support Vector Machine             Accuracy: 0.75
#Model 5 : Naive Bayes                        Accuracy: 0.75
#----------------------------------------------------
#Test on new data
#----------------------------------------------------

test = ["Type your own sentence here."]

#----------------------------------------------------

#preparing the data
from nltk.tokenize import sent_tokenize, word_tokenize

#Overwriting the previous train bow one final time to get the answer
X_test_bow = vectorizer.transform(test)
#Prediction
y_pred_logistic = PassiveAggressiveClassifier.predict(X_test_bow)

print("Your text :'",test[0], "' is considered by the AI as : ",y_pred_logistic[0])
