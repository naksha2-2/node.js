import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# Sample DataFrame of text data
data = {'text': ["I love programming!", "Python is awesome.", "I hate bugs.", "I love debugging my code.", "Coding is fun!"]}
df = pd.DataFrame(data)

# Create the TfidfVectorizer instance
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the text data into a TF-IDF matrix
X = vectorizer.fit_transform(df['text'])

# Get the words and their corresponding TF-IDF scores
words = vectorizer.get_feature_names_out()
tfidf_scores = X.sum(axis=0).A1  # Sum of TF-IDF scores for each word

# Create a DataFrame for easier analysis
word_freq_df = pd.DataFrame(list(zip(words, tfidf_scores)), columns=['Word', 'TF-IDF'])
word_freq_df = word_freq_df.sort_values(by='TF-IDF', ascending=False)

# Display the words with the highest TF-IDF scores
print(word_freq_df.head())

# Create a word cloud to visualize the most important words
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(zip(word_freq_df['Word'], word_freq_df['TF-IDF'])))

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
