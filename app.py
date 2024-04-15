import streamlit as st
import pandas as pd
import re
from transformers import pipeline
import matplotlib.pyplot as plt

# Load the first 10 rows of the DataFrame
@st.cache_data
def load_data():
    df = pd.read_csv("reviews.csv", nrows=10)
    return df

# Function to clean the review text
def clean_text(text):
    # Convert non-string values to string
    if not isinstance(text, str):
        text = str(text)
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation marks
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to perform sentiment analysis
def classify_sentiment(text):
    classifier = pipeline(
        task="zero-shot-classification",
        model="facebook/bart-large-mnli"
    )
    result = classifier(text, ["positive", "negative", 'neutral'])
    sorted_results = sorted(zip(result['labels'], result['scores']), key=lambda x: x[1], reverse=True)
    return sorted_results[0][0]

# Streamlit app
def main():
    # Set page title and favicon
    st.set_page_config(page_title="Sentiment Analysis App", page_icon=":bar_chart:")

    st.title('Sentiment Analysis')

    # Option to input text
    user_input = st.text_area("Enter text for sentiment analysis:", "")

    # Option to upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file (Note: the file should contain a review column)", type=["csv"])

    if st.button("Analyze Text") and user_input:
        # Perform sentiment analysis on the input text
        sentiment = classify_sentiment(user_input)
        st.write(f"Sentiment: {sentiment}")

    if st.button("Analyze CSV") and uploaded_file:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        df = df.head(10)
        # Perform sentiment analysis on the review column
        if 'review' in df.columns:
            df['clean_review'] = df['review'].apply(clean_text)
            df['sentiment'] = df['clean_review'].apply(classify_sentiment)

            st.write(df[['review', 'sentiment']])

            # Display the distribution of sentiments using a bar chart
            st.write('Distribution of Sentiments (Bar Chart):')
            sentiment_counts = df['sentiment'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(sentiment_counts.index, sentiment_counts.values)
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Sentiments')
            st.pyplot(fig)

            # Display the distribution of sentiments using a pie chart with percentages
            st.write('Distribution of Sentiments (Pie Chart with Percentages):')
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
            ax.set_title('Distribution of Sentiments')
            ax.axis('equal')
            st.pyplot(fig)

        else:
            st.error("CSV file does not contain a 'review' column.")

if __name__ == "__main__":
    main()
