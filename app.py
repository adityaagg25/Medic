import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Download the VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Function to analyze sentiment based on the value
def analyze_sentiment(value):
    if value > 0.7:
        return "High Attention"
    elif value >= 0.4:
        return "Medium Attention"
    else:
        return "Low Attention"

# Streamlit app title and description
st.title("BRAIN EEG ANALYZER 🧠")
st.write("This website analyzes EEG reports to determine your brain's attention level based on the values provided in a CSV file.")

# File upload
uploaded_file = st.file_uploader("Upload any CSV file with numeric data", type="csv")

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)
    
    # Display the dataset
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Identify numeric columns in the dataset
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(numeric_columns) == 0:
        st.write("No numeric columns found in the dataset.")
    else:
        # Let user select which numeric column to analyze
        selected_column = st.selectbox("Select a numeric column to analyze:", numeric_columns)

        # Apply the sentiment analysis based on the selected numeric column
        df['Attention Level'] = df[selected_column].apply(analyze_sentiment)

        # Display the dataframe with analyzed attention levels
        st.write("Dataset with Attention Levels:")
        st.dataframe(df)

        # Visualizations
        st.subheader("Sentiment Distribution (Attention Levels)")
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Attention Level', data=df)
        plt.title('Attention Level Distribution')
        plt.xlabel('Attention Level')
        plt.ylabel('Count')
        st.pyplot(plt)

        st.subheader("Sentiment Distribution - Scatter Plot")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df.index, y=selected_column, hue="Attention Level", data=df)
        plt.title('Attention Level Distribution - Scatter Plot')
        plt.xlabel('Index')
        plt.ylabel('Selected Numeric Column Values')
        st.pyplot(plt)

# Run the app
if __name__ == '__main__':
    st.write("Upload a CSV file to analyze the EEG report or any dataset.")
