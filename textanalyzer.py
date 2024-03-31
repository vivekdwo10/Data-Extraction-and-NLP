import os
import re
import requests
import pandas as pd
import pyphen
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from bs4 import BeautifulSoup
from textblob import TextBlob

class TextAnalyzer:
    """
    A class to analyze text from URLs and compute various readability metrics and sentiment scores.

    Attributes:
        stop_words (set): A set of stop words to be removed during text preprocessing.
        positive_words (set): A set of positive words used in sentiment analysis.
        negative_words (set): A set of negative words used in sentiment analysis.
        dic (Pyphen): A Pyphen object for syllable counting.
        personal_pronoun_pattern (str): A regex pattern for identifying personal pronouns.
        avg_sentence_lengths (list): List to store average sentence lengths.
        complex_word_percentages (list): List to store percentages of complex words.
        fog_indices (list): List to store Fog indices.
        positive_scores (list): List to store positive scores.
        negative_scores (list): List to store negative scores.
        polarity_scores (list): List to store polarity scores.
        subjectivity_scores (list): List to store subjectivity scores.
        complex_word_counts (list): List to store counts of complex words.
        word_counts (list): List to store total word counts.
        syllable_counts_per_word (list): List to store syllable counts per word.
        personal_pronoun_counts (list): List to store counts of personal pronouns.
        average_word_lengths (list): List to store average word lengths.
    """

    def __init__(self):
        """
        Initializes the TextAnalyzer class.
        """
        # Download NLTK data for tokenization (if not already downloaded)
        nltk.download('punkt')
        
        # Initialize sets and other attributes
        self.stop_words = set()
        self.positive_words = set()
        self.negative_words = set()
        self.dic = pyphen.Pyphen(lang='en')
        self.personal_pronoun_pattern = r'\b(?:I|we|my|ours|us)\b'
        self.avg_sentence_lengths = []
        self.complex_word_percentages = []
        self.fog_indices = []
        self.positive_scores = []
        self.negative_scores = []
        self.polarity_scores = []
        self.subjectivity_scores = []
        self.complex_word_counts = []
        self.word_counts = []
        self.syllable_counts_per_word = []
        self.personal_pronoun_counts = []
        self.average_word_lengths = []

    def extract_text(self, url):
        """
        Extracts text content from the provided URL.

        Args:
            url (str): The URL from which text content is to be extracted.

        Returns:
            str: The extracted text content.
        """
        try:
            # Send an HTTP GET request to the URL
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract the article text
            article_text = soup.find('article').get_text()
            return article_text
        except Exception as e:
            print(f'Error extracting text from URL: {str(e)}')
            return None

    def preprocess_text(self, text):
        """
        Preprocesses the text by tokenizing and removing stop words.

        Args:
            text (str): The text to be preprocessed.

        Returns:
            str: The preprocessed text.
        """
        # Tokenize the text using NLTK's word_tokenize and remove stop words
        tokens = word_tokenize(text)
        filtered_words = [word for word in tokens if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)

    def load_stop_words(self, directory='StopWords'):
        """
        Loads stop words from text files in the specified directory.

        Args:
            directory (str): The directory containing stop word files.
        """
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                with open(os.path.join(directory, filename), 'r', encoding='ISO-8859-1') as file:
                    self.stop_words.update(file.read().splitlines())

    def load_positive_negative_words(self, directory='MasterDictionary'):
        """
        Loads positive and negative word lists from files in the specified directory.

        Args:
            directory (str): The directory containing positive and negative word files.
        """
        with open(os.path.join(directory, 'positive-words.txt'), 'r', encoding='ISO-8859-1') as pos_file:
            self.positive_words.update(pos_file.read().splitlines())
        with open(os.path.join(directory, 'negative-words.txt'), 'r', encoding='ISO-8859-1') as neg_file:
            self.negative_words.update(neg_file.read().splitlines())

    def process_url(self, url):
        """
        Processes the text content from a URL and computes readability metrics and sentiment scores.

        Args:
            url (str): The URL of the webpage to be processed.
        """
        article_text = self.extract_text(url)

        if article_text:
            preprocessed_text = self.preprocess_text(article_text)
            cleaned_text = re.sub(r'[?!,.]', '', preprocessed_text)  # Remove punctuation

            tokens = word_tokenize(preprocessed_text)
            sentences = sent_tokenize(preprocessed_text)
            words = word_tokenize(cleaned_text)

            complex_word_count = sum(1 for word in words if len(self.dic.inserted(word).split('-')) > 2)
            percentage_complex_words = complex_word_count / len(tokens)
            word_count = len(words)
            syllable_count_per_word = sum(len(self.dic.inserted(word).split('-')) for word in words)
            personal_pronoun_count = len(re.findall(self.personal_pronoun_pattern, cleaned_text, flags=re.IGNORECASE))
            average_word_length = sum(len(word) for word in words) / word_count
            avg_sentence_length = len(tokens) / len(sentences)
            fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
            positive_score = sum(1 for word in tokens if word in self.positive_words)
            negative_score = sum(1 for word in tokens if word in self.negative_words)

            article_sentiment = TextBlob(preprocessed_text)
            polarity_score = article_sentiment.sentiment.polarity
            subjectivity_score = (positive_score + negative_score) / (len(tokens) + 0.000001)

            # Append computed metrics to respective lists
            self.avg_sentence_lengths.append(avg_sentence_length)
            self.complex_word_percentages.append(percentage_complex_words)
            self.fog_indices.append(fog_index)
            self.positive_scores.append(positive_score)
            self.negative_scores.append(negative_score)
            self.polarity_scores.append(polarity_score)
            self.subjectivity_scores.append(subjectivity_score)
            self.complex_word_counts.append(complex_word_count)
            self.word_counts.append(word_count)
            self.syllable_counts_per_word.append(syllable_count_per_word)
            self.personal_pronoun_counts.append(personal_pronoun_count)
            self.average_word_lengths.append(average_word_length)

    def process_urls_from_excel(self, excel_file):
        """
        Processes URLs from an Excel file and computes readability metrics and sentiment scores.

        Args:
            excel_file (str): The path to the Excel file containing URLs.
        """
        df = pd.read_excel(excel_file, skiprows=1, names=['URL_ID', 'URL'])

        self.load_stop_words()
        self.load_positive_negative_words()

        for index, row in df.iterrows():
            url = row['URL']
            self.process_url(url)

    def save_results_to_excel(self, output_excel_file='score.xlsx'):
        """
        Saves the computed metrics to an Excel file.

        Args:
            output_excel_file (str): The path to save the Excel file.
        """
        score_df = pd.DataFrame({
            'Average_Sentence_Length': self.avg_sentence_lengths,
            'Percentage_Complex_Words': self.complex_word_percentages,
            'Fog_Index': self.fog_indices,
            'Positive_Score': self.positive_scores,
            'Negative_Score': self.negative_scores,
            'Polarity_Score': self.polarity_scores,
            'Subjectivity_Score': self.subjectivity_scores,
            'Complex_Word_Count': self.complex_word_counts,
            'Word_Count': self.word_counts,
            'Syllable_Count_Per_Word': self.syllable_counts_per_word,
            'Personal_Pronouns': self.personal_pronoun_counts,
            'Average_Word_Length': self.average_word_lengths
        })

        # Save the DataFrame to an Excel file with xlsxwriter engine
        score_df.to_excel(output_excel_file, index=False)

        print(f'Saved scores and readability metrics to {output_excel_file}')


