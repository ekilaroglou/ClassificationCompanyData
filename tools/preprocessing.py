from nltk import word_tokenize
from nltk.corpus import stopwords
import re
import unicodedata
import nltk
from nltk import WordNetLemmatizer
from multiprocessing import Pool

def preprocess_text_keywords(text, lemmatize = False, extra_stop_words = None, cores=6):
     
    # Get stopwords
    sw = set(stopwords.words("english"))

    # possible values of punctuarion
    punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'
    text = unicodedata.normalize('NFD', text).encode('ascii','ignore').decode('utf-8')
    text = re.sub(r'http\S+', '', text) #remove links
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # Replace dash between words
    text = str(text).lower()  # Lowercase words
    text = re.sub(r"[0-9]", " ", text) # Remove numbers
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
    text = re.sub(r"[.,@]", " ", text) # remove dot's
    text = ''.join(ch for ch in text if ch not in set(punctuation)) #remove punctuation
    
    tokens = word_tokenize(text)  # Get tokens from text
    tokens = [t for t in tokens if not t in sw]  # Remove stopwords
    tokens = [t for t in tokens if not t in extra_stop_words]  # Remove additional stopwords
    tokens = ["" if t.isdigit() else t for t in tokens]  # Remove digits
    tokens = [t for t in tokens if len(t) > 1]  # Remove short tokens
    
    if lemmatize:
        with Pool(processes=cores) as pool:
            wnl = WordNetLemmatizer()
            tokens = pool.map(wnl.lemmatize, tokens)
                    
    text = ' '.join(tokens)
    return text

def preprocess_text_bert(text):

    # possible values of punctuarion
    punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'
    text = unicodedata.normalize('NFD', text).encode('ascii','ignore').decode('utf-8')
    text = re.sub(r'http\S+', '', text) #remove links
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # Replace dash between words
    text = re.sub(r"[0-9]", " ", text) # Remove numbers
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
    text = ''.join(ch for ch in text if ch not in set(punctuation)) #remove punctuation
                    
    return text

def lemmatize(df_column):
    # Download required resources
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Function to lemmatize text
    def lemmatize_text(text):
        words = word_tokenize(text)
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    
    # Apply the function to the DataFrame
    df_column = df_column.apply(lemmatize_text)
    
    return df_column