from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
import pickle
import os
from gensim.models import Word2Vec
import multiprocessing

embedding_path = "../embeddings/"

def train_word2vec(documents, sg_value, min_count, max_epochs=30):
    """ Generate Word2Vec model based on min_count parameter

    Args:
      documents: List of documents. Documents should be tokenized.
                    That means each document should be a list of words.
      sg_value: {0, 1} – Training algorithm: 1 for skip-gram; 
      otherwise CBOW.
      min_count: (int) – Ignores all words with total frequency lower than 
      this.
      max_epochs: (int) – Number of max iterations over the corpus.

    Returns:
      Word2Vec model
    """
    
    # Get number of cores from computer
    cores = multiprocessing.cpu_count()
    # Create word2vec model
    model = Word2Vec(min_count=min_count,
                        window=5,
                        sample=6e-5, 
                        alpha=0.03, 
                        min_alpha=0.0007, 
                        negative=20,
                        workers=cores-1,
                        sg=sg_value)
    # Build vocabulary
    model.build_vocab(documents)
    # Train the model
    model.train(documents, total_examples=model.corpus_count, 
                epochs=max_epochs, report_delay=1)
    
    return model
    
def vectorize_word2vec(documents, model):
    """ Generate vectors for list of documents using a Word Embedding

    Args:
      documents: List of documents. Documents should be tokenized.
                    That means each document should be a list of words.
      model: Gensim's Word Embedding

    Returns:
      List of document vectors based on word2vec model
    """
    document_vectors = []
    
    # For each document (a document is a list of tokens)
    for document in documents:
        # Zero vector with size equal to word2vec vector size
        zero_vector = np.zeros(model.vector_size)
        # List to append word2vec vectors 
        vectors = []
        # For each token in the document
        for token in document:
            # Get word2vec vector representation of that token
            # and append that vector to the vector list
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    # if there's not a word2vec vector representation
                    # for that token, skip that token
                    continue
                
        # If there's at least one non zero word2vec vector
        if vectors:
            # Get a vector that's the mean of all word2vec vectors
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            # That way the document is represented as the mean of the
            # word vectors that consist the document
            document_vectors.append(avg_vec)
        # If there's non any word2vec vector
        else:
            # This document is represented as the zero vector
            document_vectors.append(zero_vector)
    return document_vectors

def vectorize_lm(documents, method):
    if method == 'minilm':
        model_name = 'paraphrase-MiniLM-L3-v2'
    elif method == 'bert':
        model_name = 'all-MiniLM-L6-v2'
        
    model = SentenceTransformer(model_name)
    
    # Define a batch size
    batch_size = 64
    dataloader = DataLoader(documents, batch_size=batch_size)
    
    embeddings = []
    for i, batch in enumerate(dataloader):
        embeddings.append(model.encode(batch))
    
    X = np.vstack(embeddings)
    
    return X

def get_embeddings(df, column, method):
    
    # define filepath to save the embeddings
    file_name = f'{column}_{method}.pkl'
    filepath = embedding_path + file_name
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            X = pickle.load(f)
            return X
    
    documents = df[column].tolist()
    
    # 1. TF-IDF
    if method == 'tfidf':
        # 1. TF-IDF
        tfidf_vectorizer = TfidfVectorizer(
                        max_df=0.7,
                        max_features=500,
                        min_df=2,
                        use_idf=True
                )
        X = tfidf_vectorizer.fit_transform(df['homepage_text']).toarray()
    

    # 2. Word2Vec
    elif method == 'word2vec':
        model = train_word2vec(documents, sg_value = 0, min_count = 10)
        # Get the document embeddings
        X = vectorize_word2vec(documents, model)
        
    # 3. MiniLm OR BERT
    else:
        X = vectorize_lm(documents, method)
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save the embeddings
    with open(filepath, 'wb') as f:
            pickle.dump(X, f)
            
    return X