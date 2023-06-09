from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

data = pd.read_csv(
    "C:/Users/97155/Downloads/potential-talents - Aspiring human resources - seeking human resources.csv")
Keywords = "Human Resources HR"


def split_data(data, Keywords):
    nlp = spacy.load("en_core_web_md")

    def preprocess_and_vectorize(text):
        doc = nlp(text)
        filtered = []
        for token in doc:
            if token.is_punct or token.is_stop:
                continue
            filtered.append(token.lemma_)

        vectors = [nlp(word).vector for word in filtered]
        if vectors:
            mean_vector = np.mean(vectors, axis=0)
        else:
            mean_vector = np.zeros(nlp.vocab.vectors.shape[1])

        return mean_vector

    data["vector"] = data["job_title"].apply(lambda x: preprocess_and_vectorize(x))
    key_ward = preprocess_and_vectorize(Keywords)

    cosin_sim = []
    for i in range(data.shape[0]):
        sim = cosine_similarity(data["vector"][i].reshape(1, -1), key_ward.reshape(1, -1))
        cosin_sim.append(sim)

    data["similarity"] = cosin_sim

    data['fit'] = data['similarity'].apply(lambda x: 1 if x >= 0.6 else 0)
    x_train, x_test, y_train, y_test = train_test_split(data.vector, data.fit, test_size=0.20, random_state=1)

    x_train = np.stack(x_train)
    x_test = np.stack(x_test)

    return x_train, x_test, y_train, y_test, data