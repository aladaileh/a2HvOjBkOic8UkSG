def preprocess_data(data, Keywords):
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

    x = np.stack(data["vector"])
    return x


def load_model(model_path):
    return joblib.load(model_path)


def perform_inference(model, data, Keywords):
    # Preprocess the input data
    preprocessed_data = preprocess_data(data, Keywords)
    # Extract the features

    # Perform inference using the loaded model
    predictions = model.predict(preprocessed_data)
    return predictions


def main():
    # Load the trained model
    model_path = "RF_model.pkl"
    model = load_model(model_path)

    # Load the input data for inference
    input_data = pd.read_csv(
        "C:/Users/97155/Downloads/potential-talents - Aspiring human resources - seeking human resources.csv")

    # Perform inference using the loaded model
    predictions = perform_inference(model, data, Keywords)

    # Save the predictions to a file
    np.savetxt("predictions.csv", predictions, delimiter=",")

    print("Inference completed successfully. Predictions saved to predictions.csv")


if __name__ == "__main__":
    main()