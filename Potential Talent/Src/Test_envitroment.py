import joblib


def main():
    data = pd.read_csv(
        "C:/Users/97155/Downloads/potential-talents - Aspiring human resources - seeking human resources.csv")
    Keywords = "Human Resources HR"

    data = split_data(data, Keywords)

    model = train_model(x_train, y_train)

    metrics = model_metrics(model, x_test, y_test)

    model_name = "RF_model.pkl"

    joblib.dump(value=model, filename=model_name)


if __name__ == "__main__":
    main()

def init():
    model_path = "RF_model.pkl"  # Update with the correct path to the model file
    model = joblib.load(model_path)

    return model