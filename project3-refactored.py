
from model_evaluator import ModelEvaluator
from model_trainer import ModelTrainer
from model_creator import ModelCreator
from result_plotter import ResultPlotter

def main():
    model_creator = ModelCreator()
    model_trainer = ModelTrainer()
    model_evaluator = ModelEvaluator()
    result_plotter = ResultPlotter()

    # Create models
    cnn_models_with_labels = model_creator.create_cnn_models()
    lstm_models_with_labels = model_creator.create_lstm_models()

    # Extract models and labels
    cnn_models, cnn_labels = zip(*cnn_models_with_labels)
    lstm_models, lstm_labels = zip(*lstm_models_with_labels)

    # Train models
    # trained_cnn_models, X_cnn_test, y_cnn_test = model_trainer.train_cnn(cnn_models_with_labels)
    # trained_lstm_models, X_lstm_test, y_lstm_test = model_trainer.train_lstm(lstm_models_with_labels)

    # Load models
    trained_cnn_data = [model_trainer.load_trained_model(label, 'cnn') for label in cnn_labels]
    trained_lstm_data = [model_trainer.load_trained_model(label, 'lstm') for label in lstm_labels]

    # Extract the models and test data
    trained_cnn_models, X_cnn_test, y_cnn_test = zip(*trained_cnn_data)
    trained_lstm_models, X_lstm_test, y_lstm_test = zip(*trained_lstm_data)

    # Evaluate models
    cnn_results = [model_evaluator.evaluate_model(model, X_cnn_test[0], y_cnn_test[0]) for model in trained_cnn_models]
    lstm_results = [model_evaluator.evaluate_model(model, X_lstm_test[0], y_lstm_test[0]) for model in trained_lstm_models]

    # Plot results
    result_plotter.plot_results(cnn_results, cnn_labels, 'cnn')
    result_plotter.plot_results(lstm_results, lstm_labels, 'lstm')


if __name__ == "__main__":
    main()
