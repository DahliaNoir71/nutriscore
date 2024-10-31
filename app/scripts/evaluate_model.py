from app.modules.train_model import (prepare_and_train_model,
                                     make_predictions,
                                     evaluate_model)

model, scaler, x_test, y_test = prepare_and_train_model(saving=True)
y_prediction_test, y_prediction_prob_test = make_predictions(model, scaler, x_test)
evaluate_model(y_test, y_prediction_test, y_prediction_prob_test, model)