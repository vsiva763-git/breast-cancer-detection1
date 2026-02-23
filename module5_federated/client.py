import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from module5_federated.split_data import load_hospital_data


def get_model_parameters(model):
    """Extract model weights as numpy arrays."""
    if model.coef_ is not None:
        return [model.coef_, model.intercept_]
    return [np.zeros((1, 30)), np.zeros(1)]


def set_model_parameters(model, parameters):
    """Set model weights from numpy arrays."""
    model.coef_ = parameters[0]
    model.intercept_ = parameters[1]
    return model


class HospitalClient(fl.client.NumPyClient):
    def __init__(self, hospital_file, hospital_name):
        self.hospital_name = hospital_name
        self.X_train, self.X_test, self.y_train, self.y_test = \
            load_hospital_data(hospital_file)

        # Initialize model
        self.model = LogisticRegression(
            max_iter=1000,
            warm_start=True,
            random_state=42,
            C=1.0
        )
        # Must fit once to initialize coef_
        self.model.fit(self.X_train, self.y_train)

        print(f"{hospital_name} client ready — "
              f"Train: {len(self.X_train)} | Test: {len(self.X_test)}")

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        """Receive global model, train locally, return updated weights."""
        set_model_parameters(self.model, parameters)
        self.model.fit(self.X_train, self.y_train)
        acc = accuracy_score(self.y_train, self.model.predict(self.X_train))
        print(f"  {self.hospital_name} local train acc: {acc:.4f}")
        return get_model_parameters(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        """Evaluate global model on local test data."""
        set_model_parameters(self.model, parameters)
        y_prob = self.model.predict_proba(self.X_test)
        loss = log_loss(self.y_test, y_prob)
        acc = accuracy_score(self.y_test, self.model.predict(self.X_test))
        print(f"  {self.hospital_name} eval — loss: {loss:.4f}  acc: {acc:.4f}")
        return float(loss), len(self.X_test), {'accuracy': float(acc)}
