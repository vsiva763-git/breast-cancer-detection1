import flwr as fl
import numpy as np
import json
import os
import joblib
from sklearn.linear_model import LogisticRegression
from module5_federated.client import HospitalClient
from module5_federated.split_data import split_into_hospitals


def run_federated_simulation(num_rounds=10,
                              data_dir='data/federated',
                              output_dir='models'):
    """
    Simulate federated learning across 3 hospitals.
    Uses Flower's in-process simulation — no network needed.
    """

    # Hospital data files
    hospitals = {
        'A': f'{data_dir}/hospital_A.csv',
        'B': f'{data_dir}/hospital_B.csv',
        'C': f'{data_dir}/hospital_C.csv',
    }

    # Track metrics per round
    round_metrics = {
        'rounds': [],
        'global_accuracy': [],
        'hospital_A_acc': [],
        'hospital_B_acc': [],
        'hospital_C_acc': [],
        'global_loss': []
    }

    # Create clients
    clients = {
        name: HospitalClient(path, f'Hospital {name}')
        for name, path in hospitals.items()
    }

    print(f"\nStarting Federated Learning Simulation")
    print(f"Hospitals : 3 (A, B, C)")
    print(f"Rounds    : {num_rounds}")
    print(f"Algorithm : FedAvg")
    print("=" * 50)

    # Get initial parameters from Hospital A
    global_params = clients['A'].get_parameters(config={})

    for round_num in range(1, num_rounds + 1):
        print(f"\nRound {round_num}/{num_rounds}")

        # Each hospital trains locally and returns weights
        local_weights = []
        local_sizes = []
        round_accs = {}

        for name, client in clients.items():
            weights, size, _ = client.fit(global_params, config={})
            local_weights.append(weights)
            local_sizes.append(size)

        # FedAvg — weighted average of all client weights
        total_samples = sum(local_sizes)
        avg_coef = np.zeros_like(local_weights[0][0])
        avg_intercept = np.zeros_like(local_weights[0][1])

        for weights, size in zip(local_weights, local_sizes):
            weight_factor = size / total_samples
            avg_coef += weight_factor * weights[0]
            avg_intercept += weight_factor * weights[1]

        global_params = [avg_coef, avg_intercept]

        # Evaluate global model on each hospital
        accs = []
        losses = []
        for name, client in clients.items():
            loss, _, metrics = client.evaluate(global_params, config={})
            accs.append(metrics['accuracy'])
            losses.append(loss)
            round_accs[name] = metrics['accuracy']

        global_acc = np.mean(accs)
        global_loss = np.mean(losses)

        round_metrics['rounds'].append(round_num)
        round_metrics['global_accuracy'].append(round(global_acc, 4))
        round_metrics['global_loss'].append(round(global_loss, 4))
        round_metrics['hospital_A_acc'].append(round(round_accs['A'], 4))
        round_metrics['hospital_B_acc'].append(round(round_accs['B'], 4))
        round_metrics['hospital_C_acc'].append(round(round_accs['C'], 4))

        print(f"  Global Accuracy: {global_acc*100:.2f}%  "
              f"Loss: {global_loss:.4f}")

    print("\n" + "=" * 50)
    print("FEDERATED LEARNING COMPLETE")
    print("=" * 50)
    print(f"Final Global Accuracy : {round_metrics['global_accuracy'][-1]*100:.2f}%")
    print(f"Round 1 Accuracy      : {round_metrics['global_accuracy'][0]*100:.2f}%")
    improvement = (round_metrics['global_accuracy'][-1] -
                   round_metrics['global_accuracy'][0]) * 100
    print(f"Improvement           : +{improvement:.2f}%")

    # Save metrics for dashboard
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = f'{output_dir}/fl_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(round_metrics, f, indent=2)
    print(f"FL metrics saved to {metrics_path}")

    # Save final global model
    final_model = LogisticRegression(max_iter=1000)
    final_model.coef_ = global_params[0]
    final_model.intercept_ = global_params[1]
    final_model.classes_ = np.array([0, 1])

    model_path = f'{output_dir}/federated_model.pkl'
    joblib.dump(final_model, model_path)
    print(f"Federated model saved to {model_path}")

    return round_metrics
