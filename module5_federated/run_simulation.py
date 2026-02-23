import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module5_federated.split_data import split_into_hospitals
from module5_federated.simulate import run_federated_simulation

print("Module 5 — Federated Learning Simulation")
print("=" * 50)

# Step 1 — Split data into hospital partitions
print("\nStep 1: Splitting data into hospital partitions...")
split_into_hospitals()

# Step 2 — Run federated simulation
print("\nStep 2: Running federated simulation (10 rounds)...")
metrics = run_federated_simulation(num_rounds=10)

# Step 3 — Print summary table
print("\nROUND BY ROUND SUMMARY")
print("-" * 60)
print(f"{'Round':<8} {'Global Acc':<14} {'Hospital A':<14} {'Hospital B':<14} {'Hospital C'}")
print("-" * 60)
for i, r in enumerate(metrics['rounds']):
    print(f"{r:<8} "
          f"{metrics['global_accuracy'][i]*100:.2f}%{'':<8} "
          f"{metrics['hospital_A_acc'][i]*100:.2f}%{'':<8} "
          f"{metrics['hospital_B_acc'][i]*100:.2f}%{'':<8} "
          f"{metrics['hospital_C_acc'][i]*100:.2f}%")

print("\nModule 5 Complete!")
