import os
import numpy as np
import matplotlib.pyplot as plt

fold_number = 0

base_dir = f"/home/will/work/1-RA/src/Identifier/Supervised-Triplet-Network/results/fold_{fold_number}"
accuracies_fp = os.path.join(base_dir, "open_cows_triplet_cnn_accuracies_log_x1.npz")
losses_fp = os.path.join(base_dir, "open_cows_triplet_cnn_train_log_x1.npz")

accuracies = np.load(accuracies_fp)
losses = np.load(losses_fp)

fig, ax1 = plt.subplots()

color1 = 'tab:blue'
ax1.set_xlabel('Training steps')
ax1.set_ylabel('Accuracy', color=color1)
ax1.set_xlim((0, np.max(accuracies['steps'])))
ax1.set_ylim((80, 100))
ax1.plot(accuracies['steps'], accuracies['accuracies_all'], color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()

color2 = 'tab:red'
ax2.set_ylabel('Loss', color=color2)
ax2.plot(losses['steps'], losses['losses_sum'], color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim((0,6))

print(f"Best accuracies for fold {fold_number}:")
print(f"All: {np.max(accuracies['accuracies_all'])}")
print(f"Known: {np.max(accuracies['accuracies_known'])}")
print(f"Novel: {np.max(accuracies['accuracies_novel'])}")
print(f"Max training steps (accuracy): {np.max(accuracies['steps'])}")
print(f"Max training steps (loss): {np.max(losses['steps'])}")

plt.tight_layout()
# plt.show()
plt.savefig(f"{os.path.basename(base_dir)}.pdf")

# Finish up
plt.cla()
plt.clf()
plt.close()