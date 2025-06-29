# Obvious_IA

1. Modular Image Classification with Custom Callback
Develop a modular image classification pipeline using `Pytorch` that includes:
•	A custom callback (LRA) to manage:
o	Learning rate adjustments.
o	Early stopping after consecutive adjustments without improvement.
o	Restoration of best weights when performance degrades.
o	Dynamic learning rate adjustments based on training accuracy thresholds.
o	Interactive prompts to continue or halt training after specified epochs.
Having parameters such as:
	model: The model instance to be trained.
	patience: Number of epochs with no improvement after which learning rate will be reduced.
	stop_patience: Number of consecutive learning rate adjustments with no improvement after which training will be stopped.
	threshold: Training accuracy threshold to start adjusting learning rate based on validation loss.
	factor: Factor by which the learning rate will be reduced.
	dwell: If True, restore model weights from the epoch with the best performance when learning rate is reduced.
	model_name: Name of the model architecture, used for logging purposes.
	freeze: If True, freeze certain layers during training (optional).
	batches: Number of batches per epoch.
	epochs: Total number of epochs for training.
	ask_epoch: After this epoch, prompt the user to decide whether to continue training.
•	A modular codebase separating modules such as data loading, model architecture, training, evaluation, and utilities
•	A main function that controls training and other functionalities of codebase.
Also, Explain the Execution steps to use this pipeline in readme file.


![image](https://github.com/user-attachments/assets/fa507f40-2a42-4bdc-8acb-9af6820184b3)
