# Accelerometer-Classification
Here is the code for a project that I completed in my Neural Networks class at NC State. The goal of this project is to classify walking cadence based on accelerometer data. The categories are
- Walk hard
- Down stairs
- Up stairs
- Walk soft

In `baseline.ipynb`, a simple baseline model based on pre-extracted features is used. For the final project `final.ipynb`, I performed an architecture search over MLP's of different widths and depths as well as CNN models of different depths, channel sizes, and kernel sizes. In the end, one of the simpler MLP models `Net3` prevailed. Many trials of data were collected. I used some trials for training the model and others for a validation set. In our class, the model was evaluated on holdout trials that the instructor kept private. With this relatively simple model and an [Optuna](https://optuna.readthedocs.io/en/stable/) hyperparamter search, I achieved the highest test accuracy in my section of the class.

Note: `fncs.py` and all data was provided by our instructor, with credit being attributed to the [NCSU AROS Lab](https://research.ece.ncsu.edu/aros/).
