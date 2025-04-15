Here's what each file in the submission folder is:

kd.ipynb: 
Implements knowledge distillation.

pruning.ipynb:
Implements pruning.

model.py:
Contains implementation for BaseModel, StudentModel, and SmallStudentModel.

utils.py:
Contains helpful functions used in the project.

eval_plots.ipynb:
Notebook for creating plots for the ablation study.

report.pdf:
The written report for the project.

models folder: 
A folder containing 3 files:
	- best_model.pt: saved model from project 1 with test accuracy 80.22%.
	- best_pruned_model_0.75_77.26.pt: saved model from pruning.ipynb.
	- best_student_model_0.7_2.0_78.90: saved model from kd.ipynb.

I only included the models folder because the notebooks require the saved models to run without error.
To run the code, run kd.ipynb, and then pruning.ipynb.