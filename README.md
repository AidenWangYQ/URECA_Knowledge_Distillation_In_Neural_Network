# URECA_Knowledge_Distillation_In_Neural_Network
This repository contains my research work on the topic of model compression and knowledge distillation in neural networks. 
This is a preliminary examination and verification of the theory outlined in the paper "Distilling the Knowledge in a Neural Network" (Published Mar 9, 2015, by authors Geoffery Hinton, Oriol Vinyals, Jeff Dean), particularly for their experiments on the MNIST dataset via their specified parameters.

Structure of Repository:
- requirements.txt: Contains all the dependencies and requirements to run the models.
- data_pipeline.py: Provides functions that modify the way the input data before being trained in the models.
- distillaton.py: Outlines the distillation loss function that is used to train a student model more effectvely.
- train_student.py: Code to train student model.
- tain_teacher.py: Code to train teacher model.
- models/student_model.py: Outlines how the student model is structured.
- models/teacher_model.py: Outlines how the teacher model is structured.
- runs folder: Contains results of all the runs I did. Do not open these, refer to my documented collated results pdfs for summary.
- data folder: Contains raw data from the MNIST dataset.
- .venv folder: Virtual environment folder.
