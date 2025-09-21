> ⚠️ This repository is part of a larger URECA project titled **"Compressing Vision-Language Models (VLMs) via Knowledge Distillation (KD)"**.
> 
> ⚠️ **Work in Progress:** This repository is still under active development. Features, results, and documentation are being updated regularly.
# URECA: Knowledge Distillation in Neural Networks Research
This repository contains my research on model compression and knowledge distillation, replicating and validating the experiments from “Distilling the Knowledge in a Neural Network” (Hinton, Vinyals & Dean, 2015). The project focuses on training student networks on the MNIST dataset using teacher–student distillation strategies, and compares performance improvements across different configurations.

**Repository Structure**

**`requirements.txt`** – Dependencies to reproduce the experiments

**`data_pipeline.py`** – Functions for preprocessing and preparing MNIST data

**`distillation.py`** – Implementation of the distillation loss function

**`train_teacher.py`** – Training loop for the teacher model

**`train_student.py`** – Training loop for the student model

**`models/student_model.py`** – Architecture of the student network

**`models/teacher_model.py`** – Architecture of the teacher network

**`runs/`** – Intermediate run outputs (final summarized results in attached PDF)
