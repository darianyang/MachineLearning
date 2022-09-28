Please submit a title and brief description (~ paragraph) of your class project. If possible, indicate the approximate size of your dataset. Do not forget to specify the names of your classmates in your project team.

If you do not have a dataset/problem in mind, please contact me during class or by email. I am happy to help with a project selection.


Title: Progress Coordinate Optimization for Weighted Ensembles of Short MD Simulations

My project idea is to take an ensemble of MD trajectories where multiple candidate progress coordinates are calculated per frame, and to then use machine learning to rank each candidate coordinate and select the best progress descriptor. Another thought is to potentially standardize the coordinates, then take a linear combination of each coordinate and optimize the weight of each. I'm not sure what ML technique will be best here, but perhaps using decision trees for ranking purposes or optimizing a target function that can approximate the quality of a single or multi-dimensional coordinate. In the test/training datasets, I will use trajectories that make it to a pre-defined target state as the labeled successful input. For the actual simulation data to be used, I am not sure which dataset to choose yet but I have a few from my research projects available to choose from (all of which are proteins or protein-ligand complexes). Ideally I will try it out for multiple systems.
