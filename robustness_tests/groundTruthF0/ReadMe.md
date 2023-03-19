# eval_results_errors_template explanation

For a 4 source model evaluation, for a single seed, there are 1632 rows of values at the end of an evaluation.
To evaluate robustness, we want to compare an evaluated frame's SI-SDR with its VAD errors and average_distance errors.

F0 errors are **independent of the model** being evaluated. Some evaluations use 5 pytorch seeds but the F0 errors are independent of the seeds.
So this error template contains the pertinent strict errors (VAD errors) and average_distance errors for each mixture frame, along with the corresponding voice, mixture name and frame.

Evaluations having 5 pytorch seeds (not unets) use these error values 5 times.
