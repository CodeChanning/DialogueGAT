# DialogueGAT Research Replication

## Goal:

We aim to replicate the results of the DialogueGAT research by using the same code and data as the research. After reproducing the results, we will test the model using newer data.

## Run the experiment

* Dependencies: PyTorch, DGL
* Running all experiments: 
  ```
  sh run.sh
  ```
* Runing a single task:
  ```
  python -W ignore -u train.py --use_gpu --v_past --year ${year} --target ${target}
  ```