# Open-Event Procedure Planning in Instructional Videos

**Abstract** : Given the current visual observations, the traditional procedure planning task in instructional videos requires a model to generate goal-directed plans within a given action space. 
All previous methods for this task conduct training and inference under the same action space, and they can only plan for pre-defined events in the training set. We argue this setting is not applicable for human assistance in real lives and aim to propose a more general and practical planning paradigm. 
Specifically, in this paper, we introduce a new task named Open-event Procedure Planning (OEPP), which extends the traditional procedure planning to the open-event setting. OEPP aims to verify whether a planner can transfer the learned knowledge to similar events that have not been seen during training. 
We rebuild a new benchmark of OpenEvent for this task based on existing datasets and divide the events involved into base and novel parts. During the data collection process, we carefully ensure the transfer ability of procedural knowledge for base and novel events by evaluating the similarity between the descriptions of different event steps with multiple stages. 
Based on the collected data, we further propose a simple and general framework specifically designed for OEPP, and conduct extensive study with various baseline methods, providing a detailed and insightful analysis on the results for this task.

Dataset