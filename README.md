# A New Deep Learning Architecture withInductive Bias Balance for Transformer Oil Temperature Forecasting

This paper contains all the source code corresponding to the paper: <>.

## Abstract
Ensuring optimal performance of power transformers is a laborious task, where the insulation system is essential to decrease their deterioration. The insulation system uses the insulate oil required to control temperature. High temperatures may reduce the lifetime of transformers, leading to expensive maintenance. Deep learning architectures have been shown to obtain remarkable results in a wide range of fields. However, this improvement usually comes with an increase in computing resources, which increases the carbon footprint and hinders the optimization of the architectures. In this work, we develop a new deep learning architecture that obtain an efficacy which compete with the best current architectures in transformer oil temperature forecasting while improve the efficacy. Effective forecasting can help prevent high temperatures and monitor the future condition of power transformers, avoiding unnecessary waste. We attempt to balance the inductive bias included in our architecture through the proposed Smooth Residual Block. This mechanism divides the original problem into multiple subproblems, obtaining different representations of the time series, which collaboratively obtain the final forecasting. Our architecture is applied to the Electricity Transformer datasets, which obtain transformer insulate oil temperature measures from two transformers in China. The results achieve a 13\% improvement in MSE and a 57\% improvement in performance compared to, as far as we know, the best current architectures. Additionally, we analyze the behavior learned by this architecture to obtain an intuitive interpretation of the achieved solution.

## Usage
First, the required libraries must be installed. We recommend to use a conda virtual environment. For the instalation use the following command:

`conda env create --file=environment.yaml`

In other case, you can prepare the environment by installing the requirements.txt.

`pip install -r requirements.txt`

Then, activate the environment (if any).

Finally, execute the `experiment.sh` which is configured to reproduce the results in the paper writing the results in the `result` folder. This folder contains a folder for each experiment identified by its hash and a csv file which summarizes the results.
