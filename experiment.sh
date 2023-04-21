MODELS=('fc' 'lstm' 'hlnet' 'srcnet' 'scinet')

python main.py --model ${MODELS[@]} --results_file 'metrics'