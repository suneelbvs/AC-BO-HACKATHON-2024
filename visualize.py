import os
import matplotlib.pyplot as plt
import numpy as np

data = {}
with open('results.csv', 'r', encoding='utf-8') as input_file:
    content = [line.strip().split(',') for line in input_file.readlines()]
    for row in content:
        if row[0] not in data:
            data[row[0]] = {}
        if row[1] not in data[row[0]]:
            data[row[0]][row[1]] = {}
        if row[2] not in data[row[0]][row[1]]:
            data[row[0]][row[1]][row[2]] = {}
        if row[3] not in data[row[0]][row[1]][row[2]]:
            data[row[0]][row[1]][row[2]][row[3]] = []
        data[row[0]][row[1]][row[2]][row[3]].append(list(map(int, row[3:])))
os.makedirs('output', exist_ok=True)
for dataset, dataset_values in data.items():
    for model, model_values in dataset_values.items():
        for fingerprint, fingerprint_values in model_values.items():
            plt.figure()
            for acquisition_function, acquisition_function_values in fingerprint_values.items():
                results = np.array(acquisition_function_values)
                x = np.arange(results.shape[1])
                slope = np.diff(results.mean(axis=0)).mean()
                plt.plot(x, results.mean(axis=0), label=f'{acquisition_function} (Slope: {slope:.1f})')
                #plt.plot(x, results.mean(axis=0), label=acquisition_function)
                plt.errorbar(x, results.mean(axis=0), yerr=results.std(axis=0), alpha=0.3)
            plt.grid(alpha=0.3)
            plt.title(f'{dataset} - {model} - {fingerprint}')
            plt.xlabel('Batch number')
            plt.ylabel('Number of active compounds')
            plt.legend()
            plt.savefig(f'output/{dataset}_{model}_{fingerprint}.png')
            plt.close()
