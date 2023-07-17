from model_evaluator import ModelEvaluator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class ResultPlotter:
    def __init__(self):
        pass

    def plot_results(self, results, labels, pdfTitle):
        model_evaluator = ModelEvaluator()

        model_comparison_pages = PdfPages(f'model_comparison_{pdfTitle}.pdf')
        errors = [result[0] for result in results]
        flops = [result[1] for result in results]
        estimatedFlops = [result[2] for result in results]

        percentage_difference = [(f - e) / f * 100 for f, e in zip(flops, estimatedFlops)]

        for lambda_ in np.arange(0, 1, 0.1):
            roundLambda = round(lambda_, 2) 
            print(f"evaluating with {roundLambda} bias")
            performances = model_evaluator.get_performances_normalized(flops, errors, roundLambda)
            plt.figure(figsize=(12, 6))
            plt.bar(range(1, len(performances) + 1), performances)
            plt.xlabel('Model Number')
            plt.ylabel('Performance')
            plt.title(f'Performance of Each Model with {roundLambda} flops bias')
            plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right')
            plt.savefig(model_comparison_pages, format='pdf')

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f'Errors, FLOPs, and Percentage Difference between FLOPs and Estimated FLOPs of Each Model')

        axs[0, 0].bar(range(1, len(errors) + 1), errors)
        axs[0, 0].set_title('Errors')

        axs[0, 1].bar(range(1, len(flops) + 1), flops)
        axs[0, 1].set_title('FLOPs')

        axs[1, 0].bar(range(1, len(percentage_difference) + 1), percentage_difference)
        axs[1, 0].set_title('Percentage Difference between FLOPs and Estimated FLOPs')

        axs[1, 1].bar(range(1, len(model_evaluator.scale_data(flops)) + 1), model_evaluator.scale_data(flops))
        axs[1, 1].set_title('Scaled FLOPs')

        for i in range(2):
            for j in range(2):
                axs[i, j].set_xticks(range(1, len(labels) + 1))
                axs[i, j].set_xticklabels(labels, rotation=45, ha='right')
                axs[i, j].set(xlabel='Model Number', ylabel='Value')

        plt.tight_layout(rect=[0, 0, 1, 0.96]) # To ensure the subplot titles and x-labels don't overlap
        plt.savefig(model_comparison_pages, format='pdf')
        model_comparison_pages.close()

        performance_df = pd.DataFrame(performances, columns=['Performance'])
        performance_df.to_csv('model_performances.csv', index=False)
