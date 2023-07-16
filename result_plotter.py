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
        # Extract errors and flops from results
        errors = [result[0] for result in results]
        flops = [result[1] for result in results]
        # Plottet die Modellleistungen
        for lambda_ in np.arange(0, 1, 0.1):
            print(f"evalutating with {lambda_} bias")
            performances = model_evaluator.get_performances_normalized(flops,errors,lambda_)
            plt.figure(figsize=(12, 6))
            plt.bar(range(1, len(performances) + 1), performances)
            plt.xlabel('Model Number')
            plt.ylabel('Performance')
            plt.title(f'Performance of Each Model with {lambda_} flops bias')
            plt.savefig(model_comparison_pages, format='pdf')

        # plt.show()
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f'Errors and Flops of Each Model')

        axs[0, 0].bar(range(1, len(errors) + 1), errors)
        axs[0, 0].set_title('Errors')

        axs[0, 1].bar(range(1, len(flops) + 1), flops)
        axs[0, 1].set_title('FLOPs')

        axs[1, 0].bar(range(1, len(model_evaluator.scale_data(errors)) + 1), model_evaluator.scale_data(errors))
        axs[1, 0].set_title('Scaled Errors')

        axs[1, 1].bar(range(1, len(model_evaluator.scale_data(flops)) + 1), model_evaluator.scale_data(flops))
        axs[1, 1].set_title('Scaled FLOPs')

        # Set the x-axis tick labels
        axs[0, 0].set_xticklabels(labels)
        axs[0, 1].set_xticklabels(labels)
        axs[1, 0].set_xticklabels(labels)
        axs[1, 1].set_xticklabels(labels)

        for ax in axs.flat:
            ax.set(xlabel='Model Number', ylabel='Value')

        plt.savefig(model_comparison_pages, format='pdf')
        model_comparison_pages.close()

        performance_df = pd.DataFrame(performances, columns=['Performance'])
        performance_df.to_csv('model_performances.csv', index=False)

