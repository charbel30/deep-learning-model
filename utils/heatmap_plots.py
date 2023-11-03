
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heat_map(results_df, scores='all', separate_graphs=True):
    # Melt the DataFrame to get each score as a separate row for each target
    melted_results = results_df.melt(id_vars=['Target'], var_name='Score Type', value_name='Score')

    # If scores is 'all', plot all scores
    if scores == 'all':
        scores = ['Balanced Accuracy', 'Precision', 'Recall']

    if separate_graphs:
        # If separate_graphs is True, plot a separate graph for each specified score
        for score in scores:
            heat_map = melted_results[melted_results['Score Type'] == score].pivot(index='Target', columns='Score Type', values='Score')
            plt.figure(figsize=(14, 4))
            sns.heatmap(heat_map * 100, annot=True, cmap="YlGnBu", fmt='.2f', vmin=0, vmax=100)
            plt.title(score)
            plt.show()
    else:
        # If separate_graphs is False, plot all scores in one graph 
        heat_map = melted_results.pivot(index='Target', columns='Score Type', values='Score')
        plt.figure(figsize=(14, 8))
        sns.heatmap(heat_map * 100, annot=True, cmap="YlGnBu", fmt='.2f', vmin=0, vmax=100)
        plt.title('Scores for Each Target')
        plt.show()
# Assuming 'results' is your DataFrame
# Call the function with a list of scores or 'all'
#plot_heat_map(results, scores=['Balanced Accuracy'], separate_graphs=True)
def plot_metrics(metrics, plots):
    plt.figure(figsize=(5*len(plots),4))
    
    if 'accuracy' in plots:
        plt.subplot(1,len(plots),plots.index('accuracy')+1)
        plt.plot(metrics['train_accu'],'-o')
        plt.plot(metrics['eval_accu'],'-o')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['Train','Valid'])
        plt.title('Train vs Valid Accuracy')

    if 'loss' in plots:
        plt.subplot(1,len(plots),plots.index('loss')+1)
        plt.plot(metrics['train_losses'],'-o')
        plt.plot(metrics['eval_losses'],'-o')
        plt.xlabel('epoch')
        plt.ylabel('losses')
        plt.legend(['Train','Valid'])
        plt.title('Train vs Valid Losses')

    if 'balance_accuracy' in plots:
        plt.subplot(1,len(plots),plots.index('balance_accuracy')+1)
        plt.plot(metrics['balance_accuracy_list'],'-o')
        plt.xlabel('epoch')
        plt.ylabel('balanced accuracy')
        plt.title('Balanced Accuracy over epochs')

    if 'precision' in plots:
        plt.subplot(1,len(plots),plots.index('precision')+1)
        plt.plot(metrics['precision_list'],'-o')
        plt.xlabel('epoch')
        plt.ylabel('precision')
        plt.title('Precision over epochs')

    if 'recall' in plots:
        plt.subplot(1,len(plots),plots.index('recall')+1)
        plt.plot(metrics['recall_list'],'-o')
        plt.xlabel('epoch')
        plt.ylabel('recall')
        plt.title('Recall over epochs')

    plt.show()
    