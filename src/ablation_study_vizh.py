import matplotlib.pyplot as plt
import json


def main():
    with open('explanation_generation_rouge.json') as f:
        explanations = json.load(f)
    with open('focused_summarization_rouge.json') as f:
        summaries = json.load(f)

    damping_factors = ['0.8', '0.85', '0.9']
    similarity_thresholds = ['0.7', '0.75', '0.8', '0.85', '0.9']
    fig, axs = plt.subplots(3, 3)
    fig.set_size_inches(12,8)

    for i, damping_factor in enumerate(damping_factors):
        explanations_rouge1, explanations_rouge2, explanations_rougel = \
            [explanations[similarity_threshold][damping_factor][0] for similarity_threshold in explanations], \
            [explanations[similarity_threshold][damping_factor][1] for similarity_threshold in explanations], \
            [explanations[similarity_threshold][damping_factor][2] for similarity_threshold in explanations]
        dem_rouge1, dem_rouge2, dem_rougel = \
            [summaries[similarity_threshold][damping_factor]['democrat'][0] for similarity_threshold in summaries], \
            [summaries[similarity_threshold][damping_factor]['democrat'][1] for similarity_threshold in summaries], \
            [summaries[similarity_threshold][damping_factor]['democrat'][2] for similarity_threshold in summaries]
        rep_rouge1, rep_rouge2, rep_rougel = \
            [summaries[similarity_threshold][damping_factor]['republican'][0] for similarity_threshold in summaries], \
            [summaries[similarity_threshold][damping_factor]['republican'][1] for similarity_threshold in summaries], \
            [summaries[similarity_threshold][damping_factor]['republican'][2] for similarity_threshold in summaries]

        axs[i, 0].plot(similarity_thresholds, explanations_rouge1, 'o-', label='ROUGE-1')
        axs[i, 0].plot(similarity_thresholds, explanations_rouge2, 'o-', label='ROUGE-2')
        axs[i, 0].plot(similarity_thresholds, explanations_rougel, 'o-', label='ROUGE-L')
        if i == 0:
            axs[i, 0].set_title('Explanation Extraction')

        axs[i, 1].plot(similarity_thresholds, dem_rouge1, 'o-', label='ROUGE-1')
        axs[i, 1].plot(similarity_thresholds, dem_rouge2, 'o-', label='ROUGE-2')
        axs[i, 1].plot(similarity_thresholds, dem_rougel, 'o-', label='ROUGE-L')
        if i == 0:
            axs[i, 1].set_title('Focused Summarization: Democrat')

        axs[i, 2].plot(similarity_thresholds, rep_rouge1, 'o-', label='ROUGE-1')
        axs[i, 2].plot(similarity_thresholds, rep_rouge2, 'o-', label='ROUGE-2')
        axs[i, 2].plot(similarity_thresholds, rep_rougel, 'o-', label='ROUGE-L')
        if i == 0:
            axs[i, 2].set_title('Focused Summarization: Republican')

        axs.flat[i*3].set(ylabel='Damping Factor = {}\nROUGE-N F1'.format(damping_factor))

        for j, ax in enumerate(axs.flat):
            if j // 3 == 2:
                ax.set(xlabel='Similarity Threshold')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        # for ax in axs.flat:
        #     ax.label_outer()

    plt.legend(bbox_to_anchor=(0.6, 0.15), loc='lower left', borderaxespad=0.)
    plt.show()


if __name__ == "__main__":
    main()
