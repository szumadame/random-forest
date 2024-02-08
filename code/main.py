from scripts_and_experiments.do_experiments import id3_comparison, nbc_comparison, tree_number_influence, \
    classifier_ratio_influence, samples_percentage_influence

nbc_comparison('loan_approval')
id3_comparison('loan_approval')
tree_number_influence('glass')
samples_percentage_influence('glass')
classifier_ratio_influence('letter')
