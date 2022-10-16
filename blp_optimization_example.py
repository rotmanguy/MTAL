import numpy as np

from active_learning.blp_selection_utils import mip_single_task_selection, mip_unrestricted_disjoint_sets, \
    mip_equal_budget_disjoint_sets, mip_joint_task_selection, mip_single_task_confidence_selection

# Set task names and task length
tasks = ['deps', 'ner']
num_tasks = len(tasks)

# Set the total cost for our budget
total_cost = 25

# Randomly generate confidence scores for all tasks ranging from 0 to 1.
rand_confidences = np.random.uniform(0, 1, (1000, num_tasks))
# Build the mapping from samples to their confidence scores
sample_to_confidences = dict(enumerate(rand_confidences))

# Randomly generate costs for all tasks ranging from 0 to 1.
rand_costs = np.random.uniform(0, 1, (1000, num_tasks))
# Add the joint costs (in this case the average of all tasks) to the final column
rand_costs = np.hstack((rand_costs, np.expand_dims(np.mean(rand_costs, axis=1), axis=1)))
# Build the mapping from samples to their confidence scores
sample_to_costs = dict(enumerate(rand_costs))

# Get sample ids for single-task selection
st_sample_ids = mip_single_task_selection(sample_to_confidences,
                                          sample_to_costs,
                                          total_cost,
                                          [tasks[0]])

# Get sample ids for unrestricted disjoint sets
udjs_sample_ids = mip_unrestricted_disjoint_sets(sample_to_confidences,
                                                 sample_to_costs,
                                                 total_cost,
                                                 tasks)

# Get sample ids for equal budget disjoint sets
eqbdjs_sample_ids = mip_equal_budget_disjoint_sets(sample_to_confidences,
                                                   sample_to_costs,
                                                   total_cost,
                                                   tasks)

# Get sample ids for joint-task selection
jt_sample_ids = mip_joint_task_selection(sample_to_confidences,
                                         sample_to_costs,
                                         total_cost,
                                         tasks)

# Get sample ids for single-task confidence selection
stc_sample_ids = mip_single_task_confidence_selection(sample_to_confidences,
                                                      sample_to_costs,
                                                      total_cost,
                                                      tasks,
                                                      tasks[0])