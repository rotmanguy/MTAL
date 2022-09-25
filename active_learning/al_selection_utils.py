from copy import deepcopy

import numpy as np


def find_pareto_efficient(costs, task_weights_for_selection=None, return_mask = False):
    """
    Find the pareto-efficient points.
    :param costs: An (n_points, n_costs) array.
    :param task_weights_for_selection: task weights for selection.
    :param return_mask: True to return a mask.
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for

    # Compute task weights

    # Check if task_weights_for_selection is None or all weights are equal
    if task_weights_for_selection is None or len(set(task_weights_for_selection)) <= 1:
        pareto_weights = [1.0 for _ in range(np.shape(costs)[1])]
    # Check if only one weight is equal to 1 and the rest are zeros
    elif (sum(np.array(task_weights_for_selection) == 1) == 1 and
          sum(np.array(task_weights_for_selection) == 0) == len(task_weights_for_selection) - 1):
        pareto_weights = task_weights_for_selection
    else:
        # Setting quantiles for all tasks, but the one with the highest weight
        most_important_task = np.argmax(task_weights_for_selection)
        pareto_weights = [np.quantile(costs[:, task_idx], task_weight) if
                  task_idx != most_important_task else 1.0
                  for task_idx, task_weight in enumerate(task_weights_for_selection)]

    # Get Pareto points
    while next_point_index < len(costs):
        non_dominated_point_mask = np.any(costs < costs[next_point_index] * pareto_weights, axis=1)
        non_dominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[non_dominated_point_mask]  # Remove dominated points
        costs = costs[non_dominated_point_mask]
        next_point_index = np.sum(non_dominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

def select_by_pareto(args, sample_to_confidences):
    """
    Pareto's method for selecting uncertain samples given multiple uncertainties for a given sample.
    """
    # creating the confidence matrix of all samples (number of samples x number of uncertainties per sample)
    confidences = np.vstack([v for k, v in sample_to_confidences.items()])
    # initializing the list of sample ids to return
    sample_ids = []
    # while the number of sample ids is less than the number of training samples to add
    while len(sample_ids) < args.training_samples_to_add:
        # get the current sample ids that are in the pareto efficient frontier
        pareto_indices = find_pareto_efficient(confidences, args.task_weights_for_selection)
        # if the number of sample ids that we wish to include
        # combined with the current size of sample_ids is larger than training_samples_to_add
        # than we should choose only a subset of samples from pareto_indices
        if len(sample_ids) + len(pareto_indices) > args.training_samples_to_add:
            # computing the number of samples that we need to add
            num_pareto_samples_to_add = args.training_samples_to_add - len(sample_ids)
            step_size = int(np.floor(len(pareto_indices) / num_pareto_samples_to_add))
            # choose samples to be in an equal distance (step size) from each other
            pareto_indices = pareto_indices[::step_size]
            # at each step we drop the median sample id
            # until we get to the desired size
            while len(pareto_indices) > num_pareto_samples_to_add:
                median_index = np.argsort(pareto_indices)[len(pareto_indices) // 2]
                pareto_indices = np.delete(pareto_indices, median_index, axis=0)
        # adding the current sample ids to the list
        sample_ids += list([k for idx, k in enumerate(sample_to_confidences.keys()) if idx in pareto_indices])
        # modifying the currently added samples' confidences to 1
        # such that they will not be selected at the next iteration.
        confidences[pareto_indices] = 1
    return sample_ids

def select_by_rrf(args, sample_to_confidences, k=60):
    """Implementing the Reciprocal Rank Fusion based on
    Cormack, Gordon V., Charles LA Clarke, and Stefan Buettcher.
    "Reciprocal rank fusion outperforms condorcet and individual rank learning methods."
     Proceedings of the 32nd international ACM SIGIR conference on Research and development
     in information retrieval. 2009.
    https://dl.acm.org/doi/pdf/10.1145/1571941.1572114"""

    # Compute number of tasks
    num_tasks = len(args.tasks)

    # Rank the samples according to their confidence values per task
    sorted_ranks = []
    for idx in range(num_tasks):
        sorted_rank = {sample_id: idx + 1 for idx, (sample_id, v) in enumerate(sorted(sample_to_confidences.items(), key=lambda item: item[1][idx]))}
        sorted_ranks.append(sorted_rank)

    # Prepare task weights
    task_weights = args.task_weights_for_selection if args.task_weights_for_selection is not None else [1.0 for _ in range(num_tasks)]

    # Compute the RRF score and re-rank according to RRF
    final_ranks = {}
    for sample_id in sample_to_confidences.keys():
        rrf_score = sum([task_weights[task_idx] * 1.0 / (sorted_rank[sample_id] + k) for task_idx, sorted_rank in enumerate(sorted_ranks)])
        final_ranks[sample_id] = rrf_score

    # Get the least confident samples (those with the higher rankings)
    sample_ids = [sample_id for sample_id, v in sorted(final_ranks.items(), key=lambda item: item[1], reverse=True)][:args.training_samples_to_add]
    return sample_ids

def select_by_independent_selection(args, sample_to_confidences):
    """
    Selecting independently a fraction of unlabeled samples from each task.
    If args.task_weights_for_selection than we select samples equally from each task.
    """
    # Create a copy of the unlabeled samples
    sample_to_confidences_copy = deepcopy(sample_to_confidences)

    # Compute number of tasks
    num_tasks = len(args.tasks)

    # Set the fraction of samples to be selected for each task
    task_weights = args.task_weights_for_selection if args.task_weights_for_selection is not None else [1. / num_tasks for _ in range(num_tasks)]

    # Initialize the final selected samples
    sample_ids = set()

    while len(sample_ids) < args.training_samples_to_add:
        cur_samples_ids = set()
        # Compute the number of samples to be selected for each task
        num_samples_from_each_task = [int(np.floor((args.training_samples_to_add - len(sample_ids)) * task_weight))
                                      for task_weight in task_weights]
        # Handle an edge case where num_samples_from_each_task is an array of zeros
        if all([num_samples == 0 for num_samples in num_samples_from_each_task]):
            most_important_task_idx = np.argmax(task_weights)
            num_samples_from_each_task[most_important_task_idx] = 1

        # Start selecting from each task
        for task_idx in range(num_tasks):
            if num_samples_from_each_task[task_idx] == 0:
                continue
            task_sample_ids = [k for k, v in
                               sorted(sample_to_confidences_copy.items(),
                                      key=lambda item: item[1][task_idx])][: num_samples_from_each_task[task_idx]]
            cur_samples_ids.update(task_sample_ids)

        sample_ids.update(cur_samples_ids)

        # Remove the current selected samples from the copied dictionary
        # before entering the next iteration
        for sample_id in cur_samples_ids:
            if sample_id in sample_to_confidences_copy:
                del sample_to_confidences_copy[sample_id]

    # Remove extra samples that could have been selected in the last iteration
    sample_ids = list(sample_ids)[:args.training_samples_to_add]
    return sample_ids