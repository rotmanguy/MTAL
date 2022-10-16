from collections.abc import Iterable
from typing import Union, Dict, List

import numpy as np
from mip import Model, BINARY, maximize, xsum

# Todo: In the current implementation the joint costs take place
#  only when a sample is annotated on all tasks, but there could be a setup
#  where there are more than two tasks and we would like to formulate joint
#  costs for the case where samples are selected to be annotated on only a
#  subset of the tasks.

def mip_single_task_selection(sample_to_confidences: Dict[int, np.array],
                              sample_to_costs: Dict[int, np.array],
                              total_cost: Union[int, float],
                              tasks: List[str]) -> List[int]:
    """
    Computing BLP optimization for single-task selection:
    A sample can be annotated on the task or not.

    The BLP optimization maximizes the single-task uncertainties (1 - confidence)
    of the selected samples given the budget constraints.
    Please make sure to normalize the confidence scores such that they are in the range
    of 0 to 1.

    :param sample_to_confidences: a mapping from sample to its confidence scores
    (should be an np.array with a single cell).
    :param sample_to_costs: a mapping from sample to its cost functions
    (should be an np.array with a single cell).
    :param total_cost: the total budget cost.
    :param tasks: a list of the trained tasks (should contain a single task).
    :return: a list of the selected sample indices.
    """

    # Number of tasks should be equal to one
    num_tasks = len(tasks)
    assert num_tasks == 1

    # Prepare sample ids
    all_sample_ids = [j for j, _ in sample_to_confidences.items()]
    num_sample_ids = len(all_sample_ids)

    # Prepare confidence scores
    confidence_scores = [confidence[0] if isinstance(confidence, Iterable) else confidence
                         for _, confidence in sample_to_confidences.items()]

    # Prepare costs
    costs = [cost[0] if isinstance(cost, Iterable) else cost
             for _, cost in sample_to_costs.items()]

    # Build BLP model
    m = Model("knapsack_single_task")
    # Add variables
    x = [m.add_var(var_type=BINARY) for _ in range(num_sample_ids)]
    # Add objective function
    m.objective = maximize(xsum((1- confidence_scores[j]) * x[j] for j in range(num_sample_ids)))
    # Add budget constraints
    m += (xsum(costs[j] * x[j] for j in range(num_sample_ids)) <= total_cost)
    # Solve BLP model
    m.optimize(max_seconds=500)

    # Retrieve selected samples
    sample_ids = [all_sample_ids[j] for j in range(num_sample_ids) if x[j].x >= 0.99]
    return sample_ids

def mip_unrestricted_disjoint_sets(sample_to_confidences: Dict[int, np.array],
                                   sample_to_costs: Dict[int, np.array],
                                   total_cost: Union[int, float],
                                   tasks: List[str]) -> List[List[int]]:
    """
    Computing BLP optimization for unrestricted disjoint sets:
    A sample can be annotated on none/one/all of the tasks.

    The BLP optimization maximizes the single-task uncertainties (1 - confidence)
    of the selected samples given the budget constraints.
    Please make sure to normalize the confidence scores such that they are in the range
    of 0 to 1.

    :param sample_to_confidences: a mapping from sample to its confidence scores
    (sorted by the task order).
    :param sample_to_costs: a mapping from sample to its cost functions
    (sorted by the task order).
    :param total_cost: the total budget cost.
    :param tasks: a list of the trained tasks.
    :return: a list containing lists of the selected samples (sorted by the task order).
    """

    # Number of tasks
    num_tasks = len(tasks)

    # Prepare sample ids
    all_sample_ids = [sample_id for sample_id, _ in sample_to_confidences.items()]
    num_sample_ids = len(all_sample_ids)

    # Prepare confidence scores
    confidence_scores = [[confidence[i] for _, confidence in sample_to_confidences.items()]
                         for i in range(num_tasks)]

    # Prepare costs
    costs = [[cost[i] for _, cost in sample_to_costs.items()]
             for i in range(num_tasks)]
    joint_costs = [cost[-1] for _, cost in sample_to_costs.items()]

    # Build model
    m = Model("knapsack_unrestricted_disjoint_sets")
    # Add variables
    x = [[m.add_var('x({},{})'.format(i, j), var_type=BINARY)
      for j in range(num_sample_ids)] for i in range(num_tasks)]
    # Add dummy variables indicating that a sample will be annotated on all tasks
    y = [m.add_var(var_type=BINARY) for _ in range(num_sample_ids)]
    # Add objective function
    m.objective = maximize(xsum((1 - confidence_scores[i][j]) * x[i][j]
                                for j in range(num_sample_ids)
                                for i in range(num_tasks)))
    # Add budget constraints
    m += (xsum(costs[i][j] * (x[i][j] - y[j])
               for j in range(num_sample_ids)
               for i in range(num_tasks)) +
          xsum(joint_costs[j] * y[j]
               for j in range(num_sample_ids))
          <= total_cost)
    # Add dummy constraints
    for j in range(num_sample_ids):
        m += xsum(x[i][j] for i in range(num_tasks)) >= num_tasks * y[j], 'col({})'.format(j)
    # Solve BLP model
    m.optimize(max_seconds=500)

    # Retrieve selected samples per task
    sample_ids = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(num_sample_ids):
            if x[i][j].x >= 0.99:
                sample_ids[i].append(all_sample_ids[j])
    return sample_ids

def mip_equal_budget_disjoint_sets(sample_to_confidences: Dict[int, np.array],
                                   sample_to_costs: Dict[int, np.array],
                                   total_cost: Union[int, float],
                                   tasks: List[str]) -> List[List[int]]:
    """
    Computing BLP optimization for equal budget disjoint sets:
    We split the budget equally between all tasks and solve a BLP optimization problem
    for each task separately given its confidence scores and costs.
    A sample can be annotated on none/one/all of the tasks.

    Since the following algorithm is iterative, for relaxation purposes,
    if a sample was selected for annotation for any of the tasks
    (could be more than one) in the current iteration,
    it will be removed from the pool of unlabeled samples
    and will not be allowed to be selected for the other tasks in next iterations.

    The BLP optimization maximizes the single-task uncertainties (1 - confidence)
    of the selected samples given the budget constraints.
    Please make sure to normalize the confidence scores such that they are in the range
    of 0 to 1.

    :param sample_to_confidences: a mapping from sample to its confidence scores
    (sorted by the task order).
    :param sample_to_costs: a mapping from sample to its cost functions
    (sorted by the task order). Additionally, please make sure to place the joint cost function
    of every sample at the final cell of its array.
    :param total_cost: the total budget cost.
    :param tasks: a list of the trained tasks.
    :return: a list containing lists of the selected samples (sorted by the task order).
    """

    # Number of tasks
    num_tasks = len(tasks)

    # Prepare overall sample ids
    sample_ids = [set() for _ in range(num_tasks)]

    # Initialize the union of the sample ids that will be selected
    sample_ids_union = set()

    # Prepare total cost variables
    total_used_cost = 0
    cur_total_used_cost = 1
    total_cost_left = total_cost

    # Iterate until reach total budget
    while total_cost_left > 0 and cur_total_used_cost > 0:
        # Prepare current sample ids (those that have not been selected thus far)
        cur_all_sample_ids = [j for j, _ in sample_to_confidences.items()
                                   if j not in sample_ids_union]
        cur_num_sample_ids = len(cur_all_sample_ids)

        # Initialize the per-task sample ids that will be selected
        cur_sample_ids = [set() for _ in range(num_tasks)]

        # Prepare current confidence scores
        cur_confidence_scores = [[confidence[i] for j, confidence in sample_to_confidences.items()
                                  if j not in sample_ids_union]
                                 for i in range(num_tasks)]

        # Prepare current costs
        cur_costs = [[cost[i] for j, cost in sample_to_costs.items()
                      if j not in sample_ids_union]
                     for i in range(num_tasks)]

        # Prepare current mappings from samples to costs
        cur_sample_to_costs = [{j: cost[i] for j, cost in sample_to_costs.items()
                                if j not in sample_ids_union}
                               for i in range(num_tasks)]
        cur_sample_to_joint_costs = {j: cost[-1] for j, cost in sample_to_costs.items()
                                     if j not in sample_ids_union}

        for i in range(num_tasks):
            # Build model
            m = Model(f"knapsack_equal_budget_disjoint_sets_task_{tasks[i]}")
            # Add variables
            x = [m.add_var(var_type=BINARY) for _ in range(cur_num_sample_ids)]
            # Add objective function
            m.objective = maximize(xsum((1 - cur_confidence_scores[i][j]) * x[j] for j in range(cur_num_sample_ids)))
            # Add budget constraints
            m += (xsum(cur_costs[i][j] * x[j] for j in range(cur_num_sample_ids)) <= (total_cost_left / num_tasks))
            # Solve model
            m.optimize(max_seconds=500)

            # Update sample ids that were selected for task i
            for j in range(cur_num_sample_ids):
                if x[j].x >= 0.99:
                    cur_sample_ids[i].add(cur_all_sample_ids[j])
            sample_ids[i].update(cur_sample_ids[i])

        # Update the union of all selected sample ids
        cur_union_sample_ids = set.union(*cur_sample_ids)
        sample_ids_union.update(cur_union_sample_ids)

        # Calculate current total cost that was used in the current iteration
        cur_inter_sample_ids = set.intersection(*cur_sample_ids)
        cur_total_used_cost = 0

        # we add the single-task cost for samples that were not selected to be annotated on all tasks
        for i in range(num_tasks):
            cur_sample_ids_set = \
                cur_sample_ids[i].difference(*[set_ for k, set_ in enumerate(cur_sample_ids) if k != i])
            for j in cur_sample_ids_set:
                cur_total_used_cost += cur_sample_to_costs[i][j]

        # we add the joint cost for samples that were selected to be annotated on all tasks
        for j in cur_inter_sample_ids:
            cur_total_used_cost += cur_sample_to_joint_costs[j]

        # Update total costs
        total_used_cost += cur_total_used_cost
        total_cost_left -= cur_total_used_cost

    # Retrieve selected samples per task
    sample_ids = [list(sample_ids_set) for sample_ids_set in sample_ids]
    return sample_ids

def mip_joint_task_selection(sample_to_confidences: Dict[int, np.array],
                             sample_to_costs: Dict[int, np.array],
                             total_cost: Union[int, float],
                             tasks: List[str]) -> List[List[int]]:
    """
    Computing BLP optimization for joint task selection:
    A sample can be annotated on none/all of the tasks.
    Since we would like to annotate samples on all tasks, we use the joint costs.

    The BLP optimization maximizes the multi-task uncertainties (1 - confidence)
    of the selected samples given the budget constraints.
    Please make sure to normalize the confidence scores such that they are in the range
    of 0 to 1.

    :param sample_to_confidences: a mapping from sample to its confidence scores
    (sorted by the task order). Additionally, please make sure to place the joint confidence score
    of every sample at the final cell of its array.
    :param sample_to_costs: a mapping from sample to its cost functions
    (sorted by the task order). Additionally, please make sure to add the joint cost function
    of every sample at the final cell of its array.
    :param total_cost: the total budget cost.
    :param tasks: a list of the trained tasks.
    :return: a list containing lists of the selected samples (sorted by the task order).
    """

    # Number of tasks
    num_tasks = len(tasks)

    # Prepare sample ids
    all_sample_ids = [j for j, _ in sample_to_confidences.items()]
    num_sample_ids = len(all_sample_ids)

    # Prepare confidence scores (we take the joint confidence scores in this case,
    # and not the per-task scores as previously)
    confidence_scores = [confidence[-1] for _, confidence in sample_to_confidences.items()]

    # Prepare costs (we take the joint costs in this case,
    # and not the per-task costs as previously)
    joint_costs = [cost[-1] for _, cost in sample_to_costs.items()]

    # Build model
    m = Model("knapsack_joint_task_selection")
    # Add variables
    x = [m.add_var(var_type=BINARY) for _ in range(num_sample_ids)]
    # Add objective function
    m.objective = maximize(xsum((1 - confidence_scores[j]) * x[j] for j in range(num_sample_ids)))
    # Add budget constraints
    m += (xsum(joint_costs[j] * x[j] for j in range(num_sample_ids)) <= total_cost)
    # Solve model
    m.optimize(max_seconds=500)

    # Retrieve selected samples per task
    sample_ids = [all_sample_ids[j] for j in range(num_sample_ids) if x[j].x >= 0.99]
    sample_ids = [sample_ids for _ in range(num_tasks)]
    return sample_ids

def mip_single_task_confidence_selection(sample_to_confidences: Dict[int, np.array],
                                         sample_to_costs: Dict[int, np.array],
                                         total_cost: Union[int, float],
                                         tasks: List[str],
                                         task_for_scoring: str) -> List[List[int]]:
    """
    Computing BLP optimization for single task confidence selection:
    A sample can be annotated on none/all of the tasks.
    We only use the confidence scores of task_for_scoring for the objective function.
    Since we would like to annotate samples on all tasks, we use the joint costs.

    The BLP optimization maximizes the single-task uncertainties (1 - confidence)
    of the selected samples given the budget constraints.
    Please make sure to normalize the confidence scores such that they are in the range
    of 0 to 1.

    :param sample_to_confidences: a mapping from sample to its confidence scores
    (sorted by the task order).
    :param sample_to_costs: a mapping from sample to its cost functions
    (sorted by the task order). Additionally, please make sure to add the joint cost function
    of every sample at the final cell of its array.
    :param total_cost: the total budget cost.
    :param tasks: a list of the trained tasks.
    :param task_for_scoring: a string of the task that should be considered for the AL selection.
    :return: a list containing lists of the selected samples (sorted by the task order).
    """

    # Number of tasks
    num_tasks = len(tasks)

    # Get task index of task_for_scoring
    task_index = -1
    for index in range(num_tasks):
        if task_for_scoring == tasks[index]:
            task_index = index
            break
    if task_index == -1:
        raise ValueError(f'task_for_scoring: {task_for_scoring} was not found in tasks')

    # Prepare sample ids
    all_sample_ids = [sample_id for sample_id, conf in sample_to_confidences.items()]
    num_sample_ids = len(all_sample_ids)

    # Prepare confidence scores (of task_for_scoring)
    confidence_scores = [confidence[task_index] for _, confidence in sample_to_confidences.items()]
    # Prepare costs (we take the joint costs in this case,
    # and not the per-task costs as previously)
    joint_costs = [cost[-1] for _, cost in sample_to_costs.items()]

    # Build model
    m = Model("knapsack_single_task_confidence_selection")
    # Add variables
    x = [m.add_var(var_type=BINARY) for _ in range(num_sample_ids)]
    # Add objective function
    m.objective = maximize(xsum((1 - confidence_scores[j]) * x[j] for j in range(num_sample_ids)))
    # Add budget constraints
    m += (xsum(joint_costs[j] * x[j] for j in range(num_sample_ids)) <= total_cost)
    # Solve model
    m.optimize(max_seconds=500)

    # Retrieve selected samples per task
    sample_ids = [all_sample_ids[j] for j in range(num_sample_ids) if x[j].x >= 0.99]
    sample_ids = [sample_ids for _ in range(num_tasks)]
    return sample_ids
