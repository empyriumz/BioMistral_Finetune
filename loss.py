import torch


def _get_possible_permutation_matrix(
    events: torch.Tensor,
    durations: torch.Tensor,
    inc_censored_in_ties=True,
    eps: float = 1e-6,
):
    """
    Returns the possible permutation matrix label for the given events and durations.

    For a right-censored sample `i`, we only know that the risk must be lower than the risk of all other
    samples with an event time lower than the censoring time of `i`, i.e. they must be ranked after
    these events. We can thus assign p=0 of sample `i` being ranked before any prior events, and uniform
    probability that it has a higher ranking.

    For another sample `j` with an event at `t_j`, we know that the risk must be lower than the risk of
    other samples with an event time lower than `t_j`, and higher than the risk of other samples either
    with an event time higher than `t_j` or with a censoring time higher than `t_j`. We do not know how
    the risk compares to samples with censoring time lower than `t_j`, and thus have to assign uniform
    probability to their rankings.
    :param events: binary vector indicating if event happened or not
    :param durations: time difference between observation start and event time
    :param inc_censored_in_ties: if we want to include censored events as possible permutations in ties with events
    :return:
    """
    # Initialize the soft permutation matrix
    perm_matrix = torch.zeros(events.shape[0], events.shape[0], device=events.device)

    # eps here forces ties between censored and event to be ordred event first (ascending)
    idx = torch.argsort(durations - events * eps, descending=False)

    # Used to return to origonal order
    perm_un_ascending = torch.nn.functional.one_hot(idx).transpose(-2, -1).float()
    ordered_durations = durations[idx]
    ordered_events = events[idx]

    # events_ordered = events[idx]
    event_counts = 0

    idx_stack = list(range(idx.shape[0]))
    idx_stack.reverse()
    i_event = []
    i_censored = []
    while idx_stack:
        if ordered_events[idx_stack[-1]]:
            i_event.append(idx_stack.pop())
        else:
            i_censored.append(idx_stack.pop())

        # Handle Ties: Look ahead, if next has the same time, add next index!
        i_all = i_event + i_censored
        if (
            idx_stack
            and i_all
            and (ordered_durations[i_all[-1]] == ordered_durations[idx_stack[-1]])
        ):
            continue

        if inc_censored_in_ties and i_censored:
            # Right censored samples
            # assign 0 for all samples with event time lower than the censoring time
            # perm_matrix[i, : i[-1]] = 0
            # assign uniform probability to all samples with event time higher than the censoring time
            # includes previous censored events that happened before the event time
            perm_matrix[i_censored, event_counts:] = 1
            i_censored = []  # clear idx on  censored

        # Events
        # Assign uniform probability to an event and all censored events with shorter time,
        if i_event:
            if inc_censored_in_ties:
                perm_matrix[i_event, event_counts : max(i_all) + 1] = 1
            else:
                perm_matrix[i_event, event_counts : max(i_event) + 1] = 1
            event_counts += int(sum(ordered_events[i_event]))
            i_event = []  # reset indices no more ties

        if not inc_censored_in_ties and i_censored:
            perm_matrix[i_censored, event_counts:] = 1
            i_censored = []  # clear idx on  censored

    # Permute to match the order of the input
    perm_matrix = perm_un_ascending @ perm_matrix

    return perm_matrix


def diffsurv_loss(perm_prediction, survival_times, event_indicators):
    perm_ground_truth = _get_possible_permutation_matrix(
        event_indicators, survival_times, inc_censored_in_ties=True
    )

    possible_predictions = (perm_ground_truth * perm_prediction).sum(dim=1)

    loss = torch.nn.BCELoss()(
        torch.clamp(possible_predictions, 1e-8, 1 - 1e-8),
        torch.ones_like(possible_predictions),
    )

    return loss
