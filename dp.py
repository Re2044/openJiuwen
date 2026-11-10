import logging
import time
from collections import defaultdict

import numpy as np

log = logging.getLogger(__name__)

UNREACHABLE = np.iinfo(np.int64).max // 4


class DP:
    """Picks estimates that maximize total reward within a token budget and a time limit.

    Estimates that share a group (the same task on different models) are
    alternatives: at most one of them is picked.

    The table is indexed by total reward and total seconds and holds the fewest
    tokens that reach that state. Rewards add up to a few hundred while the
    budget runs to a million tokens, so this is far smaller than indexing by tokens.
    """

    def calc(self, budget_in_tokens, time_limit_in_seconds, tasks, groups=None):
        """Return (best total reward, 1-based indices into `tasks` of the picked estimates).

        `groups` gives a key per estimate, such as its task id. Without it every
        estimate is its own group.
        """
        started = time.monotonic()
        if groups is None:
            groups = list(range(len(tasks)))
        if len(groups) != len(tasks):
            raise ValueError("groups must have one key per estimate")

        options_by_group = defaultdict(list)
        for index, (estimate, group) in enumerate(zip(tasks, groups), start=1):
            reward = int(estimate.reward)
            if reward < 0 or estimate.tokens < 0 or estimate.seconds < 0:
                log.warning("skipping estimate %d with a negative value: %s", index, estimate)
                continue
            # An estimate that alone breaks a limit can never be picked.
            if estimate.tokens <= budget_in_tokens and estimate.seconds <= time_limit_in_seconds:
                options_by_group[group].append((index, reward, int(estimate.tokens), int(estimate.seconds)))
        option_groups = list(options_by_group.values())

        max_reward = sum(max(option[1] for option in options) for options in option_groups)
        shape = (max_reward + 1, time_limit_in_seconds + 1)
        log.info("DP: %d estimates in %d groups, table %d rewards x %d seconds", len(tasks), len(option_groups),
                 shape[0], shape[1])

        tokens = np.full(shape, UNREACHABLE, dtype=np.int64)
        tokens[0, 0] = 0
        # choices[g][r, s]: which option of group g leads to state (r, s), or -1 for none.
        choices = []
        for options in option_groups:
            updated = tokens.copy()
            choice = np.full(shape, -1, dtype=np.int16)
            for option_number, (_, reward, option_tokens, seconds) in enumerate(options):
                source = tokens[:shape[0] - reward, :shape[1] - seconds]
                candidate = np.where(source < UNREACHABLE, source + option_tokens, UNREACHABLE)
                target = updated[reward:, seconds:]
                better = candidate < target
                target[better] = candidate[better]
                choice[reward:, seconds:][better] = option_number
            tokens = updated
            choices.append(choice)

        feasible = tokens <= budget_in_tokens
        rewards_reachable = np.nonzero(feasible.any(axis=1))[0]
        best_reward = int(rewards_reachable.max())
        # Among the ways to reach the best reward, take the one using the fewest tokens.
        row = np.where(feasible[best_reward], tokens[best_reward], UNREACHABLE)
        seconds_used = int(row.argmin())

        picked = []
        reward_left, seconds_left = best_reward, seconds_used
        for options, choice in zip(reversed(option_groups), reversed(choices)):
            option_number = int(choice[reward_left, seconds_left])
            if option_number < 0:
                continue
            index, reward, _, seconds = options[option_number]
            picked.append(index)
            reward_left -= reward
            seconds_left -= seconds

        log.info("DP finished in %.2fs: reward %d using %d tokens and %d s", time.monotonic() - started,
                 best_reward, int(tokens[best_reward, seconds_used]), seconds_used)
        return best_reward, sorted(picked)
