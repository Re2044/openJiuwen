class DP:
    def calc(self, budget_in_tokens, time_limit_in_seconds, tasks):
        dp = [[[-1] * (time_limit_in_seconds + 1)] * (budget_in_tokens + 1)] * (len(tasks) + 1)
        for i in range(0, len(tasks) + 1):
            for j in range(0, budget_in_tokens + 1):
                for k in range(0, time_limit_in_seconds + 1):
                    dp[i][j][k] = 0

        for i in range(1, len(tasks) + 1):
            for j in range(0, budget_in_tokens + 1):
                for k in range(0, time_limit_in_seconds + 1):
                    new_tokens = j + tasks[i - 1].tokens
                    new_seconds = k + tasks[i - 1].seconds
                    if(new_tokens <= budget_in_tokens and new_seconds <= time_limit_in_seconds):
                        dp[i][new_tokens][new_seconds] = max(dp[i][new_tokens][new_seconds], dp[i-1][j][k] + tasks[i - 1].reward)

        return dp[len(tasks)][budget_in_tokens][time_limit_in_seconds]