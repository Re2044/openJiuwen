class DP:
    def calc(self, budget_in_tokens, time_limit_in_seconds, tasks):
        dp = [[[0] * (time_limit_in_seconds + 1)] * (budget_in_tokens + 1)] * (len(tasks) + 1)
        path = [[[(0, -1, -1)] * (time_limit_in_seconds + 1)] * (budget_in_tokens + 1)] * (len(tasks) + 1)

        for i in range(1, len(tasks) + 1):
            for j in range(1, budget_in_tokens + 1):
                for k in range(1, time_limit_in_seconds + 1):
                    if dp[i][j][k - 1] > dp[i][j][k]:
                        dp[i][j][k] = dp[i][j][k - 1]
                        path[i][j][k] = path[i][j][k-1]

                    if dp[i-1][j][k] > dp[i][j][k]:
                        dp[i][j][k] = dp[i-1][j][k]
                        path[i][j][k] = (0, j, k)

                    if dp[i][j-1][k] > dp[i][j][k]:
                        dp[i][j][k] = dp[i][j-1][k]
                        path[i][j][k] = path[i][j-1][k]

                    old_tokens = j - tasks[i - 1].tokens
                    old_seconds = k - tasks[i - 1].seconds
                    if (old_tokens >= 0 and old_seconds >= 0) and dp[i - 1][old_tokens][old_seconds] + tasks[i - 1].reward > dp[i][j][k]:
                        dp[i][j][k] = dp[i-1][old_tokens][old_seconds] + tasks[i - 1].reward
                        path[i][j][k] = (1, old_tokens, old_seconds)

        res = self.backtrack_path(path, len(tasks), budget_in_tokens, time_limit_in_seconds)
        return (dp[len(tasks)][budget_in_tokens][time_limit_in_seconds], res)

    
    def backtrack_path(self, path, tasks_len, budget_in_tokens, time_limit_in_seconds):
        res = []

        curr_i = tasks_len
        curr_j = budget_in_tokens
        curr_k = time_limit_in_seconds

        while (curr_i > 0 and curr_j != -1 and curr_k != -1):
            cell = path[curr_i][curr_j][curr_k]
            if cell[0] == 1:
                res.append(curr_i)

            curr_i = curr_i - 1
            curr_j = cell[1]
            curr_k = cell[2]
        
        return res