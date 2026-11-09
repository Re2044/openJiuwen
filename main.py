from openai import OpenAI
from data_access import TasksRepository, TaskData, TaskEstimate, APIService
from dp import DP
import os
import constants
from dotenv import load_dotenv

data_accessor = TasksRepository()

load_dotenv()

tasks = data_accessor.task_data_array

api_service = APIService(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)

budget_in_tokens = int(os.getenv("BUDGET"))

task_estimates = []
judge_model = constants.JUDGE_MODEL_NAME

for task in tasks:
    for model_name in constants.MODEL_NAME_LIST:
        task_estimate = api_service.EstimateTask(judge_model, task.instruction, model_name, budget_in_tokens)
        task_estimate.model_name = model_name
        task_estimate.reward = task.reward
        task_estimates.append(task_estimate)


dp = DP()
res = dp.calc(budget_in_tokens, constants.TIME_LIMIT_IN_SECONDS, task_estimates)