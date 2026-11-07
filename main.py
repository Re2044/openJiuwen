from openai import OpenAI
from data_access import TasksRepository, TaskData, TaskEstimate, APIService
import os
import constants

data_accessor = TasksRepository()

tasks = data_accessor.task_data_array

api_service = APIService(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)

task_estimates = []
judge_model = constants.JUDGE_MODEL_NAME

for task in tasks:
    for model_name in constants.MODEL_NAME_LIST:
        task_estimate = api_service.EstimateTask(judge_model, task.instruction, model_name)
        task_estimates.append(e)

