from dataclasses import dataclass
import os
import json
from pydantic import BaseModel, Field
import constants
from openai import OpenAI


class TasksRepository:
    def __init__(self, path_to_task_dir: str = "./task-dir"):
        if os.getenv("TASKS") is not None and os.getenv("TASKS") != "":
            path_to_task_dir = os.getenv("TASKS")

        self.task_data_array = self.get_task_data_array(path_to_task_dir)

    def get_task_data_array(self, path_to_task_dir: str):
        task_data_array = []
        folders = [os.path.join(path_to_task_dir, f) for f in os.listdir(path_to_task_dir) if os.path.isdir(os.path.join(path_to_task_dir, f))]
        for f in folders:
            task_file_path = os.path.join(f, "task.json")

            with open(task_file_path, 'r') as file:
                json_data = json.load(file)
                task_data = TaskData(
                    task_id = json_data["task_id"],
                    task_type = json_data["task_type"],
                    instruction = json_data["instruction"],
                    reward = json_data["reward"]
                )
                task_data_array.append(task_data)

        return task_data_array

class APIService:
    def __init__(self, base_url, api_key):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
    

    def EstimateTask(self, judge_model, instruction, model_name, budget_in_tokens):
        resp = self.client.chat.completions.parse(
            model=judge_model,
            messages=[
                {"role": "system", "content": constants.SYSTEM},
                {"role": "user", "content": f"Task: {instruction}\nModel: {model_name}"},
            ],
            response_format=TaskEstimate,
            temperature=0,
        )

        budget_in_tokens = budget_in_tokens - resp.usage.total_tokens

        return resp.choices[0].message.parsed

@dataclass
class TaskData:
    task_id: str
    task_type: str
    instruction: str
    reward: int

class TaskEstimate(BaseModel):
    model_name: str
    success_percent: int = Field(description="Chance of completing the task, 0-100")
    tokens: int = Field(description="Total input+output tokens expected")
    seconds: int = Field(description="Wall-clock time expected, in seconds")
    reward: int