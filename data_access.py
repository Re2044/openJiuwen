from dataclasses import dataclass
import os
import json

class DataAccessor:
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


@dataclass
class TaskData:
    task_id: str
    task_type: str
    instruction: str
    reward: int