from openai import OpenAI
from data_access import DataAccessor, TaskData
import os
import constants

data_accessor = DataAccessor()

tasks = data_accessor.task_data_array

print(tasks)

client = OpenAI(
    base_url="https://api.neuraldeep.ru/v1",
    api_key=os.getenv(constants.API_KEY),
)

r = client.chat.completions.create(
    model="qwen3.6-35b-a3b",
    messages=[{"role": "user", "content": "Привет!"}],
)
print(r.choices[0].message.content)
