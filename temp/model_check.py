import openai, os
from dotenv import load_dotenv
load_dotenv()
client = openai.Client(
    api_key=os.environ.get("CSCS_SERVING_API"),
    base_url="https://api.swissai.svc.cscs.ch/v1"
)
models = client.models.list()
for m in models.data:
    print(m.id)