import json
import openai
import os
from dotenv import load_dotenv

# initialize data and file paths
gpt4o_res = []
gpt4o_res_jsonl_file_path = 'gpt4o-gpt4oResponse.jsonl'
prompt_jsonl_file_path = 'prompt-eval-gpt4.jsonl'

# Load the environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=api_key)

# Define chatbot messages
messages = []

def GPT(user_input):
    messages.append({"role": "user", "content": user_input})

    # Use the client to create chat completions
    response = client.chat.completions.create(
        model = "gpt-4o",
         messages = messages
    )

    ChatGPT_reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply

#add data to jsonl file
def append_to_jsonl(file_path, new_data):
    with open(file_path, 'a') as jsonl_file:
        for data in new_data:
            jsonl_file.write(json.dumps(data) + '\n')

# read prompt from jsonl file
def read_prompt_jsonl(prompt_file_path):
    with open(prompt_file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            prompts = json.loads(line)
            # Process the prompts
            original_prompt = prompts['original_prompt']
            revised_prompt = prompts['revised_prompt']
             
            messages = [{"role": "system", "content": ""}]

            original_response = GPT(f"{original_prompt}")

            messages = [{"role": "system", "content": ""}]
            revised_response = GPT(f"{revised_prompt}")

            gpt4o_res.append({"task": original_prompt, "original_response": original_response, "revised_response": revised_response})

read_prompt_jsonl(prompt_jsonl_file_path)
append_to_jsonl(gpt4o_res_jsonl_file_path, gpt4o_res)
