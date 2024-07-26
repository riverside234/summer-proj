import json
import openai
import os
from dotenv import load_dotenv
import random

#add data to jsonl file
def append_to_jsonl(file_path, new_data):
    with open(file_path, 'a') as jsonl_file:
        for data in new_data:
            jsonl_file.write(json.dumps(data) + '\n')

finetuning_jsonl_file_path = 'finetuning-data.jsonl'

finetuning_new_data = []

prompt_eval_jsonl_file_path = 'prompt-eval-gpt4.jsonl'

prompt_eval_new_data = []

tasks_eval_file_path = 'tasks-eval.jsonl'

tasks_eval = []

# Load the environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=api_key)

# Define the topics
topics = ["Fitness and Exercise Routines"]

#define agents
agents = {
    "agent2": {
        "system_message": """
        You are a highly skilled Prompt Engineer and NLP Expert. Your task is to analyze a prompt intended for use with Large Language Model (LLM) like GPT-4.

        Instructions:

        -Identify the prompt purpose, context, and desired outcome to guide LLM-focused analysis.

        -Provide comprehensive and specific suggestions to optimize the prompt for enhanced LLM output quality:
        1. Define the LLM character's role based on the task, such as act as a math professor.
        2. Provide suggestions for the prompt by break complex purpose down into smaller, more specific sub-tasks or more manageable steps.
        3. Use accurate, domain-specific terminology for precision and clarity.
        4. Provide all necessary background context to make the prompt as unambiguous and well-defined as possible.
        5. Provide all necessary details in the instructions to make the prompt as complete and well-specified as possible.
        6. Consider any exceptional scenarios or edge cases not included in the prompt but related to the purpose of the prompt.
        7. Include instructions in the prompt to specify the output format, length, level of detail required, and formatting requirements.
        8. For the output format of the LLM, add instructions in the prompt such that the LLM conducts its reasoning first and then provides the answer.

        -Provide as much suggestions as possible. Prioritize suggestions based on the prompt's desired outcome.
        -For lengthy suggestions, identify overarching themes in the suggestions and consolidate related suggestions into a single, focused recommendation.

        -Provide your suggestions in the following format:

            #Prompt Info:
                ##Purpose: [purpose of the given prompt for LLM].
                ##Context: [context of the given prompt for LLM].
                ##Desired Outcome: [expected outcome or response from LLM].
            #Suggestions for improvement based on instructions: 
                1: [provide comprehensive and specific suggestions].
                2: ...
        """,
        "messages": []
    },
    "agent3": {
        "system_message": f"""
        Given the topic {topics[0]}, generate ten short tasks that can be completed by an LLM like GPT-4. Each task should be a specific, actionable item that reflects an aspect of the topic and can be expressed in a few sentences. Ensure the tasks vary in nature, demonstrating the versatility of LLMs in addressing different aspects of {topics[0]}. Make sure the descriptions of the tasks are diverse. Some descriptions should be shorter, like one sentence, while others should be longer, like a few sentences. Some descriptions should be more specific and detailed, while others can be more vague.
        """,
        "messages": []
    }
}

def CustomChatGPT(agent_name, user_input):
    agent = agents[agent_name]
    if not agent["messages"]:
        agent["messages"].append({"role": "system", "content": agent["system_message"]})
    
    agent["messages"].append({"role": "user", "content": user_input})

    # Prepare the API call parameters
    api_params = {
        "model": "gpt-4o",
        "messages": agent["messages"]
    }

    # Add response_format parameter for agent3
    if agent_name == "agent3":
        api_params["response_format"] = {"type": "json_object"}

    # Use the client to create chat completions
    response = client.chat.completions.create(**api_params)

    ChatGPT_reply = response.choices[0].message.content
    agent["messages"].append({"role": "assistant", "content": ChatGPT_reply})

    return ChatGPT_reply

#Generate dataset
topicTasks = CustomChatGPT("agent3", """
Provide your response in the following JSON format:
{
    "1": "[description of task 1 (can be a few sentences)]",
    "2": ...
    }
""")
print(topicTasks)
 
tasksLoad = json.loads(topicTasks)

random_task = random.randint(1, 10)

for i in range(1, 11):

    task = tasksLoad[str(i)]      
    #reset messages history  
    agents["agent2"]["messages"] = []

    response1 = CustomChatGPT("agent2", f"""
    prompt <{task}>
    """)

    response2 = CustomChatGPT("agent2", """
    Please provide a LLM prompt incorporating all provided suggestions. Ensure the prompt is well-structured, coherent, concise, and utilizes formatting techniques. 
    -Provide the prompt in the following format:
    #Revised Prompt: 
    [revised prompt text goes here]
    """)

    if i == random_task:
        prompt_eval_new_data.append({"original_prompt": task, "revised_prompt": response2})
        tasks_eval.append({"task": task})
    else:

        responseFinal = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a highly skilled Prompt Engineer and NLP Expert. Your task is to analyze a prompt intended for use with Large Language Model (LLM) like GPT-4.<|eot_id|><|start_header_id|>user<|end_header_id|>

    Analysis prompt:{task}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    {response1}<|eot_id|><|start_header_id|>user<|end_header_id|>

    Please provide a LLM prompt incorporating all provided suggestions.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    {response2}<|eot_id|>
    """

    finetuning_new_data.append({"text": responseFinal})


append_to_jsonl(finetuning_jsonl_file_path, finetuning_new_data)
append_to_jsonl(prompt_eval_jsonl_file_path, prompt_eval_new_data)
append_to_jsonl(tasks_eval_file_path, tasks_eval)



