import os
import sys
from openai import OpenAI
# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
SpatialAnalysisAgent_dir = os.path.join(current_script_dir, 'SpatialAnalysisAgent')
# print(LLMQGIS_dir)
if current_script_dir not in sys.path:
    sys.path.append(SpatialAnalysisAgent_dir)


# import SpatialAnalysisAgent_Constants as constants
import TESTING_HELPER as helper

OpenAI_key = helper.load_OpenAI_key()
print(OpenAI_key)



client = OpenAI(api_key=OpenAI_key)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a fun fact about space."}
]

response = client.chat.completions.create(
    model='gpt-5',   # pick an available model
    messages=messages,
    reasoning= {
        "effort":"minimal"
    },
)

content = response.choices[0].message.content
print(content)






#
# # Define the conversation messages

#
# try:
#     # Make the API call for chat completion
#     response = client.chat.completions.create(
#         model="gpt-5",  # Or "gpt-5-mini", "gpt-5-nano"
#         messages=messages,
#         reasoning_effort= {'effort':'low'},
#         input = "What is Buffer in GIS?"
#         # max_completion_tokens=150,  # Optional: limit the response length
#         #temperature=1, # Optional: control creativity
#         #verbosity="medium" # Optional: control verbosity of the response
#     )
#
#     # Print the assistant's reply
#     print(response.choices[0].message.content)
#
# except Exception as e:
#     print(f"An error occurred: {e}")

