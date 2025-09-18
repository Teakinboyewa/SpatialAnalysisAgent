import os
import sys
from openai import OpenAI
# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
SpatialAnalysisAgent_dir = os.path.join(current_script_dir, 'SpatialAnalysisAgent')

if current_script_dir not in sys.path:
    sys.path.append(SpatialAnalysisAgent_dir)


# import SpatialAnalysisAgent_Constants as constants
import TESTING_HELPER as helper

OpenAI_key = helper.load_OpenAI_key()
# print(OpenAI_key)

client = OpenAI(api_key=OpenAI_key)

prompt = "Give a brief and concise description"


response = client.responses.create(
    model="gpt-5",
    input= [{ 'role': 'developer', 'content': prompt },
            { 'role': 'user', 'content': 'What is buffer analysis in GIS?' }],
    reasoning = {
        "effort": "minimal"
    },
)

print(response)
content = response.output_text
print(content)

# # Extract model's text output
# output_text = ""
# for item in response.output:
#     if hasattr(item, "content"):
#         for content in item.content:
#             if hasattr(content, "text"):
#                 output_text += content.text
#
# # Token usage details
# usage = response.usage
#
# print("--------------------------------")
# print("Output:")
# print(output_text)

