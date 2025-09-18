import configparser
import os
import re
import sys

# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the directory to sys.path
if current_script_dir not in sys.path:
    sys.path.append(current_script_dir)

print(current_script_dir)
import QGIS_tool_creation_constants as const

def get_OpenAI_key():
    config = configparser.ConfigParser()
    config_path = os.path.join(current_script_dir, 'tool_creation_config.ini')
    config.read(config_path)
    OpenAI_key = config.get('API_Key', 'OpenAI_key')
    return OpenAI_key


# def create_CodeSample_prompt(algorithm_id, tool_synopsis, tool_flags, tool_doc, tool_document):
#     CodeSample_requirement_str = '\n'.join(
#         [f"{idx + 1}. {line}" for idx, line in enumerate(const.CodeSample_requirements)])
#
#     prompt =f"Your role: {const.CodeSample_role} \n" + \
#             f"Your mission: {const.CodeSample_prefix}: " + f"\n{tool_doc}\n\n" + \
#             f"Requirements: \n{CodeSample_requirement_str} \n\n" + \
#             f"The correct algorithm ID to be used is: {algorithm_id}\n\n" + \
#             f"The tool synopsis: {tool_synopsis}\n\n" + \
#             f"Tool flags: {tool_flags}\n\n" + f"Tool document: {tool_document}"
#             # f"Some code examples. Example 1: {const.CodeSample_example1}\n " + " OR " + f"Example 2: {const.CodeSample_reply_example2}"
#     return prompt


def create_CodeSample_prompt(algorithm_id, tool_synopsis, tool_flags, tool_doc, tool_document):
    CodeSample_requirement_str = '\n'.join(
        [f"{idx + 1}. {line}" for idx, line in enumerate(const.CodeSample_requirements)])

    prompt =f"Your role: {const.CodeSample_role} \n" + \
            f"Your mission: {const.CodeSample_prefix}: \n\n" + \
            f"The correct algorithm ID to be used is: {algorithm_id}\n\n" + \
            f"The tool synopsis: {tool_synopsis}\n\n" + \
            f"Tool flags: {tool_flags}\n\n" +\
            f"Tool document: {tool_doc}\n\n"+\
            f"Description: {tool_document}\n\n" +\
            f"Requirements: \n{CodeSample_requirement_str}"

            # f"Some code examples. Example 1: {const.CodeSample_example1}\n " + " OR " + f"Example 2: {const.CodeSample_reply_example2}"
    return prompt


async def fetch_chunks(model, prompt_str):
    chunks = []
    async for chunk in model.astream(prompt_str):
        chunks.append(chunk)
        # print(chunk.content, end="", flush=True)
    return chunks


def convert_chunks_to_str(chunks):
    LLM_reply_str = ""
    for c in chunks:
        # print(c)

        cleaned_str = c.content.replace("```json", "").replace("```", "")
        LLM_reply_str += cleaned_str
        # # Append content, remove backticks, and strip leading/trailing whitespace

    return LLM_reply_str


def extract_code (response, verbose = True):
    '''
    Extract python code from reply

    '''
    python_code = ""
    python_code_match = re.search("python", response, re.DOTALL)
    if python_code_match:
        python_code = python_code_match.group(1).strip()
    if verbose:
        print(python_code)
    return python_code