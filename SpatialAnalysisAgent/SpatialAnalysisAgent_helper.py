import ast
import asyncio
import io
import json
import sys
import re
import traceback
# import openai
from collections import deque
from io import StringIO
import nest_asyncio
import toml
from IPython.core.display_functions import clear_output
# from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
import configparser
# import networkx as nx
import logging
import time
import os
import requests
import networkx as nx
import pandas as pd
# import geopandas as gpd
from pyvis.network import Network
# import processing
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.embeddings import OpenAIEmbeddings  # Or HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, LLMChain
from langchain_core.prompts import PromptTemplate

# from SpatialAnalysisAgent.SpatialAnalysisAgent_MyScript_v2 import reasoning_effort_value

# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the directory to sys.path
if current_script_dir not in sys.path:
    sys.path.append(current_script_dir)

json_path = os.path.join(current_script_dir, 'Tools_Documentation', 'qgis_tools_for_rag.json')

def load_config():
    config = configparser.ConfigParser()
    config_path = os.path.join(current_script_dir, 'config.ini')
    config.read(config_path)
    return config


def load_OpenAI_key():
    config = load_config()  # Re-read the configuration file
    OpenAI_key = config.get('API_Key', 'OpenAI_key')
    return OpenAI_key


def create_openai_client():
    OpenAI_key = load_OpenAI_key()
    return OpenAI(api_key=OpenAI_key)

def get_question_id(user_api_key):
    import requests
    url = f"https://www.gibd.online/api/request-question-id"
    payload = {
            "service_name": "GIS Copilot",
            "user_api_key": user_api_key}
    response = requests.post(url, json=payload)
    response.text
    return response.json()["question_id"]


# def workspace_directory(path):
#     path =

import SpatialAnalysisAgent_Constants as constants
# import SpatialAnalysisAgent_Eye_Constant as eye_constant
import SpatialAnalysisAgent_Codebase as codebase


# def create_OperationIdentification_prompt(task):
#     OperationIdentification_requirement_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(constants.OperationIdentification_requirements)])
#
#     prompt =    f"Your role: {constants.OperationIdentification_role} \n" + \
#                 f"Your mission: {constants.OperationIdentification_task_prefix}: " + f"{task}\n\n" + \
#                 f"Requirements: \n{OperationIdentification_requirement_str} \n\n" + \
#                 f"List of QGIS tools: {codebase.algorithm_names} \n" +\
#                 f'Your reply example: {constants.OperationIdentification_reply_example}'
#     return prompt

# Add this function to generate the task name using specific gpt model

def generate_task_name_with_gpt(specific_model_name, task_description):
    prompt = f"Given the following task description: '{task_description}',give the best task that represents this task.\n\n" + \
             f"Provide the task name in just one or two words. \n\n" + \
             f"Underscore '_' is the only alphanumeric symbols that is allowed in a task name. A task_name must not contain quotations or inverted commas example or space. \n"
    # Fallback to basic OpenAI client
    client = create_openai_client()
    response = client.chat.completions.create(
        model=specific_model_name,
        messages=[
            {"role": "user", "content": prompt},
        ])
    print(specific_model_name)
    task_name = response.choices[0].message.content
    return task_name

# Add this function to generate the task name using UNIFIED MODEL PROVIDER
def generate_task_name_with_model_provider(request_id, model_name, task_description):
    prompt = f"Given the following task description: '{task_description}',give the best task that represents this task.\n\n" + \
             f"Provide the task name in just one or two words. \n\n" + \
             f"Underscore '_' is the only alphanumeric symbols that is allowed in a task name. A task_name must not contain quotations or inverted commas example or space. \n"

    # Use the unified model provider
    try:
        from SpatialAnalysisAgent_ModelProvider import create_unified_client
        client, provider = create_unified_client(model_name)
        messages=[
            {"role": "user", "content": prompt},
        ]
        # Generate response using the provider
        response = provider.generate_completion(

            request_id,
            client, 
            model_name,
            messages,
        )
        # # It's a regular response object
        # if hasattr(response, "choices"):
        #     task_name = response.choices[0].message.content.strip()
        # elif isinstance(response, dict):
        #     task_name = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        # else:
        #     task_name = str(response)

        task_name = streaming_openai_response(response)


    except ImportError:
        # Fallback to basic OpenAI client
        client = create_openai_client()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt},
            ]
        )

        task_name = response.choices[0].message.content
    return task_name


def create_OperationIdentification_promt(task):
    OperationIdentification_requirement_str = '\n'.join(
        [f"{idx + 1}. {line}" for idx, line in enumerate(constants.OperationIdentification_requirements)])

    prompt = f"Your role: {constants.OperationIdentification_role} \n" + \
             f"Your mission: {constants.OperationIdentification_task_prefix}: " + f"{task}\n" + \
             f"Requirements: \n{OperationIdentification_requirement_str} \n\n" + \
             f"Customized tools:\n{constants.tools_index}\n" + \
             f"Your reply examples, depending on the task. Example 1: {constants.OperationIdentification_reply_example_1}\n " + " OR " + f"Example 2: {constants.OperationIdentification_reply_example_2}\n" + " OR " + f"Example 3: {constants.OperationIdentification_reply_example_3}"
    return prompt


def create_ToolSelect_prompt(task, data_path):
    ToolSelect_requirement_str = '\n'.join(
        [f"{idx + 1}. {line}" for idx, line in enumerate(constants.ToolSelect_requirements)])
    data_path_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(data_path)])

    prompt = f"Your role: {constants.ToolSelect_role} \n" + \
             f"Your mission: {constants.ToolSelect_prefix}: " + f"{task}\n\n" + \
             f"Based on the provided data {data_path_str}\n" + \
             f"Requirements: \n{ToolSelect_requirement_str} \n\n" + \
             f"Customized tools:\n{constants.tools_index}\n" + \
             f"Example for your reply: {constants.ToolSelect_reply_example2}\n"

    return prompt


def create_operation_prompt(task, data_path, selected_tools, documentation_str, workspace_directory):
    operation_requirement_str = '\n'.join(
        [f"{idx + 1}. {line}" for idx, line in enumerate(constants.operation_requirement)])
    data_path_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(data_path)])
    prompt = f"Your role: {constants.operation_role} \n" + \
             f"Your mission: {constants.operation_task_prefix}: " + f"{task}" + "Using the following data paths: " + f"{data_path_str}" + "\nAnd this output directory: " + f"{workspace_directory}\n\n" + \
             f"Using the following Selected tool(s): {selected_tools}\n" + \
             f"Documentation of the selected tools: \n{documentation_str}\n" + \
             f"requirements: \n{operation_requirement_str}\n" + \
             f"Set: " + f"{workspace_directory}" + " as the output directory for any operation"
    return prompt


def generate_operation_code(request_id, operation_prompt_str, model_name, stream):
    """Return a fine-tuned prompt using the selected model.
    Supports: OpenAI proxy, GPT-5, and normal OpenAI"""
    return unified_llm_call(
        request_id = request_id,
        messages = [
        {"role": "user", "content": operation_prompt_str},
    ],
    model_name=model_name,
    stream=stream
    )



def code_review_prompt(extracted_code, data_path, selected_tool_dict, workspace_directory, documentation_str):
    operation_code_review_requirement_str = '\n'.join(
        [f"{idx + 1}. {line}" for idx, line in enumerate(constants.operation_code_review_requirement)])
    # print(f"Code passed to review: {extracted_code}")
    operation_code_review_prompt = f"Your role: {constants.operation_code_review_role} \n" + \
                                   f"Your mission: {constants.operation_code_review_task_prefix} \n\n" + \
                                   f"The code is: \n----------\n{extracted_code}\n----------\n\n" + \
                                   f"The properties of the data are given below:\n{data_path}\n" + \
                                   f"Using the following selected tool(s):{selected_tool_dict}\n " + \
                                   f"The code examples in the Documentation: \n{documentation_str} can be used as an example while reviewing the {extracted_code} \n\n" + \
                                   f"The requirements for the code is: \n{operation_code_review_requirement_str}\n\n" + \
                                   f"Output directory that should be used:{workspace_directory}"
    return operation_code_review_prompt



def code_review(request_id, code_review_prompt_str, model_name, stream):
    """Return a fine-tuned prompt using the selected model.
    Supports: OpenAI proxy, GPT-5, and normal OpenAI"""
    return unified_llm_call(
        request_id=request_id,
        messages=[
            {"role": "user", "content": code_review_prompt_str},
        ],
        model_name=model_name,
        stream=stream
    )


# def get_code_for_operation(task_description, data_path, selected_tool, selected_tool_ID, documentation_str, review =True):
def get_code_for_operation(model_name, task_description, data_path, selected_tool, selected_tool_ID, selected_tool_dict, documentation_str,
                           review=True, stream=True):
    """
    Generate operation code using unified LLM call.
    Supports: OpenAI proxy, GPT-5, Ollama, and normal OpenAI
    """
    operation_requirement_str = '\n'.join(
        [f"{idx + 1}. {line}" for idx, line in enumerate(constants.operation_requirement)])

    user_prompt = f"Your mission: {constants.operation_task_prefix}: {task_description}\n\n" + \
                  f"Using the following data paths: {data_path}\n" + \
                  f"Selected tool: {selected_tool}\n" + \
                  f'{selected_tool_ID} Documentation: \n{documentation_str}\n' + \
                  f'Requirements: \n{operation_requirement_str}'

    messages = [
        {"role": "system", "content": constants.operation_role},
        {"role": "user", "content": user_prompt}
    ]

    response_str = unified_llm_call(
        messages=messages,
        model_name=model_name,
        stream=stream,
        temperature=1
    )

    # Extract code from the string response
    extracted_code = extract_code_from_str(response_str)

    # Debugging: Print the operation_code to ensure it was extracted correctly
    print(f"Extracted Operation Code: {extracted_code}")

    if review:
        operation_code = ask_LLM_to_review_operation_code(model_name, extracted_code, selected_tool_ID, selected_tool_dict, documentation_str)
        return operation_code
    else:
        return extracted_code


def ask_LLM_to_review_operation_code(model_name, extracted_code, selected_tool_ID, selected_tool_dict, documentation_str, stream=False):
    """
    Review operation code using unified LLM call.
    Supports: OpenAI proxy, GPT-5, Ollama, and normal OpenAI
    """
    operation_code_review_requirement_str = '\n'.join(
        [f"{idx + 1}. {line}" for idx, line in enumerate(constants.operation_code_review_requirement)])

    print(f"Code passed to review: {extracted_code}")

    user_prompt = f"Your task: {constants.operation_code_review_task_prefix} \n\n" + \
                  f"The code is: \n----------\n{extracted_code}\n----------\n\n" + \
                  f"The selected tool(s) is: {selected_tool_dict}\n" + \
                  f'{selected_tool_ID} Documentation: \n{documentation_str} \n\n' + \
                  f"The requirements for the code is: \n{operation_code_review_requirement_str}"

    print("LLM is reviewing the operation code... \n")

    messages = [
        {"role": "system", "content": constants.operation_code_review_role},
        {"role": "user", "content": user_prompt}
    ]

    response_str = unified_llm_call(
        messages=messages,
        model_name=model_name,
        stream=stream,
        temperature=1
    )

    # Extract code from the string response
    reviewed_code = extract_code_from_str(response_str)

    return reviewed_code


def convert_chunks_to_str(chunks):
    LLM_reply_str = ""
    for c in chunks:
        # print(c)

        cleaned_str = c.content.replace("```json", "").replace("```", "")
        LLM_reply_str += cleaned_str
        # # Append content, remove backticks, and strip leading/trailing whitespace

    return LLM_reply_str


def extract_dictionary_from_response(response):
    dict_pattern = r"\{.*?\}"
    match = re.search(dict_pattern, response)
    if match:
        dict_string = match.group()  # Extract the dictionary-like string
        return dict_string
    else:
        print("No dictionary found in the response.")
        return "{}"  # Return empty dict string as fallback


def convert_chunks_to_code_str(chunks):
    LLM_reply_str = ""
    for c in chunks:
        # Append content, remove backticks, and strip leading/trailing whitespace
        LLM_reply_str += c.content
    return LLM_reply_str


def fix_json_format(incorrect_json_str):
    # Fix common JSON issues such as missing double quotes around keys
    # Example: convert {selected tool: ["Clip","Scatterplot"]} to {"selected tool": ["Clip","Scatterplot"]}
    fixed_json_str = re.sub(r'(\w+):', r'"\1":', incorrect_json_str)
    return fixed_json_str


def parse_llm_reply(LLM_reply_str):
    try:
        # Try to load the string directly as JSON
        selection_operation = json.loads(LLM_reply_str)
    except json.JSONDecodeError:
        # If it fails, try to fix the JSON format and decode again
        corrected_reply = fix_json_format(LLM_reply_str)
        try:
            selection_operation = json.loads(corrected_reply)
        except json.JSONDecodeError as e:
            # If it still fails, return None or raise an error as per your needs
            print(f"Failed to parse LLM reply: {e}")
            selection_operation = None
    except TypeError as e:
        # Catch the case where input is not a string, bytes, or bytearray
        print(f"TypeError: {e} - Input must be a valid JSON string.")
        selection_operation = None
    return selection_operation


def get_LLM_reply(prompt="Provide Python code to read a CSV file from this URL and store the content in a variable. ",
                  system_role=r'You are a professional Geo-information scientist and developer.',
                  model_name=r"gpt-4o",
                  request_id="",
                  # model=r"gpt-3.5-turbo",
                  verbose=True,
                  temperature=1,
                  stream=True,
                  retry_cnt=3,
                  sleep_sec=10,
                  reasoning_effort="medium"  # Add reasoning_effort parameter for GPT-5
                  ):

    try:
        from SpatialAnalysisAgent_ModelProvider import create_unified_client
        client, provider = create_unified_client(model_name)
        use_unified_client = True
    except ImportError:
        # Fallback to basic OpenAI client
        client = create_openai_client()
        use_unified_client = False
    
    count = 0
    isSucceed = False
    while (not isSucceed) and (count < retry_cnt):
        try:
            count += 1
            if use_unified_client:
                # Generate response using the provider
                # Add reasoning_effort for GPT-5
                kwargs = {
                    'stream': stream,
                    'temperature': temperature
                }
                if model_name == 'gpt-5':
                    kwargs['reasoning_effort'] = reasoning_effort

                response = provider.generate_completion(
                    request_id,
                    client,
                    model_name,
                    [{"role": "system", "content": system_role},
                     {"role": "user", "content": prompt}],
                    **kwargs
                )
            else:
                response = client.chat.completions.create(model=model_name,
                                                          messages=[
                                                              {"role": "system", "content": system_role},
                                                              {"role": "user", "content": prompt},
                                                          ],
                                                          temperature=temperature,
                                                          stream=stream)
            isSucceed = True  # Mark as successful if we reach here
        except Exception as e:
            # logging.error(f"Error in get_LLM_reply(), will sleep {sleep_sec} seconds, then retry {count}/{retry_cnt}: \n", e)
            print(f"Error in get_LLM_reply(), will sleep {sleep_sec} seconds, then retry {count}/{retry_cnt}: \n", e)
            time.sleep(sleep_sec)

    response_chucks = []
    if stream:
        for chunk in response:
            response_chucks.append(chunk)
            # Handle different response formats based on provider
            content = None
            if use_unified_client and hasattr(chunk, 'type'):
                # Handle GPT-5 ResponseCreatedEvent format
                if hasattr(chunk, 'response') and hasattr(chunk.response, 'body') and hasattr(chunk.response.body, 'choices'):
                    if chunk.response.body.choices and hasattr(chunk.response.body.choices[0], 'delta'):
                        content = chunk.response.body.choices[0].delta.content
                elif hasattr(chunk, 'delta') and hasattr(chunk.delta, 'content'):
                    content = chunk.delta.content
                # Try alternative GPT-5 streaming format
                elif hasattr(chunk, 'content'):
                    content = chunk.content
            else:
                # Handle standard OpenAI format
                if hasattr(chunk, 'choices') and chunk.choices:
                    if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                        content = chunk.choices[0].delta.content

            if content is not None:
                if verbose:
                    print(content, end='')
    else:
        # Handle non-streaming response
        if use_unified_client:
            # Handle different non-streaming formats for unified client
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
            elif hasattr(response, 'response') and hasattr(response.response, 'body'):
                if hasattr(response.response.body, 'choices') and response.response.body.choices:
                    content = response.response.body.choices[0].message.content
                elif hasattr(response.response.body, 'content'):
                    content = response.response.body.content
            elif hasattr(response, 'content'):
                content = response.content
        else:
            content = response.choices[0].message.content
        # print(content)
    print('\n\n')
    # print("Got LLM reply.")

    response = response_chucks  # good for saving

    return response


def extract_content_from_LLM_reply(response):
    stream = False
    if isinstance(response, list):
        stream = True

    content = ""
    if stream:
        for chunk in response:
            # Handle different response formats based on chunk type
            chunk_content = None

            # Check for GPT-5 ResponseCreatedEvent format
            if hasattr(chunk, 'type'):
                # Handle GPT-5 ResponseCreatedEvent format
                if hasattr(chunk, 'response') and hasattr(chunk.response, 'body') and hasattr(chunk.response.body, 'choices'):
                    if chunk.response.body.choices and hasattr(chunk.response.body.choices[0], 'delta'):
                        chunk_content = chunk.response.body.choices[0].delta.content
                elif hasattr(chunk, 'delta') and hasattr(chunk.delta, 'content'):
                    chunk_content = chunk.delta.content
                # Try alternative GPT-5 streaming format
                elif hasattr(chunk, 'content'):
                    chunk_content = chunk.content
            else:
                # Handle standard OpenAI format
                if hasattr(chunk, 'choices') and chunk.choices:
                    if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                        chunk_content = chunk.choices[0].delta.content

            if chunk_content is not None:
                # print(chunk_content, end='')
                content += chunk_content
                # print(content)
        # print()
    else:
        # Handle non-streaming response
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
        elif hasattr(response, 'response') and hasattr(response.response, 'body'):
            if hasattr(response.response.body, 'choices') and response.response.body.choices:
                content = response.response.body.choices[0].message.content
            elif hasattr(response.response.body, 'content'):
                content = response.response.body.content
        elif hasattr(response, 'content'):
            content = response.content
        # print(content)

    return content



#Fetching the streamed response of LLM
async def fetch_chunks(model, prompt_str):
    # print(f"\n[DEBUG] Model being used inside fetch_chunks: {model.model_name if hasattr(model, 'model_name') else model}\n")
    chunks = []
    async for chunk in model.astream(prompt_str):
        chunks.append(chunk)
        # print(chunk.content, end="", flush=True)
    return chunks


nest_asyncio.apply()


def extract_selected_tools(chunks):
    """
    Extracts and combines selected tools from a list of chunk dictionaries.

    :param chunks: List of dictionaries, each containing a "Selected tools" key.
    :return: A string of combined selected tools separated by commas.
    """
    all_tools = []

    for chunk in chunks:
        # Ensure the key exists and its value is a list
        tools = chunk.get("Selected tools", [])
        if isinstance(tools, list):
            all_tools.extend(tools)
        else:
            print(f"Warning: 'Selected tools' is not a list in chunk: {chunk}")

    # Optional: Remove duplicates while preserving order
    seen = set()
    unique_tools = []
    for tool in all_tools:
        if tool not in seen:
            seen.add(tool)
            unique_tools.append(tool)

    # Combine the tools into a single string separated by commas
    combined_tools_str = ', '.join(unique_tools)

    return combined_tools_str


def extract_code(response, verbose=False):
    '''
    Extract python code from reply
    '''
    # if isinstance(response, list):  # use OpenAI stream mode.
    #     reply_content = ""
    #     for chunk in response:
    #         chunk_content = chunk["choices"][0].get("delta", {}).get("content")
    #
    #         if chunk_content is not None:
    #             print(chunk_content, end='')
    #             reply_content += chunk_content
    #             # print(content)
    # else:  # Not stream:
    #     reply_content = response["choices"][0]['message']["content"]

    python_code = ""
    reply_content = extract_content_from_LLM_reply(response)
    python_code_match = re.search(r"```(?:python)?(.*?)```", reply_content, re.DOTALL)
    if python_code_match:
        python_code = python_code_match.group(1).strip()

    if verbose:
        print(python_code)

    return python_code


def extract_code_from_str(LLM_reply_str, verbose=False):
    '''
    Extract python code from reply string, not 'response'.
    '''

    python_code = ""
    python_code_match = re.search(r"```(?:python)?(.*?)```", LLM_reply_str, re.DOTALL)
    if python_code_match:
        python_code = python_code_match.group(1).strip()

    if verbose:
        print(python_code)

    return python_code


def execute_complete_program(request_id, code: str, try_cnt: int, task: str, model_name: str, reasoning_effort_value:str, documentation_str: str,  data_path,
                             workspace_directory, stream,
                             review=True) -> (str, str):
    count = 0
    output_capture = io.StringIO()
    original_stdout = sys.stdout  # Save the original stdout

    error_collector = []

    # Generate unique request ID to track all attempts for this user request
    # import uuid
    # request_id = str(uuid.uuid4())
    # print(f"REQUEST_ID:{request_id}")  # Use parseable format for UI to capture

    while count < try_cnt:
        print(f"\n\n-------------- Running code (trial # {count + 1}/{try_cnt}) --------------\n\n")
        original_stdout.flush()  # Ensure the message is printed immediately
        try:
            count += 1
            # Redirect stdout to capture print output
            sys.stdout = output_capture

            compiled_code = compile(code, 'Complete program', 'exec')
            exec(compiled_code, globals())  # pass only globals()

            # Restore original stdout after execution
            sys.stdout = original_stdout

            # Display the successfully executed code
            print("\nSuccessfully executed code:")
            print("```python")
            print(code)
            print("```")

            # send_error(user_api_key=load_OpenAI_key(), request_id=request_id, user_query=task, feedback="", feedback_message ="", error_msg ="", error_traceback="", generated_code=code)

            # Ensure that generated_output is returned or printed
            print(f"\n\n--------------- Done ---------------\n\n")
            return code, output_capture.getvalue(), error_collector

        except Exception as err:
            sys.stdout = original_stdout  # Restore original stdout in case of error
            # Capture full traceback for error reporting
            error_traceback = traceback.format_exc()
            error_collector.append({"attempt": count,
                                    "code_snapshot": code[:800],  # truncate long code
                                    "error_message": str(err),
                                    "error_traceback": error_traceback
                                    })


            if count == try_cnt:
                print(f"Failed to execute and debug the code within {try_cnt} times.")
                return code, output_capture.getvalue(), error_collector

            debug_prompt = get_debug_prompt(exception=err, code=code, task=task, data_path= data_path, documentation_str=documentation_str)
            print("=" * 56)
            print("AI IS DEBUGGING THE CODE...")
            print("=" * 56)
            # Use the same streaming approach that works for code generation
            # Create a LangChain model for debugging (same as code generation, but without printing config)
            # debug_model = ai_model(model_name=model_name, reasoning_effort_value=reasoning_effort_value, OpenAI_key=get_openai_key(model_name))

            # Format the debug prompt like other prompts
            formatted_debug_prompt = f"{constants.debug_role}\n\n{debug_prompt}"

            # Use the same successful streaming method as code generation
            print("DEBUGGING RESPONSE:", end="", flush=True)
            # debug_response_str = asyncio.run(stream_llm_response(debug_model, formatted_debug_prompt))

            try:
                debug_response_str = unified_llm_call(
                    request_id=request_id,
                    messages=[
                        {"role": "user", "content": formatted_debug_prompt},
                    ],
                    model_name=model_name,
                    stream=stream
                )
            except Exception as api_error:
                # If API call fails (e.g., invalid API key, network error), print error and continue to next iteration
                print(f"\n\nAPI call failed during debugging: {api_error}")
                print(f"Retrying with same code (attempt {count}/{try_cnt})...")
                continue

            # print("\n", flush=True)

            # Extract code from the string response (same as code generation)
            code = extract_code_from_str(debug_response_str)

            # Emit the debugged code to the UI
            # print("DEBUGGING COMPLETED - SENDING CODE TO UI", flush=True)
            print("=" * 56, flush=True)
            print("\nDEBUGGED CODE:")
            print("```python")
            print(code)
            print("```")

            import urllib.parse
            print("CODE_READY_URLENCODED:" + urllib.parse.quote(code), flush=True)

            # Capture full traceback for error reporting
            error_traceback = traceback.format_exc()
            # send_error(developer_api_key=load_OpenAI_key(), user_api_key=load_OpenAI_key(), user_query=task, feedback= code, error_msg=error_traceback, request_id=request_id, attempt_number=count, status="error")
            # send_error(user_api_key=load_OpenAI_key(), request_id=request_id, user_query=task, feedback="",
            #            feedback_message="", error_msg=str(err), error_traceback=error_traceback, generated_code=code)
            # TODO: Have error message -err and error traceback -error_traceback
            # send_error(developer_api_key=load_OpenAI_key(), user_api_key=load_OpenAI_key(), user_query=task, feedback="FEEDBACK", error_msg = err, error_traceback=error_traceback)

            # print("DEBUG: Code sent to UI", flush=True)
            sys.stdout.flush()  # Force flush to ensure output reaches UI
            # if review:
            #     print("=" * 56)
            #     print("AI IS REVIEWING THE DEBUG CODE...")
            #     print("=" * 56)
                # code = review_operation_code(extracted_code=code, data_path=data_path, workspace_directory=workspace_directory,documentation_str=documentation_str)
    return code, output_capture.getvalue(),error_collector,




def review_operation_code(extracted_code, data_path, workspace_directory, documentation_str):
    OpenAI_key = load_OpenAI_key()
    model = ChatOpenAI(api_key=OpenAI_key, model='gpt-4o', temperature=1)

    operation_code_review_requirement_str = '\n'.join(
        [f"{idx + 1}. {line}" for idx, line in enumerate(constants.operation_code_review_requirement)])
    operation_code_review_prompt = f"Your role: {constants.operation_code_review_role} \n" + \
                                   f"Your mission: {constants.operation_code_review_task_prefix} \n\n" + \
                                   f"The extracted code is: \n----------\n{extracted_code}\n----------\n\n" + \
                                   f"The code examples in the Documentation: \n{documentation_str} can be used as an example while reviewing the extracted_code \n\n" + \
                                   f"The requirements for the code is: \n{operation_code_review_requirement_str}\n\n" + \
                                   f"Replace the data path in the code example with:{data_path}\n\n" + \
                                   f"Set {workspace_directory} as the output directory for any operation"

    code_review_prompt_str_chunks = asyncio.run(fetch_chunks(model, operation_code_review_prompt))
    clear_output(wait=True)
    review_str_LLM_reply_str = convert_chunks_to_code_str(chunks=code_review_prompt_str_chunks)
    # EXTRACTING REVIEW_CODE

    print("\n")
    print(f"\n -------------------------- FINAL REVIEWED CODE -------------------------- \n")
    print("```python")
    reviewed_code = extract_code_from_str(LLM_reply_str=review_str_LLM_reply_str, verbose=True)
    # print(reviewed_code)
    # print("```")

    # Emit the debugged code to CodeEditor
    import urllib.parse
    print("CODE_READY_URLENCODED:" + urllib.parse.quote(reviewed_code))

    return reviewed_code


# def execute_complete_program(code: str, try_cnt: int, task: str, model_name: str, documentation_str: str) -> str:
#     count = 0
#
#     while count < try_cnt:
#         print(f"\n\n-------------- Running code (trial # {count + 1}/{try_cnt}) --------------\n\n")
#         try:
#             count += 1
#
#
#             compiled_code = compile(code, 'Complete program', 'exec')
#             exec(compiled_code, globals())  # #pass only globals() not locals()
#             # !!!!    all variables in code will become global variables! May cause huge issues!     !!!!
#
#
#             # Ensure that generated_output is returned or printed
#             print(f"\n\n--------------- Done ---------------\n\n")
#             return code
#
#         # except SyntaxError as err:
#         #     error_class = err.__class__.__name__
#         #     detail = err.args[0]
#         #     line_number = err.lineno
#         #
#         except Exception as err:
#
#             # cl, exc, tb = sys.exc_info()
#
#             # print("An error occurred: ", traceback.extract_tb(tb))
#             #
#
#             if count == try_cnt:
#                 print(f"Failed to execute and debug the code within {try_cnt} times.")
#                 return code
#
#             # print("code in execute_complete_program():", code)
#             #
#             debug_prompt = get_debug_prompt(exception=err, code=code, task=task, documentation_str=documentation_str)
#             print("Sending error information to LLM for debugging...")
#             # print("Prompt:\n", debug_prompt)
#             response = get_LLM_reply(prompt=debug_prompt,
#                                      system_role=constants.debug_role,
#                                      model=model_name,
#                                      verbose=True,
#                                      stream=True,
#                                      retry_cnt=5,
#                                      )
#             code = extract_code(response)
#
#     return code
#
#
# def capture_print_output(code: str) -> str:
#     # Create a StringIO object to capture printed output
#     output_capture = io.StringIO()
#     original_stdout = sys.stdout  # Save the original stdout
#
#     try:
#         # Redirect stdout to capture print output
#         sys.stdout = output_capture
#
#         # Compile and execute the code
#         compiled_code = compile(code, 'Complete program', 'exec')
#         exec(compiled_code, globals())  # Pass only globals()
#
#         # Restore original stdout after execution
#         sys.stdout = original_stdout
#
#         # Return captured output
#         return output_capture.getvalue()
#
#     except Exception as err:
#         # Restore original stdout in case of error
#         sys.stdout = original_stdout
#         print(f"Error occurred during code execution: {err}")
#         return ""


def get_debug_prompt(exception, code, task, data_path, documentation_str):
    etype, exc, tb = sys.exc_info()
    exttb = traceback.extract_tb(tb)  # Do not quite understand this part.
    # https://stackoverflow.com/questions/39625465/how-do-i-retain-source-lines-in-tracebacks-when-running-dynamically-compiled-cod/39626362#39626362

    print("code in get_debug_prompt:", code)
    ## Fill the missing data:
    exttb2 = [(fn, lnnr, funcname,
               (code.splitlines()[lnnr - 1] if fn == 'Complete program'
                else line))
              for fn, lnnr, funcname, line in exttb]

    # Print:
    error_info_str = 'Traceback (most recent call last):\n'
    for line in traceback.format_list(exttb2[1:]):
        error_info_str += line
    for line in traceback.format_exception_only(etype, exc):
        error_info_str += line

    print(f"Error_info_str: \n{error_info_str}")

    # print(f"traceback.format_exc():\n{traceback.format_exc()}")

    debug_requirement_str = '\n'.join(
        [f"{idx + 1}. {line}" for idx, line in enumerate(constants.debug_requirement)])

    debug_prompt = f"Your role: {constants.debug_role} \n" + \
                   f"Your task: {constants.debug_task_prefix} \n\n" + \
                   f"The properties of the data are given below:\n{data_path}\n" + \
                   f"Requirement: \n {debug_requirement_str} \n\n" + \
                   f"Your reply examples: {constants.OperationIdentification_reply_example_1} + ' or ' + '{constants.OperationIdentification_reply_example_2}'. \n\n " + \
                   f"The given code is used for this task: {task} \n\n" + \
                   f"When you are correcting the codes, Check the task again to ensure that the correct parameters (such as attributes, data paths) are being used. \n\n" + \
                   f"The technical guidelines for the code: \n {documentation_str} \n\n" + \
                   f"The error information for the code is: \n{str(error_info_str)} \n\n" + \
                   f"The code is: \n{code}"

    return debug_prompt




def has_disconnected_components(directed_graph, verbose=True):
    # Get the weakly connected components
    weakly_connected = list(nx.weakly_connected_components(directed_graph))

    # Check if there is more than one weakly connected component
    if len(weakly_connected) > 1:
        if verbose:
            print("component count:", len(weakly_connected))
        return True
    else:
        return False


def read_html_graph_content(html_graph_path: str) -> str:
    """
    Read the HTML graph file content and return as string.

    Args:
        html_graph_path: Path to the HTML graph file

    Returns:
        str: Complete HTML content as string

    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """

    if not os.path.exists(html_graph_path):
        raise FileNotFoundError(f"HTML graph file not found: {html_graph_path}")

    try:
        with open(html_graph_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # print(f"✓ HTML graph loaded: {len(html_content)} bytes")
        return html_content

    except Exception as e:
        raise IOError(f"Error reading HTML graph file: {e}")


def generate_function_def(node_name, G):
    '''
    Return a dict, includes two lines: the function definition and return line.
    parameters: operation_node
    '''
    node_dict = G.nodes[node_name]
    node_type = node_dict['node_type']

    predecessors = G.predecessors(node_name)

    # print("predecessors:", list(predecessors))

    # create parameter list with default values
    para_default_str = ''  # for the parameters with the file path
    para_str = ''  # for the parameters without the file path
    for para_name in predecessors:
        # print("para_name:", para_name)
        para_node = G.nodes[para_name]
        # print(f"para_node: {para_node}")
        # print(para_node)
        data_path = para_node.get('data_path', '')  # if there is a path, the function need to read this file

        if data_path != "":
            para_default_str = para_default_str + f"{para_name}='{data_path}', "
        else:
            para_str = para_str + f"{para_name}={para_name}, "

    all_para_str = para_str + para_default_str

    function_def = f'{node_name}({all_para_str})'
    function_def = function_def.replace(', )', ')')  # remove the last ","

    # generate the return line
    successors = G.successors(node_name)
    return_str = 'return ' + ', '.join(list(successors))

    # print("function_def:", function_def)  # , f"node_type:{node_type}"
    # print("return_str:", return_str)  # , f"node_type:{node_type}"
    # print(function_def, predecessors, successors)
    return_dict = {"function_definition": function_def,
                   "return_line": return_str,
                   'description': node_dict['description'],
                   'node_name': node_name
                   }
    return return_dict


def bfs_traversal(graph, start_nodes):
    visited = set()
    queue = deque(start_nodes)

    order = []
    while queue:
        node = queue.popleft()
        # print(node)
        if node not in visited:
            order.append(node)
            visited.add(node)
            queue.extend(neighbor for neighbor in graph[node] if neighbor not in visited)
    return order


def generate_function_def_list(G):
    '''
    Return a list, each string is the function definition and return line
    '''
    # start with the data loading, following the data flow.
    nodes = []
    # Find nodes without predecessors
    nodes_without_predecessors = [node for node in G.nodes() if G.in_degree(node) == 0]
    # print(nodes_without_predecessors)
    # Traverse the graph using BFS starting from the nodes without predecessors
    traversal_order = bfs_traversal(G, nodes_without_predecessors)

    # print("traversal_order:", traversal_order)

    def_list = []
    data_node_list = []
    for node_name in traversal_order:
        node_type = G.nodes[node_name]['node_type']
        if node_type == 'operation':
            # print(node_name, node_type)
            # predecessors = G.predecessors('Load_shapefile')
            # successors = G.successors('Load_shapefile') 

            function_def_returns = generate_function_def(node_name, G)
            def_list.append(function_def_returns)

        if node_type == 'data':
            data_node_list.append(node_name)

    return def_list, data_node_list


def get_given_data_nodes(G):
    given_data_nodes = []
    for node_name in G.nodes():
        node = G.nodes[node_name]
        in_degrees = G.in_degree(node_name)
        if in_degrees == 0:
            given_data_nodes.append(node_name)
            # print(node_name,in_degrees,  node)
    return given_data_nodes


def get_data_loading_nodes(G):
    data_loading_nodes = set()

    given_data_nodes = get_given_data_nodes(G)
    for node_name in given_data_nodes:

        successors = G.successors(node_name)
        for node in successors:
            data_loading_nodes.add(node)
            # print(node_name,in_degrees,  node)
    data_loading_nodes = list(data_loading_nodes)
    return data_loading_nodes


def get_data_sample_text(file_path, file_type="csv", encoding="utf-8"):
    """
    file_type: ["csv", "shp", "txt"]
    return: a text string
    """
    if file_type == "csv":
        df = pd.read_csv(file_path)
        text = str(df.head(3))

    if file_type == "shp":
        gdf = gpd.read_file(file_path)
        text = str(gdf.head(2))  # .drop('geomtry')

    if file_type == "txt":
        with open(file_path, 'r', encoding=encoding) as f:
            lines = f.readlines()
            text = ''.join(lines[:3])
    return text


def show_graph(G):
    if has_disconnected_components(directed_graph=G):
        print("Disconnected component, please re-generate the graph!")

    nt = Network(notebook=True,
                 cdn_resources="remote",
                 directed=True,
                 # bgcolor="#222222",
                 # font_color="white",
                 height="800px",
                 # width="100%",
                 #  select_menu=True,
                 # filter_menu=True,

                 )

    nt.from_nx(G)

    sinks = find_sink_node(G)
    sources = find_source_node(G)
    # print("sinks:", sinks)

    # Set node colors based on node type
    node_colors = []
    for node in nt.nodes:
        # print('node:', node)
        if node['node_type'] == 'data':
            # print('node:', node)
            if node['label'] in sinks:
                node_colors.append('violet')  # lightgreen
                # print(node)
            elif node['label'] in sources:
                node_colors.append('lightgreen')  #
                # print(node)
            else:
                node_colors.append('orange')

        elif node['node_type'] == 'operation':
            node_colors.append('deepskyblue')

            # Update node colorsb
    for i, color in enumerate(node_colors):
        nt.nodes[i]['color'] = color
        # nt.nodes[i]['shape'] = 'box'
        nt.nodes[i]['shape'] = 'dot'
        # nt.set_node_style(node, shape="box")
        nt.nodes[i]['font'] = {'size': 20}  # set font size

    return nt

    # # Set consistent node colors and shapes based on node type
    # for i, node in enumerate(nt.nodes):
    #     # Check if 'node_type' key exists
    #     if 'node_type' not in node:
    #         print(f"Warning: node {node['label']} does not have 'node_type'. Skipping this node.")
    #         continue
    #
    #     if node['node_type'] == 'data':
    #         # All data nodes get the same color for consistency
    #         nt.nodes[i]['color'] = '#4CAF50'  # Green for data nodes
    #         nt.nodes[i]['shape'] = 'circle'   # Circle shape for data
    #         nt.nodes[i]['size'] = 25
    #
    #     elif node['node_type'] == 'operation':
    #         # All operation nodes get the same color for consistency
    #         nt.nodes[i]['color'] = '#2196F3'  # Blue for operation nodes
    #         nt.nodes[i]['shape'] = 'box'      # Box shape for operations
    #         nt.nodes[i]['size'] = 25

    return nt


def find_sink_node(G):
    """
    Find the sink node in a NetworkX directed graph.

    :param G: A NetworkX directed graph
    :return: The sink node, or None if not found
    """
    sinks = []
    for node in G.nodes():
        if G.out_degree(node) == 0 and G.in_degree(node) > 0:
            sinks.append(node)
    return sinks


# Function to find the source node
def find_source_node(graph):
    # Initialize an empty list to store potential source nodes
    source_nodes = []

    # Iterate over all nodes in the graph
    for node in graph.nodes():
        # Check if the node has no incoming edges
        if graph.in_degree(node) == 0:
            # Add the node to the list of source nodes
            source_nodes.append(node)

    # Return the source nodes
    return source_nodes


# def Query_tuning_gpt(user_query):
#     OpenAI_key = load_OpenAI_key()
#     llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OpenAI_key)
#     cot_prompt = PromptTemplate(
#         input_variables=["query"],
#         template= constants.cot_description_prompt
#     )
#     task_chain = LLMChain(llm=llm, prompt=cot_prompt)
#     fine_tuned_request = task_chain.run(user_query)
#     # print(f"Preprocessed Task: {fine_tuned_request.strip()}")
#     return fine_tuned_request

def unified_llm_call(request_id, messages, model_name, stream=False, temperature=1, response_format=None, **kwargs):
    """
    Unified LLM API call function that handles:
    - OpenAI proxy API
    - GPT-5 (via ModelProvider)
    - Normal OpenAI
    - Streaming and non-streaming
    - Structured output (response_format)

    Args:
        messages: List of message dicts [{"role": "user", "content": "..."}]
        model_name: Model name (e.g., "gpt-4o", "gpt-5")
        stream: Whether to stream the response
        temperature: Temperature for generation
        response_format: Optional pydantic model for structured output
        **kwargs: Additional parameters (e.g., max_tokens, top_p, etc.)

    Returns:
        str: The complete response content

    Example:
        # Simple call
        response = unified_llm_call(
            messages=[{"role": "user", "content": "Hello"}],
            model_name="gpt-4o"
        )

        # With structured output
        from pydantic import BaseModel
        class MySchema(BaseModel):
            name: str
            age: int

        response = unified_llm_call(
            messages=[{"role": "user", "content": "Extract info"}],
            model_name="gpt-4o",
            response_format=MySchema
        )
    """
    import requests
    from SpatialAnalysisAgent_ModelProvider import ModelProviderFactory

    # Check if model requires local provider (Ollama) - this takes precedence
    provider_name = ModelProviderFactory._model_providers.get(model_name, 'openai')

    # Check if using proxy
    api_key = load_OpenAI_key()
    service_name = "GIS Copilot"

    if provider_name == 'ollama':
        # ===== OLLAMA/LOCAL MODEL CASE - Always use ModelProvider regardless of API key =====
        from SpatialAnalysisAgent_ModelProvider import create_unified_client
        client, provider = create_unified_client(model_name)

        # Use regular completion (Ollama doesn't support structured output via beta API)
        response = provider.generate_completion(
            request_id,
            client,
            model_name,
            messages,
            stream=stream,
            temperature=temperature,
            **kwargs
        )
        return streaming_openai_response(response)


    elif 'gibd-services' in (api_key or ''):
        # ===== PROXY CASE =====
        url = f"https://www.gibd.online/api/openai/{api_key}"
        payload = {
            "service_name": service_name,
            "question_id": request_id,
            "model": model_name,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            **kwargs
        }
        response_req = requests.post(url, json=payload, stream=stream)
        if stream:
            # Handle streaming
            def stream_generator():
                for line in response_req.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data_str)
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    content = delta.get('content')
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                pass
            # Collect streamed response
            out = ""
            for content in stream_generator():
                print(content, end="")
                out += content
            return out
        else:
            # Non-streaming
            data = response_req.json()
            content = data['choices'][0]['message']['content']
            # Clean markdown code blocks if present
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            return content

    else:
        # ===== NON-PROXY CASE (GPT-5 and Normal OpenAI) =====
        try:
            from SpatialAnalysisAgent_ModelProvider import create_unified_client
            client, provider = create_unified_client(model_name)

            # Check if structured output is requested and supported
            if response_format and client and hasattr(client, 'beta'):
                # Use structured output API
                response = client.beta.chat.completions.parse(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    response_format=response_format,
                    **kwargs
                )
                return response.choices[0].message.content
            else:
                # Use regular completion
                response = provider.generate_completion(
                    request_id,
                    client,
                    model_name,
                    messages,
                    stream=stream,
                    temperature=temperature,
                    **kwargs
                )
                return streaming_openai_response(response)

        except ImportError:
            # Direct OpenAI fallback
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            if response_format and hasattr(client, 'beta'):
                # Use structured output
                response = client.beta.chat.completions.parse(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    response_format=response_format,
                    **kwargs
                )
                return response.choices[0].message.content
            else:
                # Regular completion
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=stream,
                    temperature=temperature,
                    **kwargs
                )
                return streaming_openai_response(response)


def streaming_openai_response(response):
    # Handle GPT-5 specialized response format first
    if hasattr(response, 'output'):
        # GPT-5 responses.create() format: response.output contains the content
        output = response.output
        if isinstance(output, str):
            return output.strip()
        elif hasattr(output, 'content'):
            return str(output.content).strip()
        else:
            return str(output).strip()

    # Handle GPT-5 alternative format: response.response.body
    if hasattr(response, 'response') and hasattr(response.response, 'body'):
        body = response.response.body
        # Check for choices format in body
        if hasattr(body, 'choices') and body.choices:
            if hasattr(body.choices[0], 'message'):
                content = body.choices[0].message.content
                return content.strip() if isinstance(content, str) else str(content).strip()
        # Check for content attribute in body
        if hasattr(body, 'content'):
            return str(body.content).strip()

    # Streaming case: iterator without .choices
    if hasattr(response, '__iter__') and not hasattr(response, 'choices'):
        out = ""
        for chunk in response:
            # Proxy strings/bytes
            if isinstance(chunk, (str, bytes)):
                t = chunk.decode("utf-8", "ignore") if isinstance(chunk, bytes) else chunk
                if t:
                    print(t, end="")
                    out += t
                continue

            # Try multiple ways to extract content from chunks
            content = None

            # GPT-5 ResponseCreatedEvent format - event-based streaming
            if hasattr(chunk, 'type'):
                try:
                    event_type = getattr(chunk, 'type', '')

                    # GPT-5 output_item events contain the content
                    if 'output_item' in event_type.lower():
                        if hasattr(chunk, 'item'):
                            item = chunk.item

                            # Try to extract text from the item
                            # Method 1: item.content (for text items)
                            if hasattr(item, 'content'):
                                item_content = item.content
                                if isinstance(item_content, str):
                                    content = item_content
                                # Check if content is a list with text parts
                                elif isinstance(item_content, list):
                                    for part in item_content:
                                        if hasattr(part, 'text'):
                                            content = part.text
                                            break
                                        elif isinstance(part, dict) and 'text' in part:
                                            content = part['text']
                                            break

                            # Method 2: item.text (direct text attribute)
                            if not content and hasattr(item, 'text'):
                                content = item.text

                            # Method 3: Try to dump and look for text
                            if not content and hasattr(item, 'model_dump'):
                                try:
                                    item_data = item.model_dump()
                                    if 'content' in item_data:
                                        content = item_data['content']
                                    elif 'text' in item_data:
                                        content = item_data['text']
                                except:
                                    pass

                    # Content delta events contain the actual text
                    elif 'content' in event_type.lower() and 'delta' in event_type.lower():
                        if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'content'):
                            content = chunk.delta.content
                        elif hasattr(chunk, 'content'):
                            content = chunk.content

                    # For done events, check if there's output in response
                    elif event_type == 'response.done':
                        if hasattr(chunk, 'response'):
                            response_obj = chunk.response
                            # Check for output array
                            if hasattr(response_obj, 'output') and response_obj.output:
                                # Output is usually a list of items
                                for output_item in response_obj.output:
                                    if hasattr(output_item, 'content'):
                                        item_content = output_item.content
                                        if isinstance(item_content, str):
                                            content = item_content
                                            break
                                        elif isinstance(item_content, list):
                                            for part in item_content:
                                                if hasattr(part, 'text'):
                                                    content = part.text
                                                    break
                except Exception as e:
                    pass

            # Standard OpenAI ChatCompletionChunk format
            if not content:
                try:
                    if hasattr(chunk, 'choices') and chunk.choices:
                        content = getattr(chunk.choices[0].delta, "content", None)
                except:
                    pass

            # GPT-5 streaming format - check for delta.content directly
            if not content:
                try:
                    if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'content'):
                        content = chunk.delta.content
                except:
                    pass

            # GPT-5 streaming format - check for content attribute directly
            if not content:
                try:
                    if hasattr(chunk, 'content'):
                        content = chunk.content
                except:
                    pass

            # GPT-5 format - check for output in chunk
            if not content:
                try:
                    if hasattr(chunk, 'output'):
                        content = chunk.output
                except:
                    pass

            if content:
                print(content, end="")
                out += str(content)

        print()
        return out

    # Non-streaming case - Standard OpenAI format
    if hasattr(response, "choices"):
        c = getattr(response.choices[0].message, "content", "")
        return c.strip() if isinstance(c, str) else (c or "")

    if isinstance(response, dict):
        c = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return c.strip() if isinstance(c, str) else (c or "")

    return str(response)

# def streaming_openai_response(response):
#
#     # Check if it's a streaming response
#     if hasattr(response, '__iter__') and not hasattr(response, 'choices'):
#         # It's a streaming generator
#         complete_content = ""
#         for chunk in response:
#             if chunk and hasattr(chunk, 'choices') and chunk.choices:
#                 delta = chunk.choices[0].delta
#                 if hasattr(delta, 'content') and delta.content:
#                     content = delta.content
#                     print(content, end="")
#                     complete_content += content
#         print()  # New line after streaming is complete
#         return complete_content
#
#     else:
#         # It's a regular response object
#         if hasattr(response, "choices") and response.choices:
#             return response.choices[0].message.content.strip()
#         elif isinstance(response, dict):
#             return response.get("choices", [{}])[0].get("message", {}).get("content", "")
#         else:
#             return str(response)

def Query_tuning(request_id, user_query, model_name, stream):
    """Return a fine-tuned prompt using the selected model.
    Supports: OpenAI proxy, GPT-5, and normal OpenAI"""
    return unified_llm_call(
        request_id=request_id,
        messages = [
        {"role": "system", "content": constants.cot_description_prompt},
        {"role": "user", "content": user_query},
    ],
    model_name=model_name,
    stream=stream
    )
    # return response

    # # Check if using proxy
    # api_key = load_OpenAI_key()
    #
    # if 'gibd-services' in (api_key or ''):
    #     # PROXY CASE: Use direct requests.post
    #     import requests
    #     url = f"http://128.118.54.16:3030/api/openai/{api_key}"
    #
    #     payload = {
    #         "model": model_name,
    #         "messages": messages,
    #         "stream": stream
    #     }
    #     response = requests.post(url, json=payload, stream=stream)
    #
    #     if stream:
    #         # Handle streaming - collect all chunks without printing
    #         def stream_generator():
    #             for line in response.iter_lines():
    #                 if line:
    #                     line = line.decode('utf-8')
    #                     if line.startswith('data: '):
    #                         data_str = line[6:]
    #                         if data_str == '[DONE]':
    #                             break
    #                         try:
    #                             chunk = json.loads(data_str)
    #                             if 'choices' in chunk and len(chunk['choices']) > 0:
    #                                 delta = chunk['choices'][0].get('delta', {})
    #                                 content = delta.get('content')
    #                                 # print(content, end="")
    #                                 if content:
    #                                     yield content
    #                         except json.JSONDecodeError:
    #                             pass
    #
    #         # Collect all streamed chunks into one string
    #         out = ""
    #         for content in stream_generator():
    #             print(content, end="")
    #             # print()
    #             out += content
    #         # print(content, end="")
    #
    #
    #         return out
    #     else:
    #         # Non-streaming
    #         data = response.json()
    #         return data['choices'][0]['message']['content']
    #
    # else:
    #     # NON-PROXY: Use ModelProvider for GPT-5 and normal OpenAI
    #     try:
    #         from SpatialAnalysisAgent_ModelProvider import create_unified_client
    #         client, provider = create_unified_client(model_name)
    #         response = provider.generate_completion(client, model_name, messages, stream=stream)
    #         return streaming_openai_response(response)
    #     except ImportError:
    #         # Direct OpenAI fallback
    #         from openai import OpenAI
    #         client = OpenAI(api_key=api_key)
    #         response = client.chat.completions.create(model=model_name, messages=messages, stream=stream)
    #         return streaming_openai_response(response)

# def Query_tuning(user_query, model_name, stream):
#     """
#     Fine-tune user query using the specified model.
#     Supports both OpenAI and local models via unified provider.
#     """
#     try:
#         # Check if this is a local model
#         import SpatialAnalysisAgent_ModelProvider as ModelProvider
#         provider_name = ModelProvider.ModelProviderFactory._model_providers.get(model_name, 'openai')
#
#         if provider_name == 'ollama':
#             # Use local model with LangChain ChatOpenAI pointing to local server
#             llm = ChatOpenAI(
#                 base_url="http://128.118.54.16:11434/v1",
#                 api_key="no-api",
#                 model_name=model_name,
#                 openai_api_key="no-api"
#             )
#         else:
#             # Use OpenAI model
#             # OpenAI_key = load_OpenAI_key()
#             from SpatialAnalysisAgent_ModelProvider import create_unified_client
#             client, provider = create_unified_client(model_name)
#             # Prepare messages (OpenAI-style)
#             messages = [
#                 {"role": "system", "content": constants.cot_description_prompt},
#                 {"role": "user", "content": user_query}
#             ]
#             response = provider.generate_completion(
#                 client,
#                 model_name,
#                 messages,
#                 stream=stream
#                 )
#
#             # Process the response using the reusable function
#             fine_tuned_request = streaming_openai_response(response)
#
#             # # Check if it's a generator (streaming response)
#             # if hasattr(response, '__iter__') and not hasattr(response, 'choices'):
#             #     # It's a streaming generator, collect all tokens AND print them
#             #     fine_tuned_request = ""
#             #     for chunk in response:
#             #         if chunk:
#             #             print(chunk, end="")  # Print each chunk as it arrives
#             #             fine_tuned_request += chunk
#             #     print()  # New line after streaming is complete
#             # else:
#             #     # It's a regular response object
#             #     if hasattr(response, "choices"):
#             #         fine_tuned_request = response.choices[0].message.content.strip()
#             #     elif isinstance(response, dict):
#             #         fine_tuned_request = response.get("choices", [{}])[0].get("message", {}).get("content", "")
#             #     else:
#             #         fine_tuned_request = str(response)
#
#
#     except ImportError:
#
#         # Fallback to direct OpenAI SDK usage
#
#         OpenAI_key = load_OpenAI_key()
#         client = create_openai_client()
#         response = client.chat.completions.create(
#             model=model_name,
#             messages=[
#                 {"role": "system", "content": constants.cot_description_prompt},
#                 {"role": "user", "content": user_query}
#             ]
#         )
#
#
#         fine_tuned_request = response.choices[0].message.content
#     return fine_tuned_request




async def stream_llm_response(model, prompt_str):
    """
    Universal streaming function for all LLM calls.
    Streams output in real-time and returns the complete response.
    """
    complete_response = ""
    async for chunk in model.astream(prompt_str):
        chunk_content = chunk.content
        if chunk_content:
            print(chunk_content, end="")
            complete_response += chunk_content
    return complete_response

# async def Query_tuning_streaming(user_query, model_name="gpt-4o", stream=False):
#     """
#     Fine-tune user query using the specified model with streaming support.
#     Streams output in real-time and returns the complete response.
#     """
#     try:
#         # Check if this is a local model
#         import SpatialAnalysisAgent_ModelProvider as ModelProvider
#         provider_name = ModelProvider.ModelProviderFactory._model_providers.get(model_name, 'openai')
#
#         if provider_name == 'ollama':
#             # Use local model with LangChain ChatOpenAI pointing to local server
#             llm = OpenAI(
#                 base_url="http://128.118.54.16:11434/v1",
#                 api_key="no-api",
#                 model_name=model_name,
#                 openai_api_key="no-api"
#             )
#         else:
#             # Use OpenAI model
#             OpenAI_key = load_OpenAI_key()
#             if 'gibd-services' in (OpenAI_key or ''):
#                 url = f"http://128.118.54.16:3030/api/openai/{OpenAI_key}"
#                 payload = {"model": model_name, "messages": user_query, "stream": True}
#                 response=requests.post(url, json=payload, stream=True)
#                 # Stream or non-stream handling
#                 if stream:
#                     with response as r:
#                         for line in r.iter_lines():
#                             if line:
#                                 decoded_line = line.decode('utf-8')
#                                 print(decoded_line, flush=True)
#                     return  # Streamed output is already printed
#                 else:
#                     response = response
#                     return response.json()
#
#                 return response.text
#             else:
#                 llm = OpenAI(model_name=model_name, openai_api_key=OpenAI_key)
#
#
#     except ImportError:
#         # Fallback to OpenAI
#         OpenAI_key = load_OpenAI_key()
#         llm = OpenAI(model_name=model_name, openai_api_key=OpenAI_key)
#
#     # Create the formatted prompt
#     formatted_prompt = constants.cot_description_prompt.format(query=user_query)
#
#     # Use the universal streaming function
#     return await stream_llm_response(llm, formatted_prompt)


# def Query_tuning_streaming(user_query, model_name="gpt-4o"):
#     """
#     Fine-tune user query using the specified model with streaming support.
#     Streams output in real-time and returns the complete response.
#     """
#     try:
#         # Check if this is a local model
#         import SpatialAnalysisAgent_ModelProvider as ModelProvider
#         provider_name = ModelProvider.ModelProviderFactory._model_providers.get(model_name, 'openai')
#
#         if provider_name == 'ollama':
#             # Use local model with LangChain ChatOpenAI pointing to local server
#             llm = OpenAI(
#                 base_url="http://128.118.54.16:11434/v1",
#                 api_key="no-api",
#                 model_name=model_name,
#                 openai_api_key="no-api"
#             )
#         else:
#             try:
#                 from SpatialAnalysisAgent_ModelProvider import create_unified_client
#                 client, provider = create_unified_client(model_name)
#                 # Use OpenAI model
#                 OpenAI_key = load_OpenAI_key()
#                 # Generate response using the provider
#                 response = provider.generate_completion(
#                     client,
#                     model_name,
#                     user_query,
#                     stream=False
#                 )
#             except ImportError:
#                 # Fallback to basic OpenAI client
#                 client = create_openai_client()
#                 response = client.chat.completions.create(
#                     model=model_name,
#                     messages=[user_query]
#                 )
#             # Fallback to OpenAI
#             OpenAI_key = load_OpenAI_key()
#             llm = OpenAI(model_name=model_name, openai_api_key=OpenAI_key)
#
#     # Create the formatted prompt
#     formatted_prompt = constants.cot_description_prompt.format(query=user_query)
#
#     # Use the universal streaming function
#     return stream_llm_response(llm, formatted_prompt)


def generate_graph_response(request_id,task, task_explanation, data_path, graph_file_path, model_name='gpt-4o', stream=True, streaming_callback=None, execute=True):
    """
    Generate LLM response for solution graph creation.
    This replaces solution.get_LLM_response_for_graph() with unified LLM call support.

    Args:
        task: The user's task description
        task_explanation: Detailed explanation/breakdown of the task
        data_path: List of data locations
        graph_file_path: Path where the graph file will be saved/loaded
        model_name: Model to use (supports proxy, GPT-5, Ollama, etc.)
        stream: Whether to stream the response
        streaming_callback: Optional callback function for streaming chunks
        execute: Whether to execute the generated code and load the graph

    Returns:
        tuple: (graph_response, code_for_graph, solution_graph)
               - graph_response: The LLM response text
               - code_for_graph: Extracted Python code from the response
               - solution_graph: The loaded NetworkX graph (None if execute=False or loading failed)
    """
    # Format data paths as numbered list
    data_path_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(data_path)])

    # Build graph requirements (from constants)
    graph_requirement = constants.graph_requirement.copy()
    graph_requirement_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(graph_requirement)])

    # Build the graph prompt
    graph_prompt = f'Your role: {constants.graph_role} \n\n' + \
                   f'Your task: {constants.graph_task_prefix} \n {task_explanation} \n\n' + \
                   f'Your reply needs to meet these requirements: \n {graph_requirement_str} \n\n' + \
                   f'Your reply example: {constants.graph_reply_exmaple} \n\n' + \
                   f'Data locations (each data is a node): {data_path_str} \n'

    # Use unified LLM call with streaming support
    messages = [
        {"role": "system", "content": constants.graph_role},
        {"role": "user", "content": graph_prompt}
    ]

    response = unified_llm_call(
        request_id=request_id,
        messages=messages,
        model_name=model_name,
        stream=stream,
        temperature=1
    )
    graph_response = response

    # Extract code from LLM response (response is a string from unified_llm_call)
    try:
        code_for_graph = extract_code_from_str(LLM_reply_str=graph_response, verbose=False)
    except Exception as e:
        code_for_graph = ""
        print(f"Extract graph Python code from LLM failed: {e}")

    # Execute code and load graph if requested
    solution_graph = None
    if execute and code_for_graph:
        try:
            # Execute the code with a namespace to capture variables
            namespace = {'nx': nx}
            exec(code_for_graph, namespace)

            # The generated code typically creates a graph variable named 'G'
            if 'G' in namespace:
                G = namespace['G']
                # Save the graph to the GraphML file
                nx.write_graphml(G, graph_file_path)
                print(f"Graph saved to: {graph_file_path}")
            else:
                print("Warning: Graph variable 'G' not found in executed code")

            # Now load the graph file with metadata
            solution_graph = load_graph_file(graph_file_path)
        except Exception as e:
            print(f"Error executing graph code or loading graph: {e}")

    return graph_response, code_for_graph, solution_graph

def load_graph_file(graph_file_path):
    """
    Load a NetworkX graph from a GraphML file.
    This is a standalone function version of the kernel's load_graph_file method.

    Args:
        graph_file_path: Path to the .graphml file to load

    Returns:
        dict: A dictionary containing:
            - 'graph': The NetworkX graph object (None if file doesn't exist)
            - 'source_nodes': List of source nodes (nodes with no predecessors)
            - 'sink_nodes': List of sink nodes (nodes with no successors)
    """
    if not os.path.exists(graph_file_path):
        print(f"Graph file not found: {graph_file_path}")
        return {
            'graph': None,
            'source_nodes': [],
            'sink_nodes': []
        }

    try:
        # Load the graph from GraphML file
        G = nx.read_graphml(graph_file_path)

        # Find source and sink nodes
        source_nodes = find_source_node(G)
        sink_nodes = find_sink_node(G)

        return {
            'graph': G,
            'source_nodes': source_nodes,
            'sink_nodes': sink_nodes
        }
    except Exception as e:
        print(f"Error loading graph file {graph_file_path}: {e}")
        return {
            'graph': None,
            'source_nodes': [],
            'sink_nodes': []
        }


def RAG_tool_Select(model_name, Operation_Description):
    OpenAI_key = load_OpenAI_key()
    with open(json_path, "r", encoding="utf-8") as f:
        tool_data = json.load(f)
    # print(f"Loaded {len(tool_data)} tools for embedding.")
    docs = []
    for entry in tool_data:
        content = f"Toolname: {entry['toolname']}\nDescription: {entry['tool_description']}\nTool ID: {entry['tool_id']}"
        metadata = {"tool_id": entry["tool_id"], "toolname": entry["toolname"]}
        docs.append(Document(page_content=content, metadata=metadata))
    # print(f"Created {len(docs)} LangChain documents.")
    # Optional: Split if descriptions are large
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    #Initialize Embeddings (Choose One)
    embeddings = OpenAIEmbeddings(openai_api_key=OpenAI_key)  # Requires OPENAI_API_KEY in env
    vector_store = FAISS.from_documents(split_docs, embeddings)

    # print("Embedding complete. FAISS vector store ready.")

    # Chain for LLM response
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model_name=model_name, openai_api_key=OpenAI_key, temperature=0)


    tool_selection_prompt = constants.tool_selection_prompt


    selection_prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=tool_selection_prompt
    )

    llm_chain = LLMChain(llm=llm, prompt=selection_prompt)

    # Combine retrieved documents with the task
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )
    # question_chain = LLMChain(llm=llm, prompt=chat_prompt)
    qa_chain = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=stuff_chain
    )
    response = qa_chain.run(Operation_Description)
    return response


def tool_select(request_id,ToolSelect_prompt_str, model_name, stream):
    """Return a fine-tuned prompt using the selected model.
    Supports: OpenAI proxy, GPT-5, and normal OpenAI"""
    return unified_llm_call(
        request_id=request_id,
        messages = [
        {"role": "user", "content": ToolSelect_prompt_str},
    ],
    model_name=model_name,
    stream=stream
    )

def get_combined_documentation_from_rag(tool_ids, model="gpt-4o", json_path = json_path ):
    OpenAI_key = load_OpenAI_key()

    # Load from JSON
    with open(json_path, "r", encoding="utf-8") as f:
        tool_data = json.load(f)
    docs = []

    for tool in tool_data:
        tool_id = tool.get("tool_id", "")
        name = tool.get("toolname", "")
        desc = tool.get("tool_description", "")
        params = tool.get("parameters", "")
        example = tool.get("code_example", "")

        page = f"""
    Toolname: {name}
    Tool ID: {tool_id}

    Description:
    {desc}

    Parameters:
    {params}

    Code Example:
    {example}
    """
        docs.append(Document(page_content=page, metadata={"tool_id": tool_id}))


    # Create vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OpenAI_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("qgis_tool_documentation_faiss")

    vectorstore = FAISS.load_local(
        "qgis_tool_documentation_faiss",
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OpenAI_key)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    doc_blocks = []
    for tool_id in tool_ids:
        query = f"Show full documentation for tool: {tool_id}"
        response = qa_chain.invoke(query)
        doc_blocks.append(response["result"])

    return "\n\n".join(doc_blocks)





def get_combined_documentation_with_fallback(tool_ids, all_documentation):
    try:
        combined_doc_rag = get_combined_documentation_from_rag(tool_ids)

        # Check if RAG failed or returned generic apology
        if not combined_doc_rag.strip() or "I don't have" in combined_doc_rag or "I'm sorry" in combined_doc_rag:
            print("RAG did not return useful documentation. Switching to fallback (TOML).")
            return '\n'.join(all_documentation)

        return combined_doc_rag

    except Exception as e:
        print(f"RAG retrieval failed due to error: {e}")
        print("Switching to fallback (TOML).")
        return '\n'.join(all_documentation)


def get_openai_key(model_name: str):
    """
    Resolve the OpenAI key for the requested model.
    Uses ModelProvider to determine if the model is local (ollama) or remote.
    For remote models it loads the key from `helper.load_OpenAI_key()`
    and throws a ValueError if none is found.
    """
    try:
        # Import here to avoid import loops if helper.py is imported by other modules
        import SpatialAnalysisAgent_ModelProvider as ModelProvider
        provider_name = ModelProvider.ModelProviderFactory._model_providers.get(model_name, 'openai')
        if provider_name == 'ollama':
            return None  # Local models don't need an OpenAI key
        else:
            OpenAI_key = load_OpenAI_key()
            if not OpenAI_key:
                raise ValueError("Please enter a valid OpenAI API key for this model.")
            return OpenAI_key
    except Exception as e:
        # Fallback: try to load key, but catch errors gracefully
        try:
            return load_OpenAI_key()
        except Exception:
            print(f"Warning: Could not load OpenAI key - {e}")
            return None

def initialize_ai_model(model_name, reasoning_effort_value, OpenAI_key):
    print("=" * 56)
    print("MODEL CONFIGURATION INFO")
    print("=" * 56)
    print(f"Selected Model: {model_name}")

    # Import ModelProvider to determine the correct provider
    try:
        import SpatialAnalysisAgent_ModelProvider as ModelProvider
        from langchain_openai import ChatOpenAI
        provider = ModelProvider.ModelProviderFactory.get_provider(model_name)
        provider_name = ModelProvider.ModelProviderFactory._model_providers.get(model_name, 'openai')

        if model_name == 'gpt-5':
            print("Model Type: GPT-5 (Specialized Provider)")
            reasoning_effort_value = globals().get('reasoning_effort', f'{reasoning_effort_value}')
            print(f"Reasoning Effort: {reasoning_effort_value}")
            print(f"Provider Class: {type(provider).__name__}")
            print(f"Provider Type: Specialized GPT-5 Provider")
            print(f"API Method: client.responses.create() with reasoning parameter")
            print(f"Reasoning Parameter: {{'effort': '{reasoning_effort_value}'}}")

            # Create model and store reasoning effort

            model = ChatOpenAI(api_key=OpenAI_key, model=model_name, temperature=1)
            reasoning_effort = reasoning_effort_value

        elif provider_name == 'ollama':
            print(f"Model Type: Local Model via Ollama Provider")
            print(f"Provider Class: {type(provider).__name__}")
            print(f"Provider Type: Ollama Local Provider")
            print(f"API Method: OpenAI-compatible endpoint")
            print(f"Base URL: http://128.118.54.16:11434/v1")

            # Create LangChain ChatOpenAI that points to local server
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(
                base_url="http://128.118.54.16:11434/v1",
                api_key="no-api",
                model=model_name,
                temperature=1
            )

        else:
            print("Model Type: Standard OpenAI Model")
            print(f"Provider Class: {type(provider).__name__}")
            print("API Method: client.chat.completions.create()")
            model = ChatOpenAI(api_key=OpenAI_key, model=model_name, temperature=1)

    except ImportError as e:
        print(f"WARNING: Could not import ModelProvider: {e}")
        print("Falling back to standard ChatOpenAI")
        model = ChatOpenAI(api_key=OpenAI_key, model=model_name, temperature=1)

    # Display API Key status
    if 'gibd-services' in (OpenAI_key or ''):
        # print("API Key: ✓ Loaded (Provided by GIBD-services - http://128.118.54.16:3030/)")
        print("API Key (Provided by GIBD-services): ✓ Loaded")
    elif OpenAI_key:
        print("API Key: ✓ Loaded")
    else:
        print("API Key: Not required")
    return model

def ai_model(model_name, reasoning_effort_value, OpenAI_key):

    # Import ModelProvider to determine the correct provider
    try:
        import SpatialAnalysisAgent_ModelProvider as ModelProvider
        from langchain_openai import ChatOpenAI
        provider = ModelProvider.ModelProviderFactory.get_provider(model_name)
        provider_name = ModelProvider.ModelProviderFactory._model_providers.get(model_name, 'openai')

        if model_name == 'gpt-5':

            reasoning_effort_value = globals().get('reasoning_effort', f'{reasoning_effort_value}')
            # Create model and store reasoning effort
            model = ChatOpenAI(api_key=OpenAI_key, model=model_name, temperature=1)
            reasoning_effort = reasoning_effort_value

        elif provider_name == 'ollama':
            # Create LangChain ChatOpenAI that points to local server
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(
                base_url="http://128.118.54.16:11434/v1",
                api_key="no-api",
                model=model_name,
                temperature=1
            )

        else:
            model = ChatOpenAI(api_key=OpenAI_key, model=model_name, temperature=1)

    except ImportError as e:
        model = ChatOpenAI(api_key=OpenAI_key, model=model_name, temperature=1)
    return model





# **************************************************************************************
# DATA EYE
# *******************************************************************************

import sys
import os
import time
import pandas as pd
import geopandas as gpd
import rasterio
from pydantic import BaseModel
from openai import OpenAI
import configparser
import json
from langchain_openai import ChatOpenAI
from collections import OrderedDict

DataEye_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_eye_constants')
if DataEye_path not in sys.path:
    sys.path.append(DataEye_path)

plugin_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# import data_eye_constants as eye_constants


# import SpatialAnalysisAgent_helper as helper

# def load_config_file():
#     config = configparser.ConfigParser()
#     config_path = os.path.join(plugin_root, 'config.ini')
#     config.read(config_path)
#     return config
#
#
# def load_OpenAI_key():
#     config = load_config_file()  # Re-read the configuration file
#     OpenAI_key = config.get('API_Key', 'OpenAI_key')
#     return OpenAI_key


# current_script_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# config_path = os.path.join(parent_dir, 'config.ini')

# config = configparser.ConfigParser()
# config.read(config_path)
# OpenAI_key = config.get('API_Key', 'OpenAI_key')

# Note: OpenAI key and client are now loaded dynamically to avoid caching issues
#
def create_client():
    """Create OpenAI client with fresh API key from config file"""
    api_key = load_OpenAI_key()
    return OpenAI(api_key=api_key)


def get_data_overview(data_location_dict):
    data_locations = data_location_dict['data_locations']
    # print()
    for data in data_locations:
        try:
            meta_str = ''
            format_ = data['format']
            data_path = data['location']

            # print("data_path:", data_path)

            if format_ in constants.vector_formats:
                meta_str = see_vector(data_path)

            if format_ in constants.table_formats:
                meta_str = see_table(data_path)

            if format_ in constants.raster_formats:
                meta_str = see_raster(data_path)

            data['meta_str'] = meta_str

        except Exception as e:
            print("Error in get_data_overview()", data, e)
    return data_location_dict

#
def add_data_overview_to_data_location(task, data_location_list, model_name=r'gpt-4o-2024-08-06', stream=False):
    # Supports: OpenAI proxy, GPT-5, and normal OpenAI
    prompt = get_prompt_to_pick_up_data_locations(task=task,
                                                  data_locations=data_location_list)

    # Check if using proxy by examining API key
    api_key = load_OpenAI_key()

    if 'gibd-services' in (api_key or ''):
        # PROXY CASE: Use direct requests.post (same as data_eye.py)
        import requests
        url = f"https://www.gibd.online/api/openai/{api_key}"

        # Add JSON format instruction for proxy
        enhanced_prompt = prompt + "\n\nIMPORTANT: Respond with ONLY valid JSON matching this schema: {\"data_locations\": [{\"location\": \"path\", \"format\": \"format_type\"}, ...]}"

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": constants.role},
                {"role": "user", "content": enhanced_prompt},
            ],
            "stream": False,
            "temperature": 1
        }
        response_req = requests.post(url, json=payload)
        data = response_req.json()

        # Extract and clean content (remove markdown code blocks)
        content = data['choices'][0]['message']['content']
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()

        attributes_json = json.loads(content)

    else:
        # NON-PROXY CASE: Use ModelProvider for GPT-5 and normal OpenAI
        try:
            from SpatialAnalysisAgent_ModelProvider import create_unified_client
            client, provider = create_unified_client(model_name)

            # Use beta.chat.completions.parse for structured output
            if client and hasattr(client, 'beta'):
                response = client.beta.chat.completions.parse(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": constants.role},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=1,
                    response_format=constants.Data_locations,
                )
                attributes_json = json.loads(response.choices[0].message.content)
            else:
                # Fallback for models that don't support beta API
                response = provider.generate_completion(
                    client,
                    model_name,
                    [
                        {"role": "system", "content": constants.role},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=1,
                    stream=False
                )
                # Clean markdown if present
                if hasattr(response, 'choices') and response.choices:
                    content = response.choices[0].message.content
                else:
                    content = str(response)
                content = content.strip()
                if content.startswith('```json'):
                    content = content[7:]
                if content.startswith('```'):
                    content = content[3:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()
                attributes_json = json.loads(content)

        except ImportError:
            # Fallback to direct OpenAI client
            client = create_openai_client()
            response = client.beta.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "system", "content": constants.role},
                    {"role": "user", "content": prompt},
                ],
                temperature=1,
                response_format=constants.Data_locations,
            )
            attributes_json = json.loads(response.choices[0].message.content)

    get_data_overview(attributes_json)

    for idx, data in enumerate(attributes_json.get('data_locations', [])):
        meta_str = data.get('meta_str','')
        if idx < len(data_location_list):  # Ensure index is valid
            if meta_str:  # Only append if meta_str is not empty
                data_location_list[idx] += ". Data overview: " + meta_str
        else:
            # Log or handle index out of range issue (optional)
            print(f"Index {idx} out of range for data_location_list.")
    return attributes_json, data_location_list





def get_prompt_to_pick_up_data_locations(task, data_locations):
    data_locations_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(data_locations)])
    prompt = f'Your mission: {constants.mission_prefix} \n\n' + \
             f'Given task description: {task} \n' + \
             f'Data location: \n{data_locations_str}'
    return prompt


def see_table(file_path):
    # print("OK")
    # print(file_path)
    # print(file_path[-3:])
    df = None
    if file_path[-4:].lower() == '.csv':
        # print(file_path)
        df = pd.read_csv(file_path)
        sample_df = pd.read_csv(file_path, dtype=str)

    # Use the formatted version for better readability
    column_lines = []
    for col, dtype in df.dtypes.items():
        sample_value = sample_df.iloc[0][col]
        column_lines.append(f"  - {col}: {dtype} (sample: {sample_value})")

    meta_str = "Columns:\n" + "\n".join(column_lines)
    return meta_str


def _get_df_types_str(df):
    samples = df.sample(1)
    # Format each column on a new line for better readability
    column_lines = []
    for col, dtype in df.dtypes.items():
        sample_value = samples.iloc[0][col]
        column_lines.append(f"  - {col}: {dtype} (sample: {sample_value})")

    types_str = "Columns:\n" + "\n".join(column_lines)
    return types_str


def see_vector(file_path):
    gdf = gpd.read_file(file_path)
    types_str = _get_df_types_str(gdf.drop(columns='geometry'))
    # print(gdf.crs)
    crs_summary = str(gdf.crs)  # will be "EPSG:4326", but the original information would be long
    crs_summary = crs_summary.replace('\n', ' ')

    # Format the metadata in a more readable way
    meta_str = f"{types_str}\n\nCoordinate Reference System: {crs_summary}"

    return meta_str


def see_raster(file_path, statistics=False, approx=False):
    with rasterio.open(file_path) as dataset:
        raster_str = _get_raster_str(dataset, statistics=statistics, approx=approx)
    return raster_str


def _get_raster_str(dataset, statistics=False, approx=False):  # receive rasterio object
    raster_dict = dataset.meta
    raster_dict["band_count"] = raster_dict.pop("count")  # rename the key
    raster_dict["bounds"] = dataset.bounds
    if statistics:
        band_stat_dict = {}
        for i in range(1, raster_dict["band_count"] + 1):
            # need time to do that
            band_stat_dict[f"band_{i}"] = dataset.stats(indexes=i, approx=approx)
        raster_dict["statistics"] = band_stat_dict

    resolution = (dataset.transform[0], dataset.transform[4])
    raster_dict["resolution"] = resolution
    # print("dataset.crs:", dataset.crs)

    crs = dataset.crs

    if crs:
        if dataset.crs.is_projected:
            raster_dict["unit"] = dataset.crs.linear_units
        else:
            raster_dict["unit"] = "degree"
    else:
        raster_dict["Coordinate reference system"] = "unknown"
    # print("dataset.crs:", dataset.crs)

    raster_str = str(raster_dict)
    return raster_str


# beta vervsion of using structured output. # https://cookbook.openai.com/examples/structured_outputs_intro
# https://platform.openai.com/docs/guides/structured-outputs/examples
def get_LLM_reply(prompt,
                  model=r"gpt-4o",
                  verbose=True,
                  temperature=1,
                  stream=True,
                  retry_cnt=3,
                  sleep_sec=10,
                  ):

    count = 0
    isSucceed = False
    # response = None  # Initialize response variable
    while (not isSucceed) and (count < retry_cnt):
        try:
            count += 1
            # Create fresh client with updated API key
            client = create_client()
            response = client.beta.chat.completions.parse(model=model,
                                                          messages=[
                                                              {"role": "system", "content": constants.role},
                                                              {"role": "user", "content": prompt},
                                                          ],
                                                          temperature=temperature,
                                                          response_format=constants.Data_locations,
                                                          )
            isSucceed = True  # Mark as successful if we reach here
        except Exception as e:
            # logging.error(f"Error in get_LLM_reply(), will sleep {sleep_sec} seconds, then retry {count}/{retry_cnt}: \n", e)
            print(f"Error in get_LLM_reply(), will sleep {sleep_sec} seconds, then retry {count}/{retry_cnt}: \n",
                  e)
            time.sleep(sleep_sec)

    return response

def send_feedback(user_api_key, request_id, user_query, feedback, feedback_message, error_msg, error_traceback, generated_code, data_overview,
               task_breakdown=None, workflow_html_path=None, selected_tools=None):
# def send_error(user_api_key, request_id, user_query, feedback, feedback_message, error_msg, error_traceback, generated_code, data_overview,
#                task_breakdown=None, workflow_html_path=None, selected_tools=None):

    # Only send error reports if using gibd-services API key
    if 'gibd-services' not in (user_api_key or ''):
        # Return a mock response object for compatibility
        class MockResponse:
            status_code = 200
            text = "Error reporting skipped (not using gibd-services API key)"
            def json(self):
                return {}
        return MockResponse()

    url = f"https://www.gibd.online/api/feedback/{user_api_key}"

    # Data to send
    data = {
        "service": "GIS Copilot",
        "question_id": request_id,
        "question": user_query,
        "feedback": feedback,
        "feedback_message": feedback_message,
        "error_msg": str(error_msg),  # Convert error to string for JSON serialization
        "error_traceback": error_traceback,
        "generated_code": generated_code,
        "data_overview":data_overview,
        "task_breakdown": task_breakdown,
        "workflow": workflow_html_path,
        "selected_tools": selected_tools
    }
    # Send POST request
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=data
    )

    # # Handle response
    # if response.status_code == 201:
    #     result = response.json()
    #     print("Record created successfully!")
    #     print(json.dumps(result, indent=2))
    # else:
    #     print(f"Error {response.status_code}: {response.text}")

    return response






def get_model_for_operation(current_model):
    """
    Returns the current model if it's from Ollama provider, otherwise returns 'gpt-4o'
    """
    try:
        # from SpatialAnalysisAgent_ModelProvider import ModelProviderFactory
        # provider_name = ModelProviderFactory._model_providers.get(current_model, 'openai')
        # # If the model uses gpt-5, use gpt-4o as operation model
        if current_model == 'gpt-5':
            return 'gpt-4o'
        else:
            return current_model

        # return current_model if provider_name != 'gpt-5' else 'gpt-4o'
    except:
        # Fallback to gpt-4o if there's any issue
        return current_model


# *****************************************************************************************************************
# Data Eye
# ****************************************************************************************************************
import sys
import os
import time
import pandas as pd
import geopandas as gpd
import rasterio
from pydantic import BaseModel
from openai import OpenAI
import configparser
import json


def get_data_overview(data_location_dict):
    data_locations = data_location_dict['data_locations']
    # print()
    for data in data_locations:
        try:
            meta_str = ''
            format_ = data['format']
            data_path = data['location']

            # print("data_path:", data_path)

            if format_ in constants.vector_formats:
                meta_str = see_vector(data_path)

            if format_ in constants.table_formats:
                meta_str = see_table(data_path)

            if format_ in constants.raster_formats:
                meta_str = see_raster(data_path)

            data['meta_str'] = meta_str

        except Exception as e:
            print("Error in get_data_overview()", data, e)
    return data_location_dict


def add_data_overview_to_data_location(request_id, task, data_location_list, model_name=r'gpt-4o-2024-08-06'):
    # Uses direct OpenAI client with structured output (works reliably)
    # The get_LLM_reply function uses client.beta.chat.completions.parse
    # which enforces JSON format via pydantic model
    prompt = get_prompt_to_pick_up_data_locations(task=task,
                                                  data_locations=data_location_list)
    enhanced_prompt = prompt + "\n\nIMPORTANT: Respond with ONLY valid JSON matching this schema: {\"data_locations\": [{\"location\": \"path\", \"format\": \"format_type\"}, ...]}"
    messages = [
        {"role": "system", "content": constants.eye_role},
        {"role": "user", "content": enhanced_prompt}
    ]
    response = unified_llm_call(
        request_id=request_id,
        messages=messages,
        model_name=model_name,
        stream=False,
        temperature=1)
    # unified_llm_call returns a string, not a response object
    response_str = response.strip()
    # Clean markdown code blocks if present
    if response_str.startswith('```json'):
        response_str = response_str[7:]
    if response_str.startswith('```'):
        response_str = response_str[3:]
    if response_str.endswith('```'):
        response_str = response_str[:-3]
    response_str = response_str.strip()
    attributes_json = json.loads(response_str)
    get_data_overview(attributes_json)

    for idx, data in enumerate(attributes_json.get('data_locations', [])):
        meta_str = data.get('meta_str','')
        if idx < len(data_location_list):  # Ensure index is valid
            if meta_str:  # Only append if meta_str is not empty
                data_location_list[idx] += ". Data overview: " + meta_str
        else:
            # Log or handle index out of range issue (optional)
            print(f"Index {idx} out of range for data_location_list.")
    return attributes_json, data_location_list


def get_prompt_to_pick_up_data_locations(task, data_locations):
    data_locations_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(data_locations)])
    prompt = f'Your mission: {constants.mission_prefix} \n\n' + \
             f'Given task description: {task} \n' + \
             f'Data location: \n{data_locations_str}'
    return prompt
def see_table(file_path):
    # print("OK")
    # print(file_path)
    # print(file_path[-3:])
    df = None
    if file_path[-4:].lower() == '.csv':
        # print(file_path)
        df = pd.read_csv(file_path)
        sample_df = pd.read_csv(file_path, dtype=str)

    # Use the formatted version for better readability
    column_lines = []
    for col, dtype in df.dtypes.items():
        sample_value = sample_df.iloc[0][col]
        column_lines.append(f"  - {col}: {dtype} (sample: {sample_value})")

    meta_str = "Columns:\n" + "\n".join(column_lines)
    return meta_str

def _get_df_types_str(df):
    samples = df.sample(1)
    # Format each column on a new line for better readability
    column_lines = []
    for col, dtype in df.dtypes.items():
        sample_value = samples.iloc[0][col]
        column_lines.append(f"  - {col}: {dtype} (sample: {sample_value})")

    types_str = "Columns:\n" + "\n".join(column_lines)
    return types_str

def see_vector(file_path):
    gdf = gpd.read_file(file_path)
    types_str = _get_df_types_str(gdf.drop(columns='geometry'))
    # print(gdf.crs)
    crs_summary = str(gdf.crs)  # will be "EPSG:4326", but the original information would be long
    crs_summary = crs_summary.replace('\n', ' ')

    # Format the metadata in a more readable way
    meta_str = f"{types_str}\n\nCoordinate Reference System: {crs_summary}"

    return meta_str

def see_raster(file_path, statistics=False, approx=False):
    with rasterio.open(file_path) as dataset:
        raster_str = _get_raster_str(dataset, statistics=statistics, approx=approx)
    return raster_str


def _get_raster_str(dataset, statistics=False, approx=False):  # receive rasterio object
    raster_dict = dataset.meta
    raster_dict["band_count"] = raster_dict.pop("count") # rename the key
    raster_dict["bounds"] = dataset.bounds
    if statistics:
        band_stat_dict = {}
        for i in range(1, raster_dict["band_count"] + 1):
              # need time to do that
            band_stat_dict[f"band_{i}"] = dataset.stats(indexes=i, approx=approx)
        raster_dict["statistics"] = band_stat_dict

    resolution = (dataset.transform[0], dataset.transform[4])
    raster_dict["resolution"] = resolution
    # print("dataset.crs:", dataset.crs)

    crs = dataset.crs

    if crs:
        if dataset.crs.is_projected:
            raster_dict["unit"] = dataset.crs.linear_units
        else:
            raster_dict["unit"] = "degree"
    else:
        raster_dict["Coordinate reference system"] = "unknown"
    # print("dataset.crs:", dataset.crs)

    raster_str = str(raster_dict)
    return raster_str