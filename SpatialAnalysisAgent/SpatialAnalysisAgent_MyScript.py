#%% Import packages
import os
import re
import sys
import json
import time
from io import StringIO
import requests
import networkx as nx
from PyQt5.QtWidgets import QMessageBox
from pyvis.network import Network
from openai import OpenAI
from IPython.display import display, HTML, Code
from IPython.display import clear_output
from langchain_openai import ChatOpenAI

import asyncio
import nest_asyncio
import processing
from IPython.display import clear_output
from IPython import get_ipython
from qgis.utils import iface

#%% Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
SpatialAnalysisAgent_dir = os.path.join(current_script_dir, 'SpatialAnalysisAgent')
# print(LLMQGIS_dir)
if current_script_dir not in sys.path:
    sys.path.append(SpatialAnalysisAgent_dir)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
SpatialAnalysisAgent_dir = os.path.join(current_script_dir, 'SpatialAnalysisAgent')
DataEye_path = os.path.join(SpatialAnalysisAgent_dir,'SpatialAnalysisAgent_DataEye')
if DataEye_path not in sys.path:
    sys.path.append(DataEye_path)

#%% IMPORT NECCESSARY MODULES
import SpatialAnalysisAgent_Constants as constants
import SpatialAnalysisAgent_helper as helper
import SpatialAnalysisAgent_ToolsDocumentation as ToolsDocumentation
from SpatialAnalysisAgent_kernel import Solution
import SpatialAnalysisAgent_Codebase as codebase
from SpatialAnalysisAgent_DataEye import data_eye



#%% DEFINING MAIN FUNCTION
def main(task, data_path):

    data_path = data_path.split(';')  # Assuming data locations are joined by a semicolon
    task = task

if __name__ == "__main__":
    # task_name = sys.argv[1]
    task = sys.argv[1]
    data_path = sys.argv[2]
    # OpenAI_key = sys.argv[3]
    model_name = sys.argv[3]
    workspace_directory = sys.argv[4]
    is_review = sys.argv[5]
    use_rag = sys.argv[6]
    reasoning_effort_value = sys.argv[7]
    main(task, data_path,workspace_directory, model_name, is_review, use_rag, reasoning_effort_value)

#%% Printing the Task
print("=" * 56)
print(f"User: {task}")
print("=" * 56)
time.sleep(3)

#%% INITIALIZE THE AI MODEL
OpenAI_key = helper.get_openai_key(model_name)
# model = ChatOpenAI(api_key=OpenAI_key, model=model_name, temperature=1)
model = helper.initialize_ai_model(model_name=model_name, reasoning_effort_value=reasoning_effort_value, OpenAI_key=OpenAI_key)


#%% ANALYZING THE USER REQUEST
time.sleep(1)
print("=" * 56)
print("AI IS ANALYZING THE TASK ...")
print("=" * 56)
# time.sleep(2)

# GENERATE TASK NAME
task_name = helper.generate_task_name_with_model_provider(model_name='gpt-4o', task_description=task)
# task_name = helper.generate_task_name_with_gpt(specific_model_name='gpt-3.5-turbo', task_description=task)


# OPERATION IDENTIFICATION
OperationIdentification_prompt_str = helper.create_OperationIdentification_promt(task=task)
# print(f"OperationIdentification PROMPT ----------{OperationIdentification_prompt_str}")

from IPython.display import clear_output

chunks = asyncio.run(helper.fetch_chunks(model,  OperationIdentification_prompt_str))

LLM_reply_str = helper.convert_chunks_to_str(chunks=chunks)


## ************************************FINE TUNING THE USER REQUEST************************************************************************
User_Query_Tuning = True
if User_Query_Tuning:
    # Use streaming version for real-time output
    print("TASK_BREAKDOWN:", end="")
    task_breakdown = asyncio.run(helper.Query_tuning_streaming(user_query=task, model_name=model_name))
    print("\n_")  # End marker for task breakdown
else:
    task_breakdown = LLM_reply_str

#%% DATA ATTRIBUTES
time.sleep(1)
print("=" * 56)
print("AI IS ANALYZING THE DATA ATTRIBUTES ...")
print("=" * 56)
time.sleep(1)

data_path_str = data_path.split('\n')
attributes_json, DATA_LOCATIONS = data_eye.add_data_overview_to_data_location(task=task, data_location_list=data_path_str, model=r'gpt-4o-2024-08-06')
print(f"DATA_LOCATIONS with data overviews: {DATA_LOCATIONS}")
print("\n__")

#%% TOOL SELECTION
time.sleep(1)
print("=" * 56)
print("AI IS SELECTING THE APPROPRIATE TOOL(S) ...")
print("=" * 56)
time.sleep(1)

tool_model = ChatOpenAI(api_key=OpenAI_key, model=r'gpt-4o', temperature=1)

##*USING RAG or NOT USING RAG****************************************************************
# use_rag = False
if use_rag:
    # Selected_Tools_reply = helper.RAG_tool_Select(Query_tuning_prompt_str)
    Selected_Tools_reply = helper.RAG_tool_Select(task_breakdown)

    print(f"Selected Tools Reply: {Selected_Tools_reply}")

    # # print(Selected_Tools_reply)
    response_str = Selected_Tools_reply
    tools_list = json.loads(response_str)

    selected_tool_IDs_list = []
    # selectedTools = {}
    all_documentation = []
    for selected_tool in tools_list:
        selected_tool_ID = selected_tool['tool_id']
        # selectedTools[selected_tool] = selected_tool_ID
        selected_tool_IDs_list.append(selected_tool_ID)
        selected_tool_file_ID = re.sub(r'[ :?\/]', '_', selected_tool_ID)

        selected_tool_file_path = None
        # Walk through all subdirectories and files in the given directory
        Tools_Documentation_dir = os.path.join(current_script_dir, 'SpatialAnalysisAgent', 'Tools_Documentation')
        for root, dirs, files in os.walk(Tools_Documentation_dir):
            for file in files:
                if file == f"{selected_tool_file_ID}.toml":
                    selected_tool_file_path = os.path.join(root, file)
                    break
            if selected_tool_file_path:
                break
        if not selected_tool_file_path:
            print(f"Tool documentation for {selected_tool_file_ID}.toml is not provided")
            continue

        if ToolsDocumentation.check_toml_file_for_errors(selected_tool_file_path):
            # If no errors, get the documentation
            print(f"File {selected_tool_file_ID} is free from errors.")
            documentation_str = ToolsDocumentation.tool_documentation_collection(tool_ID=selected_tool_file_ID)
        else:
            # Step 2: If there are errors, fix the file and then get the documentation
            print(f"File {selected_tool_file_ID} has errors. Attempting to fix...")
            ToolsDocumentation.fix_toml_file(selected_tool_file_path)

            # After fixing, try to retrieve the documentation
            print(f"Retrieving documentation after fixing {selected_tool_file_ID}.")
            documentation_str = ToolsDocumentation.tool_documentation_collection(tool_ID=selected_tool_file_ID)

            # Append the retrieved documentation to the list
        all_documentation.append(documentation_str)
        # Add the selected tool and its ID to the SelectedTools dictionary
        # SelectedTools[selected_tool] = selected_tool_ID
    # for tool in tools_list:
    #     print(tool['tool_id'])
    # Print the list of all selected tool IDs after the loop is complete
    print(f"List of selected tool IDs: {selected_tool_IDs_list}")
    # Step 3: Join all the collected documentation into a single string
    combined_documentation_str = helper.get_combined_documentation_with_fallback(selected_tool_IDs_list, all_documentation)

else:
    ToolSelect_prompt_str = helper.create_ToolSelect_prompt(task=task_breakdown, data_path=DATA_LOCATIONS)

    print(f"TOOL SELECT PROMPT ---------------------: {ToolSelect_prompt_str}")
    print("SELECTED TOOLS:", end="")
    Selected_Tools_reply = asyncio.run(helper.stream_llm_response(tool_model, ToolSelect_prompt_str))
    Refined_Selected_Tools_reply = helper.extract_dictionary_from_response(response=Selected_Tools_reply)
    import ast
    # Convert the string to an actual dictionary
    try:
        Selected_Tools_Dict = ast.literal_eval(Refined_Selected_Tools_reply)
        print(f"\nSELECTED TOOLS: {Selected_Tools_Dict}\n")
    except (SyntaxError, ValueError) as e:
        print("Error parsing the dictionary:", e)

    selected_tools = Selected_Tools_Dict['Selected tool']

    # # Check if the selected_tools is a string or a list
    if isinstance(selected_tools, str):
        selected_tools = [selected_tools]
    print(selected_tools)

    Tools_Documentation_dir = os.path.join(current_script_dir, 'SpatialAnalysisAgent', 'Tools_Documentation')
    # Iterate over each selected tool
    selected_tool_IDs_list = []
    SelectedTools = {}
    all_documentation =[]

    for selected_tool in selected_tools:
        if selected_tool in codebase.algorithm_names:
            selected_tool_ID = codebase.algorithms_dict[selected_tool]['ID']

        elif selected_tool in constants.tool_names_lists:
            selected_tool_ID = constants.CustomTools_dict[selected_tool]['ID']
            # print(f"Selected a tool from the customized folder")
        else:
            selected_tool_ID = selected_tool

        # Add the selected tool and its ID to the SelectedTools dictionary
        SelectedTools[selected_tool] = selected_tool_ID

        selected_tool_IDs_list.append(selected_tool_ID)
        # print(f"SELECTED TOOLS ID: {selected_tool_ID}")
        selected_tool_file_ID = re.sub(r'[ :?\/]', '_', selected_tool_ID)
        # print(F"TOOL_ID: {selected_tool_ID}")
        # print(f"Selected tool filename: {selected_tool_file_ID}")

        selected_tool_file_path = None
        # Walk through all subdirectories and files in the given directory
        for root, dirs, files in os.walk(Tools_Documentation_dir):
            for file in files:
                if file == f"{selected_tool_file_ID}.toml":
                    selected_tool_file_path = os.path.join(root, file)
                    break
            if selected_tool_file_path:
                break
        if not selected_tool_file_path:
            print(f"Tool documentation for {selected_tool_file_ID}.toml is not provided")
            continue

        if selected_tool_file_path:
            # Print the tool information
            print(f"TOOL_ID: {selected_tool_ID}")
            print(f"Selected tool filename: {selected_tool_file_ID}")

        # Define the path to the file (you'll need to adjust this path as needed)
        # selected_file_path = os.path.join(Tools_Documentation_dir, f"{selected_tool_file_ID}.toml")

        # Step 1: Check if the file is free from errors
        if ToolsDocumentation.check_toml_file_for_errors(selected_tool_file_path):
            # If no errors, get the documentation
            print(f"File {selected_tool_file_ID} is free from errors.")
            documentation_str = ToolsDocumentation.tool_documentation_collection(tool_ID=selected_tool_file_ID)
        else:
            # Step 2: If there are errors, fix the file and then get the documentation
            print(f"File {selected_tool_file_ID} has errors. Attempting to fix...")
            ToolsDocumentation.fix_toml_file(selected_tool_file_path)

            # After fixing, try to retrieve the documentation
            print(f"Retrieving documentation after fixing {selected_tool_file_ID}.")
            documentation_str = ToolsDocumentation.tool_documentation_collection(tool_ID=selected_tool_file_ID)

        # Append the retrieved documentation to the list
        all_documentation.append(documentation_str)
    # Print the list of all selected tool IDs after the loop is complete
    print(f"List of selected tool IDs: {selected_tool_IDs_list}")
    # Step 3: Join all the collected documentation into a single string
    combined_documentation_str = '\n'.join(all_documentation)
    print(combined_documentation_str)


#******************************************** SOLUTION GRAPH *******************************************************************************
print ('\n---------- AI IS GENERATING THE GEOPROCESSING WORKFLOW FOR THE TASK ----------\n')

model_name2 = r'gpt-4o'
script_directory = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_directory, "graphs")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
solution = Solution(
    task=task,
    task_explanation= task_breakdown,
    task_name = task_name,
    save_dir=save_dir,
    data_path=DATA_LOCATIONS,
    model= model_name2,
    streaming_callback=streaming_callback if 'streaming_callback' in locals() else None,
)
# print (f'\n---MODEL : {solution.model}\n')
task_explanation = task_breakdown
response_for_graph = solution.get_LLM_response_for_graph()
solution.graph_response = response_for_graph
solution.save_solution()

clear_output(wait=True)
exec(solution.code_for_graph)
solution_graph = solution.load_graph_file()

# # Show the graph
G = nx.read_graphml(solution.graph_file)
nt = helper.show_graph(G)
graphs_directory = save_dir
html_graph_path = os.path.join(graphs_directory ,f"{task_name}_solution_graph.html")
counter = 1
while os.path.exists(html_graph_path):
    html_graph_path = os.path.join(graphs_directory, f"{task_name}_solution_graph_{counter}.html")
    counter += 1
# nt.show_graph(html_graph_path)
nt.save_graph(html_graph_path)
print(f"GRAPH_SAVED:{html_graph_path}")

#%%***************************************** #Get code for operation without Solution graph ************************
# Create and print the operation prompt string for each selected tool
operation_prompt_str = helper.create_operation_prompt(task = task_breakdown, data_path =DATA_LOCATIONS, workspace_directory =workspace_directory, selected_tools = selected_tool_IDs_list, documentation_str=combined_documentation_str)
print(f"OPERATION PROMPT: {operation_prompt_str}")
print ('\n---------- AI IS GENERATING THE OPERATION CODE ----------\n')

print("GENERATED CODE:", end="")
LLM_reply_str = asyncio.run(helper.stream_llm_response(model, operation_prompt_str))
# print(LLM_reply_str)
#EXTRACTING CODE

print("\n -------------------------- GENERATED CODE --------------------------------------------\n")
print("```python")
extracted_code = helper.extract_code_from_str(LLM_reply_str, task)
print("```")

# Emit the initially generated code to CodeEditor immediately after generation
generated_code = extracted_code
print("OPERATION CODE GENERATED SUCCESSFULLY")
import urllib.parse
print("CODE_READY_URLENCODED:" + urllib.parse.quote(generated_code))

if is_review:


    #%% --------------------------------------------- CODE REVIEW ------------------------------------------------------
    # Print the message and apply a waiting time with progress dots
    print("\n ----AI IS REVIEWING THE GENERATED CODE (YOU CAN DISABLE CODE REVIEW IN THE SETTINGS TAB)----", end="")
    # print(f'\n---Model name : {model_name}\n')

    code_review_prompt_str = helper.code_review_prompt(extracted_code = extracted_code, data_path = DATA_LOCATIONS, selected_tool_dict= selected_tool_IDs_list, workspace_directory = workspace_directory, documentation_str=combined_documentation_str)
    # print(code_review_prompt_str)
    code_review_prompt_str_chunks = asyncio.run(helper.fetch_chunks(model, code_review_prompt_str))
    clear_output(wait=False)
    review_str_LLM_reply_str = helper.convert_chunks_to_code_str(chunks=code_review_prompt_str_chunks)



    for i in range(1):
        sys.stdout.flush()
        time.sleep(1)  # Adjust the number of seconds as needed
    print()  # Move to the next line


    #EXTRACTING REVIEW_CODE
    print("\n\n")
    print(f"-------------------------- REVIEWED CODE --------------------------\n")
    print("```python")
    reviewed_code = helper.extract_code_from_str(review_str_LLM_reply_str, task_explanation)
    print("```")

    # Emit the reviewed code to CodeEditor
    generated_code = reviewed_code
    print(f"OPERATION CODE GENERATED AND REVIEWED SUCCESSFULLY")

    import urllib.parse
    print("CODE_READY_URLENCODED:" + urllib.parse.quote(generated_code))




    #%% EXECUTION OF THE CODE
    # print("Running code2")
    code, output = helper.execute_complete_program(code=reviewed_code, try_cnt=5, task=task, model_name=model_name,
                                                   reasoning_effort_value=reasoning_effort_value, documentation_str=combined_documentation_str,
                                                   data_path= data_path, workspace_directory=workspace_directory, review=True)

    # display(Code(code, language='python'))
else:
    # In non-review mode, the code is already emitted above, just proceed to execution
    generated_code = extracted_code

    # print("Running code2")
    code, output = helper.execute_complete_program(code= extracted_code, try_cnt=5, task=task, model_name=model_name,
                                                   reasoning_effort_value=reasoning_effort_value, documentation_str=combined_documentation_str,
                                                   data_path=data_path, workspace_directory=workspace_directory, review=True)



generated_code = code
# Update the CodeEditor with the final code (which may have been debugged)
import urllib.parse
print("CODE_READY_URLENCODED2:" + urllib.parse.quote(generated_code))

# Display the captured output (like the file path) in your GUI or terminal
for line in output.splitlines():
    print(f"Output: {line}")

# print("-----Script completed-----")