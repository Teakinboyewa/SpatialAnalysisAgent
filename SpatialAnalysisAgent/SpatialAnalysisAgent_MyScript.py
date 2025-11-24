#%% Import packages
import os
import re
import sys
import time
import requests
import uuid


#%% Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
SpatialAnalysisAgent_dir = os.path.join(current_script_dir, 'SpatialAnalysisAgent')
# print(LLMQGIS_dir)
if current_script_dir not in sys.path:
    sys.path.append(SpatialAnalysisAgent_dir)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
SpatialAnalysisAgent_dir = os.path.join(current_script_dir, 'SpatialAnalysisAgent')


#%% IMPORT NECCESSARY MODULES
import SpatialAnalysisAgent_Constants as constants
import SpatialAnalysisAgent_helper as helper
import SpatialAnalysisAgent_ToolsDocumentation as ToolsDocumentation
import SpatialAnalysisAgent_Codebase as codebase




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
    reasoning_effort_value = sys.argv[6]
    main(task, data_path,workspace_directory, model_name, is_review, reasoning_effort_value)

#%% Printing the Task
print("=" * 56)
print(f"User: {task}")
print("=" * 56)
time.sleep(3)
#%% INITIALIZE THE AI MODEL
API_Key = helper.get_openai_key(model_name)


if 'gibd-services'  in (API_Key or ''):
    request_id = helper.get_question_id(API_Key)
    print(f"RequestID:{request_id}")
else:
    request_id = str(uuid.uuid4())

model = helper.initialize_ai_model(model_name=model_name, reasoning_effort=reasoning_effort_value, OpenAI_key=API_Key)


#%% ANALYZING THE USER REQUEST
time.sleep(1)
print("=" * 56)
print("AI IS ANALYZING THE TASK ...")
print("=" * 56)
# time.sleep(2)

# GENERATE TASK NAME
operation_model = helper.get_model_for_operation(model_name)

task_name = helper.generate_task_name_with_model_provider(request_id=request_id,model_name=operation_model, stream = False, task_description=task, reasoning_effort=reasoning_effort_value)
# task_name = helper.generate_task_name_with_gpt(specific_model_name='gpt-3.5-turbo', task_description=task)
print(f"task_name: {task_name}")


#%% DATA OVERVIEW
time.sleep(1)
print("=" * 56)
print("AI IS EXAMINING THE DATA ...")
print("=" * 56)

time.sleep(1)

data_path_str = data_path.split('\n')
# attributes_json, DATA_LOCATIONS = data_eye.add_data_overview_to_data_location(task=task, data_location_list=data_path_str, model_name=operation_model)
attributes_json, data_overview = helper.add_data_overview_to_data_location(request_id=request_id,task=task, data_location_list=data_path_str, model_name=operation_model, reasoning_effort=reasoning_effort_value)

print(f"data overview: {data_overview}")

# print(DATA_LOCATIONS)
# print("\n__")
print(attributes_json)




## ************************************FINE TUNING THE USER REQUEST************************************************************************
User_Query_Tuning = True

Query_tuning_prompt_str = helper.create_Query_tuning_prompt(task=task, data_overview=data_overview)

print(Query_tuning_prompt_str)


if User_Query_Tuning:
    # Use streaming version for real-time output
    print("TASK_BREAKDOWN:", end="")
    task_breakdown = helper.Query_tuning(request_id=request_id, Query_tuning_prompt_str = Query_tuning_prompt_str, model_name= model_name, stream=True, reasoning_effort=reasoning_effort_value)
    print("\n_")  # End marker for task breakdown
else:
    # OPERATION IDENTIFICATION
    OperationIdentification_prompt_str = helper.create_OperationIdentification_promt(task=task)
    Selected_OperationIdentification_reply = helper.OperationIdentification(request_id=request_id, OperationIdentification_prompt_str=OperationIdentification_prompt_str,
                                              model_name=operation_model, stream=True, reasoning_effort=reasoning_effort_value)
    # # print(f"OperationIdentification PROMPT ----------{OperationIdentification_prompt_str}")
    # from IPython.display import clear_output
    # chunks = asyncio.run(helper.fetch_chunks(model, OperationIdentification_prompt_str))
    # LLM_reply_str = helper.convert_chunks_to_str(chunks=chunks)
    # task_breakdown = LLM_reply_str



#%% TOOL SELECTION
time.sleep(1)
print("=" * 56)
print("AI IS SELECTING THE APPROPRIATE TOOL(S) ...")
print("=" * 56)
time.sleep(1)


ToolSelect_prompt_str = helper.create_ToolSelect_prompt(task=task_breakdown, data_path=data_overview)
# print(ToolSelect_prompt_str)



print(f"TOOL SELECT PROMPT ---------------------: {ToolSelect_prompt_str}")
print("SELECTED TOOLS:", end="")
# Selected_Tools_reply = asyncio.run(helper.stream_llm_response(tool_model, ToolSelect_prompt_str))
Selected_Tools_reply = helper.tool_select(request_id=request_id,ToolSelect_prompt_str = ToolSelect_prompt_str, model_name=operation_model, stream=True, reasoning_effort=reasoning_effort_value)
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


# model_name2 = r'gpt-4o'
script_directory = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_directory, "graphs")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Define graph file path
graph_file_path = os.path.join(save_dir, f"{task_name}.graphml")

# Use helper function instead of Solution class
task_explanation = task_breakdown
graph_response, code_for_graph, solution_graph_dict = helper.generate_graph_response(
    request_id=request_id,
    task=task,
    task_explanation=task_explanation,
    data_path=data_overview,
    graph_file_path=graph_file_path,
    model_name=operation_model,
    stream=True,
    execute=True,  # This will execute the code and load the graph,
    reasoning_effort=reasoning_effort_value
)

# clear_output(wait=True)

# Access the graph from the returned dictionary
if solution_graph_dict and solution_graph_dict['graph']:
    G = solution_graph_dict['graph']
    source_nodes = solution_graph_dict['source_nodes']
    sink_nodes = solution_graph_dict['sink_nodes']

    # Visualize and save the graph
    nt = helper.show_graph(G)
    graphs_directory = save_dir
    html_graph_path = os.path.join(graphs_directory, f"{task_name}_solution_graph.html")
    counter = 1
    while os.path.exists(html_graph_path):
        html_graph_path = os.path.join(graphs_directory, f"{task_name}_solution_graph_{counter}.html")
        counter += 1
    # nt.show_graph(html_graph_path)
    nt.save_graph(html_graph_path)
    print(f"GRAPH_SAVED:{html_graph_path}")
else:
    print("Failed to generate or load solution graph")



#%%***************************************** #Get code for operation without Solution graph ************************
# Create and print the operation prompt string for each selected tool
operation_prompt_str = helper.create_operation_prompt(task = task_breakdown, data_path =data_overview, workspace_directory =workspace_directory, selected_tools = selected_tool_IDs_list, documentation_str=combined_documentation_str)
print(f"OPERATION PROMPT: {operation_prompt_str}")
print ('\n---------- AI IS GENERATING THE OPERATION CODE ----------\n')

print("GENERATED CODE:", end="")

LLM_reply_str = helper.generate_operation_code(request_id=request_id,operation_prompt_str = operation_prompt_str, model_name=model_name, stream=True, reasoning_effort=reasoning_effort_value)
# LLM_reply_str = asyncio.run(helper.stream_llm_response(model, operation_prompt_str))
# print(LLM_reply_str)
#EXTRACTING CODE

print("\n -------------------------- GENERATED CODE --------------------------------------------\n")
print("```python")
extracted_code = helper.extract_code_from_str(LLM_reply_str, task)
print("```")


if is_review:


    #%% --------------------------------------------- CODE REVIEW ------------------------------------------------------
    # Print the message and apply a waiting time with progress dots
    print("\n ----AI IS REVIEWING THE GENERATED CODE (YOU CAN DISABLE CODE REVIEW IN THE SETTINGS TAB)----", end="")
    # print(f'\n---Model name : {model_name}\n')

    code_review_prompt_str = helper.code_review_prompt(extracted_code = extracted_code, data_path = data_overview, selected_tool_dict= selected_tool_IDs_list, workspace_directory = workspace_directory, documentation_str=combined_documentation_str)
    # print(code_review_prompt_str)
    # code_review_prompt_str_chunks = asyncio.run(helper.fetch_chunks(model, code_review_prompt_str))
    # clear_output(wait=False)
    # review_str_LLM_reply_str = helper.convert_chunks_to_code_str(chunks=code_review_prompt_str_chunks)

    review_str_LLM_reply_str = helper.code_review(request_id=request_id, code_review_prompt_str = code_review_prompt_str, model_name=model_name, stream=True, reasoning_effort=reasoning_effort_value)




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
    code, output, error_collector = helper.execute_complete_program(request_id=request_id,code=reviewed_code, try_cnt=5, task=task, model_name=model_name,
                                                   reasoning_effort_value=reasoning_effort_value, documentation_str=combined_documentation_str,
                                                   data_path= data_path, workspace_directory=workspace_directory, review=True, stream=True, reasoning_effort=reasoning_effort_value)

    # display(Code(code, language='python'))
else:
    # In non-review mode, the code is already emitted above, just proceed to execution
    generated_code = extracted_code

    # print("Running code2")
    code, output, error_collector = helper.execute_complete_program(request_id=request_id, code= extracted_code, try_cnt=5, task=task, model_name=model_name,
                                                   reasoning_effort_value=reasoning_effort_value, documentation_str=combined_documentation_str,
                                                   data_path=data_path, workspace_directory=workspace_directory, stream=True, review=True, reasoning_effort=reasoning_effort_value)



generated_code = code
# Update the CodeEditor with the final code (which may have been debugged)
import urllib.parse
print("CODE_READY_URLENCODED2:" + urllib.parse.quote(generated_code))

# Send completion report with all collected information
# try:
# Convert selected_tools list to string for sending
selected_tools_str = ', '.join(selected_tools) if isinstance(selected_tools, list) else str(selected_tools)



# Read the HTML content
try:
    html_graph_content = helper.read_html_graph_content(html_graph_path)
except (FileNotFoundError, IOError) as e:
    print(f"ERROR: {e}")
    html_graph_content = ""

# Only send error reports if using gibd-services API key
if 'gibd-services' not in (API_Key or ''):
    print("Error reporting skipped (not using gibd-services API key)")

url = f"https://www.gibd.online/api/feedback/{API_Key}"

# feedback to send
feedback = {
    "service_name": "GIS Copilot",
    "question_id": request_id,
    "question": task,
    # "feedback":"GOOD/BAD",
    # "feedback_message":"",
    "error_msg": "Collected execution errors",
    "error_traceback":str(error_collector),
    "generated_code":generated_code,
    "data_overview": data_overview,
    "task_breakdown":task_breakdown,
    "selected_tools":selected_tools_str,
    "workflow": html_graph_content
}

# Send POST request
response = requests.post(
    url,
    headers={"Content-Type": "application/json"},
    json=feedback
)





# Display the captured output (like the file path) in your GUI or terminal
for line in output.splitlines():
    print(f"Output: {line}")

# print("-----Script completed-----")
