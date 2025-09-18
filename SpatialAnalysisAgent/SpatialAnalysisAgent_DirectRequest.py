
## Install package
# ! pip install pyvis
# ! pip install networkx
# ! pip install openai
# ! pip install pandas


##Import package
import os
import sys
import requests
import networkx as nx
#import pandas as pd
#import geopandas as gpd
from pyvis.network import Network
from openai import OpenAI
from IPython.display import display, HTML, Code
from IPython.display import clear_output
from langchain_openai import ChatOpenAI


custom_module_dir = r"D:\LLM_Geo_QGIS\LLMQgs"  # REPLACE WITH THE PATH THAT CONTAINS ALL THE MODULES
if custom_module_dir not in sys.path:sys.path.append(custom_module_dir)

import SpatialAnalysisAgent_Constants as constants
import helper
from SpatialAnalysisAgent_kernel import Solution
OpenAI_key =helper.load_OpenAI_key()



# Enable autoreload
from IPython import get_ipython
ipython = get_ipython()
if ipython:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')

import SpatialAnalysisAgent_Constants as constants
import helper
from SpatialAnalysisAgent_kernel import Solution





#CASE STUDY 1: DATA LOADING


#CASE STUDY 2: Extract by attribute
# task_name ='idw'
# output_path = r'D:\LLM_Geo_QGIS\data\Joined_NC_Tracts\Selected_Feature20.shp'
# task = rf'''1) Select the Area above 4. Area is contained in 'Area' column. Output_path : {output_path}
# '''
# DATA_LOCATIONS = ["NC tract population CSV file: r'D:\LLM_Geo_QGIS\data\Joined_NC_Tracts\joined_tract_37.shp' "
#                 ]


# task_name ='idw'
# output_path = r'D:\LLM_Geo_QGIS\data\Joined_NC_Tracts\Selected_Feature20.shp'
# task = rf'''1) Generate an Inverse Distance Weighted interpolation of a point vector layer using the column "Coronary_h".
# '''
#
# DATA_LOCATIONS = ['D:/LLM_Geo_QGIS/data/Joined_NC_Tracts/joined_tract_37.shp'
#                 ]

task_name ='idR'
output_path = r'D:\LLM_Geo_QGIS\data\Joined_NC_Tracts\Selected_Feature20.shp'
# task = rf'''in a map matrix，12 months in total. All monthly maps need to use the same colorbar range (color scheme: coolwarm). The base of the change rate is January 2020.
# '''
task = Rf'''Create an heatmap to Identify the places where there are hotspots of Cholera in Nigeria?
'''
DATA_LOCATIONS = [r'D:\LLM_Geo_QGIS\data\Joined_NC_Tracts\joined_tract_37.shp'
                ]



if os.path.exists(output_path):
    os.remove(output_path)

save_dir = os.path.join(os.getcwd(), task_name)
os.makedirs(save_dir, exist_ok=True)

# create graph
# model=r"gpt-4"
model = r'gpt-4o'
# model = ChatOpenAI(api_key=OpenAI_key, model=model_name, temperature=0)


#
solution = Solution(
    task=task,
    task_name=task_name,
    save_dir=save_dir,
    data_locations=DATA_LOCATIONS,
    model=model,
)

direct_request_LLM_response = solution.get_direct_request_LLM_response(review=False)

# clear_output(wait=True)
# display(Code(solution.direct_request_code, language='python'))

code = solution.execute_complete_program(code=solution.direct_request_code, try_cnt=10)
# solution.direct_request_code = code
# display(Code(code, language='python'))




# # def main(task_name, task, DATA_LOCATIONS):
# #
# #
# #     # Convert the data locations string back to a list if needed
# #     DATA_LOCATIONS = DATA_LOCATIONS.split(';')  # Assuming data locations are joined by a semicolon
# #
# # if __name__ == "__main__":
# #     task_name = sys.argv[1]
# #     TASK = sys.argv[2]
# #     DATA_LOCATIONS = sys.argv[3]
# #     main(task_name, task, DATA_LOCATIONS)
# #
# # print("Prompt to get solution graph:\n")
# # print(solution.graph_prompt)
# #
# # #%%
# # ##Get graph code from GPT API
# # response_for_graph = solution.get_LLM_response_for_graph()
# # solution.graph_response = response_for_graph
# # solution.save_solution()
# #
# # clear_output(wait=True)
# # display(Code(solution.code_for_graph, language='python'))
# #
# # #%%
# # #Execute code to generate the solution graph
# # exec(solution.code_for_graph)
# # solution_graph = solution.load_graph_file()
# #
# # # Show the graph
# # G = nx.read_graphml(solution.graph_file)
# # nt = helper.show_graph(G)
# # # html_name = os.path.join(os.getcwd(), solution.task_name + '.html')
# # html_name = os.path.join(r'D:\LLM_Geo_QGIS\Case_Study1\SlnGraph', 'solution_graph.html')
# # # HTML file should in the same directory. See:
# # # https://stackoverflow.com/questions/65564916/error-displaying-pyvis-html-inside-jupyter-lab-cell
# # nt.show(name=html_name)
# # # html_name
#
#
# # #%%
# # ## Define Solution class
# # # %load_ext autoreload
# # # %autoreload 2
# # import helper
# #
# # # from LLM_Geo_kernel import Solution
# # from IPython import get_ipython
# #
# # # Enable autoreload
# # ipython = get_ipython()
# # if ipython:
# #     ipython.run_line_magic('load_ext', 'autoreload')
# #     ipython.run_line_magic('autoreload', '2')
# #
# # import LLM_Geo_Constants as constants
# # import helper
# # from LLM_Geo_kernel import Solution
#
#
#
# import helper
# OperationIdentification_prompt_str = helper.create_OperationIdentification_promt(task=task)
# # print(OperationIdentification_prompt_str)
#
#
# #tool identification
# from IPython.display import clear_output
# async def fetch_chunks(model, OperationIdentification_prompt_str):
#     chunks = []
#     async for chunk in model.astream(OperationIdentification_prompt_str):
#         chunks.append(chunk)
#         print(chunk.content, end="", flush=True)
#     return chunks
#
# # Call the async function
# import asyncio
# chunks = asyncio.run(fetch_chunks(model, OperationIdentification_prompt_str))
#
# # Clear output
# clear_output(wait=False)
#
#
# # Convert chunks to string
# LLM_reply_str = helper.convert_chunks_to_str(chunks=chunks)
# clean_str =LLM_reply_str.strip().replace('```json','').replace('```','')
#
#
# import ast
# identify_QGIS_Tool = ast.literal_eval(clean_str)
# selected_QGIS_Tool = identify_QGIS_Tool['Selected QGIS Tool']
# Tool_Identification_response = identify_QGIS_Tool['Explanation']
#
#
# QGIS_Operation_str = helper.create_operaton_prompt(task, selected_QGIS_Tool)
# # print(QGIS_Operation_str)
# print(Tool_Identification_response)
# print("selected_QGIS_Tool:", selected_QGIS_Tool)
#
# #------------------SOLUTION GRAPH
#
#
#
# # # create graph
# # model=r"gpt-4"
# model = r'gpt-4o'
# solution = Solution(
#     task=task,
#     task_name=task_name,
#     save_dir=save_dir,
#     data_locations=DATA_LOCATIONS,
#     model=model,
# )
# # print("Prompt to get solution graph:\n")
# # print(solution.graph_prompt)
#
#
#
# ##Get graph code from GPT API
# response_for_graph = solution.get_LLM_response_for_graph()
# solution.graph_response = response_for_graph
# solution.save_solution()
# #
# clear_output(wait=True)
# # display(Code(solution.code_for_graph, language='python'))
#
#
# #Execute code to generate the solution graph
# exec(solution.code_for_graph)
# solution_graph = solution.load_graph_file()
# #
# # Show the graph
# G = nx.read_graphml(solution.graph_file)
# nt = helper.show_graph(G)
# html_name = os.path.join(os.getcwd(), solution.task_name + '.html')
# # HTML file should in the same directory. See:
# # https://stackoverflow.com/questions/65564916/error-displaying-pyvis-html-inside-jupyter-lab-cell
# nt.show(name=html_name)
# # html_name
#
#
# #%%
# ##Generate prompts and code for operations (functions)
# operations = solution.get_LLM_responses_for_operations(review=isReview)
# solution.save_solution()
#
# all_operation_code_str = '\n'.join([operation['operation_code'] for operation in operations])
#
# clear_output(wait=True)
# display(Code(all_operation_code_str, language='python'))
# #
# #
# #%%
# #Generate prompts and code for assembly program
# assembly_LLM_response = solution.get_LLM_assembly_response(review=isReview)
# solution.assembly_LLM_response = assembly_LLM_response
# solution.save_solution()
# #
# clear_output(wait=True)
# display(Code(solution.code_for_assembly, language='python'))
# #
# #
# #
# # #%%
# # #Execute assembly code
# # all_code = all_operation_code_str + '\n' + solution.code_for_assembly
# #
# # # display(Code(all_code, language='python'))
# #
# # all_code = solution.execute_complete_program(code=all_code, try_cnt=10)
# # #
# # #
# # # # Assuming `all_code` contains the generated code as a string
# # #
# # # # Define the filename
# # # filename = 'generated_code.py'
# # #
# # # # Write the code to the file
# # # with open(filename, 'w') as file:
# # #     file.write(all_code)
# # #
# # # print(f"Code has been saved to {filename}")
#
#
#
#
