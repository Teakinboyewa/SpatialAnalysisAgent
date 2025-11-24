import os
import sys
import configparser
from pydantic import BaseModel
from openai import OpenAI

# from SpatialAnalysisAgent.SpatialAnalysisAgent_helper import Query_tuning

# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the directory to sys.path
if current_script_dir not in sys.path:
    sys.path.append(current_script_dir)

import SpatialAnalysisAgent_Codebase as codebase
import SpatialAnalysisAgent_SmartDebugger as smart_debugger


def load_config():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_script_dir, 'config.ini')
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

# Use the loaded configuration
config = load_config()

OpenAI_key = config.get('API_Key', 'OpenAI_key')
client = OpenAI(api_key=OpenAI_key)



folder_path  = os.path.join(current_script_dir, 'Tools_Documentation', 'Customized_tools')
# tools_list, other_tools_dict = codebase.index_tools(folder_path)
tools_index, CustomTools_dict, tool_names_lists = codebase.index_tools(folder_path)

# carefully change these prompt parts!

#********************************************************************************************************************************************************************
# -------------------------------------- Query Tuning  ----------------------------------------------------------------------------------

cot_description_prompt = """ You are a GIS expert. Convert the following user request into a **short GIS task description**.

Think step-by-step about what the user is asking, and then write a **concise, domain-specific description** of the GIS task they want to perform.

INSTRUCTIONS: 
- Do NOT include steps related to data acquisition or downloading data.
- Do NOT mention specific software (e.g., GIS software, ArcGIS, QGIS).
- Focus ONLY on spatial analysis or GIS operations.
- Use technical GIS terms where appropriate (e.g., Buffer, Clip, Reproject, Attribute Query).
- Check the Data overview and select the suitable layers, attributes, and field information that would be needed for the User Query.
- Please include the layer name in the task description.
- Please always check the Data overview for projection information and the User Query before deciding whether reprojection of data is needed is needed.
-Only do the reprojection as needed when 1) e.g., calculating distances/buffers that needs projected CRS, and the layers have different projections.
- Start each operation with a label
- Output ONLY the GIS task description - do NOT explain your reasoning.

User Query:
"{query}"

Data Overview:
f"{data_overview}"

Let's think step-by-step:
1. What is the user’s goal?
2. What data is available (review the Data overview for layers, attributes, and field information)?
3. What GIS operations are needed to achieve it?
4. Write a concise summary of the GIS task
5. List labeled operations to perform.

Output Sample: 
Perform a spatial analysis to identify and quantify the counties in Pennsylvania with suitability for tree planting based on annual rainfall. Specifically, execute the following tasks:
1. **Attribute Query**: Filter the counties of Pennsylvania using an attribute query to select those with annual rainfall greater than 2.5 inches.
2. **Calculate Area**: Determine the total area of the selected counties to assess the percentage of Pennsylvania suitable for tree planting.
3. **Count Features**: Count the number of counties meeting the rainfall criteria to identify how many are suitable for tree planting.

"""

Query_tuning_role = """You are a GIS expert. Convert the following user request into a **short GIS task description**."""
Query_tuning_prefix = """Think step-by-step about what the user is asking, and then write a **concise, domain-specific description** of the GIS task they want to perform."""

Query_tuning_requirement = [
                        "Do NOT include steps related to data acquisition or downloading data.",
                        "Do NOT mention specific software (e.g., GIS software, ArcGIS, QGIS).",
                        "Focus ONLY on spatial analysis or GIS operations.",
                        "Use technical GIS terms where appropriate (e.g., Buffer, Clip, Reproject, Attribute Query).",
                        "Check the Data overview and select the suitable layers, attributes, and field information that would be needed for the User Query.",
                        "Please include the layer name in the task description.",
                        "Please always check the {data_overview} for projection information and the {User_Query} before deciding whether reprojection of data is needed is needed.",
                        "Only do the reprojection as needed when 1) e.g., calculating distances/buffers that needs projected CRS, and the layers have different projections.",
                        "Start each operation with a label",
                        "Output ONLY the GIS task description - do NOT explain your reasoning."
    ]
Query_tuning_instructions=[ """
    Let's think step-by-step:
    1. What is the user’s goal?
    2. What data is available (review the Data overview for layers, attributes, and field information)?
    3. What GIS operations are needed to achieve it?
    4. Write a concise summary of the GIS task
    5. List labeled operations to perform.
"""]
Output_Sample = ["""
Perform a spatial analysis to identify and quantify the counties in Pennsylvania with suitability for tree planting based on annual rainfall. Specifically, execute the following tasks:
1. **Attribute Query**: Filter the counties of Pennsylvania using an attribute query to select those with annual rainfall greater than 2.5 inches.
2. **Calculate Area**: Determine the total area of the selected counties to assess the percentage of Pennsylvania suitable for tree planting.
3. **Count Features**: Count the number of counties meeting the rainfall criteria to identify how many are suitable for tree planting.

"""
]






# *********************************************************************************************************************************************************************
# --------------------------------Tool Selection -----------------------------------------------------------------------------------------------------------------------
tool_selection_prompt = """
You are a GIS assistant. Based on the GIS operation description and the provided tool documentation, select the **most appropriate QGIS tool**.

INSTRUCTIONS:
- Choose the **best-fit tool** based on the GIS Operation description.
- "NOTE: You are not limited to QGIS tools only, you can also make use of python libraries".
- There may be some operations that require multiple steps and multiple tools. In that case recommend the tools for each operation.
- When creating charts or plots such as barchart, barplot, scatterplot etc., you should make use of `seaborn` by default except another method is specified.
- "You are not limited to QGIS python functions, you can also use other python functions such as geopandas, numpy, scipy etc.",
- "NOTE:  Algorithm `native:rastercalculator` is not the correct ID for Raster Calculator, the correct ID is `native:rastercalc`",
- f"If a task directly mention creation of thematic map. NOTE: Thematic map creation is to be used. DO NOT select any existing QGIS tool for thematic map creation, rather select from the 'Customized tools' provided. E.g, do not select 'categorized renderer from styles'",
-Do not provide explaination on why a tool is been selected.
- Output MUST follow the format shown below.
- Only do the reprojection as needed when 1) e.g., calculating distances/buffers that needs projected CRS, and the layers have different projections.
User Task:
'{question}'

Available Tools:
{context}

OUTPUT FORMAT (JSON list of tools):
[
{{
"toolname": "<Toolname>",
"tool_id": "<tool_id>",
"description": "<short description>"
}}
]
"""

#*********************************************************************************************************************************************************************
#---------------------------------Identify Operation type------------------------------------------------------------------------------------------------------------
OperationIdentification_role = r''' aaaA professional Geo-information scientist with high proficiency in Geographic Information System (GIS) operations. You also have excellent proficiency in QGIS to perform GIS operations. You are very familiar with QGIS Processing toolbox. You have super proficency in python programming. 
You are very good at providing explanation to a task and  identifying QGIS tools or other tools and functions that can be used to address a problem.
'''
OperationIdentification_task_prefix = rf' Provide a brief explanation on which tool that can be used to perform this task. Identify the most appropriate tools from QGIS processing tool algorithms or any other algorithm or python libraries in order to perform this task (***Note: You are not limited to QGIS tools only***):'


OperationIdentification_requirements = [
    "Think step by step and skip any step that is not applicable for the task at hand",
    "Identify the most appropriate and the best tool for the task",
    "NOTE: You are not limited to QGIS tools only, you can also make use of python libraries",
    "The identification of the most appropriate tool should be guided by the properties of the data provided",
    f"You can Look through the available qgis processing tool algorithms in here and specify if any of the tools can be used for the task: {codebase.algorithm_names}. NOTE: DO NOT return the tool ID",# e.g, 'qgis:heatmapkerneldensityestimation'. This is not a tool name, it is an ID.",
    "You are not limited to QGIS python functions, you can also use other python functions asuch as geopandas, numpy, scipy etc.",
    "NOTE:  Algorithm `native:rastercalculator` is not the correct ID for Raster Calculator, the correct ID is `native:rastercalc`",
    "DO NOT provide Additional details of any tool",
    f"DO NOT make fake tool. If you cannot find any suitable qgis tool, return any tool name that you think is most appropriate based on the descriptions of tools listed in the 'Customized tool' ptovided and if you cannot find other tools, provide any other tools that is suitable",#select from the return 'Unknown' as for the 'Selected tool' key in the reply JSON format. DO NOT use ```json and ```",
    f"If a task directly mention creation of thematic map. NOTE: Thematic map creation is to be used. DO NOT select any existing QGIS tool for thematic map creation, rather select from the 'Customized tools' provided. E.g, do not select 'categorized renderer from styles'",
    "If creating charts or plots such as barchart, barplot, scatterplot etc., you should make use of `seaborn` by default except another method is specified",
    f"If a task involve the use of kernel density map estimation, DO NOT select any existing QGIS tool for density map creation, rather select Density map (Kernel Density Estimation) listed in the 'Customized tools' provided",
    "When using `gdal:proximity`, ensure all shapefiles are rasterized before using them",
    # f"if a task involve the use of Inverse Distance Weighted (IDW) interpolation, DO NOT select any existing QGIS tool, rather select from other tools contained in the 'Customized tools' provided."
]

OperationIdentification_reply_example_1 = "To select the tracts with population above 3000, the tool suitable for the operation is found in the qgis processing tools and the name is  'Extract by attribute' tool. This tool create a new vector layer that only contains matching features from an input layer"

OperationIdentification_reply_example_2 = "To create a thematic map there is no suitable tool within the qgis processing tool. Therefore, I will be performing operation using other tool different from qgis technique. I will be using 'Thematic map creation' tool to perform this task. This operation enables rendering a map using a specified attribute"

OperationIdentification_reply_example_3 = "To extract the counties with Median household income below 50,000 in Pennsylvania, the tool suitable for this operation is found in the QGIS processing tools. The steps to be followed are Use the 'Extract by attribute' tool to select counties where the 'Median_hou' field is below 50,000. Then, use the 'Extract by attribute' tool again to select counties where the 'STATEFP' field is 42, which corresponds to Pennsylvania. If multiple conditions can be combined, then the 'Select by expression' tool will achieve this in one step using an expression."




#*********************************************************************************************************************************************************************
#---------------------------------Tool selection------------------------------------------------------------------------------------------------------------
ToolSelect_role = r''' A professional Geo-information scientist with high proficiency in Geographic Information System (GIS) operations. You also have excellent proficiency in QGIS to perform GIS operations. You are very familiar with QGIS Processing toolbox. You have super proficency in python programming. 
You are very good at identifying QGIS tools and functions that can be used to address a problem.
'''
ToolSelect_prefix = rf' You are to provide a structured response to contain the tool mentioned in this explanation and analysis of the tools to be used to perform a task: '

ToolSelect_reply_example1 = """ {'Selected tool': "Select by attribute"}"""
ToolSelect_reply_example2 = """ {'Selected tool': ["Select by expression", "Select by location"]}"""



ToolSelect_requirements = [
                        f"Look through the available qgis processing tool algorithms in here {codebase.algorithm_names}. NOTE: DO NOT return the tool ID",# e.g, 'qgis:heatmapkerneldensityestimation'. This is not a tool name, it is an ID.",
                        "NOTE: You are not limited to QGIS python functions, you can also use other python functions asuch as geopandas, numpy, scipy etc.",
                        f"DO NOT make fake tool. If you cannot find any qgis tool that match, return any tool name that you think is most appropriate based on the descriptions of tools listed in the 'Customized tools' provided. And if you cannot still find suitable tool just use the name of the tool or python library mentioned in the explanation provided",#other tools, provide any other tools that is suitable"#select from the return 'Unknown' as for the 'Selected tool' key in the reply JSON format. DO NOT use ```json and ```",
                        # # f"If a task involve the use of kernel density map estimation, DO NOT select any existing QGIS tool for density map creation, rather select Density map (Kernel Density Estimation) listed in the 'Customized tools' provided",#{other_tools}.",
                        # f"if a task involve the use of Inverse Distance Weighted (IDW) interpolation, DO NOT select any existing QGIS tool, rather select from other tools contained in the 'Customized tools' provided",#the Other tools ({tools_list})"
                        f"If a task directly mention creation of thematic map. NOTE: Thematic map creation is to be used. DO NOT select any existing QGIS tool for thematic map creation, rather select from the 'Customized tools' provided. E.g, do not select 'categorized renderer from styles'",
                        f"For a single tool, your response should be in form of this example: {ToolSelect_reply_example1}",
                        "When creating charts or plots such as barchart, barplot, scatterplot etc., you should make use of `seaborn` by default except another method is specified",
                        f"If the tools mentioned in the explanation is more than one, then the tools should be in the list 'Selected tool'. For example; {ToolSelect_reply_example2}",
                        "NOTE:  Algorithm `native:rastercalculator` is not the correct ID for Raster Calculator, the correct ID is `native:rastercalc`",
                        "NOTE: You are not limited to QGIS python functions, you can also use other python functions asuch as geopandas, numpy, scipy etc.",
                        "DO NOT provide Additional details of any tool",
                        "When using `gdal:proximity`, ensure all shapefiles are rasterized before using them",
                        "Do NOT provide any explanation for your response",
                        "DO NOT include ' ```json' and ' ``` ' in your reply",
                       "Only do the reprojection as needed when 1) e.g., calculating distances/buffers that needs projected CRS, and the layers have different projections",
                        # f"DO NOT make fake tool. If you cannot find any suitable qgis tool, return any tool you think is most appropriate from the list in {other_tools}" ,#select from the return 'Unknown' as for the 'Selected tool' key in the reply JSON format. DO NOT use ```json and ```",
                        # f"Your response should be strictly in the format example: {ToolSelect_reply_example2}.Do not add any other explanation or comments."

]






# # ***********************************************************************************************************************************************************************
# #----------------------------------- Step ----------------------------------------

#--------------- CONSTANTS FOR GRAPH GENERATION -----------------------------------------------------------------------------------------------------------
# graph_role = r'''A professional Geo-information scientist with high proficiency in using QGIS and programmer good at Python. You have worked on Geographic information science more than 20 years, and know every detail and pitfall when processing spatial data and coding. You know which QGIS tool suitable for a particular spatial analysis such as Spatial Join, vector selection, Buffering, overlay analysis and thematic map rendering. You have significant experence on graph theory, application, and implementation. You will be creating a solution graph based on the specific task.
# '''

# graph_task_prefix = r'Generate a graph (qgis graphical modeler) only, whose nodes are the operations/steps to solve this question: '

# graph_reply_exmaple = r"""
# ```python
# import networkx as nx
#
# G = nx.DiGraph()
#
#
# # Step 1: Access Loaded Data
# G.add_node("network_data", node_type="data", description="Loaded network data")
# G.add_edge("load_network_data", "network_data")
#
# # Step 2: Set Speed Limit as Edge Weight
# G.add_node("set_speed_limit_weight", node_type="operation", description="Assign speed limits as edge weights for road segments in the network")
# G.add_edge("network_data", "set_speed_limit_weight")
#
# G.add_node("weighted_edges", node_type="data", description="Road network with speed limits as edge weights")
# G.add_edge("set_speed_limit_weight", "weighted_edges")
#
#
#
# ...
# ```
# """

# graph_requirement = ["The graph should contain the basic steps to carry out the specified task",
#
#                      "The task will be addressed by using python codes, particularly PYQGIS in QGIS enviroment. Therefore DO NOT include steps that involve opening of QGIS Software or opening of any tool in the QGIS software",
#                      "DO NOT include the steps that involve QGIS environment setup and library installation",
#                      "The input can be data paths or variables, and the output are temporary layer",
#                      "If the data is already loaded in the QGIS instance, you can directly access the layer by its name and proceed with the other steps.",
#                      "Intending to use the QGIS processing tool to perform tasks",
#                      "The output should not be saved but should be loaded as a temporary layer within the QGIS",
#                      "Display Output in QGIS",
#                      "Steps and data (both input and output) form a graph stored in NetworkX. Disconnected components are NOT allowed.",
#                      "There are two types of nodes: a) operation node, and b) data node (both input and output data). These nodes are also input nodes for the next operation node.",
#                     "There must not be disconnected components",
#                      "The input of each operation is the output of the previous operations, except the those need to load data from a path or need to collect data.",
#                      "DO NOT alter or change the name of any given data path or variable that are given.  E.g, '/Data/Shape1' shold not be changed to '/Data/Shape_1'"
#                      "You need to carefully name the output data node, making they human readable but not to long.",
#                      "The data and operation form a graph.",
#
#                      'The first operations are data loading or collection, and the output of the last operation is the final answer to the task.',
#                      'Operation nodes need to connect via output data nodes, DO NOT connect the operation node directly.',
#                      "The node attributes include: 1) node_type (data or operation), 2) data_path (data node only, set to '' if not given ), and description. E.g., {‘name’: 'County boundary', 'data_type': 'data', 'data_path': 'D:\Test\county.shp',  'description': 'County boundary for the study area'}.",
#                      "The connection between a node and an operation node is an edge.",
#                      "Add all nodes and edges, including node attributes to a NetworkX instance, DO NOT change the attribute names.",
#                      "DO NOT generate code to implement the steps.",
#                      # 'Join the attribute to the vector layer via a common attribute if necessary.',
#                      "Put your reply into a Python code block, NO explanation or conversation outside the code block(enclosed by ```python and ```).",
#                      "Note that GraphML writer does not support class dict or list as data values.",
#                      # 'You need spatial data (e.g., vector or raster) to make a map.',
#                      "Do not put the GraphML writing process as a step in the graph.",
#                     "Do not put the graph creation as a step in the graph",
#                      "Keep the graph concise, DO NOT use too many operation nodes.",
#                      'Keep the graph concise, DO NOT over-split task into too many small steps'
#                      ]
#
# # other requirements prone to errors, not used for now
# """
# 'DO NOT over-split task into too many small steps, especially for simple problems. For example, data loading and data transformation/preprocessing should be in one step.',
# """

#****************************************************************************************************************************************************************
## NEW CONSTANTS FOR GRAPH GENERATION
graph_role = r'''A professional Geo-information scientist with high proficiency in using QGIS and programmer good at Python. You have worked on Geographic information science more than 20 years, and know every detail and pitfall when processing spatial data and coding. You know well how to set up workflows for spatial analysis tasks. You have significant experence on graph theory, application, and implementation. You know which QGIS tool suitable for a particular spatial analysis such as Spatial Join, vector selection, Buffering, overlay analysis and thematic map rendering. You have significant experence on graph theory, application, and implementation. You are also experienced on generating map using Matplotlib and GeoPandas.
'''

graph_task_prefix = r'Generate a graph (data structure) only, whose nodes are (1) a series of consecutive steps and (2) data to solve this question: '


graph_reply_exmaple = r"""
```python
import networkx as nx
G = nx.DiGraph()
# Add nodes and edges for the graph
# 1 Load hazardous waste site shapefile
G.add_node("haz_waste_shp_url", node_type="data", path="https://github.com/gladcolor/LLM-Geo/raw/master/overlay_analysis/Hazardous_Waste_Sites.zip", description="Hazardous waste facility shapefile URL")
G.add_node("load_haz_waste_shp", node_type="operation", description="Load hazardous waste facility shapefile")
G.add_edge("haz_waste_shp_url", "load_haz_waste_shp")
G.add_node("haz_waste_gdf", node_type="data", description="Hazardous waste facility GeoDataFrame")
G.add_edge("load_haz_waste_shp", "haz_waste_gdf")
...
```
"""

graph_requirement = ['Think step by step.',
                        'Steps and data (both input and output) form a graph stored in NetworkX. Disconnected components are NOT allowed.',
                        'Each step is a data process operation: the input can be data paths or variables, and the output can be data paths or variables.',
                        'There are two types of nodes: a) operation node, and b) data node (both input and output data). These nodes are also input nodes for the next operation node.',
                        'The input of each operation is the output of the previous operations, except the those need to load data from a path or need to collect data.',
                        'You need to carefully name the output data node, making they human readable but not to long.',
                        'The data and operation form a graph.',
                        'The first operations are data loading or collection, and the output of the last operation is the final answer to the task.'
                        'Operation nodes need to connect via output data nodes, DO NOT connect the operation node directly.',
                        'The node attributes include: 1) node_type (data or operation), 2) data_path (data node only, set to "" if not given ), and description. E.g., {"name": "County boundary", "data_type": "data", "data_path": "D:\\Test\\county.shp",  "description": "County boundary for the study area"}.',
                        'The connection between a node and an operation node is an edge.',
                        'Add all nodes and edges, including node attributes to a NetworkX instance, DO NOT change the attribute names.',
                        'DO NOT generate code to implement the steps.',
                        'Join the attribute to the vector layer via a common attribute if necessary.',
                        'Put your reply into a Python code block, NO explanation or conversation outside the code block(enclosed by ```python and ```).',
                        'Note that GraphML writer does not support class dict or list as data values.',
                        'You need spatial data (e.g., vector or raster) to make a map.',
                        'Do not put the GraphML writing process as a step in the graph.',
                        'Keep the graph concise, DO NOT use too many operation nodes.',
                     ]

# other requirements prone to errors, not used for now
"""
'DO NOT over-split task into too many small steps, especially for simple problems. For example, data loading and data transformation/preprocessing should be in one step.',
"""

#****************************************************************************************************************************************************************
## CONSTANTS FOR OPERATION GENERATION ------------------------------------------

operation_role = r'''A professional Geo-information scientist with high proficiency in GIS operations. You are also proficient in using QGIS processing tool python functions and other python functions such as geopandas, numpy etc. to solve a particular task. You know when to use a particular tool and when not to. You are not limited to QGIS tools
'''
operation_task_prefix = r'You need to generate Python function to do: '

operation_reply_example = """
```python',
def perform_idw_interpolation(input_layer_path, z_field): #cell_size=100, power=2.0):
'''
    #Perform IDW interpolation on a given point layer.
    
    #Define the parameters for IDW interpolation
    # Run the IDW interpolation algorithm
    # Add the output raster layer to the QGIS project
perform_idw_interpolation()

```
"""
operation_requirement = [
    "Think step by step",
    "If you need to perform more than one operation, you must perform the operations step by step",
    "Use the selected tools provided",
    "DO NOT include the QGIS initialization code in the script",
    f"When using QGIS processing algorithm, use `QgsVectorLayer` to load shapefiles. For example `output_layer = QgsVectorLayer(result['OUTPUT'], 'Layer Name', 'ogr')`",
    "Put your reply into a Python code block, Explanation or conversation can be Python comments at the begining of the code block(enclosed by ```python and ```).",
    "The python code is only in a function named in with the operation name e.g 'perform_idw_interpolation()'. The last line is to execute this function.",
    "Only do the reprojection as needed when 1) e.g., calculating distances/buffers that needs projected CRS, and the layers have different projections",
    "If you need to use `QVariant` should be imported from `PyQt5.Qtcore` and NOT `qgis.core`",
    "If you need to use `QColor` should be imported from `PyQt5.QtGui`",
    "Put your reply into a Python code block (enclosed by python and ), NO explanation or conversation outside the code block.",
    "You are not limited to QGIS python functions/tools, you can also use other python functions asuch as geopandas, numpy, scipy etc.",
    "If you need to use `QgsVectorLayer`, it should always be imported from qgis.core.",
    "DO NOT add validity check and DO NOT raise any exception.",
    "DO NOT raise exceptions messages.",
    "When performing any operation that generates an output vector or raster layer , include the code to load the resulting output layer into QGIS",
    "When performing any operation such as counting of features, generating plots (scatter plot, bar plot), etc., which do not require creation of new layers, do not include load the resulting output layer into QGIS rather print the result",
    "If you need to use any field from the input shapefile layer, first access the fields (example code: `fields = input_layer.fields()`), then select the appropriate field carefully from the list of fields in the layer.",
    "If you need to load a raster layer, use this format `output_layer = QgsRasterLayer(output_path, 'Slope Output')`",
    "When using Raster calculator 'native:rastercalculator' is wrong rather the correct ID for the Raster Calculator algorithm is 'native:rastercalc'.",
    "When creating plots such as barplot, scatterplot etc., usually their result is a html or image file. Always save the file into the specified output directory and print the output layer. Do not Load the output HTML/ or image in QGIS as a standalone resource. ",
    "When creating charts or plots such as barchart, barplot, scatterplot etc., you should make use of `seaborn` by default except another method is specified",
    "When creating a scatterplot, 'native:scatterplot' and 'qgis:scatterplot' are not supported. The correct tool is qgis:vectorlayerscatterplot.",
    "When using tool that is used to generate counts e.g 'Vector information(gdal:ogrinfo), Count points in polygon(native:countpointsinpolygon), etc., don't just print the file path (e.g the html path) but also ensure you print the count(e.g Number of conties)",
    "NOTE: `vector_layer.featureCount()` can be use to generate the count of features",
    "If you are printing any file path (e.g html, png, etc.), Do not include any additional information. just print the file path",
    "When loading a CSV layer as a layer, use this: `'f'file///{csv_path}?delimeter=,''`, assuming the csv is comma-separated, but use the csv_path directly for the Input parameter in join operations.",
    "If you are to use processing algorithm, you do not need to include the code to load a data",
    "For tasks that contains interrogative words such as ('how', 'what', 'why', 'when', 'where', 'which'), ensure that no layers are loaded into the QGIS, instead the result should be printed",    "If you are creating plots such as barplot, scatterplot etc., usually their result is a html file. Always save the html file into the specified output directory and print the output layer. Do not Load the output HTML in QGIS as a standalone resource.",
    "If you are using the processing algorithm, make the output parameter to be the user's specified output directory . And use `QgsVectorLayer` to load the feature as a new layer: For example `Buffer_layer = QgsVectorLayer(result['OUTPUT'], 'Buffered output', 'ogr')` for the case of a shapefile.",
    "Similarly, if you used geopandas to generate a new layer, use `QgsVectorLayer` to load the feature as a new layer: For example `Buffer_layer = QgsVectorLayer(result['OUTPUT'], 'Buffered output', 'ogr')` for the case of a shapefile.",
    "Whenever a new layer is being saved, ensure the code first checks if a file with the same name already exists in the output directory, and if it doesn't, go ahead and save with the original name, but if same name exist, append a number to the filename to create a unique name, thereby avoiding any errors related to overwriting or saving the layer.",
    "When naming any output layer, choose a name that is concise, descriptive, easy to read, and free of spaces.",
    "Ensure that temporary layer is not used as the output parameter",
    "When using `gdal:proximity`, ensure all shapefiles are rasterized before using them",
    "When performing multi-step tasks that involve creating intermediary layers, ensure there is a waiting period before proceeding to the next step. This allows enough time for the intermediary layers to be fully created, preventing errors such as 'data not found.'",
    "When adding a new field to the a shapefile, it should be noted that the maximum length for field name is 10, so avoid mismatch in the fieldname in the data and in the calculation."
    "When creating a thematic map after joining attributes to a shapefile, ensure that the field name length for the attribute use for thematic map do not exceed 10, if it exceed 10, truncate the field name (E.g, 'White_Population' can be truncated to 'White_Popu'). Adhering to the 10 field name length limit ensures consistency and prevents errors during thematic map creation."
]

# ------------- OPERATION_CODE REVIEW------------------------------------------------------
operation_code_review_role = r''' A professional Geo-information scientist and Python developer with over 20 years of experience in Geographic Information Science (GIS). You are highly knowledgeable about spatial data processing and coding, and you specialize in code review. You are meticulous and enjoy identifying potential bugs and data misunderstandings in code.
'''

operation_code_review_task_prefix = r'''Review the code of a function to determine whether it meets its associated requirements and documentation. If it does not, correct it and return the complete corrected code.'''
operation_code_review_requirement = ["Review the codes very carefully to ensure it meets its requirement.",
                                    "NOTE: You are not limited to QGIS tools only, you can also make use of python libraries",
                                    "Compare the code with the code example of any tool being used which is contained in the tool documentation (if provided), and ensure the parameters are set correctly.",
                                    "Ensure that QGIS initialization code are not included in the script.",
                                    "If you need to use `QColor` should be imported from `PyQt5.QtGui`",
                                    "Put your reply into a Python code block, Explanation or conversation can be Python comments at the begining of the code block(enclosed by ```python and ```).",
                                     "The python code is only in a function named in with the operation name e.g 'perform_idw_interpolation()'. The last line is to execute this function.",
                                    "Only do the reprojection as needed when 1) e.g., calculating distances/buffers that needs projected CRS, and the layers have different projections",
                                    "When performing any operation that generates an output vector or raster layer , include the code to load the resulting output layer into QGIS",
                                    "When performing any operation such as counting of features, generating plots (scatter plot, bar plot), etc., which do not require creation of new layers, do not include load the resulting output layer into QGIS rather print the result",
                                    "For tasks that contains interrogative words such as ('how', 'what', 'why', 'when', 'where', 'which'), ensure that no layers are loaded into the QGIS, instead the result should be printed",
                                    "The data needed for the task are already loaded in the qgis environment, so there is no need to load data; just use the provided data path.",
                                    "The code should not contain any validity check.",
                                    "The code is designed to be run within the QGIS Python environment, where the relevant QGIS libraries are available. However, if any third-party libraries needed, it should always be imported.",
                                    f"When using QGIS processing algorithm, use `QgsVectorLayer` to load shapefiles. For example `output_layer = QgsVectorLayer(result['OUTPUT'], 'Layer Name', 'ogr')`",
                                    "Ensure that the data paths in the code examples are replaced with the data paths provided by the user approprately",
                                    "When using Raster calculator, 'native:rastercalculator' is wrong rather the correct ID for the Raster Calculator algorithm is 'native:rastercalc'.",
                                    "When creating plots such as barplot, scatterplot etc., usually their result is a html or image file. Always save the file into the specified output directory and print the output layer. Do not Load the output HTML in QGIS as a standalone resource. ",# Always print out the result"
                                    "When creating charts or plots such as barchart, barplot, scatterplot etc., you should make use of `seaborn` by default except another method is specified",
                                    "When printing the result of plots e.g barplot,scatterplot, boxplot etc, always print out the file path of the result only, ensure any description or comment is not added.",
                                    "When using tool that is used to generate counts e.g 'Vector information(gdal:ogrinfo), Count points in polygon(native:countpointsinpolygon), etc., don't just print the file path (e.g the html path) but also ensure you print the count(e.g Number of conties)",
                                    "NOTE: `vector_layer.featureCount()` can be use to generate the count of features",
                                     "If you are printing any file path (e.g html, png, etc.), Do not include any additional information. just print the file path",
                                    "Do not generate a layer for tasks that only require printing the answer, like questions of how, what, why, etc. e.g., for tasks like 'How many counties are there in PA?', 'What is the distance from A to B', etc.",
                                    "When creating a scatter plot, 'native:scatterplot' and 'qgis:scatterplot' are not supported. The correct tool is qgis:vectorlayerscatterplot, ensure the correct tool is used",
                                    "When creating density maps, do not use `matplotlib` to visualize the result, ensure the result is saved as 'tif' and loaded to QGIS",
                                    f"When using the processing algorithm, make the output parameter to be the user's specified output directory . And use `QgsVectorLayer` to load the feature as a new layer: For example `output_layer = QgsVectorLayer(result['OUTPUT'], 'Layer Name', 'ogr')` for the case of a shapefile.",
                                    f"Similarly, if you used geopandas to generate a new layer, use `QgsVectorLayer` to load the feature as a new layer: For example `buffer_layer = QgsVectorLayer(result['OUTPUT'], 'Buffered output', 'ogr')` for the case of a shapefile.",
                                    "Whenever a new layer is being saved, ensure the code first checks if a file with the same name already exists in the output directory, and if it doesn't, go ahead and save with the original name, but if same name exist, append a number to the filename to create a unique name, thereby avoiding any errors related to overwriting or saving the layer.",
                                    "When naming any output layer, choose a name that is concise, descriptive, easy to read, and free of spaces.",
                                     "Ensure that temporary layer is not used as the output parameter",
                                    "When using `gdal:proximity`, ensure all shapefiles are rasterized before using them",
                                    "When performing multi-step tasks that involve creating intermediary layers, ensure there is a waiting period before proceeding to the next step. This allows enough time for the intermediary layers to be fully created, preventing errors such as 'data not found.'",
                                    "When adding a new field to the a shapefile, it should be noted that the maximum length for field name is 10, so avoid mismatch in the fieldname in the data and in the calculation.",
                                    "When creating a thematic map after joining attributes to a shapefile, ensure that the field name length for the attribute use for thematic map do not exceed 10, if it exceed 10, truncate the field name. Adhering to the 10 field name length limit ensures consistency and prevents errors during thematic map creation."

                                     ]



# --------------- SMART DEBUGGING CONSTANTS ---------------
# Initialize smart debugger instance
debugger_instance = smart_debugger.SmartDebugger()

# Enhanced debugging role with smart capabilities
debug_role = r'''A professional Geo-information scientist with high proficiency in using QGIS and programmer good at Python. You have worked on Geographic information science more than 20 years, and know every detail and pitfall when processing spatial data and coding. You have advanced debugging capabilities with pattern recognition, contextual analysis, and adaptive learning from debugging history. You analyze errors intelligently and provide targeted solutions.
'''

debug_task_prefix = r'You need to correct the code of a program based on the given error information, then return the complete corrected code. Use smart debugging techniques to analyze the error pattern and provide contextual solutions.'

def get_smart_debug_requirements(error_msg="", code="", operation_type=None):
    """Generate dynamic debugging requirements based on error analysis"""

    # Get smart suggestions from the debugger
    smart_suggestions = debugger_instance.generate_debug_suggestions(error_msg, code, operation_type)

    # Base requirements that are always included
    base_requirements = [
        "Analyze the error pattern and apply contextual debugging strategies",
        "Elaborate your reasons for revision based on error analysis",
        "You must return the entire corrected program in only one Python code block(enclosed by ```python and ```); DO NOT return the revised part only.",
        "If you need to perform more than one operation, you must perform the operations step by step",
        "NOTE: You are not limited to QGIS tools only, you can also make use of python libraries",
        "When using `QgsVectorLayer`, it should always be imported from `qgis.core`.",
        "When using QGIS processing algorithm, use `QgsVectorLayer` to load shapefiles. For example `output_layer = QgsVectorLayer(result['OUTPUT'], 'Layer Name', 'ogr')`",
        "If you need to use `QColor` should be imported from `PyQt5.QtGui`",
        "DO NOT include the QGIS initialization code in the script",
        "Make your codes to be concise/short and accurate",
        "`QVariant` should be imported from `PyQt5.Qtcore` and NOT `qgis.core`",
        "When running processing algorithms, use `processing.run('algorithm_id', {parameter_dictionary})`",
        "Put your reply into a Python code block (enclosed by python and ), NO explanation or conversation outside the code block.",
        "When using Raster calculator 'native:rastercalculator' is wrong rather the correct ID for the Raster Calculator algorithm is 'native:rastercalc'.",
        "When loading a CSV layer as a layer, use this: `'f'file///{csv_path}?delimeter=,''`, assuming the csv is comma-separated, but use the csv_path directly for the Input parameter in join operations.",
        "For tasks that contains interrogative words such as ('how', 'what', 'why', 'when', 'where', 'which'), ensure that no layers are loaded into the QGIS, instead the result should be printed",
        "When creating plots such as barplot, scatterplot etc., usually their result is a html or image file. Always save the file into the specified output directory and print the output layer. Do not Load the output HTML in QGIS as a standalone resource.",
        "When printing the result of plots e.g barplot,scatterplot, boxplot etc, always print out the file path of the result only, ensure any description or comment is not added.",
        "When creating a scatter plot, 'native:scatterplot' and 'qgis:scatterplot' are not supported. The correct tool is qgis:vectorlayerscatterplot, ensure the correct tool is used",
        "When using tool that is used to generate counts e.g 'Vector information(gdal:ogrinfo), Count points in polygon(native:countpointsinpolygon), etc., ensure you print the count",
        "NOTE: `vector_layer.featureCount()` can be use to generate the count of features",
        "Whenever a new layer is being saved, ensure the code first checks if a file with the same name already exists in the output directory, and if it does, append a number (e.g filename_1, filename_2, etc) to the filename to create a unique name, thereby avoiding any errors related to overwriting or saving the layer.",
        "When naming any output layer, choose a name that is concise, descriptive, easy to read, and free of spaces.",
        "Ensure that temporary layer is not used as the output parameter",
        "When adding a new field to the a shapefile, it should be noted that the maximum length for field name is 10, so avoid mismatch in the fieldname in the data and in the calculation."
    ]

    # Add smart suggestions as requirements
    smart_requirements = [f"Smart Debug Suggestion: {suggestion}" for suggestion in smart_suggestions]

    return base_requirements + smart_requirements

# Legacy support - static requirements for backward compatibility
debug_requirement = [
    "Elaborate your reasons for revision.",
    "If same error persist, please fallback to the best function/tools you are mostly familiar with",
    "You must return the entire corrected program in only one Python code block(enclosed by ```python and ```); DO NOT return the revised part only.",
    "If you need to perform more than one operation, you must perform the operations step by step",
    "NOTE: You are not limited to QGIS tools only, you can also make use of python libraries",
    "If the generated codes for the selected tools provided are not working you can use other python functions such as geopandas, numpy, scipy etc.",
    "When using `QgsVectorLayer`, it should always be imported from `qgis.core`.",
    "When using QGIS processing algorithm, use `QgsVectorLayer` to load shapefiles. For example `output_layer = QgsVectorLayer(result['OUTPUT'], 'Layer Name', 'ogr')`",
    "If you need to use `QColor` should be imported from `PyQt5.QtGui`",
    "Use the latest qgis libraries and methods.",
    "DO NOT include the QGIS initialization code in the script",
    "Make your codes to be concise/short and accurate",
    "`QVariant` should be imported from `PyQt5.Qtcore` and NOT `qgis.core`",
    "NOTE: `QgsVectorJoinInfo` may not always be available or accessible in recent QGIS installations, thus use `QgsVectorLayerJoinInfo` instead",
    "When running processing algorithms, use `processing.run('algorithm_id', {parameter_dictionary})`",
    "Put your reply into a Python code block (enclosed by python and ), NO explanation or conversation outside the code block.",
    "When using `QgsVectorLayer `, it should always be imported from qgis.core.",
    "When using Raster calculator 'native:rastercalculator' is wrong rather the correct ID for the Raster Calculator algorithm is 'native:rastercalc'.",
    "NOTE: When saving a file (e.g shapefile, csv file etc) to the any path/directory, first check if the the filename already exists in the specified path/directory. If it does, overwrite the file. If the file does not exist, then save the new file directly",
    "NOTE, when a one data path is provided, you DO NOT need to perform join.",
    "If you need to use any field from the input shapefile layer, first access the fields (example code: `fields = input_layer.fields()`), then select the appropriate field carefully from the list of fields in the layer.",
    "When loading a CSV layer as a layer, use this: `'f'file///{csv_path}?delimeter=,''`, assuming the csv is comma-separated, but use the csv_path directly for the Input parameter in join operations.",
    "For tasks that contains interrogative words such as ('how', 'what', 'why', 'when', 'where', 'which'), ensure that no layers are loaded into the QGIS, instead the result should be printed",
    "When creating plots such as barplot, scatterplot etc., usually their result is a html or image file. Always save the file into the specified output directory and print the output layer. Do not Load the output HTML in QGIS as a standalone resource. Always print out the file path of the result only without adding any comment.",
    "When printing the result of plots e.g barplot,scatterplot, boxplot etc, always print out the file path of the result only, ensure any description or comment is not added.",
    "When creating a scatter plot, 'native:scatterplot' and 'qgis:scatterplot' are not supported. The correct tool is qgis:vectorlayerscatterplot, ensure the correct tool is used",
    "When using tool that is used to generate counts e.g 'Vector information(gdal:ogrinfo), Count points in polygon(native:countpointsinpolygon), etc., ensure you print the count",
    "NOTE: `vector_layer.featureCount()` can be use to generate the count of features",
    "When using the processing algorithm, make the output parameter to be the user's specified output directory . And use `QgsVectorLayer` to load the feature as a new layer: For example `output_layer = QgsVectorLayer(result['OUTPUT'], 'Layer Name', 'ogr')` for the case of a shapefile.",
    "Similarly, if you used geopandas to generate a new layer, use `QgsVectorLayer` to load the feature as a new layer: For example `output_layer = QgsVectorLayer(result['OUTPUT'], 'Layer Name', 'ogr')` for the case of a shapefile.",
    "Whenever a new layer is being saved, ensure the code first checks if a file with the same name already exists in the output directory, and if it does, append a number (e.g filename_1, filename_2, etc) to the filename to create a unique name, thereby avoiding any errors related to overwriting or saving the layer.",
    "When naming any output layer, choose a name that is concise, descriptive, easy to read, and free of spaces.",
    "Ensure that temporary layer is not used as the output parameter",
    "When adding a new field to the a shapefile, it should be noted that the maximum length for field name is 10, so avoid mismatch in the fieldname in the data and in the calculation."
]



# *******************************************************************************
#DATA EYE CONSTANT
# ********************************************************************************



table_formats = ["CSV", 'Parquet', "TXT"]
vector_formats = ["ESRI shapefile", "GeoPackage", "KML", "geojson"]
raster_formats = ["Tiff", "JPEG", "PNG", "ERDAS IMG", "JP2", "HDF5" "HDF"]

support_formats = table_formats + vector_formats + raster_formats


eye_role = r'''A professional Geo-information scientist and programmer good at Python. You have worked on Geographic information science more than 20 years, and know every detail and pitfall when processing spatial data and coding. You are a very careful person to follow instruction exactly in work.
'''

mission_prefix = r'''You will be provided with brief geospatial data description and locations for a spatial analysis task.
You need to extract the data path, URL, API, an format from a task and data description. Every given data should be included, and keep the original order.
Below are the description of your reply parameters:
- location: the disk path, URL, or API to access the data. Such as r"C:\test.zip".
- format: the format of data, which belongs one of ['TXT', 'CSV', 'Parquet', 'ESRI shapefile', 'KML', 'geojson', 'HDF', 'HDF5', 'LAS/LAZ', 'XLS', 'GML', 'GeoPackage', 'Tiff', 'JPEG', 'PNG', 'URL', 'REST API', 'other']
'''

class Data(BaseModel):
    location: str
    format: str


class Data_locations(BaseModel):
    data_locations: list[Data]
