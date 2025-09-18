import copy
import os
import requests
import networkx as nx
import pandas as pd
import geopandas as gpd
# from pyvis.network import Network
from openai import OpenAI
import configparser
import pickle
import time
import sys
import traceback
from IPython.display import clear_output
# Get the directory of the current script
# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the directory to sys.path
if current_script_dir not in sys.path:
    sys.path.append(current_script_dir)

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
    """Legacy function - kept for compatibility"""
    OpenAI_key = load_OpenAI_key()
    return OpenAI(api_key=OpenAI_key)

def create_unified_client_for_model(model: str):
    """Create appropriate client based on model type"""
    try:
        import SpatialAnalysisAgent_ModelProvider as ModelProvider
        return ModelProvider.create_unified_client(model)
    except ImportError:
        # Fallback to OpenAI client if ModelProvider not available
        return create_openai_client(), None



import SpatialAnalysisAgent_Constants as constants
import SpatialAnalysisAgent_helper as helper




class Solution():
    """
    class for the solution. Carefully maintain it.  
    
    by Huan Ning, 2023-04-28
    """

    def __init__(self,
                 task,
                 task_explanation,
                 task_name,
                 save_dir,
                 role=constants.graph_role,
                 model = None,
                 data_path=[],
                 stream=True,
                 verbose=True,
                 streaming_callback=None,
                 ):
        # self.async_methods_to_remove = ['get_LLM_reply']
        self.task = task
        self.task_explanation = task_explanation
        self.solution_graph = None
        self.graph_response = None
        self.role = role
        self.data_path = data_path
        self.task_name = task_name
        self.save_dir = save_dir
        self.code_for_graph = ""
        self.graph_file = os.path.join(self.save_dir, f"{self.task_name}.graphml")
        self.source_nodes = None
        self.sink_nodes = None
        self.operations = []  # each operation is an element:
        # {node_name: "", function_descption: "", function_definition:"", return_line:""
        # operation_prompt:"", operation_code:""}
        self.assembly_prompt = ""

        self.parent_solution = None
        self.model = model

        self.stream = stream
        self.verbose = verbose
        self.streaming_callback = streaming_callback

        self.assembly_LLM_response = ""
        self.code_for_assembly = ""
        self.graph_prompt = ""

        self.data_path_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(self.data_path)])

        graph_requirement = constants.graph_requirement.copy()
        graph_requirement.append(f"Save the network into GraphML format, save it at: {self.graph_file}")
        graph_requirement_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(graph_requirement)])

        graph_prompt = f'Your role: {self.role} \n\n' + \
                       f'Your task: {constants.graph_task_prefix} \n {self.task_explanation} \n\n' + \
                       f'Your reply needs to meet these requirements: \n {graph_requirement_str} \n\n' + \
                       f'Your reply example: {constants.graph_reply_exmaple} \n\n' + \
                       f'Data locations (each data is a node): {self.data_path_str} \n'
        self.graph_prompt = graph_prompt

        # self.direct_request_prompt = ''
        self.direct_request_LLM_response = ''
        self.direct_request_code = ''

        self.chat_history = [{'role': 'system', 'content': role}]

    # def clear_console(self):
    #     # Check if the system is Windows
    #     if os.name == 'nt':
    #         os.system('cls')
    #     else:
    #         os.system('clear')

    def get_code_for_operation(self, task_description, selected_tool, selected_tool_ID, documentation_str):
        """
        Generate operation code using GPT for a given task description, tool, and documentation.
        """



        operation_requirement_str = '\n'.join(
            [f"{idx + 1}. {line}" for idx, line in enumerate(constants.operation_requirement)])

        prompt = f"Your role: {constants.operation_role} \n" + \
                 f"Your mission: {constants.operation_task_prefix}: {task_description}\n\n" + \
                 f"Using the following data paths: {self.data_path_str}\n" + \
                 f"Selected tool: {selected_tool}\n" + \
                 f"{selected_tool_ID} Documentation: \n{documentation_str}\n\n" + \
                 f"Requirements: \n{operation_requirement_str}"

        response = self.get_LLM_reply(
            prompt=prompt,
            system_role=constants.operation_role,
            model=self.model,
        )
        operation_response = response
        try:
            operation_code = helper.extract_code(response = operation_response, verbose = False)
            print("operation_code:", operation_code)
        except Exception as e:
            operation_code = ""
        operation_code = operation_code

        return operation_code





    def ask_LLM_to_review_operation_code_generated(self, extracted_code, selected_tool_ID, documentation_str):
        """
        Ask the LLM to review the generated operation code.
        """

        operation_code_review_requirement_str = '\n'.join(
            [f"{idx + 1}. {line}" for idx, line in enumerate(constants.operation_code_review_requirement)])

        operation_code_review_prompt = f"Your role: {constants.operation_code_review_role} \n" + \
                                       f"Your task: {constants.operation_code_review_task_prefix} \n\n" + \
                                       f"The code is: \n----------\n{extracted_code}\n----------\n\n" + \
                                       f"{selected_tool_ID} Documentation: \n{documentation_str} \n\n" + \
                                       f"The requirements for the code are: \n{operation_code_review_requirement_str}"

        print("LLM is reviewing the operation code... \n")
        print(operation_code_review_prompt)

        # Send the prompt to the LLM for review
        response = self.get_LLM_reply(
            prompt=operation_code_review_prompt,
            system_role=constants.operation_code_review_role,
            model=self.model,
            verbose=True,
            stream=True,
            retry_cnt=5,
        )

        # Extract the new code or confirm the review passed
        reviewed_code = helper.extract_code(response)
        return reviewed_code




    def get_LLM_reply(self,
                      prompt,
                      verbose=True,
                      temperature=1,
                      stream=True,
                      retry_cnt=3,
                      sleep_sec=10,
                      system_role=None,
                      model=None,
                      streaming_callback=None,
                      ):

        if system_role is None:
            system_role = self.role

        if model is None:
            model = self.model

        # Use unified client that supports multiple providers
        try:
            client, provider = create_unified_client_for_model(model)
        except Exception:
            # Fallback to traditional OpenAI client
            client = create_openai_client()
            provider = None
        
        count = 0
        isSucceed = False
        response = None  # Initialize response to avoid UnboundLocalError
        self.chat_history.append({'role': 'user', 'content': prompt})
        while (not isSucceed) and (count < retry_cnt):
            try:
                count += 1
                if provider:
                    response = provider.generate_completion(client, model, [
                        {"role": "system", "content": system_role},
                        {"role": "user", "content": prompt},
                    ], temperature=temperature, stream=stream)
                else:
                    response = client.chat.completions.create(model= model,
                                                          # messages=self.chat_history,  # Too many tokens to run.
                                                          messages=[
                                                              {"role": "system", "content": system_role},
                                                              {"role": "user", "content": prompt},
                                                          ],
                                                          temperature=temperature,
                                                          stream=stream

                                                          )
                isSucceed = True
            except Exception as e:
                # logging.error(f"Error in get_LLM_reply(), will sleep {sleep_sec} seconds, then retry {count}/{retry_cnt}: \n", e)
                print(f"Error in get_LLM_reply(), will sleep {sleep_sec} seconds, then retry {count}/{retry_cnt}: \n",
                      e)
                time.sleep(sleep_sec)


        response_chucks = []
        accumulated_response = ""
        if stream:

            for chunk in response:
                response_chucks.append(chunk)
                content = getattr(chunk.choices[0].delta, 'content', '')


                if content is not None:
                    accumulated_response += content
                    # Emit streaming chunks if callback is provided (use instance callback if not provided)
                    callback_to_use = streaming_callback if streaming_callback else self.streaming_callback
                    if callback_to_use and content:
                        callback_to_use(content)

            if verbose:
                print(accumulated_response)
                #
                # response_chucks.append(content)
        # if verbose:
        #     self.clear_console()  # Clear previous output
        #     print(accumulated_response)

        else:
            content = response.choices[0].message.content
            print(content)

        print('\n\n')
        # print("Got LLM reply.")

        response = response_chucks  # good for saving

        content = helper.extract_content_from_LLM_reply(response)

        self.chat_history.append({'role': 'assistant', 'content': content})

        return response

    # def get_LLM_reply(self,
    #                   prompt,
    #                   verbose=True,
    #                   temperature=1,
    #                   stream=True,
    #                   retry_cnt=3,
    #                   sleep_sec=10,
    #                   system_role=None,
    #                   model=None,
    #                   ):
    #
    #     # Default values for system_role and model
    #     if system_role is None:
    #         system_role = self.role
    #     if model is None:
    #         model = self.model
    #
    #     # Create the client and initialize response container
    #     client = create_openai_client()
    #     response = None
    #     self.chat_history.append({'role': 'user', 'content': prompt})
    #
    #     # Retry logic to handle failures
    #     for attempt in range(retry_cnt):
    #         try:
    #             # Request completion from GPT with or without streaming
    #             response = client.chat.completions.create(
    #                 model=model,
    #                 messages=[
    #                     {"role": "system", "content": system_role},
    #                     {"role": "user", "content": prompt},
    #                 ],
    #                 temperature=temperature,
    #                 stream=stream
    #             )
    #             break  # If successful, exit the retry loop
    #         except Exception as e:
    #             print(f"Error during attempt {attempt + 1}/{retry_cnt}: {e}")
    #             time.sleep(sleep_sec)
    #     else:
    #         raise ValueError("Failed to get a response from the LLM after several retries")
    #
    #     # Handle streaming response
    #     response_chunks = []
    #     final_content = ""
    #
    #     if stream:
    #         for chunk in response:
    #             response_chunks.append(chunk)
    #             content = getattr(chunk.choices[0].delta, 'content', '').strip()
    #
    #             if content:
    #                 final_content += content
    #                 if verbose:
    #                     print(content, end='', flush=True)  # Display in real-time
    #
    #     else:
    #         # Handle non-streaming response
    #         final_content = response.choices[0].message.content.strip()
    #         print(final_content)
    #
    #     print('\n\n')
    #
    #     # Store response history and extract final content
    #     response = response_chunks if stream else response
    #     extracted_content = helper.extract_content_from_LLM_reply(response)
    #
    #     # Append to chat history for future interactions
    #     self.chat_history.append({'role': 'assistant', 'content': extracted_content})
    #
    #     return final_content if stream else response

    # def generate_task_name_with_gpt(task, client, model="gpt-3.5-turbo", max_tokens=100):
    #     prompt = f"Given the following task description: '{task}', extract the key action or operation word that represents this task. Provide the task name in one or two words."
    #     response = client.chat.completions.create(
    #         model=model,
    #         messages=[
    #             {"role": "system", "content": "You are a helpful assistant."},
    #             {"role": "user", "content": prompt},
    #         ],
    #         max_tokens=max_tokens,
    #         temperature=0.5
    #     )
    #
    #     task_name = response.choices[0].message['content'].strip()
    #     return task_name

    def get_LLM_response_for_graph(self, execuate=True):
        # self.chat_history.append()
        response = self.get_LLM_reply(
            prompt=self.graph_prompt,
            system_role=self.role,
            model=self.model,
        )
        self.graph_response = response
        try:
            self.code_for_graph = helper.extract_code(response=self.graph_response, verbose=False)
        except Exception as e:
            self.code_for_graph = ""
            print("Extract graph Python code rom LLM failed.")
        if execuate:
            exec(self.code_for_graph)
            self.load_graph_file()
        return self.graph_response

    def load_graph_file(self, file=""):
        G = None
        if os.path.exists(file):
            self.graph_file = file
            G = nx.read_graphml(self.graph_file)
        else:

            if file == "" and os.path.exists(self.graph_file):
                G = nx.read_graphml(self.graph_file)
            else:
                print("Do not find the given graph file:", file)
                return None

        self.solution_graph = G

        self.source_nodes = helper.find_source_node(self.solution_graph)
        self.sink_nodes = helper.find_sink_node(self.solution_graph)

        return self.solution_graph

    @property
    def operation_node_names(self):
        opera_node_names = []
        assert self.solution_graph, "The Solution class instance has no solution graph. Please generate the graph"
        for node_name in self.solution_graph.nodes():
            node = self.solution_graph.nodes[node_name]
            if node['node_type'] == 'operation':
                opera_node_names.append(node_name)
        return opera_node_names

    def get_ancestor_operations(self, node_name):
        ancestor_operation_names = []
        ancestor_node_names = nx.ancestors(self.solution_graph, node_name)
        # for ancestor_node_name in ancestor_node_names:
        ancestor_operation_names = [node_name for node_name in ancestor_node_names if
                                    node_name in self.operation_node_names]

        ancestor_operation_nodes = []
        for oper in self.operations:
            oper_name = oper['node_name']
            if oper_name in ancestor_operation_names:
                ancestor_operation_nodes.append(oper)

        return ancestor_operation_nodes



    def get_descendant_operations(self, node_name):
        descendant__operation_names = []
        descendant_node_names = nx.descendants(self.solution_graph, node_name)
        # for descendant_node_name in descendant_node_names:
        descendant__operation_names = [node_name for node_name in descendant_node_names if
                                       node_name in self.operation_node_names]
        # descendant_codes = '\n'.join([oper['operation_code'] for oper in descendant_node_names])
        descendant_operation_nodes = []
        for oper in self.operations:
            oper_name = oper['node_name']
            if oper_name in descendant__operation_names:
                descendant_operation_nodes.append(oper)

        return descendant_operation_nodes

    def get_descendant_operations_definition(self, descendant_operations):

        keys = ['node_name', 'description', 'function_definition', 'return_line']
        operation_def_list = []
        for node in descendant_operations:
            operation_def = {key: node[key] for key in keys}
            operation_def_list.append(str(operation_def))
        defs = '\n'.join(operation_def_list)
        return defs

    def get_prompt_for_an_opearation(self, operation):
        assert self.solution_graph, "Do not find solution graph!"
        # operation_dict = function_def.copy()

        node_name = operation['node_name']

        # get ancestors code
        ancestor_operations = self.get_ancestor_operations(node_name)
        ancestor_operation_codes = '\n'.join([oper['operation_code'] for oper in ancestor_operations])
        descendant_operations = self.get_descendant_operations(node_name)
        descendant_defs = self.get_descendant_operations_definition(descendant_operations)
        descendant_defs_str = str(descendant_defs)

        pre_requirements = [
            f'The function description is: {operation["description"]}',
            f'The function definition is: {operation["function_definition"]}',
            f'The function return line is: {operation["return_line"]}'
        ]

        operation_requirement_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(
            pre_requirements + constants.operation_requirement)])

        operation_prompt = f'Your role: {constants.operation_role} \n\n' + \
                           f'operation_task: {constants.operation_task_prefix} {operation["description"]} \n\n' + \
                           f'This function is one step to solve the question/task: {self.task} \n\n' + \
                           f"This function is a operation node in a solution graph for the question/task, the Python code to build the graph is: \n{self.code_for_graph} \n\n" + \
                           f'Data locations: {self.data_path_str} \n\n' + \
                           f'Your reply example: {constants.operation_reply_example} \n\n' + \
                           f'Your reply needs to meet these requirements: \n {operation_requirement_str} \n\n' + \
                           f"The ancestor function code is (need to follow the generated file names and attribute names): \n {ancestor_operation_codes} \n\n" + \
                           f"The descendant function (if any) definitions for the question are (node_name is function name): \n {descendant_defs_str}"

        operation['operation_prompt'] = operation_prompt
        return operation_prompt
        # self.operations.append(operation_dict)





    # initial the oepartion list
    def initial_operations(self):
        self.operations = []
        operation_names = self.operation_node_names
        for node_name in operation_names:
            function_def_returns = helper.generate_function_def(node_name, self.solution_graph)
            self.operations.append(function_def_returns)

    def get_LLM_responses_for_operations(self, review=True):
        # def_list, data_node_list = helper.generate_function_def_list(self.solution_graph)
        self.initial_operations()
        for idx, operation in enumerate(self.operations):
            node_name = operation['node_name']
            print(
                f"{idx + 1} / {len(self.operations)}, LLM is generating code for operation node: {operation['node_name']}")
            prompt = self.get_prompt_for_an_opearation(operation)

            response = self.get_LLM_reply(
                prompt=prompt,
                system_role=constants.operation_role,
                model=self.model,
                # model=r"gpt-4",
            )
            # print(response)
            operation['response'] = response
            try:
                operation_code = helper.extract_code(response=operation['response'], verbose=False)
                # print("operation_code:", operation_code)
            except Exception as e:
                operation_code = ""
            operation['operation_code'] = operation_code

            if review:
                operation = self.ask_LLM_to_review_operation_code(operation)

        return self.operations





    def prompt_for_assembly_program(self):
        all_operation_code_str = '\n'.join([operation['operation_code'] for operation in self.operations])
        # operation_code = solution.operations[-1]['operation_code']
        # assembly_prompt = f"" + \

        assembly_requirement = '\n'.join(
            [f"{idx + 1}. {line}" for idx, line in enumerate(constants.assembly_requirement)])

        assembly_prompt = f"Your role: {constants.assembly_role} \n\n" + \
                          f"Your task is: use the given Python functions, return a complete Python program to solve the question: \n {self.task}" + \
                          f"Requirement: \n {assembly_requirement} \n\n" + \
                          f"Data location: \n {self.data_path_str} \n" + \
                          f"Code: \n {all_operation_code_str}"

        self.assembly_prompt = assembly_prompt
        return self.assembly_prompt

    def get_LLM_assembly_response(self, review=True):
        self.prompt_for_assembly_program()
        assembly_LLM_response = helper.get_LLM_reply(self.assembly_prompt,
                                                     system_role=constants.assembly_role,
                                                     model=self.model,
                                                     # model=r"gpt-4",
                                                     )
        self.assembly_LLM_response = assembly_LLM_response
        self.code_for_assembly = helper.extract_code(self.assembly_LLM_response)

        try:
            code_for_assembly = helper.extract_code(response=self.assembly_LLM_response, verbose=False)
        except Exception as e:
            code_for_assembly = ""

        self.code_for_assembly = code_for_assembly

        if review:
            self.ask_LLM_to_review_assembly_code()

        return self.assembly_LLM_response

    # def __getstate__(self):
    #     # Get the current state of the instance
    #     state = self.__dict__.copy()
    #
    #     for method in self.async_methods_to_remove:
    #         if method in state:
    #             del state[method]
    #
    #     return state
    #
    # def __setstate__(self, state):
    #     # Restore instance state (default behavior)
    #     self.__dict__.update(state)
    #
    def save_solution(self):
        #         , graph=True
        # Temporarily store and remove the streaming_callback before pickling
        temp_callback = self.streaming_callback
        self.streaming_callback = None
        
        try:
            new_name = os.path.join(self.save_dir, f"{self.task_name}.pkl")
            with open(new_name, "wb") as f:
                pickle.dump(self, f)
        finally:
            # Restore the streaming_callback after pickling
            self.streaming_callback = temp_callback

    def load_solution(self, file_path):
        # Load the pickled instance
        with open(file_path, "rb") as f:
            solution = pickle.load(f)
        return solution


    def get_solution_at_one_time(self):
        pass



    @property
    def direct_request_prompt(self):

        direct_request_requirement_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(
            constants.direct_request_requirement)])

        direct_request_prompt = f'Your role: {constants.direct_request_role} \n' + \
                                f'Your task: {constants.direct_request_task_prefix} to address the question or task: {self.task} \n' + \
                                f'Location for data you may need: {self.data_path_str} \n' + \
                                f'Your reply needs to meet these requirements: \n {direct_request_requirement_str} \n'
        return direct_request_prompt

    def get_direct_request_LLM_response(self, review=True):

        response = helper.get_LLM_reply(prompt=self.direct_request_prompt,
                                        model=self.model,
                                        stream=self.stream,
                                        verbose=self.verbose,
                                        )

        self.direct_request_LLM_response = response

        self.direct_request_code = helper.extract_code(response=response)

        if review:
            self.ask_LLM_to_review_direct_code()

        return self.direct_request_LLM_response

    def execute_complete_program(self, code: str, try_cnt: int = 10) -> str:

        count = 0
        while count < try_cnt:
            print(f"\n\n-------------- Running code (trial # {count + 1}/{try_cnt}) --------------\n\n")
            try:
                count += 1
                compiled_code = compile(code, 'Complete program', 'exec')
                exec(compiled_code, globals())  # #pass only globals() not locals()
                #!!!!    all variables in code will become global variables! May cause huge issues!     !!!!

                print("\n\n--------------- Done ---------------\n\n")
                return code

            # except SyntaxError as err:
            #     error_class = err.__class__.__name__
            #     detail = err.args[0]
            #     line_number = err.lineno
            #
            except Exception as err:

                # cl, exc, tb = sys.exc_info()

                # print("An error occurred: ", traceback.extract_tb(tb))

                if count == try_cnt:
                    print(f"Failed to execute and debug the code within {try_cnt} times.")
                    return code

                debug_prompt = self.get_debug_prompt(exception=err, code=code)
                print("Sending error information to LLM for debugging...")
                # print("Prompt:\n", debug_prompt)
                response = helper.get_LLM_reply(prompt=debug_prompt,
                                                system_role=constants.debug_role,
                                                model=self.model,
                                                verbose=True,
                                                stream=True,
                                                retry_cnt=5,
                                                )
                code = helper.extract_code(response)

        return code

    def get_debug_prompt(self, exception, code):
        etype, exc, tb = sys.exc_info()
        exttb = traceback.extract_tb(tb)  # Do not quite understand this part.
        # https://stackoverflow.com/questions/39625465/how-do-i-retain-source-lines-in-tracebacks-when-running-dynamically-compiled-cod/39626362#39626362

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
                       f"Your task: correct the code of a program according to the error information, then return the corrected and completed program. \n\n" + \
                       f"Requirement: \n {debug_requirement_str} \n\n" + \
                       f"The given code is used for this task: {self.task} \n\n" + \
                       f"The data location associated with the given code: \n {self.data_path_str} \n\n" + \
                       f"The error information for the code is: \n{str(error_info_str)} \n\n" + \
                       f"The code is: \n{code}"

        return debug_prompt

    def ask_LLM_to_review_operation_code(self, operation):
        code = operation['operation_code']
        operation_prompt = operation['operation_prompt']
        review_requirement_str = '\n'.join(
            [f"{idx + 1}. {line}" for idx, line in enumerate(constants.operation_review_requirement)])
        review_prompt = f"Your role: {constants.operation_review_role} \n" + \
                        f"Your task: {constants.operation_review_task_prefix} \n\n" + \
                        f"Requirement: \n{review_requirement_str} \n\n" + \
                        f"The code is: \n----------\n{code}\n----------\n\n" + \
                        f"The requirements for the code is: \n----------\n{operation_prompt} \n----------\n"

        # {node_name: "", function_descption: "", function_definition:"", return_line:""
        # operation_prompt:"", operation_code:""}
        print("LLM is reviewing the operation code... \n")
        # print(f"review_prompt:\n{review_prompt}")
        response = helper.get_LLM_reply(prompt=review_prompt,
                                        system_role=constants.operation_review_role,
                                        model=self.model,
                                        verbose=True,
                                        stream=True,
                                        retry_cnt=5,
                                        )
        new_code = helper.extract_code(response)
        reply_content = helper.extract_content_from_LLM_reply(response)
        if (reply_content == "PASS") or (new_code == ""):  # if no modification.
            print("Code review passed, no revision.\n\n")
            new_code = code
        operation['code'] = new_code

        return operation

    def ask_LLM_to_review_assembly_code(self):
        code = self.code_for_assembly
        assembly_prompt = self.assembly_prompt
        review_requirement_str = '\n'.join(
            [f"{idx + 1}. {line}" for idx, line in enumerate(constants.assembly_review_requirement)])
        review_prompt = f"Your role: {constants.assembly_review_role} \n" + \
                        f"Your task: {constants.assembly_review_task_prefix} \n\n" + \
                        f"Requirement: \n{review_requirement_str} \n\n" + \
                        f"The code is: \n----------\n{code} \n----------\n\n" + \
                        f"The requirements for the code is: \n----------\n{assembly_prompt} \n----------\n\n"

        print("LLM is reviewing the assembly code... \n")
        # print(f"review_prompt:\n{review_prompt}")
        response = helper.get_LLM_reply(prompt=review_prompt,
                                        system_role=constants.assembly_review_role,
                                        model=self.model,
                                        verbose=True,
                                        stream=True,
                                        retry_cnt=5,
                                        )
        new_code = helper.extract_code(response)
        if (new_code == "PASS") or (new_code == ""):  # if no modification.
            print("Code review passed, no revision.\n\n")
            new_code = code

        self.code_for_assembly = new_code

    def ask_LLM_to_review_direct_code(self):
        code = self.direct_request_code
        direct_prompt = self.direct_request_prompt
        review_requirement_str = '\n'.join(
            [f"{idx + 1}. {line}" for idx, line in enumerate(constants.direct_review_requirement)])
        review_prompt = f"Your role: {constants.direct_review_role} \n" + \
                        f"Your task: {constants.direct_review_task_prefix} \n\n" + \
                        f"Requirement: \n{review_requirement_str} \n\n" + \
                        f"The code is: \n----------\n{code} \n----------\n\n" + \
                        f"The requirements for the code is: \n----------\n{direct_prompt} \n----------\n\n"

        print("LLM is reviewing the direct request code... \n")
        # print(f"review_prompt:\n{review_prompt}")
        response = helper.get_LLM_reply(prompt=review_prompt,
                                        system_role=constants.direct_review_role,
                                        model=self.model,
                                        verbose=True,
                                        stream=True,
                                        retry_cnt=5,
                                        )
        new_code = helper.extract_code(response)
        if (new_code == "PASS") or (new_code == ""):  # if no modification.
            print("Code review passed, no revision.\n\n")
            new_code = code

        self.direct_request_code = new_code

    def ask_LLM_to_sample_data(self, operation_code):

        sampling_data_requirement_str = '\n'.join(
            [f"{idx + 1}. {line}" for idx, line in enumerate(constants.sampling_data_requirement)])
        sampling_data_review_prompt = f"Your role: {constants.sampling_data_role} \n" + \
                                      f"Your task: {constants.sampling_task_prefix} \n\n" + \
                                      f"Requirement: \n{sampling_data_requirement_str} \n\n" + \
                                      f"The function code is: \n----------\n{code} \n----------\n\n"  #+ \
        # f"The requirements for the code is: \n----------\n{sampling_data_requirement_str} \n----------\n\n"

        print("LLM is reviewing the direct request code... \n")
        # print(f"review_prompt:\n{review_prompt}")
        response = helper.get_LLM_reply(prompt=sampling_data_review_prompt,
                                        system_role=constants.sampling_data_role,
                                        model=self.model,
                                        verbose=True,
                                        stream=True,
                                        retry_cnt=5,
                                        )
        code = helper.extract_code(response)
        return code
        # if (new_code == "PASS") or (new_code == ""):  # if no modification.
        #     print("Code review passed, no revision.\n\n")
        #     new_code = code

        # self.direct_request_code = new_code
