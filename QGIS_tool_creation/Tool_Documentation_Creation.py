import asyncio
import os
import re
import shutil
import sys
import tomllib

import toml
import tomli_w
from langchain_openai import ChatOpenAI

# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the directory to sys.path
if current_script_dir not in sys.path:
    sys.path.append(current_script_dir)

import QGIS_tool_creation_Helper as Helper

# toml_directory = r"D:\Onedrive\OneDrive - The Pennsylvania State University\PhD Work\SpatialAnalysisAgent_Reasearch\Plugin\GRASS_toml_withHTML_MD\Others"
toml_directory = r"C:\Users\AKINBOYEWA TEMITOPE\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\SpatialAnalysisAgent-master\SpatialAnalysisAgent\Tools_Documentation\QGIS_Tools"
def rename_vgrass_toml_filename(directory):
    files_in_directory = os.listdir(directory)
    toml_files = [file for file in files_in_directory if file.endswith(".toml")]
    for file_name in toml_files:
        new_file_name = f"grass7_{file_name}"
        os.rename(os.path.join(directory, file_name), os.path.join(directory, new_file_name))
        print(f"{file_name} renamed to {new_file_name}")
    renamed_files = os.listdir(directory)
    return renamed_files


# folder_path = r"D:\Onedrive\OneDrive - The Pennsylvania State University\PhD Work\SpatialAnalysisAgent_Reasearch\Plugin\GRASS_toml"
# folder_path = r"D:\Onedrive\OneDrive - The Pennsylvania State University\PhD Work\SpatialAnalysisAgent_Reasearch\Plugin\grass_toml_test"
# rename_vgrass_toml_filename(toml_directory)


def rename_vgrass_toml_tool_ID(folder_path):
    # Iterate over each TOML file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".toml"):
            file_path = os.path.join(folder_path, file_name)

            # Open the TOML file and read its contents
            with open(file_path, 'r') as file:
                data = toml.load(file)

            # Check if 'tool_ID' exists and modify it
            if 'tool_ID' in data:
                data['tool_ID'] = f"grass7_{data['tool_ID']}"

            # Write the modified content back to the TOML file
            with open(file_path, 'w') as file:
                toml.dump(data, file)

    print("tool_ID fields updated successfully.")


def tool_documentation_collection(tool_ID, tool_dir=toml_directory):
    tool_file = os.path.join(tool_dir, f'{tool_ID}.toml')

    # Check if the file exists
    if not os.path.exists(tool_file):
        return ""

    with open(tool_file, "rb") as f:
        tool = tomllib.load(f)
    # tool_parameter_str = tool['parameters']
    tool_parameter_str = tool['parameters']
    algorithm_id = tool['tool_ID']
    tool_synopsis = tool['synopsis']
    tool_flags = tool['flags']
    tool_document = tool['document']
    # algorithm_id = "qgis:buffer"

    tool_parameter_lines = tool_parameter_str.strip().split('\n')
    numbered_tool_parameter_str = ''
    for idx, line in enumerate(tool_parameter_lines):
        line = line.strip(' ')
        numbered_tool_parameter_str += f"{idx + 1}. {line}\n"

    return numbered_tool_parameter_str, tool_synopsis, algorithm_id, tool_flags, tool_document


def append_code_to_toml(tool_ID, code_sample, tool_dir=toml_directory):
    tool_file = os.path.join(tool_dir, f'{tool_ID}.toml')

    # Check if the file exists
    if not os.path.exists(tool_file):
        print(f"TOML file for tool {tool_ID} does not exist.")
        return

    # Load the existing TOML file
    with open(tool_file, "rb") as f:
        tool_data = tomllib.load(f)

    # # Replace the sample code in the code_example section if it exists, otherwise add it
    # tool_data['code_example'] = code_sample

    # Append the sample code to the code_example section
    if 'code_example' in tool_data:
        tool_data['code_example'] += f"\n{code_sample}"
    else:
        tool_data['code_example'] = code_sample

    # Write the updated content back to the file using tomli_w
    with open(tool_file, "wb") as f:
        tomli_w.dump(tool_data, f)


# def formatting_toml_file(tool_ID, tool_dir=Tools_Documentation_dir):
#     #file_path = r"D:\Onedrive\OneDrive - The Pennsylvania State University\PhD Work\SpatialAnalysisAgent_Reasearch\Plugin\toml\toml\gdal_slope.toml"
#     # Read the original file
#     tool_file = os.path.join(tool_dir, f'{tool_ID}.toml')
#     with open(tool_file, 'r', encoding = 'utf-8') as file:
#         # tool = tomllib.load(f)
#         # tool_parameter_str = tool['parameters']
#         toml_content = file.read()
#
#     # Function to replace single quotes with triple quotes for specific sections
#     def replace_single_with_triple_quotes(section_name, content):
#         # Pattern to match the section content that uses single quotes
#         pattern = rf'({section_name}\s*=\s*)\"(.*?)\"'
#         # Replace with triple quotes
#         return re.sub(pattern, r'\1"""\2"""', content, flags=re.DOTALL)
#
#     # Sections that need to be reformatted
#     sections_to_format = ['brief_description', 'full_description', 'parameters', 'code_example']
#
#     # Apply reformatting to each section
#     for section in sections_to_format:
#         toml_content = replace_single_with_triple_quotes(section, toml_content)
#
#
#     # Replace occurrences of `n\` (escaped newline in TOML) with proper multiline format
#     # Specifically, we will place strings in between triple double-quotes for basic multiline handling
#     updated_code_example = toml_content.replace('\\n', '\n')
#
#     # Write the updated content back into the file
#     # updated_file_path = r'D:\Onedrive\OneDrive - The Pennsylvania State University\PhD Work\SpatialAnalysisAgent_Reasearch\Plugin\toml\toml\gdal_slope.toml'
#     with open(tool_file, 'w', encoding='utf-8') as updated_file:
#         updated_file.write(updated_code_example)


def formatting_toml_file(tool_ID, tool_dir):
    #file_path = r"D:\Onedrive\OneDrive - The Pennsylvania State University\PhD Work\SpatialAnalysisAgent_Reasearch\Plugin\toml\toml\gdal_slope.toml"
    # Read the original file
    tool_file = os.path.join(tool_dir, f'{tool_ID}.toml')
    with open(tool_file, 'r', encoding='utf-8') as file:
        # tool = tomllib.load(f)
        # tool_parameter_str = tool['parameters']
        toml_content = file.read()

    # Function to replace single quotes with triple quotes for specific sections
    def replace_single_with_triple_quotes(section_name, content):
        # Pattern to match the section content that uses single quotes
        pattern = rf'({section_name}\s*=\s*)\"(.*?)\"'
        # Replace with triple quotes
        return re.sub(pattern, r'\1"""\2"""', content, flags=re.DOTALL)

    # Sections that need to be reformatted
    # sections_to_format = ['full_description', 'synopsis', 'parameters', 'flags', 'code_example']
    sections_to_format = ['synopsis', "flags", "code_example", "document"]

    # Apply reformatting to each section
    for section in sections_to_format:
        toml_content = replace_single_with_triple_quotes(section, toml_content)

    # Replace occurrences of `n\` (escaped newline in TOML) with proper multiline format
    # Specifically, we will place strings in between triple double-quotes for basic multiline handling
    updated_code_example = toml_content.replace('\\n', '\n')

    # Write the updated content back into the file
    # updated_file_path = r'D:\Onedrive\OneDrive - The Pennsylvania State University\PhD Work\SpatialAnalysisAgent_Reasearch\Plugin\toml\toml\gdal_slope.toml'
    with open(tool_file, 'w', encoding='utf-8') as updated_file:
        updated_file.write(updated_code_example)


#
# tool_ID = "grass7_i.albedo"
# formatting_toml_file(tool_ID)


def format_toml_files_in_directory(directory):
    for file_name in os.listdir(directory):
        if file_name.endswith(".toml"):
            tool_file_name = os.path.splitext(file_name)[0]

            formatting_toml_file(tool_file_name, directory)
            print(f"Processed file: {tool_file_name}")


#


def process_toml_files_in_directory(directory):
    for file_name in os.listdir(directory):
        if file_name.endswith(".toml"):
            tool_file_name = os.path.splitext(file_name)[0]
            algorithm_id, tool_synopsis, tool_flags, numbered_tool_parameter_str, tool_document = tool_documentation_collection(
                tool_ID=tool_file_name)
            Tooldoc_prompt_str = Helper.create_CodeSample_prompt(algorithm_id=algorithm_id, tool_synopsis=tool_synopsis,
                                                                 tool_flags=tool_flags, tool_doc=numbered_tool_parameter_str,
                                                                 tool_document=tool_document)

            # Generate code sample using OpenAI model
            OpenAI_key = Helper.get_OpenAI_key()

            model = ChatOpenAI(api_key=OpenAI_key, model="gpt-4o", temperature=1)
            CodeSample_prompt_str_chunks = asyncio.run(Helper.fetch_chunks(model, Tooldoc_prompt_str))
            LLM_reply_str = Helper.convert_chunks_to_str(chunks=CodeSample_prompt_str_chunks)

            # Extract Python code and clean it
            if 'python' in LLM_reply_str:
                sample_code = LLM_reply_str.split('python', 1)[1].strip()
            sample_code = sample_code.replace("\\n", "\n")

            # Append the sample code to the TOML file
            append_code_to_toml(tool_ID=tool_file_name, code_sample=sample_code)

            # Reformat the TOML file for multiline strings
            # formatting_toml_file(tool_ID =tool_file_name)
            # format_toml_files_in_directory(directory=toml_directory)

            print(f"Processed file: {tool_file_name}.toml")


def fix_toml_file(tool_dir, tool_ID):
    """
    Attempts to fix common errors in the TOML file, such as unterminated strings at the end of the file.
    Specifically looks for single or double quotes at the end of the file and replaces them with triple quotes.
    """
    try:
        tool_file = os.path.join(tool_dir, f'{tool_ID}.toml')
        with open(tool_file, 'r', encoding='utf-8') as f:
            content = f.read()
        # Split content into lines
        lines = content.splitlines()

        # Iterate over the lines to find and fix 'example_code = ' issues
        for i in range(len(lines)):
            # Check if the line contains 'example_code ='
            if 'code_example =' in lines[i]:
                # Check for single quotes or triple single quotes on the same line
                if lines[i].strip().endswith("'") or lines[i].strip().endswith("'''"):
                    # print(f"Fixing 'example_code =' in {file_path} (same line)...")
                    # Replace single quotes or triple single quotes with triple double quotes
                    lines[i] = lines[i].replace("'", '"')

                # If 'example_code =' is followed by single quotes or triple single quotes on the next line
                elif i + 1 < len(lines) and (lines[i + 1].strip() == "'" or lines[i + 1].strip() == "'''"):
                    # print(f"Fixing 'example_code =' in {file_path} (next line)...")
                    # Replace the following line's quotes with triple double quotes
                    lines[i + 1] = lines[i + 1].replace("'", '"')

        # Check the last line for dangling quotes and clean them first
        if lines and (lines[-1].endswith('"') or lines[-1].endswith("'")):
            print(f"Fixing unterminated string in {tool_file}...")
            # Remove any trailing single or double quotes
            lines[-1] = lines[-1].rstrip('"').rstrip("'")
            # Add the triple quotes
            lines[-1] += '"""'

        # Fix other sections by ensuring triple quotes around the content
        # fixed_content = fix_section_content("\n".join(lines))
        #
        # Join lines back to content
        fixed_content = "\n".join(lines)

        # Write the fixed content back to the file
        with open(tool_file, "w", encoding="utf-8") as f:
            f.write(fixed_content)
    #
    #     print(f"File {tool_file} has been fixed.")
    #
    except Exception as fix_error:
        print(f"Failed to fix file {tool_dir}: {fix_error}")


def fix_section_content(content):
    """
    Ensures the sections brief_description, full_description, and parameters
    have their contents stripped of single or double quotes, and encloses the content
    in triple quotes as a block.
    """
    # Define the sections to fix and their order
    sections_to_fix = ['parameters']
    other_sections = ['tool_ID', 'tool_name', 'brief_description',
                      'code_example']  # Sections to identify the end of each section

    lines = content.splitlines()
    fixed_lines = []
    current_section = None
    section_content = []

    def add_line_breaks_to_parameters(text):
        """
        Adds line breaks before each fully uppercase word (excluding 'OGR' and 'GDAL'),
        but skips adding a line break before the first parameter.
        """
        words = text.split()  # Split the content into words
        result = []
        first_param = True  # Flag to track the first parameter

        for word in words:
            if word not in ['OGR',
                            'GDAL'] and word.isupper():  # Check if the word is fully uppercase and not in the exclusions
                if first_param:
                    result.append(word)  # Don't add a newline before the first parameter
                    first_param = False
                else:
                    result.append("\n" + word)  # Add a newline before subsequent parameters
            else:
                result.append(word)  # No change for non-uppercase or excluded words

        return " ".join(result)  # Join the words back into a string

    for line in lines:
        # Check if the line matches one of the other sections (end of the current section)
        if any(re.match(rf'^{section}\s*=', line) for section in other_sections):
            # If we're leaving a section, add the content block with triple quotes
            if current_section:
                # For parameters section, apply the special line break logic
                if current_section == 'parameters':
                    # Ensure that the opening triple quotes are on the same line as 'parameters ='
                    #fixed_lines.append(f'{add_line_breaks_to_parameters("\n".join(section_content))}\n"""')
                    fixed_lines.append(add_line_breaks_to_parameters("\n".join(section_content)) + '\n"""')
                else:
                    #fixed_lines.append(f'"""\n{"\n".join(section_content)}\n"""')
                    fixed_lines.append('"""\n' + "\n".join(section_content) + '\n"""')
                section_content = []
            current_section = None

        # If we are inside a section we need to fix
        if current_section:
            content_inside = line.strip()

            # Remove any single or double quotes within the content
            content_inside = content_inside.replace('"', '').replace("'", '')

            # Collect the cleaned content (but don't add triple quotes yet)
            section_content.append(content_inside)

        else:
            # Match section names and start fixing them
            for section in sections_to_fix:
                pattern = rf'^{section}\s*=\s*["\']?(.*)["\']?$'
                match = re.match(pattern, line)

                if match:
                    content_inside = match.group(1).strip()

                    # Remove any single or double quotes within the content
                    content_inside = content_inside.replace('"', '').replace("'", '')

                    # Start collecting the section's content and keep triple quotes on the same line
                    fixed_lines.append(f'{section} = """')
                    section_content.append(content_inside)
                    current_section = section  # Mark this as the active section
                    break
            else:
                # If the line doesn't match any section, just append it as is
                fixed_lines.append(line)

    # If we exit the loop still inside a section, close it off
    if current_section:
        if current_section == 'parameters':
            # Ensure the opening triple quotes stay on the same line
            fixed_lines.append(f'{add_line_breaks_to_parameters(" ".join(section_content))}\n"""')
        else:
            #fixed_lines.append(f'"""\n{"\n".join(section_content)}\n"""')
            fixed_lines.append('"""\n' + "\n".join(section_content) + '\n"""')

    return "\n".join(fixed_lines)


def check_toml_files_for_errors(directory):
    problematic_files = []
    # Walk through all subdirectories and files in the given directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.toml'):  # Check only .toml files
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "rb") as f:
                        tomllib.load(f)  # Try loading the TOML file
                except Exception as e:
                    # If there's an error, add the file to problematic_files
                    print(f"Error in file {file}: {e}")
                    problematic_files.append(file_path)
                    # problematic_files.append(os.path.basename(file_path))
                    # problematic_file_ID.append(os.path.basename(file_path))

                    # # Destination path for the problematic file
                    # dest_path = os.path.join(error_folder, file)
                    #
                    # # Move (cut) the file to the error folder
                    # shutil.move(file_path, dest_path)
                    # print(f"Moved {file_path} to {dest_path}")

    return problematic_files




def check_toml_files_for_errors_and_move(directory, error_folder):
    problematic_files = []

    # Walk through all subdirectories and files in the given directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.toml'):  # Check only .toml files
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "rb") as f:
                        tomllib.load(f)  # Try loading the TOML file
                except Exception as e:
                    # If there's an error, add the file to problematic_files
                    print(f"Error in file {file}: {e}")
                    problematic_files.append(file_path)
                    # problematic_files.append(os.path.basename(file_path))
                    # problematic_file_ID.append(os.path.basename(file_path))

                    # Destination path for the problematic file
                    dest_path = os.path.join(error_folder, file)

                    # Move (cut) the file to the error folder
                    shutil.move(file_path, dest_path)
                    print(f"Moved {file_path} to {dest_path}")

    return problematic_files


import re
import os


def escape_backslashes_and_fix_newlines_in_toml_folder(folder_path):
    # Check if the folder path exists
    if not os.path.isdir(folder_path):
        print(f"Error: The folder {folder_path} does not exist.")
        return

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.toml'):  # Process only TOML files
            file_path = os.path.join(folder_path, filename)
            print(f'Processing {file_path}...')

            with open(file_path, 'r') as file:
                content = file.read()

            # Escape unescaped backslashes
            corrected_content = re.sub(r'(?<!\\)\\(?!\\)', r'\\\\', content)

            # Ensure there is a newline after each statement if not already present
            # corrected_content = re.sub(r'([^\n])(\s*)([^\n])', r'\1\n\3', content)

            # Save the corrected content back to the TOML file
            with open(file_path, 'w') as file:
                file.write(corrected_content)

            print(f'Backslashes and newlines corrected in {file_path}')


import os


def add_code_example_to_all_toml_files_in_folder(folder_path):
    def add_code_example_section(content):
        # Check if "code_example" section already exists
        if 'document' not in content:
            # Add the code_example section at the end of the file
            content += 'document = "" '
        return content

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.toml'):  # Process only TOML files
            file_path = os.path.join(folder_path, filename)
            print(f'Processing {file_path}...')

            with open(file_path, 'r') as file:
                content = file.read()

            # Add the code_example section if it does not exist
            updated_content = add_code_example_section(content)

            # Save the updated content back to the TOML file if changes were made
            if updated_content != content:
                with open(file_path, 'w') as file:
                    file.write(updated_content)
                print(f'"code_example" section added to {file_path}')
            else:
                print(f'"code_example" section already exists in {file_path}')



import os
import re

import os
import re

import os
import re

def remove_document_key_from_all_toml_files_in_folder(folder_path):
    def remove_document_key(content):
        # Use regex to find the 'document =' key and remove everything up to 'code_example ='
        updated_content = re.sub(r'document\s*=\s*".*?"\s*\n.*?(?=code_example\s*=)', '', content, flags=re.DOTALL)
        return updated_content

    # Check if the folder path exists
    if not os.path.isdir(folder_path):
        print(f"Error: The folder {folder_path} does not exist.")
        return

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.toml'):  # Process only TOML files
            file_path = os.path.join(folder_path, filename)
            print(f'Processing {file_path}...')

            with open(file_path, 'r') as file:
                content = file.read()

            # Remove the 'document =' key and its content up to 'code_example ='
            updated_content = remove_document_key(content)

            # Save the updated content back to the TOML file if changes were made
            if updated_content != content:
                with open(file_path, 'w') as file:
                    file.write(updated_content)
                print(f'"document" key removed from {file_path}')
            else:
                print(f'No "document" key found in {file_path}')



def format_first_parameters_in_all_toml_files_in_folder(folder_path):
    def format_first_parameters(content):
        # Use regex to find and format only the first 'parameters' key
        # This regex captures the first occurrence of the 'parameters' key and its content
        updated_content = re.sub(r'(parameters\s*=\s*".*?")', lambda m: m.group(1).replace('"', '"""'), content, count=1)
        return updated_content

    # Check if the folder path exists
    if not os.path.isdir(folder_path):
        print(f"Error: The folder {folder_path} does not exist.")
        return

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.toml'):  # Process only TOML files
            file_path = os.path.join(folder_path, filename)
            print(f'Processing {file_path}...')

            with open(file_path, 'r') as file:
                content = file.read()

            # Format only the first 'parameters' key
            updated_content = format_first_parameters(content)

            # Save the updated content back to the TOML file if changes were made
            if updated_content != content:
                with open(file_path, 'w') as file:
                    file.write(updated_content)
                print(f'First "parameters" formatted in {file_path}')
            else:
                print(f'No formatting needed for "parameters" in {file_path}')

# Usage example:











# check_toml_files_for_errors(toml_directory)

# import tomli
# import tomli_w
# import os
#
# # The fields that need to be checked and corrected
# fields_to_correct = ['synopsis', 'parameters', 'flags', 'code_example']
#
# def fix_quotes_in_field(content):
#     # Check if the field starts with six quotes (""""""), replace them with three (""")
#     if content.startswith('""""""'):
#         content = content.replace('""""""', '"""', 1)  # Replace only the first occurrence
#     return content
#
# def update_toml_file_quotes(tool_ID, tool_dir=toml_directory):
#     tool_file = os.path.join(tool_dir, f'{tool_ID}.toml')
#
#     # Check if the file exists
#     if not os.path.exists(tool_file):
#         print(f"TOML file for tool {tool_ID} does not exist.")
#         return
#
#     # Load the existing TOML file
#     with open(tool_file, "rb") as f:
#         tool_data = tomli.load(f)
#
#     # Check each field and fix the quotes if necessary
#     for field in fields_to_correct:
#         if field in tool_data:
#             tool_data[field] = fix_quotes_in_field(tool_data[field])
#
#     # Write the updated content back to the TOML file
#     temp_file = tool_file + '.tmp'  # Use a temp file for safer writing
#
#     with open(temp_file, "wb") as f:
#         tomli_w.dump(tool_data, f)
#
#     # Replace the original file with the temp file (atomic operation)
#     os.replace(temp_file, tool_file)
#
#     print(f"Updated quotes in {tool_ID}.toml")
#
# def process_toml_files_in_directory(directory):
#     for file_name in os.listdir(directory):
#         if file_name.endswith(".toml"):
#             tool_file_name = os.path.splitext(file_name)[0]
#             update_toml_file_quotes(tool_ID=tool_file_name)

# Run the processing function for all TOML files in the directory
# process_toml_files_in_directory(toml_directory)


# STEP1 [OPTIONAL]
# check_toml_files_for_errors(toml_directory)
#### Correct Unescaped '\' error

# escape_backslashes_and_fix_newlines_in_toml_folder(folder)


#### Run the processing function for all TOML files in the directory
# rename_vgrass_toml_filename(toml_directory)
# rename_vgrass_toml_tool_ID(toml_directory)


# STEP2 [OPTIONAL] --- #GENERATE SAMPLE_CODE

# add_code_example_to_all_toml_files_in_folder(toml_directory)
# process_toml_files_in_directory(toml_directory)

#STEP3 --- FORMAT MULTILINES

# remove_document_key_from_all_toml_files_in_folder(toml_directory)    #---OPTIONAL
# format_first_parameters_in_all_toml_files_in_folder(toml_directory)
# format_toml_files_in_directory(directory=toml_directory)


# STEP 4 --FORMAT THE TOML FILE ____CODE
# for file_name in os.listdir(toml_directory):
#     if file_name.endswith(".toml"):
#         tool_file_name = os.path.splitext(file_name)[0]
#         print(f"Processing TOML file: {tool_file_name}")
#         fix_toml_file(tool_dir=toml_directory,tool_ID=tool_file_name)
#         print(f"Fixed: {tool_file_name}")


#CHECK IF ANY ERROR EXIST
error_folder = r"C:\Users\AKINBOYEWA TEMITOPE\Downloads\landsat_toar"
check_toml_files_for_errors(error_folder)
# error_folder = r"D:\Onedrive\OneDrive - The Pennsylvania State University\PhD Work\SpatialAnalysisAgent_Reasearch\Plugin\GRASS_toml_withHTML_MD\NOT GOOD"

# check_toml_files_for_errors_and_move(toml_directory, error_folder)

# escape_backslashes_and_fix_newlines_in_toml_folder(toml_directory)