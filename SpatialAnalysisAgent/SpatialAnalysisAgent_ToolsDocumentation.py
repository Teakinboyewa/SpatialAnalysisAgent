import os
import sys
import tomli as tomllib
import re


current_script_dir = os.path.dirname(os.path.abspath(__file__))
Tools_Documentation_dir = os.path.join(current_script_dir, 'Tools_Documentation')



def check_and_fix_toml_file(file_path):
    """
    Attempts to load and fix any common formatting errors in the TOML file.
    Specifically checks for the sections brief_description, full_description, and parameters,
    and ensures their contents are enclosed in triple quotes after removing any single or double quotes.
    """
    try:
        with open(file_path, "rb") as f:
            tomllib.load(f)  # Try loading the TOML file to check for errors
        return True  # No issues
    except tomllib.TOMLDecodeError as e:
        print(f"Attempting to fix file {file_path} due to error: {e}")

        # Try to fix common issues by ensuring all relevant section contents are wrapped in triple quotes
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Split content into lines
            lines = content.splitlines()

            # Check the last line for dangling quotes
            if lines and (lines[-1].endswith('"') or lines[-1].endswith("'")):
                print(f"Fixing unterminated string in {file_path}...")
                # # Replace the last line's single or double quote with triple quotes
                # lines[-1] = lines[-1].replace('"', '"""').replace("'", '"""')
                # Remove any trailing single or double quotes
                lines[-1] = lines[-1].rstrip('"').rstrip("'")
                # Add the triple quotes
                lines[-1] += '"""'

            # Fix other sections by ensuring triple quotes around the content
            fixed_content = fix_section_content("\n".join(lines))

            # # Fix each section by ensuring triple quotes around the content of the specific sections
            # fixed_content = fix_section_content(content)

            # Write the fixed content back to the file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(fixed_content)

            # Try loading the fixed file again
            with open(file_path, "rb") as f:
                tomllib.load(f)
            return True  # If this works, return True

        except Exception as fix_error:
            print(f"Failed to fix file {file_path}: {fix_error}")
            return False  # If fixing failed, return False


def fix_section_content(content):
    """
    Ensures the sections brief_description, full_description, and parameters
    have their contents stripped of single or double quotes, and encloses the content
    in triple quotes as a block.
    """
    # Define the sections to fix and their order
    sections_to_fix = ['parameters']
    other_sections = ['tool_ID', 'tool_name', 'brief_description', 'full_description', 'code_example']  # Sections to identify the end of each section

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
            if word not in ['OGR', 'GDAL'] and word.isupper():  # Check if the word is fully uppercase and not in the exclusions
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

                    # fixed_lines.append(f'{add_line_breaks_to_parameters("\n".join(section_content))}\n"""')
                    fixed_lines.append(add_line_breaks_to_parameters("\n".join(section_content)) + '\n"""')
                    # processed_line = add_line_breaks_to_parameters("\n".join(section_content))
                    # fixed_lines.append(f'{processed_line}\n"""')

                else:
                    fixed_lines.append('"""\n' + "\n".join(section_content) + '\n"""')
                    # fixed_lines.append(f'"""\n{"\n".join(section_content)}\n"""')
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
            # fixed_lines.append(f'{add_line_breaks_to_parameters(" ".join(section_content))}\n"""')
            fixed_lines.append(add_line_breaks_to_parameters("\n".join(section_content)) + '\n"""')
        else:
            # fixed_lines.append(f'"""\n{"\n".join(section_content)}\n"""')
            fixed_lines.append('"""\n' + "\n".join(section_content) + '\n"""')

    return "\n".join(fixed_lines)

# **************************************************
#
def tool_documentation_collection(tool_ID, tool_dir=Tools_Documentation_dir):
    # Initialize the tool file path variable
    tool_file = None

    # Walk through all subdirectories and files in the Tools_Documentation folder
    for root, dirs, files in os.walk(tool_dir):
        # Check if the tool file exists in any subdirectory
        for file in files:
            if file ==f'{tool_ID}.toml':
                tool_file = os.path.join(root, file)
                break # Stop once the file is found

        if tool_file:  # Exit the loop early if the file was found
            break

    # tool_file = os.path.join(tool_dir, f'{tool_ID}.toml')

    ## If the file is not found, return an empty string
    if not tool_file:
        return ""
    # #Check if the file exists
    # if not os.path.exists(tool_file):
    #     return ""

    with open(tool_file, "rb") as f:
        tool = tomllib.load(f)
    tool_parameter_str = tool['parameters']
    tool_total_str = str(tool)
    tool_total_str_lines = tool_total_str.strip().split('\n')
    # numbered_tool_total_str = ''
    # for idx, line in enumerate(tool_total_str_lines):
    #     line = line.strip(' ')
    #     numbered_tool_total_str += f"{idx + 1}. {line}\n"
    # Dynamically print each section's name and content
    output = ''
    for section, content in tool.items():
        # Automatically print section names in title case with content below it
        output += f"{section.replace('_', ' ').title()}:\n"
        output += f"{content.strip()}\n\n" if isinstance(content, str) else f"{content}\n\n"

    return output

def check_toml_file_for_errors(file_path):

    try:
        with open(file_path, "rb") as f:
            tomllib.load(f)
        # print(f"File {file_path} is free from errors.")
        return file_path
    except Exception as e:
        # If there's an error, add the file to problematic_files
        print(f"Error in file {file_path}: {e}")
        return False

# ************************************************************************************************************************

def fix_toml_file(file_path):
    """
    Attempts to fix common errors in the TOML file, such as unterminated strings at the end of the file.
    Specifically looks for single or double quotes at the end of the file and replaces them with triple quotes.
    """
    try:
        # Read the file contents
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.splitlines()

        for i in range(len(lines)):
            if 'code_example =' in lines[i]:
                # Check for single quotes or triple single quotes on the same line
                if lines[i].strip().endswith("'") or lines[i].strip().endswith("'''"):
                    lines[i] = lines[i].replace("'", '"')

                # If 'example_code =' is followed by single quotes or triple single quotes on the next line
                elif i + 1 < len(lines) and (lines[i + 1].strip() == "'" or lines[i + 1].strip() == "'''"):
                    lines[i + 1] = lines[i + 1].replace("'", '"')

        # Check the last line for dangling quotes and clean them first
        if lines and (lines[-1].endswith('"') or lines[-1].endswith("'")):
            print(f"Fixing unterminated string in {file_path}...")
            # Remove any trailing single or double quotes
            lines[-1] = lines[-1].rstrip('"').rstrip("'")
            # Add the triple quotes
            lines[-1] += '"""'

        # Fix other sections by ensuring triple quotes around the content
        fixed_content = fix_section_content("\n".join(lines))

        # Write the fixed content back to the file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(fixed_content)

        print(f"File {file_path} has been fixed.")

    except Exception as fix_error:
        print(f"Failed to fix file {file_path}: {fix_error}")

def fix_section_content(content):
    """
    Ensures the sections brief_description, full_description, and parameters
    have their contents stripped of single or double quotes, and encloses the content
    in triple quotes as a block.
    """
    # Define the sections to fix and their order
    sections_to_fix = ['parameters']
    other_sections = ['tool_ID', 'tool_name', 'brief_description', 'full_description', 'code_example']  # Sections to identify the end of each section

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
            if word not in ['OGR', 'GDAL'] and word.isupper():  # Check if the word is fully uppercase and not in the exclusions
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
                    # fixed_lines.append(f'{add_line_breaks_to_parameters("\n".join(section_content))}\n"""')
                    fixed_lines.append(add_line_breaks_to_parameters("\n".join(section_content)) + '\n"""')
                else:
                    # fixed_lines.append(f'"""\n{"\n".join(section_content)}\n"""')
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
            # fixed_lines.append(f'{add_line_breaks_to_parameters(" ".join(section_content))}\n"""')
            fixed_lines.append(add_line_breaks_to_parameters("\n".join(section_content)) + '\n"""')
        else:
            # fixed_lines.append(f'"""\n{"\n".join(section_content)}\n"""')
            fixed_lines.append('"""\n' + "\n".join(section_content) + '\n"""')

    return "\n".join(fixed_lines)

