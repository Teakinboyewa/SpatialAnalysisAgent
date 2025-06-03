import processing
from qgis.core import QgsVectorLayer, QgsProject
import os

def apply_affine_transformation():
    # Define the input and output paths
    input_path = 'D:/Case_Studies/Data/Penn_State_Buildings.shp'
    output_dir = 'C:/Users/AKINBOYEWA TEMITOPE/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/SpatialAnalysisAgent-master/Default_workspace'
    
    # Create a unique output file name based on its presence in the directory
    output_file_name = "Output.shp"
    output_path = os.path.join(output_dir, output_file_name)
    base_name = "Output"
    extension = ".shp"
    
    index = 1
    while os.path.exists(output_path):
        output_file_name = f"{base_name}_{index}{extension}"
        output_path = os.path.join(output_dir, output_file_name)
        index += 1

    # Set transformation processing parameters for the affine transformation
    transformation_params = {
        'input': input_path,
        'layer': -1,  # Apply to all layers
        'output': output_path,
        'xshift': 0,
        'yshift': 0,
        'zshift': 0,
        'xscale': 2.0,
        'yscale': 2.0,
        'zscale': 1.0,
        'zrotation': 30.0,
        'columns': ''
    }
    
    # Run the affine transformation using the grass7:v.transform tool
    result = processing.run("grass7:v.transform", transformation_params)
    
    # Load the resulting transformed layer into QGIS
    transformed_layer = QgsVectorLayer(result['output'], 'PennStateBuildingsTransformed', 'ogr')
    QgsProject.instance().addMapLayer(transformed_layer)

# Execute the function
apply_affine_transformation()

