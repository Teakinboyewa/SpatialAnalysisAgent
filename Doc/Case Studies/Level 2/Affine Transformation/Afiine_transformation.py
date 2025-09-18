from qgis.core import QgsVectorLayer, QgsProject
import os
import processing

def apply_affine_transformation():
    # Define input and output paths
    input_path = 'D:/Case_Studies/Data/Penn_State_Buildings.shp'
    output_directory = r'C:\Users\AKINBOYEWA TEMITOPE\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\SpatialAnalysisAgent-master\Default_workspace'
    output_filename = 'Output.shp'
    output_path = os.path.join(output_directory, output_filename)

    # Check if the output file already exists, and if so, append a number to the filename
    counter = 1
    while os.path.exists(output_path):
        output_filename = f'Penn_State_Buildings_Transformed_{counter}.shp'
        output_path = os.path.join(output_directory, output_filename)
        counter += 1

    # Load input shapefile as a vector layer
    input_layer = QgsVectorLayer(input_path, 'Penn_State_Buildings', 'ogr')

    # Define Parameters for the transformation
    parameters = {
        'INPUT': input_layer,
        'DELTA_X': 0,       # No shift on x-axis
        'DELTA_Y': 0,       # No shift on y-axis
        'DELTA_Z': 0,       # No shift on z-axis
        'DELTA_M': 0,       # No shift on m-values
        'SCALE_X': 2,       # Scale by 2 on the x-axis
        'SCALE_Y': 2,       # Scale by 2 on the y-axis
        'SCALE_Z': 1,       # No scaling on z-axis
        'SCALE_M': 1,       # No scaling on m-values
        'ROTATION_Z': 30,   # Rotate by 30 degrees counter-clockwise around the z-axis
        'OUTPUT': output_path
    }

    # Apply affine transformation using processing.run
    result = processing.run('native:affinetransform', parameters)

    # Load the transformed layer into QGIS
    transformed_layer = QgsVectorLayer(result['OUTPUT'], 'Transformed_Buildings', 'ogr')
    QgsProject.instance().addMapLayer(transformed_layer)

# Execute the function
apply_affine_transformation()