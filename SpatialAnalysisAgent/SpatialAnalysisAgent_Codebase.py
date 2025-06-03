import processing

Thematic_map_creation_sample = r'''
#Importing necessary modules
from qgis.core import QgsVectorLayer, QgsProject, QgsVectorLayerJoinInfo, QgsGraduatedSymbolRenderer, QgsMapSettings, QgsSymbol
from qgis.PyQt.QtCore import QVariant, QSize
from qgis.PyQt.QtGui import QImage, QPainter, QColor
    
def Thematic_Map_Creation():
    # Load the shapefile
    path_to_layer = 'D:\\Data\\Data.shp'
    layer = QgsVectorLayer(path_to_layer, "Population_Distribution", "ogr")
    QgsProject.instance().addMapLayer(layer)
    
    # Create a Graduated Symbol Renderer using 'Population' field
    symbol = QgsSymbol.defaultSymbol(layer.geometryType())
    renderer = QgsGraduatedSymbolRenderer('', [])
    renderer.setClassAttribute('Population')
    renderer.setMode(QgsGraduatedSymbolRenderer.Quantile)
    renderer.updateClasses(layer, 5)
    
    # Set the color ramp (green gradient)
    from qgis.core import QgsGradientColorRamp
    color1 = QColor(144, 238, 144) # light green
    color2 = QColor(0, 128, 0)     # dark green
    color_ramp = QgsGradientColorRamp(color1, color2)
    renderer.updateColorRamp(color_ramp)

    layer.setRenderer(renderer)
    layer.triggerRepaint()
    QgsProject.instance().addMapLayer(layer)

# Execute the function
Thematic_Map_Creation()

'''
Extract_by_attribute_example = r'''
import processing
from qgis.core import QgsProject

# Define the parameters
input_layer_path = 'path/to/input_layer.shp' # path to the input shapefile
output_path = 'path/to/save/selected_features.shp'  # Path to save the new shapefile

input_layer = QgsVectorLayer(input_layer_path, "Input Layer", "ogr")
QgsProject.instance().addMapLayer(input_layer)

#Define the parameters Example below:
field_name = 'Area'
operator = 4  # Select the appropriate operator based on the task. Many different operators are available: ['0': '=', '1': '!=', '2': '>', '3':'>=', '4':'<', '5':'<=', '6': 'begins with'  etc] 
value = '4'
parameters = {
    'INPUT': input_layer_path,
    'FIELD': field_name,
    'OPERATOR': operator,
    'VALUE': value,
    'OUTPUT': 'memory:'  # Use 'memory:' to create a temporary layer in memory
}
# Perform the extract by attribute operation
result = processing.run("native:extractbyattribute", parameters)

# Load the selected features as a new layer
output_layer = result['OUTPUT']
QgsProject.instance().addMapLayer(output_layer)

# Save the selected features to a new shapefile
options = QgsVectorFileWriter.SaveVectorOptions()
options.driverName = "ESRI Shapefile"
options.fileEncoding = "UTF-8"
transform_context = QgsProject.instance().transformContext()

# Write the vector layer to a new shapefile
QgsVectorFileWriter.writeAsVectorFormatV3(output_layer, output_path, transform_context, options)

#loading the saved_selecteded layer
selected_layer = QgsVectorLayer(output_path, "Selected Layer", "ogr")
QgsProject.instance().addMapLayer(selected_layer)

'''

Extract_by_attribute_parameters = r'''
INPUT - Layer to extract features from
FIELD - Filtering field of the layer
OPERATOR - Many different operators are available: ['0': '=', '1': '!=', '2': '>', '3':'>=', '4':'<', '5':'<=', '6': 'begins with'  etc] 
VALUE - Value to be evaluated (Optional)
OUTPUT - Default: [Create Temporary Layer], specify the output vector layer for matching features.
'''



attribute_join = r"""
```
from qgis.core import QgsVectorLayer, QgsProject, QgsVectorLayerJoinInfo, QgsVectorFileWriter
from PyQt5.QtCore import QVariant

# Load PA tract boundary shapefile
# tract_boundary_path = 'D:/LLM_Geo_QGIS/Case_Study1/data/tract_42_EPSG4326/tract_42_EPSG4326.shp'
tract_boundary_path = r"D:/SpatialAnalysisAgent/Data/SouthCarolinaCounties/SouthCarolinaCounties.shp"
tract_boundary = QgsVectorLayer(tract_boundary_path, "PA Tract Boundary", "ogr")
QgsProject.instance().addMapLayer(tract_boundary)

# Load CSV data file
# csv_data_path = 'D:/LLM_Geo_QGIS/Case_Study1/data/PA_Data.csv'
csv_data_path = 'D:/SpatialAnalysisAgent/Data/SouthCarolinaCounties/CensusData.csv'
csv_layer = QgsVectorLayer(f'file:///{csv_data_path}?delimiter=,', 'csv_data', 'delimitedtext')
QgsProject.instance().addMapLayer(csv_layer)

# Setup join information
join_info = QgsVectorLayerJoinInfo()
join_info.setJoinLayer(csv_layer)
join_info.setTargetFieldName("GEOID")
join_info.setJoinFieldName("GEOID")
join_info.setPrefix("")

# Apply join to the tract boundary layer
tract_boundary.addJoin(join_info)

# Export the joined layer as a new shapefile and load it into the project
# output_path = 'D:/LLM_Geo_QGIS/Case_Study1/data/joined_tract_boundary.shp'
output_path =os.path.join("D:/SpatialAnalysisAgent/Data/SouthCarolinaCounties/Temporary_Joined_Layer.shp")
options = QgsVectorFileWriter.SaveVectorOptions()
options.driverName = 'ESRI Shapefile'
options.fileEncoding = "UTF-8"

QgsVectorFileWriter.writeAsVectorFormatV3(tract_boundary, output_path, QgsProject.instance().transformContext(), options)

joined_layer = QgsVectorLayer(output_path, "Joined Tract Boundary", "ogr")
QgsProject.instance().addMapLayer(joined_layer)
```
"""
from qgis import processing
from qgis.core import *
# Algorithm name to search for
from qgis.core import QgsApplication




from qgis.core import QgsApplication

# Create a list to store the algorithm names
algorithm_names = []
for alg in QgsApplication.processingRegistry().algorithms():
    algorithm_names.append(alg.displayName())



# algorithm_IDs = []
# for alg in QgsApplication.processingRegistry().algorithms():
#     algorithm_names.append(alg.id())


algorithms_dict = {}
# Iterate through the algorithms in the processing registry
for alg in QgsApplication.processingRegistry().algorithms():
    algorithms_dict[alg.displayName()] = {'ID': alg.id()}
# print(algorithms_dict)

# Check if the selected tool exists in the dictionary
def documentation (selected_tool_ID,algorithm_names):
    if selected_tool_ID in algorithm_names:
        selected_tool_handbook = processing.algorithmHelp(selected_tool_ID)
        print(f"Handbook for {selected_tool_ID} ):\n{selected_tool_handbook}")
    else:
        print("Tool not found")


#************************************************************************************************************************


algorithms_dict = {}
# Iterate through the algorithms in the processing registry
for alg in QgsApplication.processingRegistry().algorithms():
    algorithms_dict[alg.displayName()] = {'ID': alg.id()}

#*******************************************************************************************************************

import os
import toml

def list_files_in_folder(folder_path):
    try:
        # List the .toml files in the directory
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.toml')]
        return files
    except FileNotFoundError:
        return "Folder not found."

def extract_tool_info(file_path):
    # Load the .toml file and extract the required information
    try:
        with open(file_path, 'r') as file:
            tool_data = toml.load(file)
            tool_name = tool_data.get('tool_name', 'Unknown')
            tool_description = tool_data.get('brief_description', 'No description provided')
            return tool_name, tool_description
    except Exception as e:
        return None, None



import os
# Get the directory of the current script
# current_script_dir = os.path.dirname(os.path.abspath(__file__))
current_script_dir = os.getcwd()
folder_path  = os.path.join(current_script_dir, 'Tools_Documentation', 'Customized_tools')

# folder_path = r"C:\Users\AKINBOYEWA TEMITOPE\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\SpatialAnalysisAgent-master\SpatialAnalysisAgent\Tools_Documentation\Customized_tools"
# def index_tools(folder_path=folder_path):
# tools_index = []
# files = list_files_in_folder(folder_path)
# print(f"Folder path: {folder_path}")
# print(f"Files found: {files}")
# for file in files:
#     tool_ID = os.path.splitext(file)[0]  # The file name without extension is the tool_ID
#     file_path = os.path.join(folder_path, file)
#
#     tool_name, tool_description = extract_tool_info(file_path)
#     print(f"Tool name: {tool_name}, Tool description: {tool_description}")
#
#     if tool_name and tool_description:
#         tools_index.append({
#             'tool_ID': tool_ID,
#             'tool_name': tool_name,
#             'tool_description': tool_description
#
#         })


# In SpatialAnalysisAgent_Codebase.py
def index_tools(folder_path):

    tools_index = []
    CustomTools_dict = {}
    tool_names_lists = []
    files = list_files_in_folder(folder_path)

    for file in files:
        tool_ID = os.path.splitext(file)[0]
        file_path = os.path.join(folder_path, file)

        tool_name, tool_description = extract_tool_info(file_path)

        tool_names_lists.append(tool_name)
        tools_index.append({
            'tool_ID': tool_ID,
            'tool_name': tool_name,
            'tool_description': tool_description

        })
        # Populate the separate dictionary with tool_ID and tool_name only
        # CustomTools_dict[tool_name] = tool_ID
        CustomTools_dict[tool_name] = {'ID': tool_ID}

    return tools_index, CustomTools_dict, tool_names_lists

current_script_dir = os.getcwd()
# folder_path  = os.path.join(current_script_dir, 'Tools_Documentation', 'Customized_tools')
#
# tools_index, CustomTools_dict, tool_names_lists = index_tools(folder_path)
# # print(CustomTools_dict)
# tool_name, tool_description = extract_tool_info(folder_path )
# print(tool_names_lists)



QGISalgorithm_names = ['Tessellate', 'Aspect', 'Assign projection', 'Buffer vectors', 'Build virtual raster', 'Build virtual vector', 'Clip raster by extent', 'Clip raster by mask layer', 'Clip vector by extent', 'Clip vector by mask layer', 'Color relief', 'Contour', 'Contour Polygons', 'Convert format', 'Dissolve', 'Execute SQL', 'Extract projection', 'Fill nodata', 'gdal2tiles', 'gdal2xyz', 'Raster information', 'Grid (Moving average)', 'Grid (Data metrics)', 'Grid (Inverse distance to a power)', 'Grid (IDW with nearest neighbor searching)', 'Grid (Linear)', 'Grid (Nearest neighbor)', 'Hillshade', 'Export to PostgreSQL (available connections)', 'Export to PostgreSQL (new connection)', 'Merge', 'Near black', 'Offset curve', 'Vector information', 'One side buffer', 'Build overviews (pyramids)', 'Pansharpening', 'PCT to RGB', 'Points along lines', 'Polygonize (raster to vector)', 'Proximity (raster distance)', 'Raster calculator', 'Rasterize (vector to raster)', 'Rasterize (overwrite with attribute)', 'Rasterize (overwrite with fixed value)', 'Rearrange bands', 'Retile', 'RGB to PCT', 'Roughness', 'Sieve', 'Slope', 'Tile index', 'Topographic Position Index (TPI)', 'Translate (convert format)', 'Terrain Ruggedness Index (TRI)', 'Viewshed', 'Warp (reproject)', 'Add autoincremental field', 'Add field to attributes table', 'Add unique value index field', 'Add X/Y fields to layer', 'Affine transform', 'Aggregate', 'Align rasters', 'Align raster', 'Align points to features', 'Geodesic line split at antimeridian', 'Array of offset (parallel) lines', 'Array of translated features', 'Aspect', 'Assign projection', 'Export atlas layout as image', 'Export atlas layout as PDF (multiple files)', 'Export atlas layout as PDF (single file)', 'Convert B3DM to GLTF', 'Batch Nominatim geocoder', 'Convert spatial bookmarks to layer', 'Boundary', 'Bounding boxes', 'Buffer', 'Variable width buffer (by M value)', 'Calculate expression', 'Overlap analysis', 'Create categorized renderer from styles', 'Cell stack percentile', 'Cell stack percentrank from raster layer', 'Cell stack percent rank from value', 'Cell statistics', 'Centroids', 'Clip', 'Collect geometries', 'Combine style databases', 'Concave hull', 'Conditional branch', 'Convert GPS data', 'Convert GPX feature type', 'Convert to curved geometries', 'Convex hull', 'Count points in polygon', 'Create attribute index', 'Create constant raster layer', 'Create directory', 'Create grid', 'Create points layer from table', 'Create random raster layer (binomial distribution)', 'Create random raster layer (exponential distribution)', 'Create random raster layer (gamma distribution)', 'Create random raster layer (geometric distribution)', 'Create random raster layer (negative binomial distribution)', 'Create random raster layer (normal distribution)', 'Create random raster layer (poisson distribution)', 'Create random raster layer (uniform distribution)', 'Create spatial index', 'DBSCAN clustering', 'Delaunay triangulation', 'Drop field(s)', 'Delete duplicate geometries', 'Delete holes', 'Densify by count', 'Densify by interval', 'Detect dataset changes', 'Difference', 'Dissolve', 'Download GPS data from device', 'Download vector tiles', 'Drop geometries', 'Drop M/Z values', 'DTM filter (slope-based)', 'Export layers to DXF', 'Equal to frequency', 'Explode HStore Field', 'Explode lines', 'Export layer(s) information', 'Export mesh edges', 'Export mesh faces', 'Export mesh on grid', 'Export mesh vertices', 'Export to spreadsheet', 'Extend lines', 'Create layer from extent', 'Extract binary field', 'Extract by attribute', 'Extract by expression', 'Extract/clip by extent', 'Extract by location', 'Extract labels', 'Extract M values', 'Extract specific vertices', 'Extract vertices', 'Extract within distance', 'Extract Z values', 'Field calculator', 'Download file', 'Fill NoData cells', 'Feature filter', 'Filter by geometry type', 'Filter layers by type', 'Filter vertices by M value', 'Filter vertices by Z value', 'Fix geometries', 'Flatten relationship', 'Force right-hand-rule', 'Fuzzify raster (gaussian membership)', 'Fuzzify raster (large membership)', 'Fuzzify raster (linear membership)', 'Fuzzify raster (near membership)', 'Fuzzify raster (power membership)', 'Fuzzify raster (small membership)', 'Generate points (pixel centroids) inside polygons', 'Geometry by expression', 'Convert GLTF to vector features', 'Greater than frequency', 'Highest position in raster stack', 'Hillshade', 'Join by lines (hub lines)', 'Export to PostgreSQL', 'Import geotagged photos', 'Interpolate point on line', 'Intersection', 'Join attributes by location', 'Join attributes by field value', 'Join attributes by location (summary)', 'Join attributes by nearest', 'Keep N biggest parts', 'K-means clustering', 'Convert layer to spatial bookmarks', 'Less than frequency', 'Line density', 'Line intersections', 'Line substring', 'Load layer into project', 'Lowest position in raster stack', 'Mean coordinate(s)', 'Merge lines', 'Merge vector layers', 'Export contours', 'Export cross section dataset values on lines from mesh', 'Export time series values from points of a mesh dataset', 'Rasterize mesh dataset', 'Minimum enclosing circles', 'Raster calculator', 'Raster calculator (virtual)', 'Difference (multiple)', 'Intersection (multiple)', 'Multipart to singleparts', 'Multi-ring buffer (constant distance)', 'Union (multiple)', 'Nearest neighbour analysis', 'Offset lines', 'Order by expression', 'Oriented minimum bounding box', 'Orthogonalize', 'Package layers', 'Raster pixels to points', 'Raster pixels to polygons', 'Point on surface', 'Points along geometry', 'Points to path', 'Create layer from point', 'Pole of inaccessibility', 'Extract layer extent', 'Polygonize', 'Polygons to lines', 'PostgreSQL execute SQL', 'Print layout map extent to layer', 'Export print layout as image', 'Export print layout as PDF', 'Project points (Cartesian)', 'Promote to multipart', 'Raise exception', 'Raise message', 'Raise warning', 'Random extract', 'Random points in extent', 'Random points in polygons', 'Random points on lines', 'Raster boolean AND', 'Raster calculator', 'Convert map to raster', 'Raster layer properties', 'Raster layer statistics', 'Raster layer unique values report', 'Raster layer zonal statistics', 'Raster boolean OR', 'Sample raster values', 'Raster surface volume', 'Reclassify by layer', 'Reclassify by table', 'Rectangles, ovals, diamonds', 'Refactor fields', 'Delete duplicates by attribute', 'Remove duplicate vertices', 'Remove null geometries', 'Rename layer', 'Rename field', 'Repair Shapefile', 'Reproject layer', 'Rescale raster', 'Retain fields', 'Reverse line direction', 'Rotate', 'Roundness', 'Round raster', 'Ruggedness index', 'Save vector features to file', 'Save log to file', 'Extract selected features', 'Segmentize by maximum angle', 'Segmentize by maximum distance', 'Select by location', 'Select within distance', 'Service area (from layer)', 'Service area (from point)', 'Set layer encoding', 'Set layer style', 'Set M value from raster', 'Set M value', 'Set project variable', 'Drape (set Z value from raster)', 'Set Z value', 'Shortest line between features', 'Shortest path (layer to point)', 'Shortest path (point to layer)', 'Shortest path (point to point)', 'Extract Shapefile encoding', 'Simplify', 'Single sided buffer', 'Slope', 'Smooth', 'Snap geometries to layer', 'Snap points to grid', 'SpatiaLite execute SQL', 'SpatiaLite execute SQL (registered DB)', 'Split features by character', 'Split lines by maximum length', 'Split vector layer', 'Split with lines', 'ST-DBSCAN clustering', 'String concatenation', 'Create style database from project', 'Subdivide', 'Sum line lengths', 'Swap X and Y coordinates', 'Symmetrical difference', 'Tapered buffers', 'Generate XYZ tiles (Directory)', 'Generate XYZ tiles (MBTiles)', 'TIN Mesh Creation', 'Transect', 'Transfer annotations from main layer', 'Translate', 'Truncate table', 'Union', 'Upload GPS data to device', 'Raster calculator (virtual)', 'Voronoi polygons', 'Create wedge buffers', 'Write Vector Tiles (MBTiles)', 'Write Vector Tiles (XYZ)', 'Zonal histogram', 'Zonal statistics (in place)', 'Zonal statistics', 'Assign projection', 'Boundary', 'Clip', 'Convert format', 'Create COPC', 'Density', 'Export to raster', 'Export to raster (using triangulation)', 'Export to vector', 'Filter', 'Information', 'Merge', 'Reproject', 'Thin (by skipping points)', 'Thin (by sampling radius)', 'Tile', 'Build virtual point cloud (VPC)', 'Advanced Python field calculator', 'Bar plot', 'Basic statistics for fields', 'Box plot', 'Check validity', 'Climb along line', 'Convert geometry type', 'Define Shapefile projection', 'Distance matrix', 'Distance to nearest hub (line to hub)', 'Distance to nearest hub (points)', 'Eliminate selected polygons', 'Execute SQL', 'Add geometry attributes', 'Find projection', 'Generate points (pixel centroids) along line', 'Heatmap (Kernel Density Estimation)', 'Hypsometric curves', 'IDW interpolation', 'Export to SpatiaLite', 'Concave hull (k-nearest neighbor)', 'Lines to polygons', 'List unique values', 'Mean and standard deviation plot', 'Minimum bounding geometry', 'Points displacement', 'Polar plot', 'PostgreSQL execute and load SQL', 'Random extract within subsets', 'Random points along line', 'Random points in layer bounds', 'Random points inside polygons', 'Random selection', 'Random selection within subsets', 'Raster calculator', 'Raster layer histogram', 'Rectangles, ovals, diamonds (variable)', 'Regular points', 'Relief', 'Vector layer scatterplot 3D', 'Select by attribute', 'Select by expression', 'Set style for raster layer', 'Set style for vector layer', 'Statistics by categories', 'Text to float', 'TIN interpolation', 'Topological coloring', 'Variable distance buffer', 'Vector layer histogram', 'Vector layer scatterplot']
QGISalgorithms_dict = {'Tessellate': {'ID': '3d:tessellate'}, 'Aspect': {'ID': 'native:aspect'}, 'Assign projection': {'ID': 'pdal:assignprojection'}, 'Buffer vectors': {'ID': 'gdal:buffervectors'}, 'Build virtual raster': {'ID': 'gdal:buildvirtualraster'}, 'Build virtual vector': {'ID': 'gdal:buildvirtualvector'}, 'Clip raster by extent': {'ID': 'gdal:cliprasterbyextent'}, 'Clip raster by mask layer': {'ID': 'gdal:cliprasterbymasklayer'}, 'Clip vector by extent': {'ID': 'gdal:clipvectorbyextent'}, 'Clip vector by mask layer': {'ID': 'gdal:clipvectorbypolygon'}, 'Color relief': {'ID': 'gdal:colorrelief'}, 'Contour': {'ID': 'gdal:contour'}, 'Contour Polygons': {'ID': 'gdal:contour_polygon'}, 'Convert format': {'ID': 'pdal:convertformat'}, 'Dissolve': {'ID': 'native:dissolve'}, 'Execute SQL': {'ID': 'qgis:executesql'}, 'Extract projection': {'ID': 'gdal:extractprojection'}, 'Fill nodata': {'ID': 'gdal:fillnodata'}, 'gdal2tiles': {'ID': 'gdal:gdal2tiles'}, 'gdal2xyz': {'ID': 'gdal:gdal2xyz'}, 'Raster information': {'ID': 'gdal:gdalinfo'}, 'Grid (Moving average)': {'ID': 'gdal:gridaverage'}, 'Grid (Data metrics)': {'ID': 'gdal:griddatametrics'}, 'Grid (Inverse distance to a power)': {'ID': 'gdal:gridinversedistance'}, 'Grid (IDW with nearest neighbor searching)': {'ID': 'gdal:gridinversedistancenearestneighbor'}, 'Grid (Linear)': {'ID': 'gdal:gridlinear'}, 'Grid (Nearest neighbor)': {'ID': 'gdal:gridnearestneighbor'}, 'Hillshade': {'ID': 'native:hillshade'}, 'Export to PostgreSQL (available connections)': {'ID': 'gdal:importvectorintopostgisdatabaseavailableconnections'}, 'Export to PostgreSQL (new connection)': {'ID': 'gdal:importvectorintopostgisdatabasenewconnection'}, 'Merge': {'ID': 'pdal:merge'}, 'Near black': {'ID': 'gdal:nearblack'}, 'Offset curve': {'ID': 'gdal:offsetcurve'}, 'Vector information': {'ID': 'gdal:ogrinfo'}, 'One side buffer': {'ID': 'gdal:onesidebuffer'}, 'Build overviews (pyramids)': {'ID': 'gdal:overviews'}, 'Pansharpening': {'ID': 'gdal:pansharp'}, 'PCT to RGB': {'ID': 'gdal:pcttorgb'}, 'Points along lines': {'ID': 'gdal:pointsalonglines'}, 'Polygonize (raster to vector)': {'ID': 'gdal:polygonize'}, 'Proximity (raster distance)': {'ID': 'gdal:proximity'}, 'Raster calculator': {'ID': 'qgis:rastercalculator'}, 'Rasterize (vector to raster)': {'ID': 'gdal:rasterize'}, 'Rasterize (overwrite with attribute)': {'ID': 'gdal:rasterize_over'}, 'Rasterize (overwrite with fixed value)': {'ID': 'gdal:rasterize_over_fixed_value'}, 'Rearrange bands': {'ID': 'gdal:rearrange_bands'}, 'Retile': {'ID': 'gdal:retile'}, 'RGB to PCT': {'ID': 'gdal:rgbtopct'}, 'Roughness': {'ID': 'gdal:roughness'}, 'Sieve': {'ID': 'gdal:sieve'}, 'Slope': {'ID': 'native:slope'}, 'Tile index': {'ID': 'gdal:tileindex'}, 'Topographic Position Index (TPI)': {'ID': 'gdal:tpitopographicpositionindex'}, 'Translate (convert format)': {'ID': 'gdal:translate'}, 'Terrain Ruggedness Index (TRI)': {'ID': 'gdal:triterrainruggednessindex'}, 'Viewshed': {'ID': 'gdal:viewshed'}, 'Warp (reproject)': {'ID': 'gdal:warpreproject'}, 'Add autoincremental field': {'ID': 'native:addautoincrementalfield'}, 'Add field to attributes table': {'ID': 'native:addfieldtoattributestable'}, 'Add unique value index field': {'ID': 'native:adduniquevalueindexfield'}, 'Add X/Y fields to layer': {'ID': 'native:addxyfields'}, 'Affine transform': {'ID': 'native:affinetransform'}, 'Aggregate': {'ID': 'native:aggregate'}, 'Align rasters': {'ID': 'native:alignrasters'}, 'Align raster': {'ID': 'native:alignsingleraster'}, 'Align points to features': {'ID': 'native:angletonearest'}, 'Geodesic line split at antimeridian': {'ID': 'native:antimeridiansplit'}, 'Array of offset (parallel) lines': {'ID': 'native:arrayoffsetlines'}, 'Array of translated features': {'ID': 'native:arraytranslatedfeatures'}, 'Export atlas layout as image': {'ID': 'native:atlaslayouttoimage'}, 'Export atlas layout as PDF (multiple files)': {'ID': 'native:atlaslayouttomultiplepdf'}, 'Export atlas layout as PDF (single file)': {'ID': 'native:atlaslayouttopdf'}, 'Convert B3DM to GLTF': {'ID': 'native:b3dmtogltf'}, 'Batch Nominatim geocoder': {'ID': 'native:batchnominatimgeocoder'}, 'Convert spatial bookmarks to layer': {'ID': 'native:bookmarkstolayer'}, 'Boundary': {'ID': 'pdal:boundary'}, 'Bounding boxes': {'ID': 'native:boundingboxes'}, 'Buffer': {'ID': 'native:buffer'}, 'Variable width buffer (by M value)': {'ID': 'native:bufferbym'}, 'Calculate expression': {'ID': 'native:calculateexpression'}, 'Overlap analysis': {'ID': 'native:calculatevectoroverlaps'}, 'Create categorized renderer from styles': {'ID': 'native:categorizeusingstyle'}, 'Cell stack percentile': {'ID': 'native:cellstackpercentile'}, 'Cell stack percentrank from raster layer': {'ID': 'native:cellstackpercentrankfromrasterlayer'}, 'Cell stack percent rank from value': {'ID': 'native:cellstackpercentrankfromvalue'}, 'Cell statistics': {'ID': 'native:cellstatistics'}, 'Centroids': {'ID': 'native:centroids'}, 'Clip': {'ID': 'pdal:clip'}, 'Collect geometries': {'ID': 'native:collect'}, 'Combine style databases': {'ID': 'native:combinestyles'}, 'Concave hull': {'ID': 'native:concavehull'}, 'Conditional branch': {'ID': 'native:condition'}, 'Convert GPS data': {'ID': 'native:convertgpsdata'}, 'Convert GPX feature type': {'ID': 'native:convertgpxfeaturetype'}, 'Convert to curved geometries': {'ID': 'native:converttocurves'}, 'Convex hull': {'ID': 'native:convexhull'}, 'Count points in polygon': {'ID': 'native:countpointsinpolygon'}, 'Create attribute index': {'ID': 'native:createattributeindex'}, 'Create constant raster layer': {'ID': 'native:createconstantrasterlayer'}, 'Create directory': {'ID': 'native:createdirectory'}, 'Create grid': {'ID': 'native:creategrid'}, 'Create points layer from table': {'ID': 'native:createpointslayerfromtable'}, 'Create random raster layer (binomial distribution)': {'ID': 'native:createrandombinomialrasterlayer'}, 'Create random raster layer (exponential distribution)': {'ID': 'native:createrandomexponentialrasterlayer'}, 'Create random raster layer (gamma distribution)': {'ID': 'native:createrandomgammarasterlayer'}, 'Create random raster layer (geometric distribution)': {'ID': 'native:createrandomgeometricrasterlayer'}, 'Create random raster layer (negative binomial distribution)': {'ID': 'native:createrandomnegativebinomialrasterlayer'}, 'Create random raster layer (normal distribution)': {'ID': 'native:createrandomnormalrasterlayer'}, 'Create random raster layer (poisson distribution)': {'ID': 'native:createrandompoissonrasterlayer'}, 'Create random raster layer (uniform distribution)': {'ID': 'native:createrandomuniformrasterlayer'}, 'Create spatial index': {'ID': 'native:createspatialindex'}, 'DBSCAN clustering': {'ID': 'native:dbscanclustering'}, 'Delaunay triangulation': {'ID': 'native:delaunaytriangulation'}, 'Drop field(s)': {'ID': 'native:deletecolumn'}, 'Delete duplicate geometries': {'ID': 'native:deleteduplicategeometries'}, 'Delete holes': {'ID': 'native:deleteholes'}, 'Densify by count': {'ID': 'native:densifygeometries'}, 'Densify by interval': {'ID': 'native:densifygeometriesgivenaninterval'}, 'Detect dataset changes': {'ID': 'native:detectvectorchanges'}, 'Difference': {'ID': 'native:difference'}, 'Download GPS data from device': {'ID': 'native:downloadgpsdata'}, 'Download vector tiles': {'ID': 'native:downloadvectortiles'}, 'Drop geometries': {'ID': 'native:dropgeometries'}, 'Drop M/Z values': {'ID': 'native:dropmzvalues'}, 'DTM filter (slope-based)': {'ID': 'native:dtmslopebasedfilter'}, 'Export layers to DXF': {'ID': 'native:dxfexport'}, 'Equal to frequency': {'ID': 'native:equaltofrequency'}, 'Explode HStore Field': {'ID': 'native:explodehstorefield'}, 'Explode lines': {'ID': 'native:explodelines'}, 'Export layer(s) information': {'ID': 'native:exportlayersinformation'}, 'Export mesh edges': {'ID': 'native:exportmeshedges'}, 'Export mesh faces': {'ID': 'native:exportmeshfaces'}, 'Export mesh on grid': {'ID': 'native:exportmeshongrid'}, 'Export mesh vertices': {'ID': 'native:exportmeshvertices'}, 'Export to spreadsheet': {'ID': 'native:exporttospreadsheet'}, 'Extend lines': {'ID': 'native:extendlines'}, 'Create layer from extent': {'ID': 'native:extenttolayer'}, 'Extract binary field': {'ID': 'native:extractbinary'}, 'Extract by attribute': {'ID': 'native:extractbyattribute'}, 'Extract by expression': {'ID': 'native:extractbyexpression'}, 'Extract/clip by extent': {'ID': 'native:extractbyextent'}, 'Extract by location': {'ID': 'native:extractbylocation'}, 'Extract labels': {'ID': 'native:extractlabels'}, 'Extract M values': {'ID': 'native:extractmvalues'}, 'Extract specific vertices': {'ID': 'native:extractspecificvertices'}, 'Extract vertices': {'ID': 'native:extractvertices'}, 'Extract within distance': {'ID': 'native:extractwithindistance'}, 'Extract Z values': {'ID': 'native:extractzvalues'}, 'Field calculator': {'ID': 'native:fieldcalculator'}, 'Download file': {'ID': 'native:filedownloader'}, 'Fill NoData cells': {'ID': 'native:fillnodata'}, 'Feature filter': {'ID': 'native:filter'}, 'Filter by geometry type': {'ID': 'native:filterbygeometry'}, 'Filter layers by type': {'ID': 'native:filterlayersbytype'}, 'Filter vertices by M value': {'ID': 'native:filterverticesbym'}, 'Filter vertices by Z value': {'ID': 'native:filterverticesbyz'}, 'Fix geometries': {'ID': 'native:fixgeometries'}, 'Flatten relationship': {'ID': 'native:flattenrelationships'}, 'Force right-hand-rule': {'ID': 'native:forcerhr'}, 'Fuzzify raster (gaussian membership)': {'ID': 'native:fuzzifyrastergaussianmembership'}, 'Fuzzify raster (large membership)': {'ID': 'native:fuzzifyrasterlargemembership'}, 'Fuzzify raster (linear membership)': {'ID': 'native:fuzzifyrasterlinearmembership'}, 'Fuzzify raster (near membership)': {'ID': 'native:fuzzifyrasternearmembership'}, 'Fuzzify raster (power membership)': {'ID': 'native:fuzzifyrasterpowermembership'}, 'Fuzzify raster (small membership)': {'ID': 'native:fuzzifyrastersmallmembership'}, 'Generate points (pixel centroids) inside polygons': {'ID': 'native:generatepointspixelcentroidsinsidepolygons'}, 'Geometry by expression': {'ID': 'native:geometrybyexpression'}, 'Convert GLTF to vector features': {'ID': 'native:gltftovector'}, 'Greater than frequency': {'ID': 'native:greaterthanfrequency'}, 'Highest position in raster stack': {'ID': 'native:highestpositioninrasterstack'}, 'Join by lines (hub lines)': {'ID': 'native:hublines'}, 'Export to PostgreSQL': {'ID': 'native:importintopostgis'}, 'Import geotagged photos': {'ID': 'native:importphotos'}, 'Interpolate point on line': {'ID': 'native:interpolatepoint'}, 'Intersection': {'ID': 'native:intersection'}, 'Join attributes by location': {'ID': 'native:joinattributesbylocation'}, 'Join attributes by field value': {'ID': 'native:joinattributestable'}, 'Join attributes by location (summary)': {'ID': 'native:joinbylocationsummary'}, 'Join attributes by nearest': {'ID': 'native:joinbynearest'}, 'Keep N biggest parts': {'ID': 'native:keepnbiggestparts'}, 'K-means clustering': {'ID': 'native:kmeansclustering'}, 'Convert layer to spatial bookmarks': {'ID': 'native:layertobookmarks'}, 'Less than frequency': {'ID': 'native:lessthanfrequency'}, 'Line density': {'ID': 'native:linedensity'}, 'Line intersections': {'ID': 'native:lineintersections'}, 'Line substring': {'ID': 'native:linesubstring'}, 'Load layer into project': {'ID': 'native:loadlayer'}, 'Lowest position in raster stack': {'ID': 'native:lowestpositioninrasterstack'}, 'Mean coordinate(s)': {'ID': 'native:meancoordinates'}, 'Merge lines': {'ID': 'native:mergelines'}, 'Merge vector layers': {'ID': 'native:mergevectorlayers'}, 'Export contours': {'ID': 'native:meshcontours'}, 'Export cross section dataset values on lines from mesh': {'ID': 'native:meshexportcrosssection'}, 'Export time series values from points of a mesh dataset': {'ID': 'native:meshexporttimeseries'}, 'Rasterize mesh dataset': {'ID': 'native:meshrasterize'}, 'Minimum enclosing circles': {'ID': 'native:minimumenclosingcircle'}, 'Raster calculator (virtual)': {'ID': 'native:virtualrastercalc'}, 'Difference (multiple)': {'ID': 'native:multidifference'}, 'Intersection (multiple)': {'ID': 'native:multiintersection'}, 'Multipart to singleparts': {'ID': 'native:multiparttosingleparts'}, 'Multi-ring buffer (constant distance)': {'ID': 'native:multiringconstantbuffer'}, 'Union (multiple)': {'ID': 'native:multiunion'}, 'Nearest neighbour analysis': {'ID': 'native:nearestneighbouranalysis'}, 'Offset lines': {'ID': 'native:offsetline'}, 'Order by expression': {'ID': 'native:orderbyexpression'}, 'Oriented minimum bounding box': {'ID': 'native:orientedminimumboundingbox'}, 'Orthogonalize': {'ID': 'native:orthogonalize'}, 'Package layers': {'ID': 'native:package'}, 'Raster pixels to points': {'ID': 'native:pixelstopoints'}, 'Raster pixels to polygons': {'ID': 'native:pixelstopolygons'}, 'Point on surface': {'ID': 'native:pointonsurface'}, 'Points along geometry': {'ID': 'native:pointsalonglines'}, 'Points to path': {'ID': 'native:pointstopath'}, 'Create layer from point': {'ID': 'native:pointtolayer'}, 'Pole of inaccessibility': {'ID': 'native:poleofinaccessibility'}, 'Extract layer extent': {'ID': 'native:polygonfromlayerextent'}, 'Polygonize': {'ID': 'native:polygonize'}, 'Polygons to lines': {'ID': 'native:polygonstolines'}, 'PostgreSQL execute SQL': {'ID': 'native:postgisexecutesql'}, 'Print layout map extent to layer': {'ID': 'native:printlayoutmapextenttolayer'}, 'Export print layout as image': {'ID': 'native:printlayouttoimage'}, 'Export print layout as PDF': {'ID': 'native:printlayouttopdf'}, 'Project points (Cartesian)': {'ID': 'native:projectpointcartesian'}, 'Promote to multipart': {'ID': 'native:promotetomulti'}, 'Raise exception': {'ID': 'native:raiseexception'}, 'Raise message': {'ID': 'native:raisemessage'}, 'Raise warning': {'ID': 'native:raisewarning'}, 'Random extract': {'ID': 'native:randomextract'}, 'Random points in extent': {'ID': 'native:randompointsinextent'}, 'Random points in polygons': {'ID': 'native:randompointsinpolygons'}, 'Random points on lines': {'ID': 'native:randompointsonlines'}, 'Raster boolean AND': {'ID': 'native:rasterbooleanand'}, 'Convert map to raster': {'ID': 'native:rasterize'}, 'Raster layer properties': {'ID': 'native:rasterlayerproperties'}, 'Raster layer statistics': {'ID': 'native:rasterlayerstatistics'}, 'Raster layer unique values report': {'ID': 'native:rasterlayeruniquevaluesreport'}, 'Raster layer zonal statistics': {'ID': 'native:rasterlayerzonalstats'}, 'Raster boolean OR': {'ID': 'native:rasterlogicalor'}, 'Sample raster values': {'ID': 'native:rastersampling'}, 'Raster surface volume': {'ID': 'native:rastersurfacevolume'}, 'Reclassify by layer': {'ID': 'native:reclassifybylayer'}, 'Reclassify by table': {'ID': 'native:reclassifybytable'}, 'Rectangles, ovals, diamonds': {'ID': 'native:rectanglesovalsdiamonds'}, 'Refactor fields': {'ID': 'native:refactorfields'}, 'Delete duplicates by attribute': {'ID': 'native:removeduplicatesbyattribute'}, 'Remove duplicate vertices': {'ID': 'native:removeduplicatevertices'}, 'Remove null geometries': {'ID': 'native:removenullgeometries'}, 'Rename layer': {'ID': 'native:renamelayer'}, 'Rename field': {'ID': 'native:renametablefield'}, 'Repair Shapefile': {'ID': 'native:repairshapefile'}, 'Reproject layer': {'ID': 'native:reprojectlayer'}, 'Rescale raster': {'ID': 'native:rescaleraster'}, 'Retain fields': {'ID': 'native:retainfields'}, 'Reverse line direction': {'ID': 'native:reverselinedirection'}, 'Rotate': {'ID': 'native:rotatefeatures'}, 'Roundness': {'ID': 'native:roundness'}, 'Round raster': {'ID': 'native:roundrastervalues'}, 'Ruggedness index': {'ID': 'native:ruggednessindex'}, 'Save vector features to file': {'ID': 'native:savefeatures'}, 'Save log to file': {'ID': 'native:savelog'}, 'Extract selected features': {'ID': 'native:saveselectedfeatures'}, 'Segmentize by maximum angle': {'ID': 'native:segmentizebymaxangle'}, 'Segmentize by maximum distance': {'ID': 'native:segmentizebymaxdistance'}, 'Select by location': {'ID': 'native:selectbylocation'}, 'Select within distance': {'ID': 'native:selectwithindistance'}, 'Service area (from layer)': {'ID': 'native:serviceareafromlayer'}, 'Service area (from point)': {'ID': 'native:serviceareafrompoint'}, 'Set layer encoding': {'ID': 'native:setlayerencoding'}, 'Set layer style': {'ID': 'native:setlayerstyle'}, 'Set M value from raster': {'ID': 'native:setmfromraster'}, 'Set M value': {'ID': 'native:setmvalue'}, 'Set project variable': {'ID': 'native:setprojectvariable'}, 'Drape (set Z value from raster)': {'ID': 'native:setzfromraster'}, 'Set Z value': {'ID': 'native:setzvalue'}, 'Shortest line between features': {'ID': 'native:shortestline'}, 'Shortest path (layer to point)': {'ID': 'native:shortestpathlayertopoint'}, 'Shortest path (point to layer)': {'ID': 'native:shortestpathpointtolayer'}, 'Shortest path (point to point)': {'ID': 'native:shortestpathpointtopoint'}, 'Extract Shapefile encoding': {'ID': 'native:shpencodinginfo'}, 'Simplify': {'ID': 'native:simplifygeometries'}, 'Single sided buffer': {'ID': 'native:singlesidedbuffer'}, 'Smooth': {'ID': 'native:smoothgeometry'}, 'Snap geometries to layer': {'ID': 'native:snapgeometries'}, 'Snap points to grid': {'ID': 'native:snappointstogrid'}, 'SpatiaLite execute SQL': {'ID': 'native:spatialiteexecutesql'}, 'SpatiaLite execute SQL (registered DB)': {'ID': 'native:spatialiteexecutesqlregistered'}, 'Split features by character': {'ID': 'native:splitfeaturesbycharacter'}, 'Split lines by maximum length': {'ID': 'native:splitlinesbylength'}, 'Split vector layer': {'ID': 'native:splitvectorlayer'}, 'Split with lines': {'ID': 'native:splitwithlines'}, 'ST-DBSCAN clustering': {'ID': 'native:stdbscanclustering'}, 'String concatenation': {'ID': 'native:stringconcatenation'}, 'Create style database from project': {'ID': 'native:stylefromproject'}, 'Subdivide': {'ID': 'native:subdivide'}, 'Sum line lengths': {'ID': 'native:sumlinelengths'}, 'Swap X and Y coordinates': {'ID': 'native:swapxy'}, 'Symmetrical difference': {'ID': 'native:symmetricaldifference'}, 'Tapered buffers': {'ID': 'native:taperedbuffer'}, 'Generate XYZ tiles (Directory)': {'ID': 'native:tilesxyzdirectory'}, 'Generate XYZ tiles (MBTiles)': {'ID': 'native:tilesxyzmbtiles'}, 'TIN Mesh Creation': {'ID': 'native:tinmeshcreation'}, 'Transect': {'ID': 'native:transect'}, 'Transfer annotations from main layer': {'ID': 'native:transferannotationsfrommain'}, 'Translate': {'ID': 'native:translategeometry'}, 'Truncate table': {'ID': 'native:truncatetable'}, 'Union': {'ID': 'native:union'}, 'Upload GPS data to device': {'ID': 'native:uploadgpsdata'}, 'Voronoi polygons': {'ID': 'native:voronoipolygons'}, 'Create wedge buffers': {'ID': 'native:wedgebuffers'}, 'Write Vector Tiles (MBTiles)': {'ID': 'native:writevectortiles_mbtiles'}, 'Write Vector Tiles (XYZ)': {'ID': 'native:writevectortiles_xyz'}, 'Zonal histogram': {'ID': 'native:zonalhistogram'}, 'Zonal statistics (in place)': {'ID': 'native:zonalstatistics'}, 'Zonal statistics': {'ID': 'native:zonalstatisticsfb'}, 'Create COPC': {'ID': 'pdal:createcopc'}, 'Density': {'ID': 'pdal:density'}, 'Export to raster': {'ID': 'pdal:exportraster'}, 'Export to raster (using triangulation)': {'ID': 'pdal:exportrastertin'}, 'Export to vector': {'ID': 'pdal:exportvector'}, 'Filter': {'ID': 'pdal:filter'}, 'Information': {'ID': 'pdal:info'}, 'Reproject': {'ID': 'pdal:reproject'}, 'Thin (by skipping points)': {'ID': 'pdal:thinbydecimate'}, 'Thin (by sampling radius)': {'ID': 'pdal:thinbyradius'}, 'Tile': {'ID': 'pdal:tile'}, 'Build virtual point cloud (VPC)': {'ID': 'pdal:virtualpointcloud'}, 'Advanced Python field calculator': {'ID': 'qgis:advancedpythonfieldcalculator'}, 'Bar plot': {'ID': 'qgis:barplot'}, 'Basic statistics for fields': {'ID': 'qgis:basicstatisticsforfields'}, 'Box plot': {'ID': 'qgis:boxplot'}, 'Check validity': {'ID': 'qgis:checkvalidity'}, 'Climb along line': {'ID': 'qgis:climbalongline'}, 'Convert geometry type': {'ID': 'qgis:convertgeometrytype'}, 'Define Shapefile projection': {'ID': 'qgis:definecurrentprojection'}, 'Distance matrix': {'ID': 'qgis:distancematrix'}, 'Distance to nearest hub (line to hub)': {'ID': 'qgis:distancetonearesthublinetohub'}, 'Distance to nearest hub (points)': {'ID': 'qgis:distancetonearesthubpoints'}, 'Eliminate selected polygons': {'ID': 'qgis:eliminateselectedpolygons'}, 'Add geometry attributes': {'ID': 'qgis:exportaddgeometrycolumns'}, 'Find projection': {'ID': 'qgis:findprojection'}, 'Generate points (pixel centroids) along line': {'ID': 'qgis:generatepointspixelcentroidsalongline'}, 'Heatmap (Kernel Density Estimation)': {'ID': 'qgis:heatmapkerneldensityestimation'}, 'Hypsometric curves': {'ID': 'qgis:hypsometriccurves'}, 'IDW interpolation': {'ID': 'qgis:idwinterpolation'}, 'Export to SpatiaLite': {'ID': 'qgis:importintospatialite'}, 'Concave hull (k-nearest neighbor)': {'ID': 'qgis:knearestconcavehull'}, 'Lines to polygons': {'ID': 'qgis:linestopolygons'}, 'List unique values': {'ID': 'qgis:listuniquevalues'}, 'Mean and standard deviation plot': {'ID': 'qgis:meanandstandarddeviationplot'}, 'Minimum bounding geometry': {'ID': 'qgis:minimumboundinggeometry'}, 'Points displacement': {'ID': 'qgis:pointsdisplacement'}, 'Polar plot': {'ID': 'qgis:polarplot'}, 'PostgreSQL execute and load SQL': {'ID': 'qgis:postgisexecuteandloadsql'}, 'Random extract within subsets': {'ID': 'qgis:randomextractwithinsubsets'}, 'Random points along line': {'ID': 'qgis:randompointsalongline'}, 'Random points in layer bounds': {'ID': 'qgis:randompointsinlayerbounds'}, 'Random points inside polygons': {'ID': 'qgis:randompointsinsidepolygons'}, 'Random selection': {'ID': 'qgis:randomselection'}, 'Random selection within subsets': {'ID': 'qgis:randomselectionwithinsubsets'}, 'Raster layer histogram': {'ID': 'qgis:rasterlayerhistogram'}, 'Rectangles, ovals, diamonds (variable)': {'ID': 'qgis:rectanglesovalsdiamondsvariable'}, 'Regular points': {'ID': 'qgis:regularpoints'}, 'Relief': {'ID': 'qgis:relief'}, 'Vector layer scatterplot 3D': {'ID': 'qgis:scatter3dplot'}, 'Select by attribute': {'ID': 'qgis:selectbyattribute'}, 'Select by expression': {'ID': 'qgis:selectbyexpression'}, 'Set style for raster layer': {'ID': 'qgis:setstyleforrasterlayer'}, 'Set style for vector layer': {'ID': 'qgis:setstyleforvectorlayer'}, 'Statistics by categories': {'ID': 'qgis:statisticsbycategories'}, 'Text to float': {'ID': 'qgis:texttofloat'}, 'TIN interpolation': {'ID': 'qgis:tininterpolation'}, 'Topological coloring': {'ID': 'qgis:topologicalcoloring'}, 'Variable distance buffer': {'ID': 'qgis:variabledistancebuffer'}, 'Vector layer histogram': {'ID': 'qgis:vectorlayerhistogram'}, 'Vector layer scatterplot': {'ID': 'qgis:vectorlayerscatterplot'}}


Customized_tool_names = ['Heatmap (Kernel Density Estimation) ',  'Inverse Distance Weighted interpolation', 'Thematic Map Creation']

# Customized_tool_names = [{'tool_ID': 'densitymap_kerneldensityestimation', 'tool_name': 'Heatmap (Kernel Density Estimation) ', 'tool_description': ' Creates a density (heatmap) raster of an input point vector layer using kernel density estimation. Heatmaps allow easy identification of hotspots and clustering of points.\n  The density is calculated based on the number of points in a location, with larger numbers of clustered points resulting in larger values.\n'}, {'tool_ID': 'idw_interpolation', 'tool_name': 'Inverse Distance Weighted interpolation', 'tool_description': ' Produces an interpolated surface by estimating values at unsampled locations based on nearby points and their values. It uses the inverse of the distance to weight nearby points more than distant ones.\n'}, {'tool_ID': 'test_1', 'tool_name': 'Testing this file', 'tool_description': 'This creates a map that shows one or more specific data themes or attributes. Examples of themes or attributes include population density, climatic patterns, economic activities, vegetation etc.\nAn example of a thematic map is a choropleth map which uses different colors or shades to represent data ranges.\n'}, {'tool_ID': 'thematic_map_creation', 'tool_name': 'Thematic Map Creation', 'tool_description': 'This creates a map that shows one or more specific data themes or attributes. Examples of themes or attributes include population density, climatic patterns, economic activities, vegetation etc.\nAn example of a thematic map is a choropleth map which uses different colors or shades to represent data ranges.\n'}]

Customized_tools_dict = {
    "Thematic Map Creation": {"ID": "thematic_map_creation"},
    "Density map (Kernel Density Estimation)":{"ID":"densitymap_kerneldensityestimation"},
    "Inverse Distance Weighted interpolation":{"ID":"idw_interpolation"}
}