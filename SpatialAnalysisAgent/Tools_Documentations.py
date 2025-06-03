#--------------------------------Tool Documentation
documentation = {

#------Documentation for IDW Interpolation
'qgis:idwinterpolation': [r"""
        '''
        IDW interpolation (qgis:idwinterpolation)
        
        Generates an Inverse Distance Weighted (IDW) interpolation of a point vector layer.
        Sample points are weighted during interpolation such that the influence of one point relative to another declines with distance from the unknown point you want to create.
        
        ----------------
        Input parameters
        ----------------
        
-INTERPOLATION_DATA: Vector layer(s) and field(s) to use for the interpolation. The following elements are provided to compose the interpolation data:  Vector layer [vector: any] and Interpolation attribute (i.e Attribute to use in the interpolation)  [tablefield: numeric]. In the strig, the layer-field elements are separated by '::|::'. The sub-elements of the layer-field elements are separated by '::~::'
-DISTANCE_COEFFICIENT: Sets the distance coefficient for the interpolation. Minimum: 0.0, maximum: 100.0.
-EXTENT: Extent of the output raster layer.
        Available methods are:
            Calculate from layer…: uses extent of a layer loaded in the current project,
            Calculate from layout map…: uses extent of a layout map item in the active project,
            Calculate from bookmark…: uses extent of a saved bookmark,
            Use map canvas extent,
            Draw on canvas: click and drag a rectangle delimiting the area to take into account,
            Enter the coordinates as xmin, xmax, ymin, ymax
-PIXEL_SIZE: Output raster size
-OUTPUT: Raster layer of interpolated values.
            One of- Save to a Temporary File ('memory') or Save to File...     
'''
'''
        ----------------
        Sample Python Code
        ----------------
Python code format:
```
from qgis.core import QgsVectorLayer, QgsProject, QgsRasterLayer, QgsCoordinateReferenceSystem
from PyQt5.QtCore import QVariant
import processing
def IDW_Interpolation(input_layer_path, attribute):
    # Load the input vector layer
    input_layer = QgsVectorLayer(input_layer_path, "Input Layer", "ogr")

     #Determining the field index of the attribute
    fields = input_layer.fields()
    attribute_index = fields.indexOf(attribute) #obtaining the index of the the attribute to be interpolated

    #Define the parameters for IDW interpolation
    parameters = {
        'INTERPOLATION_DATA': f"{input_layer_path}::~::0::~::{attribute_index}::~::0",
        'DISTANCE_COEFFICIENT': 2,
        'EXTENT': input_layer.extent(),
        'PIXEL_SIZE': 0.1,
        'OUTPUT': 'memory'
    }
    # Run the IDW interpolation algorithm
    processing.run("qgis:idwinterpolation", parameters)
    # Add the output raster layer to the QGIS project
    output_raster_layer = QgsRasterLayer(parameters['OUTPUT'], "IDW Interpolation")
    QgsProject.instance().addMapLayer(output_raster_layer)
input_layer_path = "D:/Data/PA_Data_EPSG4326/PA_Data_EPSG4326.shp" #path to the input shapefile
attribute = "Coronary_h" # attribute to be interpolated
IDW_Interpolation(input_layer_path, attribute)
```
'''

"""

],

    # ***************************************************************************************************************************************************************************************

'qgis:tininterpolation': [r"""
        '''
        TIN interpolation (qgis:qgis:tininterpolation)

            Generates a Triangulated Irregular Network (TIN) interpolation of a point vector layer.
            With the TIN method you can create a surface formed by triangles of nearest neighbor points. To do this, circumcircles around selected sample points are created and their intersections are connected to a network of non overlapping and as compact as possible triangles. The resulting surfaces are not smooth.
            The algorithm creates both the raster layer of the interpolated values and the vector line layer with the triangulation boundaries.

        ----------------
        Input parameters
        ----------------

    -INTERPOLATION_DATA: Vector layer(s) and field(s) to use for the interpolation. The following elements are provided to compose the interpolation data:  Vector layer [vector: any] and Interpolation attribute (i.e Attribute to use in the interpolation)  [tablefield: numeric]. In the strig, the layer-field elements are separated by '::|::'. The sub-elements of the layer-field elements are separated by '::~::'
    -METHOD: Set the interpolation method to be used. One of:
                0 - Linear (Default)
                1 - Clough-Toucher (cubic)
    -EXTENT: Extent of the output raster layer.
            Available methods are:
                Calculate from layer…: uses extent of a layer loaded in the current project,
                Calculate from layout map…: uses extent of a layout map item in the active project,
                Calculate from bookmark…: uses extent of a saved bookmark,
                Use map canvas extent,
                Draw on canvas: click and drag a rectangle delimiting the area to take into account,
                Enter the coordinates as xmin, xmax, ymin, ymax
    -PIXEL_SIZE: Output raster size
    -OUTPUT: Raster layer of interpolated values. One of:
                - Save to a Temporary File ('memory')
                - Save to File...     
    '''
    
            ----------------
            Python code format
            ----------------
    Python code format:
```
    from PyQt5.QtCore import QVariant
    import processing
    def TIN_Interpolation(input_layer_path, attribute):
        # Load the input vector layer
        input_layer = QgsVectorLayer(input_layer_path, "Input Layer", "ogr")
    
         #Determining the field index of the attribute
        fields = input_layer.fields()
        attribute_index = fields.indexOf(attribute) #obtaining the index of the the attribute to be interpolated
    
        #Define the parameters for IDW interpolation
        parameters = {
            'INTERPOLATION_DATA': f"{input_layer_path}::~::0::~::{attribute_index}::~::0",
            'METHOD':0,
            'EXTENT': input_layer.extent(),
            'PIXEL_SIZE': 0.1,
            'OUTPUT': 'TEMPORARY_OUTPUT'
        }
        # Run the IDW interpolation algorithm
        result = processing.run("qgis:tininterpolation", parameters)
        # Add the output raster layer to the QGIS project
        output_raster_layer = QgsRasterLayer(result['OUTPUT'], "TIN Interpolation")
        QgsProject.instance().addMapLayer(output_raster_layer)
    input_layer_path = "D:/Data/PA.shp" #path to the input shapefile
    attribute = "Coronary_h" # attribute to be interpolated
    TIN_Interpolation(input_layer_path, attribute)
```
"""],

    #*****************************************************************************************************************************************************************************
'qgis:heatmapkerneldensityestimation': [r'''
    Heatmap (Kernel Density Estimation) (qgis:heatmapkerneldensityestimation)
    
    Creates a density (heatmap) raster of an input point vector layer using kernel density estimation. Heatmaps allow easy identification of hotspots and clustering of points.
    The density is calculated based on the number of points in a location, with larger numbers of clustered points resulting in larger values.
    
    
    
    ----------------
    Input parameters
    ----------------
    
    INPUT: Point layer
    
        Description: Point vector layer to use for the heatmap
        Parameter type:	QgsProcessingParameterFeatureSource
    
        Accepted data types:
            - str: layer ID
            - str: layer name
            - str: layer source
            - QgsProcessingFeatureSourceDefinition
            - QgsProperty
            - QgsVectorLayer
    
    RADIUS: Radius
        Description: Heatmap search radius (or kernel bandwidth) in map units. The radius specifies the distance around a point at which the influence of the point will be felt. Larger values result in greater smoothing, but smaller values may show finer details and variation in point density.
        Parameter type:	QgsProcessingParameterDistance
        Accepted data types:
            - int
            - float
            - QgsProperty
    
    RADIUS_FIELD: Radius from field (optional)
        Description: Sets the search radius for each feature from an attribute field in the input layer.
        Parameter type:	QgsProcessingParameterField
        Accepted data types:
            - str
            - QgsProperty
    
    PIXEL_SIZE: Output raster size
        Description: Pixel size of the output raster layer in layer units. In the GUI, the size can be specified by the number of rows (Number of rows) / columns (Number of columns) or the pixel size( Pixel Size X / Pixel Size Y). Increasing the number of rows or columns will decrease the cell size and increase the file size of the output raster. The values in Rows, Columns, Pixel Size X and Pixel Size Y will be updated simultaneously - doubling the number of rows will double the number of columns, and the cell size will be halved. The extent of the output raster will remain the same (approximately).
        Accepted data types: [number] 
            Default: 0.1
    
    WEIGHT_FIELD: Weight from field (Optional)
        Description: Allows input features to be weighted by an attribute field. This can be used to increase the influence certain features have on the resultant heatmap.   
        Accepted data types:
            [tablefield: numeric]
    
    KERNEL: Kernel shape
        Description: Controls the rate at which the influence of a point decreases as the distance from the point increases. Different kernels decay at different rates, so a triweight kernel gives features greater weight for distances closer to the point then the Epanechnikov kernel does. Consequently, triweight results in “sharper” hotspots and Epanechnikov results in “smoother” hotspots.
        Available values:
            - 0: Quartic
            - 1: Triangular
            - 2: Uniform
            - 3: Triweight
            - 4: Epanechnikov
    
        Accepted data types:
            [enumeration] - Default: 0
    
    DECAY: Decay ratio (Triangular kernels only) - Optional
        Description: Can be used with Triangular kernels to further control how heat from a feature decreases with distance from the feature.\n
                    A value of 0 (=minimum) indicates that the heat will be concentrated in the center of the given radius and completely extinguished at the edge. \n
                    A value of 0.5 indicates that pixels at the edge of the radius will be given half the heat as pixels at the center of the search radius. \n
                    A value of 1 means the heat is spread evenly over the whole search radius circle. (This is equivalent to the ‘Uniform’ kernel.). \n
                    A value greater than 1 indicates that the heat is higher towards the edge of the search radius than at the center.
        
        Accepted data types:
            	[number]
            	Default: 0.0
    
    OUTPUT_VALUE: Output value scaling
        Description: Allow to change the values of the output heatmap raster.
        
        Available values:
            - 0: Raw
            - 1: Scaled
    
        Accepted data types:
            [enumeration]
            Default: Raw
    
    OUTPUT: Heatmap
        Description: Specify the output raster layer with kernel density values. One of: -Save to a Temporary File ('memory'), -Save to file...
    
        Accepted data types:
            [raster]
            Default: [Save to Temporary File ('memory')]
            
     ----------------
    Sample Python Code
    ----------------
    
    
    
            '''],



#*********************************************************************************************************************************************************************

    'qgis:selectbyattribute':[r"""
        '''
    Select by attribute (qgis:selectbyattribute)
    This algorithm creates a selection in a vector layer. The criteria for selected features is defined based on the values of an attribute from the input layer.
    
        ----------------
    Input parameters
    ----------------
    
    INPUT: Input layer
    
        Parameter type:	QgsProcessingParameterVectorLayer
    
        Accepted data types:
            - str: layer ID
            - str: layer name
            - str: layer source
            - QgsProperty
            - QgsVectorLayer
    
    FIELD: Selection attribute
    
        Parameter type:	QgsProcessingParameterField
    
        Accepted data types:
            - str
            - QgsProperty
    
    OPERATOR: Operator
    
        Parameter type:	QgsProcessingParameterEnum
    
        Available values:
            - 0: =
            - 1: ≠
            - 2: >
            - 3: ≥
            - 4: <
            - 5: ≤
            - 6: begins with
            - 7: contains
            - 8: is null
            - 9: is not null
            - 10: does not contain
    
        Accepted data types:
            - int
            - str: as string representation of int, e.g. '1'
            - QgsProperty
    
    VALUE: Value
    
        Parameter type:	QgsProcessingParameterString
    
        Accepted data types:
            - str
            - QgsProperty
    
    METHOD: Modify current selection by
    
        Parameter type:	QgsProcessingParameterEnum
    
        Available values:
            - 0: creating new selection
            - 1: adding to current selection
            - 2: removing from current selection
            - 3: selecting within current selection
    
        Accepted data types:
            - int
            - str: as string representation of int, e.g. '1'
            - QgsProperty
            
    OUTPUT: Specify the output vector layer for matching features. One of:

            Create Temporary Layer ('memory')
            Save to File…
            Save to Geopackage…
            Save to Database Table…

    ------------------
    Sample Python Code
    ------------------ 
    ```
    import processing
    from qgis.core import QgsProject,QgsVectorLayer
    def select_by_attribute(input_layer_path):
        # Define the parameters
        input_layer = QgsVectorLayer(input_layer_path, "Input Layer", "ogr")
    
        # Define the parameters Example below:
        field_name = 'Population'
        operator = 4  # Select the appropriate operator based on the task. Many different operators are available: ['0': '=', '1': '!=', '2': '>', '3':'>=', '4':'<', '5':'<=', '6': 'begins with'  etc]
        value = '3000'
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
    input_layer_path = "D:/Data/PrevalenceData.shp"  # path to the input shapefile
    select_by_attribute(input_layer_path)
    ```
    '''
    """
    ],

#***************************************************************************************************************************************************************************************
'native:extractbyattribute': [r"""
        '''
    Select by attribute (qgis:extractbyattribute)
    This algorithm creates a selection in a vector layer. The criteria for selected features is defined based on the values of an attribute from the input layer.

    ----------------
    Input parameters
    ----------------

    INPUT: Input layer

        Parameter type:	QgsProcessingParameterVectorLayer

        Accepted data types:
            - str: layer ID
            - str: layer name
            - str: layer source
            - QgsProperty
            - QgsVectorLayer

    FIELD: Selection attribute

        Parameter type:	QgsProcessingParameterField

        Accepted data types:
            - str
            - QgsProperty

    OPERATOR: Operator

        Parameter type:	QgsProcessingParameterEnum

        Available values:
            - 0: =
            - 1: ≠
            - 2: >
            - 3: ≥
            - 4: <
            - 5: ≤
            - 6: begins with
            - 7: contains
            - 8: is null
            - 9: is not null
            - 10: does not contain

        Accepted data types:
            - int
            - str: as string representation of int, e.g. '1'
            - QgsProperty

    VALUE: Value

        Parameter type:	QgsProcessingParameterString

        Accepted data types:
            - str
            - QgsProperty

    METHOD: Modify current selection by

        Parameter type:	QgsProcessingParameterEnum

        Available values:
            - 0: creating new selection
            - 1: adding to current selection
            - 2: removing from current selection
            - 3: selecting within current selection

        Accepted data types:
            - int
            - str: as string representation of int, e.g. '1'
            - QgsProperty
    
    OUTPUT: Specify the output vector layer for matching features. One of:

            Create Temporary Layer ('memory')
            Save to File…
            Save to Geopackage…
            Save to Database Table…
   
    ------------------
    Sample Python Code
    ------------------ 
    ```
    import processing
    from qgis.core import QgsProject,QgsVectorLayer
    def extract_by_attribute(input_layer_path):
        # Define the parameters
        input_layer = QgsVectorLayer(input_layer_path, "Input Layer", "ogr")
    
        # Define the parameters Example below:
        field_name = 'Population'
        operator = 4  # Select the appropriate operator based on the task. Many different operators are available: ['0': '=', '1': '!=', '2': '>', '3':'>=', '4':'<', '5':'<=', '6': 'begins with'  etc]
        value = '3000'
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
    input_layer_path = "D:/Data/PrevalenceData.shp"  # path to the input shapefile
    extract_by_attribute(input_layer_path)
    ```
    '''
    """
    ],

#****************************************************************************************************************************
'native:extractbyexpression' : [r'''
    Extract by expression (native:extractbyexpression)

    This algorithm creates a new vector layer that only contains matching features from an input layer. The criteria for adding features to the resulting layer is based on a QGIS expression.
    
    For help with QGIS expression functions, see the inbuilt help for specific functions which is available in the expression builder.
    
    
    ----------------
    Input parameters
    ----------------
    
    INPUT: Input layer
    
        Parameter type:	QgsProcessingParameterFeatureSource
    
        Accepted data types:
            - str: layer ID
            - str: layer name
            - str: layer source
            - QgsProcessingFeatureSourceDefinition
            - QgsProperty
            - QgsVectorLayer
    
    EXPRESSION: Expression
    
        Parameter type:	QgsProcessingParameterExpression
    
        Accepted data types:
            - str
            - QgsProperty
    
    OUTPUT: Matching features
    
        Parameter type:	QgsProcessingParameterFeatureSink
    
        Accepted data types:
            - str: destination vector file, e.g. 'd:/test.shp'
            - str: 'memory:' to store result in temporary memory layer
            - str: using vector provider ID prefix and destination URI, e.g. 'postgres:…' to store result in PostGIS table
            - QgsProcessingOutputLayerDefinition
            - QgsProperty
    
    FAIL_OUTPUT: Non-matching
    
        Parameter type:	QgsProcessingParameterFeatureSink
    
        Accepted data types:
            - str: destination vector file, e.g. 'd:/test.shp'
            - str: 'memory:' to store result in temporary memory layer
            - str: using vector provider ID prefix and destination URI, e.g. 'postgres:…' to store result in PostGIS table
            - QgsProcessingOutputLayerDefinition
            - QgsProperty
    
    ----------------
    Outputs
    ----------------
    
    OUTPUT:  <QgsProcessingOutputVectorLayer>
        Matching features
    
    FAIL_OUTPUT:  <QgsProcessingOutputVectorLayer>
        Non-matching
        
    ------------------
    Sample Python Code
    ------------------ 
    ```
    import processing
    from qgis.core import QgsProject,QgsVectorLayer
    def extract_by_expression(input_layer_path):
        # Define the parameters
        input_layer = QgsVectorLayer(input_layer_path, "Input Layer", "ogr")
    
        # Define the parameters Example below:
        parameters = {
            'INPUT': input_layer_path,
            'EXPRESSION':' "STATEFP"  =\'42\' AND  "Obesity" <30',
            'OUTPUT': 'memory:Obesity<30'  # Use 'memory:' to create a temporary layer in memory
        }
        # Perform the extract by attribute operation
        result = processing.run("native:extractbyexpression", parameters)
        # Load the selected features as a new layer
        output_layer = result['OUTPUT']
        QgsProject.instance().addMapLayer(output_layer)
    input_layer_path = "D:/Data/Data.shp"  # path to the input shapefile
    extract_by_expression(input_layer_path)
    ```    
    '''],

    'native:extractbylocation': [r'''
    Extract by location (native:extractbylocation)

    This algorithm creates a new vector layer that only contains matching features from an input layer. The criteria for adding features to the resulting layer is defined based on the spatial relationship between each feature and the features in an additional layer.

    ----------------
    Input parameters
    ----------------

    INPUT: Extract features from

        Parameter type:	QgsProcessingParameterFeatureSource

        Accepted data types:
            - str: layer ID
            - str: layer name
            - str: layer source
            - QgsProcessingFeatureSourceDefinition
            - QgsProperty
            - QgsVectorLayer

    PREDICATE: Where the features (geometric predicate)

        Parameter type:	QgsProcessingParameterEnum

        Available values:
            - 0: intersect
            - 1: contain
            - 2: disjoint
            - 3: equal
            - 4: touch
            - 5: overlap
            - 6: are within
            - 7: cross

        Accepted data types:
            - int
            - str: as string representation of int, e.g. '1'
            - QgsProperty

    INTERSECT: By comparing to the features from

        Parameter type:	QgsProcessingParameterFeatureSource

        Accepted data types:
            - str: layer ID
            - str: layer name
            - str: layer source
            - QgsProcessingFeatureSourceDefinition
            - QgsProperty
            - QgsVectorLayer

    OUTPUT: Extracted (location)

        Parameter type:	QgsProcessingParameterFeatureSink

        Accepted data types:
            - str: destination vector file, e.g. 'd:/test.shp'
            - str: 'memory:' to store result in temporary memory layer
            - str: using vector provider ID prefix and destination URI, e.g. 'postgres:…' to store result in PostGIS table
            - QgsProcessingOutputLayerDefinition
            - QgsProperty

    ----------------
    Outputs
    ----------------

    OUTPUT:  <QgsProcessingOutputVectorLayer>


    ------------------
    Sample Python Code
    ------------------ 
    ```
import processing
from qgis.core import QgsProject,QgsVectorLayer
def extract_by_location():


    parameters = {
        'INPUT':'D:/Data/Data.shp', #Extract features from
        'PREDICATE':[1], #Available values: 0:intersect, 1:contain, 2: disjoint, 3: equal, 4:touch, 5:overlap, 6: are within, 7: cross
        'INTERSECT':'D:/Data/HW_Sites_EPSG4326/HW_Sites_EPSG4326.shp', #By comparing to the features from
        'OUTPUT': 'memory:'  # Use 'memory:' to create a temporary layer in memory
    }
    # Perform the extract by attribute operation
    result = processing.run("native:extractbylocation", parameters)
    # Load the selected features as a new layer
    output_layer = result['OUTPUT']
    QgsProject.instance().addMapLayer(output_layer)

extract_by_location()  
    ```
    '''],

    # ****************************************************************************************************************************
    'native:kmeansclustering': [r'''
    K-means clustering (native:kmeansclustering)

    Calculates the 2D distance based k-means cluster number for each input feature.
    K-means clustering aims to partition the features into k clusters in which each feature belongs to the cluster with the nearest mean. The mean point is represented by the barycenter of the clustered features.
    If input geometries are lines or polygons, the clustering is based on the centroid of the feature.

    ----------------
    Input parameters
    ----------------
    
    INPUT: Input layer
    
        Description: Layer to analyze
       
        Parameter type: [vector:any]
            
    CLUSTERS: Number of clusters
        Description: Number of clusters to create with the features
        Parameter type:	[number] - Default: 5
        
    
    FIELD_NAME: Cluster field name
        Description: Name of the field where the associated cluster number shall be stored
        Parameter type:	[string] - Default: ‘CLUSTER_ID’
    
    SIZE_FIELD_NAME: Cluster size field name
        Description: Name of the field with the count of features in the same cluster
        Parameter type:	[string] - Default: ‘CLUSTER_SIZE’
    
    OUTPUT: Radius from field (optional)
        Description: Specify the output vector layer for generated the clusters. One of: n\
                        -Create Temporary Layer (TEMPORARY_OUTPUT)
                        -Save to File…
                        -Save to Geopackage…
                        -Save to Database Table…

        Parameter type:	[vector: any] - Default:[Create temporary layer]
       
     ------------------
    Sample Python Code
    ------------------ 
    ```
    import processing
    from qgis.core import QgsProject,QgsVectorLayer
    def k_means_clustering():
        data_path = r"D:\Data\PA.shp"
        parameters = {
            'INPUT':data_path, #Extract features from
            'CLUSTERS':5,
            'FIELD_NAME':'CLUSTER_ID',
            'SIZE_FIELD_NAME':'CLUSTER_SIZE',
            'OUTPUT': 'TEMPORARY_OUTPUT'  # Use 'memory:' to create a temporary layer in memory
        }
        # Perform the kmeansclustering 
        result = processing.run("native:kmeansclustering", parameters)
        # Load the selected features as a new layer
       
        output_layer = result['OUTPUT']
        QgsProject.instance().addMapLayer(output_layer)
    
    
    
        # Apply categorized symbology to the output layer based on the cluster IDs
        field_name = 'CLUSTER_ID'
        categories = []
    
        # Define colors for each category
        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF'] #This can be changed
    
        for i in range(5):
            symbol = QgsSymbol.defaultSymbol(output_layer.geometryType())
            symbol.setColor(QColor(colors[i]))
            category = QgsRendererCategory(i, symbol, str(i))
            categories.append(category)
    
        renderer = QgsCategorizedSymbolRenderer(field_name, categories)
        output_layer.setRenderer(renderer)
        output_layer.triggerRepaint()
    k_means_clustering()
    ```    
    '''],

#****************************************************************************************************************************************************************************************
'Thematic_Map_Creation' : [r'''
    This creates a map that shows one or more specific data themes or attributes. Examples of themes or attributes include population density, climatic patterns, economic activities, vegetation etc.
    An example of a thematic map is a choropleth map which uses different colors or shades to represent data ranges.
    
    --------------------
     Python sample code
    --------------------
    ```
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
```
'''

],
"qgis:rastercalculator": ['''
Performing algebraic operations using raster layers.

Parameters:
        INPUT: List of input raster layers
        EXPRESSION:Raster-based expression that will be used to calculate the output raster layer.
            - In writing the expression, raster layers are referred by their name and the number of the band, e.g., `layer_name@1`. For instance, the first band from a layer named DEM will be referred as DEM@1.
            - Operators contains a number of calculation operators for pixels manipulation: 
                -Arithmetical: +, -, *, sqrt, abs, ln, … 
                -Trigonometric: sin, cos, tan, …
                -Comparison: =, !=, <, >=, …
                - Logical: IF, AND, OR, (, )
                - Statistical: min, max
--------------------
 sample code:
--------------------
def rastercalc():
    slope_path = '/vsimem/slope.tif'
    slope_layer = QgsRasterLayer(slope_path, 'Slope Layer')
   
    parameters = {
    'LAYERS':[slope_layer], # These are the list of all the layers loaded, these ae not the path
    'EXPRESSION':'"Slope Layer@1"<0.3',
    'EXTENT': None,
    'CELL_SIZE':None,
    'OUTPUT':'/vsimem/raster_calc_output.tif'}
    result = processing.run("native:rastercalc", parameters)
    
    raster_calc_layer = QgsRasterLayer(result['OUTPUT'], 'Raster Calculator Result')
    QgsProject.instance().addMapLayer(raster_calc_layer)
    
rastercalc()
    '''],

"native:rastercalc": ['''
Performing algebraic operations using raster layers.

Parameters:
        INPUT: List of input raster layers
        EXPRESSION:Raster-based expression that will be used to calculate the output raster layer.
            - In writing the expression, raster layers are referred by their name and the number of the band, e.g., `layer_name@1`. For instance, the first band from a layer named DEM will be referred as DEM@1.
            - Operators contains a number of calculation operators for pixels manipulation: 
                -Arithmetical: +, -, *, sqrt, abs, ln, … 
                -Trigonometric: sin, cos, tan, …
                -Comparison: =, !=, <, >=, …
                - Logical: IF, AND, OR, (, )
                - Statistical: min, max
--------------------
 sample code:
--------------------
def rastercalc():
    slope_path = '/vsimem/slope.tif'
    slope_layer = QgsRasterLayer(slope_path, 'Slope Layer')
   
    parameters = {
    'LAYERS':[slope_layer], # These are the list of all the layers loaded, these ae not the path
    'EXPRESSION':'"Slope Layer@1"<0.3',
    'EXTENT': None,
    'CELL_SIZE':None,
    'OUTPUT':'/vsimem/raster_calc_output.tif'}
    result = processing.run("native:rastercalc", parameters)
    
    raster_calc_layer = QgsRasterLayer(result['OUTPUT'], 'Raster Calculator Result')
    QgsProject.instance().addMapLayer(raster_calc_layer)
    
rastercalc()
    '''],

'gdal:rastercalculator': ['''
Performing algebraic operations using raster layers.

Parameters:
        INPUT: List of input raster layers
        EXPRESSION:Raster-based expression that will be used to calculate the output raster layer.
            - In writing the expression, raster layers are referred by their name and the number of the band, e.g., `layer_name@1`. For instance, the first band from a layer named DEM will be referred as DEM@1.
            - Operators contains a number of calculation operators for pixels manipulation: 
                -Arithmetical: +, -, *, sqrt, abs, ln, … 
                -Trigonometric: sin, cos, tan, …
                -Comparison: =, !=, <, >=, …
                - Logical: IF, AND, OR, (, )
                - Statistical: min, max
--------------------
 sample code:
--------------------
def rastercalc():
    slope_path = '/vsimem/slope.tif'
    slope_layer = QgsRasterLayer(slope_path, 'Slope Layer')
   
    parameters = {
    'LAYERS':[slope_layer], # These are the list of all the layers loaded, these ae not the path
    'EXPRESSION':'"Slope Layer@1"<0.3',
    'EXTENT': None,
    'CELL_SIZE':None,
    'OUTPUT':'/vsimem/raster_calc_output.tif'}
    result = processing.run("native:rastercalc", parameters)
    
    raster_calc_layer = QgsRasterLayer(result['OUTPUT'], 'Raster Calculator Result')
    QgsProject.instance().addMapLayer(raster_calc_layer)
    
rastercalc()
    '''],




# This function generates the shortest path from City A to City B using QGIS processing tool in Python

'native:shortestpathpointtopoint': ["""

Shortest path (point to point) (native:shortestpathpointtopoint)

This algorithm computes optimal (shortest or fastest) route between given start and end points.


----------------
Input parameters
----------------

INPUT: Vector layer representing network

	Description:	Line vector layer representing the network to be covered

STRATEGY: Path type to calculate

	Parameter type:	QgsProcessingParameterEnum
	Description: The type of path to calculate.	Available values:
		- 0: Shortest
		- 1: Fastest

DIRECTION_FIELD: Direction field

	Parameter type:	QgsProcessingParameterField

	Accepted data types:
		- str
		- QgsProperty

VALUE_FORWARD: Value for forward direction

	Parameter type:	QgsProcessingParameterString

	Accepted data types:
		- str
		- QgsProperty

VALUE_BACKWARD: Value for backward direction

	Parameter type:	QgsProcessingParameterString

	Accepted data types:
		- str
		- QgsProperty

VALUE_BOTH: Value for both directions

	Parameter type:	QgsProcessingParameterString

	Accepted data types:
		- str
		- QgsProperty

DEFAULT_DIRECTION: Default direction

	Parameter type:	QgsProcessingParameterEnum

	Available values:
		- 0: Forward direction
		- 1: Backward direction
		- 2: Both directions

	Accepted data types:
		- int
		- str: as string representation of int, e.g. '1'
		- QgsProperty

SPEED_FIELD: Speed field

	Parameter type:	QgsProcessingParameterField

	Accepted data types:
		- str
		- QgsProperty

DEFAULT_SPEED: Default speed (km/h)

	Parameter type:	QgsProcessingParameterNumber

	Accepted data types:
		- int
		- float
		- QgsProperty

TOLERANCE: Topology tolerance

	Parameter type:	QgsProcessingParameterDistance

	Accepted data types:
		- int
		- float
		- QgsProperty

START_POINT: Start point

	Parameter type:	QgsProcessingParameterPoint

	Accepted data types:
		- str: as an 'x,y' string, e.g. '1.5,10.1'
		- QgsPointXY
		- QgsProperty
		- QgsReferencedPointXY
		- QgsGeometry: centroid of geometry is used

END_POINT: End point

	Parameter type:	QgsProcessingParameterPoint

	Accepted data types:
		- str: as an 'x,y' string, e.g. '1.5,10.1'
		- QgsPointXY
		- QgsProperty
		- QgsReferencedPointXY
		- QgsGeometry: centroid of geometry is used

OUTPUT: Shortest path

	

--------------------
 Python code template
--------------------
```
def calculate_shortest_path():
    from qgis.core import QgsVectorLayer, QgsProject
    import processing

    # Define data paths
    network_points_path = 'D:/SpatialAnalysisAgent/LLMQGIS_Data/QGIS-Training-Data-master/QGIS-Training-Data-master/exercise_data/network_analysis/network_ponts.shp'
    network_lines_path = 'D:/SpatialAnalysisAgent/LLMQGIS_Data/QGIS-Training-Data-master/QGIS-Training-Data-master/exercise_data/network_analysis/network_lines.shp'

    # Load layers
    network_points_layer = QgsVectorLayer(network_points_path, 'Network Points', 'ogr')
    network_lines_layer = QgsVectorLayer(network_lines_path, 'Network Lines', 'ogr')

    # Add layers to the QGIS project
    QgsProject.instance().addMapLayer(network_points_layer)
    QgsProject.instance().addMapLayer(network_lines_layer)

    # Get field names in network points layer
    fields = network_points_layer.fields()

    # Assuming the points for City A and City B are identified by attributes, we need to find their coordinates
    field_names = [field.name() for field in fields]

    city_a_feature = None
    city_b_feature = None

    for feature in network_points_layer.getFeatures():
        if feature['name'] == 'State College':  # Assuming the attribute 'name' holds the city names
            city_a_feature = feature
        elif feature['name'] == 'Altoona':
            city_b_feature = feature

        if city_a_feature and city_b_feature:
            break

    city_a_coords = (city_a_feature.geometry().asPoint().x(), city_a_feature.geometry().asPoint().y())
    city_b_coords = (city_b_feature.geometry().asPoint().x(), city_b_feature.geometry().asPoint().y())

    # Use the QGIS processing tool to calculate the shortest path
    shortest_path_params = {
        'INPUT': network_lines_layer,
        'START_POINT': f'{city_a_coords[0]},{city_a_coords[1]}',
        'END_POINT': f'{city_b_coords[0]},{city_b_coords[1]}',
        'STRATEGY': 0,  # Select the appropriate strategy based on the task. Available values: 0:shortest,  1:fastest
        'OUTPUT': 'memory:'
    }

    shortest_path_layer = processing.run('native:shortestpathpointtopoint', shortest_path_params)['OUTPUT']

    # Add the shortest path result to the QGIS project
    QgsProject.instance().addMapLayer(shortest_path_layer)

calculate_shortest_path()
```
"""],

'native:fieldcalculator': ["""

Field calculator (native:fieldcalculator)

This algorithm computes a new vector layer with the same features of the input layer, but either overwriting an 
existing attribute or adding an additional attribute. The values of this field are computed from each feature using 
an expression, based on the properties and attributes of the feature. Note that if "Field name" is an existing field 
in the layer then all the rest of the field settings are ignored.


----------------
Input parameters
----------------

    INPUT: Input layer

        Parameter type:	QgsProcessingParameterFeatureSource
    
        Accepted data types:
            - str: layer ID
            - str: layer name
            - str: layer source
            - QgsProcessingFeatureSourceDefinition
            - QgsProperty
            - QgsVectorLayer
    
    FIELD_NAME: Field name
    
        Parameter type:	QgsProcessingParameterString
    
        Accepted data types:
            - str
            - QgsProperty
    
    FIELD_TYPE: Result field type
    
        Parameter type:	QgsProcessingParameterEnum
    
        Available values:
            - 0: Decimal (double)
            - 1: Integer (32 bit)
            - 2: Text (string)
            - 3: Date
            - 4: Time
            - 5: Date & Time
            - 6: Boolean
            - 7: Binary Object (BLOB)
            - 8: String List
            - 9: Integer List
            - 10: Decimal (double) List
    
        Accepted data types:
            - int
            - str: as string representation of int, e.g. '1'
            - QgsProperty
    
    FIELD_LENGTH: Result field length
    
        Parameter type:	QgsProcessingParameterNumber
    
        Accepted data types:
            - int
            - float
            - QgsProperty
    
    FIELD_PRECISION: Result field precision
    
        Parameter type:	QgsProcessingParameterNumber
    
        Accepted data types:
            - int
            - float
            - QgsProperty
    
    FORMULA: Formula
    
        Parameter type:	QgsProcessingParameterExpression
    
        Accepted data types:
            - str
            - QgsProperty
    
    OUTPUT: Calculated
    
        Parameter type:	QgsProcessingParameterFeatureSink
    
        Accepted data types:
            - str: destination vector file, e.g. 'd:/test.shp'
            - str: 'memory:' to store result in temporary memory layer
            - str: using vector provider ID prefix and destination URI, e.g. 'postgres:…' to store result in PostGIS table
            - QgsProcessingOutputLayerDefinition
            - QgsProperty

--------------------
 Python code template
--------------------
from qgis.core import QgsProject,QgsVectorLayer
def fieldcalculator():
    # Paths to input layers
    csv_path = "D:/SpatialAnalysisAgent/Data/SouthCarolinaCounties/CensusData.csv"
    csv_uri = f"file:///{csv_path}?delimiter=,"
    census_layer = QgsVectorLayer(csv_uri, 'CensusData', 'delimitedtext')
    # Add CSV layer to the project
    QgsProject.instance().addMapLayer(census_layer)
    # Define the parameters
    # input_layer = QgsVectorLayer(census_layer, "Input Layer", "ogr")

    # Define the parameters Example below:
    parameters = {
        'INPUT': census_layer,
        'FIELD_NAME':'PopChange',
        'FIELD_TYPE':0, # Available values- 0: Decimal (double), 1: Integer (32 bit), 2: Text (string), 3: Date, 4: Time, 5: Date & Time, 6: Boolean, 7: Binary Object (BLOB), 8: String List, 9: Integer List, 10: Decimal (double) List
        'FIELD_LENGTH':0,
        'FIELD_PRECISION':0,
        # 'NEW_FIELD': True,
        'FORMULA':' ("TPOP2008" - "TPOP2005")/ "TPOP2005" ',
        'OUTPUT':'memory:PopChange'}
    # Perform the extract by attribute operation
    result = processing.run("native:fieldcalculator", parameters)
    # Load the selected features as a new layer
    output_layer = result['OUTPUT']
    QgsProject.instance().addMapLayer(output_layer)
fieldcalculator()
"""],


'native:countpointsinpolygon': ["""
Count points in polygon (native:countpointsinpolygon)

Counts point features located within polygon features. This algorithm takes a points layer and a polygon layer and counts the number of points from the first one in each polygons of the second one. A new polygons layer is generated, with the exact same content as the input polygons layer, but containing an additional field with the points count corresponding to each polygon.

An optional weight field can be used to assign weights to each point. If set, the count generated will be the sum of the weight field for each point contained by the polygon.

Alternatively, a unique class field can be specified. If set, points are classified based on the selected attribute, and if several points with the same attribute value are within the polygon, only one of them is counted. The final count of the point in a polygon is, therefore, the count of different classes that are found in it.

Both the weight field and unique class field cannot be specified. If they are, the weight field will take precedence and the unique class field will be ignored.


----------------
Input parameters
----------------

    POLYGONS: Polygons
    
        Parameter type:	QgsProcessingParameterFeatureSource
    
        Accepted data types:
            - str: layer ID
            - str: layer name
            - str: layer source
            - QgsProcessingFeatureSourceDefinition
            - QgsProperty
            - QgsVectorLayer
    
    POINTS: Points
    
        Parameter type:	QgsProcessingParameterFeatureSource
    
        Accepted data types:
            - str: layer ID
            - str: layer name
            - str: layer source
            - QgsProcessingFeatureSourceDefinition
            - QgsProperty
            - QgsVectorLayer
    
    WEIGHT: Weight field
    
        Parameter type:	QgsProcessingParameterField
    
        Accepted data types:
            - str
            - QgsProperty
    
    CLASSFIELD: Class field
    
        Parameter type:	QgsProcessingParameterField
    
        Accepted data types:
            - str
            - QgsProperty
    
    FIELD: Count field name
    
        Parameter type:	QgsProcessingParameterString
    
        Accepted data types:
            - str
            - QgsProperty
    
    OUTPUT: Count
    
        Parameter type:	QgsProcessingParameterFeatureSink
    
        Accepted data types:
            - str: destination vector file, e.g. 'd:/test.shp'
            - str: 'memory:' to store result in temporary memory layer
            - str: using vector provider ID prefix and destination URI, e.g. 'postgres:…' to store result in PostGIS table
            - QgsProcessingOutputLayerDefinition
            - QgsProperty

--------------------
 Python code template
--------------------
```
import processing
from qgis.core import QgsProject,QgsVectorLayer
def CountPointsinPolygon():
    # Paths to input layers
    point_path = "D:/SpatialAnalysisAgent/Data/SpatialQueryData/DamagedHouses.shp"
    polygon_path = "D:/SpatialAnalysisAgent/Data/SpatialQueryData/SAF_SpecialStudyZone.shp"
    point_layer = QgsVectorLayer(point_path, 'PointLayer', 'ogr')   
    polygon_layer = QgsVectorLayer(polygon_path, 'PolygonLayer', 'ogr')

    # Define the parameters Example below:
    parameters = {
        'POLYGONS': polygon_layer,
        'POINTS': point_layer,
        'WEIGHT':'',
        'CLASSFIELD':'',
        'FIELD':'NUMPOINTS',
        'OUTPUT':'TEMPORARY_OUTPUT'}
    # Perform the extract by attribute operation
    result = processing.run("native:countpointsinpolygon", parameters)
    # Load the selected features as a new layer
    output_layer = result['OUTPUT']
    QgsProject.instance().addMapLayer(output_layer)
CountPointsinPolygon()
```
"""],

"gdal:buffervectors" : ["""
--------------------
 Python code template
--------------------

import processing
from qgis.core import QgsProject,QgsVectorLayer
def buffervectors():
    # Paths to input layers
    data_path = "D:/SpatialAnalysisAgent/Data/Exercise12Data/flood2015may.shp"
    data_layer = QgsVectorLayer(data_path, 'dataLayer', 'ogr')

    # Define the parameters Example below:
    parameters = {
        'INPUT': data_layer,
        'GEOMETRY':'geometry',
        'DISTANCE':8,
        'FIELD':'',
        'DISSOLVE':False,
        'EXPLODE_COLLECTIONS':False,
        'OPTIONS':'',
        'OUTPUT':'TEMPORARY_OUTPUT'}
    # Perform the extract by attribute operation
    result = processing.run("gdal:buffervectors", parameters)
    # Load the selected features as a new layer
    output_layer = QgsVectorLayer(result['OUTPUT'],"buffer", 'ogr')
    QgsProject.instance().addMapLayer(output_layer)
buffervectors()

"""],

'native:buffer': ["""
Buffer (native:buffer)

This algorithm computes a buffer area for all the features in an input layer, using a fixed or dynamic distance.

The segments parameter controls the number of line segments to use to approximate a quarter circle when creating rounded offsets.

The end cap style parameter controls how line endings are handled in the buffer.

The join style parameter specifies whether round, miter or beveled joins should be used when offsetting corners in a line.

The miter limit parameter is only applicable for miter join styles, and controls the maximum distance from the offset curve to use when creating a mitered join.

--------------------
 Python code template
--------------------
import processing
from qgis.core import QgsProject,QgsVectorLayer
def buffer():
    # Paths to input layers
    data_path = "D:/SpatialAnalysisAgent/Data/Point_data.shp"
    data_layer = QgsVectorLayer(data_path, 'dataLayer', 'ogr')

    # Define the parameters Example below:
    parameters = {
        'INPUT': data_layer,
        'DISTANCE':10,
        'SEGMENTS':5,
        'END_CAP_STYLE':0, # Available values - 0:Round, 1:Flat, 2:Square
        'JOIN_STYLE':0, #Available values-  0: Round, 1: Miter, 2: Bevel
        'MITER_LIMIT':2,
        'DISSOLVE':False,
        'SEPARATE_DISJOINT':False,
        'OUTPUT':'memory: buffered'}
    # Perform the extract by attribute operation
    result = processing.run("native:buffer", parameters)
    # Load the selected features as a new layer
    output_layer = result['OUTPUT']
    QgsProject.instance().addMapLayer(output_layer)
buffer()
"""],

"native:joinattributesbylocation":["""

Join attributes by location (native:joinattributesbylocation)

Join attributes from one vector layer to another by location.

This algorithm takes an input vector layer and creates a new vector layer that is an extended version of the input one, with additional attributes in its attribute table.

The additional attributes and their values are taken from a second vector layer. A spatial criteria is applied to select the values from the second layer that are added to each feature from the first layer in the resulting one.



--------------------
 Python code template
--------------------
import processing
from qgis.core import QgsProject,QgsVectorLayer
def joinattributesbylocation():

    # Define the parameters Example below:
    parameters = {
        'INPUT':'D:/SpatialAnalysisAgent/Data/flooddata.shp',
        'PREDICATE':[5], #Available values - 0: intersect, 1: contain, 2: equal, 3: touch, 4: overlap, 5: are within, 6: cross
        'JOIN':'D:/SpatialAnalysisAgent/Data/roads.shp',
        'JOIN_FIELDS':[], # Available values- 0: Create separate feature for each matching feature (one-to-many), 1: Take attributes of the first matching feature only (one-to-one), 2: Take attributes of the feature with largest overlap only (one-to-one)
        'METHOD':0,
        'DISCARD_NONMATCHING':True,
        'PREFIX':'',
        'OUTPUT':'memory:joined attributes'}
    # Perform the extract by attribute operation
    result = processing.run("native:joinattributesbylocation", parameters)
    # Load the selected features as a new layer
    output_layer = result['OUTPUT']
    QgsProject.instance().addMapLayer(output_layer)
joinattributesbylocation()


"""],

"native:intersection": ["""

Intersection (native:intersection)

This algorithm extracts the overlapping portions of features in the Input and Overlay layers. Features in the output 
Intersection layer are assigned the attributes of the overlapping features from both the Input and Overlay layers.

--------------------
 Python code template
--------------------

import processing
from qgis.core import QgsProject,QgsVectorLayer
def intersection():

    # Define the parameters Example below:
    parameters = {
        'INPUT':'D:/SpatialAnalysisAgent/Data/Exercise12Data/flood2015may_ESRI102965.shp', #Layer to extract (parts of) features from.
        'OVERLAY':'D:/SpatialAnalysisAgent/Data/Exercise12Data/rivers_ESRI102965.shp', # Layer containing the features to check for overlap. Its features’ geometry is expected to have at least as many dimensions (point: 0D, line: 1D, polygon: 2D, volume: 3D) as the input layer’s.
        'INPUT_FIELDS':[], #  Field(s) of the input layer to keep in the output. If no fields are chosen all fields are taken.
        'OVERLAY_FIELDS':[], # Field(s) of the overlay layer to keep in the output. If no fields are chosen all fields are taken. Duplicate field names will be appended a count suffix to avoid collision.
        'OVERLAY_FIELDS_PREFIX':'',
        'OUTPUT':'TEMPORARY_OUTPUT',
        'GRID_SIZE':None}
    # Perform the extract by attribute operation
    result = processing.run("native:intersection", parameters)
    # Load the selected features as a new layer
    output_layer = result['OUTPUT']
    QgsProject.instance().addMapLayer(output_layer)
intersection()

"""],

"native:native:extractwithindistancenative:extractwithindistance":["""Creates a new vector layer that only contains matching features from an input 
layer. Features are copied wherever they are within the specified maximum distance from the features in an additional 
reference layer.

--------------------
 Python code template
--------------------

import processing
from qgis.core import QgsProject,QgsVectorLayer
def extractwithindistance():

    # Define the parameters Example below:
    parameters = {
        'INPUT':'D:/SpatialAnalysisAgent/Data/Exercise12Data/flood2015may_ESRI102965.shp', # Input vector layer to copy features from
        'REFERENCE':'D:/SpatialAnalysisAgent/Data/Exercise12Data/rivers.shp', # Vector layer whose features closeness is used
        'DISTANCE':100, # The maximum distance around reference features to select input features within
        'OUTPUT':'TEMPORARY_OUTPUT'}
    # Perform the extract by attribute operation
    result = processing.run("native:extractwithindistance", parameters)
    # Load the selected features as a new layer
    output_layer = result['OUTPUT']
    QgsProject.instance().addMapLayer(output_layer)
extractwithindistance()

"""],

"native:clip": ["""
Clip (native:clip)

--------------------
 Python code template
--------------------
import processing
from qgis.core import QgsProject,QgsVectorLayer
def clip():

    # Define the parameters Example below:
    parameters = {
        'INPUT': 'D:/SpatialAnalysisAgent/Data/BG.shp',
        'OVERLAY': 'D:/SpatialAnalysisAgent/Data/bd.shp',
        'OUTPUT':'TEMPORARY_OUTPUT',
        }
    # Perform the extract by attribute operation
    result = processing.run("native:clip", parameters)
    # Load the selected features as a new layer
    output_layer = result['OUTPUT']
    QgsProject.instance().addMapLayer(output_layer)
clip()
"""
],

"native:reprojectlayer": ["""
Reproject layer (native:reprojectlayer)

This algorithm reprojects a vector layer. It creates a new layer with the same features as the input one, but with geometries reprojected to a new CRS.

Attributes are not modified by this algorithm.


----------------
Input parameters
----------------

INPUT: Input layer

	Parameter type:	QgsProcessingParameterFeatureSource

	Accepted data types:
		- str: layer ID
		- str: layer name
		- str: layer source
		- QgsProcessingFeatureSourceDefinition
		- QgsProperty
		- QgsVectorLayer

TARGET_CRS: Target CRS

	Parameter type:	QgsProcessingParameterCrs

	Accepted data types:
		- str: 'ProjectCrs'
		- str: CRS auth ID (e.g. 'EPSG:3111')
		- str: CRS PROJ4 (e.g. 'PROJ4:…')
		- str: CRS WKT (e.g. 'WKT:…')
		- str: layer ID. CRS of layer is used.
		- str: layer name. CRS of layer is used.
		- str: layer source. CRS of layer is used.
		- QgsCoordinateReferenceSystem
		- QgsMapLayer: CRS of layer is used
		- QgsProcessingFeatureSourceDefinition: CRS of source is used
		- QgsProperty

CONVERT_CURVED_GEOMETRIES: Convert curved geometries to straight segments

	If checked, curved geometries will be converted to straight segments. Otherwise, they will be kept as curves. This can fix distortion issues.

	Parameter type:	QgsProcessingParameterBoolean

	Accepted data types:
		- bool
		- int
		- str
		- QgsProperty

OPERATION: Coordinate operation

	Parameter type:	QgsProcessingParameterCoordinateOperation

	Accepted data types:
		- str: string representation of a Proj coordinate operation

OUTPUT: Reprojected

	Parameter type:	QgsProcessingParameterFeatureSink

	Accepted data types:
		- str: destination vector file, e.g. 'd:/test.shp'
		- str: 'memory:' to store result in temporary memory layer
		- str: using vector provider ID prefix and destination URI, e.g. 'postgres:…' to store result in PostGIS table
		- QgsProcessingOutputLayerDefinition
		- QgsProperty

--------------------
 Python code template
--------------------

import processing
from qgis.core import QgsProject,QgsVectorLayer
def reprojectlayer():

    # Define the parameters Example below:
    parameters = {
        'INPUT': 'D:/LLMQGIS/Data/network_analysis/Network_Analyst_Pro_Tutorial/Cleaned/Roads.shp',
        'TARGET_CRS':QgsCoordinateReferenceSystem('ESRI:102965'),
        'CONVERT_CURVED_GEOMETRIES':False,
        'OPERATION':'+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +ellps=GRS80',
        'OUTPUT':'TEMPORARY_OUTPUT'}

    # Perform the extract by attribute operation
    result = processing.run("native:reprojectlayer", parameters)
    # Load the selected features as a new layer
    output_layer = result['OUTPUT']
    QgsProject.instance().addMapLayer(output_layer)
reprojectlayer()
"""],

'native:addfieldtoattributestable': ["""


Add field to attributes table (native:addfieldtoattributestable)

This algorithm adds a new attribute to a vector layer.

The name and characteristics of the attribute are defined as parameters. The new attribute is not added to the input layer but a new layer is generated instead.


----------------
Input parameters
----------------

INPUT: Input layer

	Parameter type:	QgsProcessingParameterFeatureSource

	Accepted data types:
		- str: layer ID
		- str: layer name
		- str: layer source
		- QgsProcessingFeatureSourceDefinition
		- QgsProperty
		- QgsVectorLayer

FIELD_NAME: Field name

	Parameter type:	QgsProcessingParameterString

	Accepted data types:
		- str
		- QgsProperty

FIELD_TYPE: Field type

	Parameter type:	QgsProcessingParameterEnum

	Available values:
		- 0: Integer (32 bit)
		- 1: Decimal (double)
		- 2: Text (string)
		- 3: Boolean
		- 4: Date
		- 5: Time
		- 6: Date & Time
		- 7: Binary Object (BLOB)
		- 8: String List
		- 9: Integer List
		- 10: Decimal (double) List

	Accepted data types:
		- int
		- str: as string representation of int, e.g. '1'
		- QgsProperty

FIELD_LENGTH: Field length

	Parameter type:	QgsProcessingParameterNumber

	Accepted data types:
		- int
		- float
		- QgsProperty

FIELD_PRECISION: Field precision

	Parameter type:	QgsProcessingParameterNumber

	Accepted data types:
		- int
		- float
		- QgsProperty

FIELD_ALIAS: Field alias

	Parameter type:	QgsProcessingParameterString

	Accepted data types:
		- str
		- QgsProperty

FIELD_COMMENT: Field comment

	Parameter type:	QgsProcessingParameterString

	Accepted data types:
		- str
		- QgsProperty

OUTPUT: Added

	Parameter type:	QgsProcessingParameterFeatureSink

	Accepted data types:
		- str: destination vector file, e.g. 'd:/test.shp'
		- str: 'memory:' to store result in temporary memory layer
		- str: using vector provider ID prefix and destination URI, e.g. 'postgres:…' to store result in PostGIS table
		- QgsProcessingOutputLayerDefinition
		- QgsProperty

--------------------
 Python code template
--------------------

    import processing
    from qgis.core import QgsProject,QgsVectorLayer
    def addfieldtoattributestable():
    
        # Define the parameters Example below:
        parameters = {
            'INPUT':'D:/LLMQGIS/Roads.shp',
            'FIELD_NAME':'Length',
            'FIELD_TYPE':1, #Available values - 0: Integer (32 bit), 1: Decimal (double), 2: Text (string), 3: Boolean, 4: Date, 5: Time, 6: Date & Time, 7: Binary Object (BLOB), 8: String List, 9: Integer List, 10: Decimal (double) List
            'FIELD_LENGTH':10,
            'OUTPUT':'TEMPORARY_OUTPUT'}
    
        # Perform the extract by attribute operation
        result = processing.run("native:addfieldtoattributestable", parameters)
        # Load the selected features as a new layer
        output_layer = result['OUTPUT']
        QgsProject.instance().addMapLayer(output_layer)
    addfieldtoattributestable()
"""],
'qgis:exportaddgeometrycolumns':["""

Add geometry attributes (qgis:exportaddgeometrycolumns)

This algorithm computes geometric properties of the features in a vector layer. It generates a new vector layer with the same content as the input one, but with additional attributes in its attributes table, containing geometric measurements. Depending on the geometry type of the vector layer, the attributes added to the table will be different.

--------------------
 Python code template
--------------------
import processing
from qgis.core import QgsProject,QgsVectorLayer
def exportaddgeometrycolumns():

    # Define the parameters Example below:
    parameters = {
        'INPUT':'D:/LLMQGIS/Data/network_analysis/Network_Analyst_Pro_Tutorial/Cleaned/Roads.shp',
        'CALC_METHOD':0,
        'OUTPUT':'TEMPORARY_OUTPUT'}

    # Perform the extract by attribute operation
    result = processing.run("qgis:exportaddgeometrycolumns", parameters)
    # Load the selected features as a new layer
    output_layer = result['OUTPUT']
    QgsProject.instance().addMapLayer(output_layer)
exportaddgeometrycolumns()


"""],

"native:joinattributestable":["""

Join attributes by field value (native:joinattributestable)
This algorithm takes an input vector layer and creates a new vector layer that is an extended version of the input one, with additional attributes in its attribute table.
The additional attributes and their values are taken from a second vector layer. An attribute is selected in each of them to define the join criteria.

--------------------
 Python code template
--------------------

import processing
from qgis.core import QgsProject,QgsVectorLayer
def joinattributestable():
    csv_path = 'D:/Data/D1.csv'
    # Define the parameters Example below:
    parameters = {
        'INPUT':'D:/Data/PA.gpkg',
        'FIELD':'FIPS',
        'INPUT_2':csv_path, # use the csv_path directly for the Input parameter in join operations.
        'FIELD_2':'FIPS',
        'FIELDS_TO_COPY':[],
        'METHOD':1,
        'DISCARD_NONMATCHING':False,
        'PREFIX':'','OUTPUT':'TEMPORARY_OUTPUT'}

    # Perform the extract by attribute operation
    result = processing.run("native:joinattributestable", parameters)
    # Load the selected features as a new layer
    output_layer = result['OUTPUT']
    QgsProject.instance().addMapLayer(output_layer)
joinattributestable()

"""],

"native:joinbylocationsummary":["""

Join attributes by location (summary) (native:joinbylocationsummary)
This algorithm takes an input vector layer and creates a new vector layer that is an extended version of the input one, with additional attributes in its attribute table.
The additional attributes and their values are taken from a second vector layer. A spatial criteria is applied to select the values from the second layer that are added to each feature from the first layer in the resulting one.
The algorithm calculates a statistical summary for the values from matching features in the second layer( e.g. maximum value, mean value, etc ).

--------------------
 Python code template
--------------------

import processing
from qgis.core import QgsProject,QgsVectorLayer
def joinbylocationsummary():
    # Define the parameters Example below:
    parameters = {
        'INPUT':'D:/LLMQGIS/Data/network_analysis/input.shp',
        'PREDICATE':[0],
        'JOIN':'D:/LLMQGIS/Data/network_analysis/Join_data.shp',
        'JOIN_FIELDS':[],
        'SUMMARIES':[],
        'DISCARD_NONMATCHING':False,
        'OUTPUT':'memory:Joined layer'}
    # Perform the extract by attribute operation
    result = processing.run("native:joinbylocationsummary", parameters)
    # Load the selected features as a new layer
    output_layer = result['OUTPUT']
    QgsProject.instance().addMapLayer(output_layer)
joinbylocationsummary()

"""],

"native:serviceareafromlayer":["""
Service area (from layer) (native:serviceareafromlayer)
This algorithm creates a new vector with all the edges or parts of edges of a network line layer that can be reached within a distance or a time, starting from features of a point layer. The distance and the time (both referred to as "travel cost") must be specified respectively in the network layer units or in hours.

--------------------
 Python code template
--------------------
import processing
from qgis.core import QgsProject,QgsVectorLayer
def serviceareafromlayer():
    # Define the parameters Example below:
    parameters = {
        'INPUT':'D:/LLMQGIS/Data/network_analysis/network_line.shp', #Vector layer representing network
        'STRATEGY':0, #Use 0 for "Shortest", 1 for "Fastest"
        'DIRECTION_FIELD':'',
        'VALUE_FORWARD':'',
        'VALUE_BACKWARD':'',
        'VALUE_BOTH':'',
        'DEFAULT_DIRECTION':2,
        'SPEED_FIELD':'',
        'DEFAULT_SPEED':50,
        'TOLERANCE':0,
        'START_POINTS':'D:/LLMQGIS/Data/network_analysis/Place.shp', #Vector layer with start points
        'TRAVEL_COST2':100, #Travel cost (distance for 'Shortest', time for 'Fastest')
        'INCLUDE_BOUNDS':False,
        'OUTPUT_LINES':'memory:Service area(lines)',
        'OUTPUT':'memory:Service area (boundary nodes'}
    # Perform the extract by attribute operation
    result = processing.run("native:serviceareafromlayer", parameters)
    # Load the selected features as a new layer
    output_layer = result['OUTPUT']
    QgsProject.instance().addMapLayer(output_layer)
serviceareafromlayer()

"""],

"native:createpointslayerfromtable" : ["""

--------------------
 Python code template
--------------------

def createpointslayerfromtable(output_path):
    parameters = {
                'INPUT':'D:/Data/PovertyData/PovertyData.csv',
                'XFIELD':'INTPTLON',
                'YFIELD':'INTPTLAT',
                'ZFIELD':'',
                'MFIELD':'',
                'TARGET_CRS':QgsCoordinateReferenceSystem('EPSG:4326'),
                'OUTPUT': output_path
                }
    result = processing.run("native:createpointslayerfromtable", parameters)
    output_layer = QgsVectorLayer(output_path, 'Obesity', 'ogr')
    QgsProject.instance().addMapLayer(output_layer)
output_path = 'D:/Data/PovertyData/Poverty3.shp'
createpointslayerfromtable(output_path)
"""],

"qgis:barplot": ["""


----------------------------
 Python sample code
-----------------------------
def barplot ():
    parameters = {
    'INPUT':'D:/Data/PovertyData/Poverty3.shp',
    'NAME_FIELD':'Poverty',
    'VALUE_FIELD':'Poverty',
    'OUTPUT':'output_path'
    }
    result = processing.run("qgis:barplot", parameters)
    output_layer = result['OUTPUT']
    print(output_layer)
output_path = D:/Data/PovertyData/barplot.html #use an output directory
barplot()

"""],
"qgis:vectorlayerscatterplot" :["""


----------------------------
 Python sample code
-----------------------------
def scatterplot():
    parameters = {
        'INPUT': 'D:/Data/PovertyData/PovertyLayerWithXY.shp',
        'XFIELD': 'x',
        'YFIELD': 'y',
        'OUTPUT': 'output_path'
    }
    processing.run("qgis:vectorlayerscatterplot", parameters)
    output_layer = result['OUTPUT']
    print(output_layer)
output_path = C:/Data/PovertyData/scatterplot.html #use an output directory
scatterplot()
"""],


'scatterplot': [r"""
    
    --------------------
     Python sample code
    --------------------
    ```
   def scatterplot():
    import plotly.express as px
    import pandas as pd

    # Sample data
    data = {'field1': [5, 7, 8, 7, 2, 17, 2, 9, 4, 11],
            'field2': [99, 86, 87, 88, 100, 86, 103, 87, 94, 78]}

    df = pd.DataFrame(data)

    # Create scatter plot
    fig = px.scatter(df, x='x', y='y', title='Sample Scatter Plot')

    # Save the plot to an HTML file
    fig.write_html('scatter_plot.html')
    
scatterplot()
"""]

}