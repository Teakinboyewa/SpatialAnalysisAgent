# Case Studies
The case studies demonstrate the capabilities of the Spatial Analysis Agent (GIS Copilot) across three levels:

1. Basic level: At this level, the tasks are straightforward, involving a single tool along with one (or occasionally two) data layers. The agent is expected to perform a straightforward operation that usually requires a single step, such as calculating the area of polygons or selecting features based on attributes.
2. Intermediate level : At this level, tasks become more complex and involve multiple steps and tools. The agent is guided with specific instructions or a list of steps to perform the task. Although the steps are outlined, the agent still needs to generate the correct code for each step and link them together to perform the analysis.
3. Advanced level : In the advanced level, tasks are multistep, and the agent is expected to determine the appropriate steps independently without explicit instructions, to devise the best approach for achieving the desired outcome.Evaluates the agent's ability to select the appropriate tool, generate the correct code, and successfully execute the complete operation.

### You can access over 100 test cases/analysis examples [here](https://giscience.psu.edu/copilot_test/) and [here](https://giscience.psu.edu/gis-copilot-demonstrations/).

## Level 1: Basic level
### Case 1.1: Can you please create 2000-feet zones around each health facilities in Washington DC to identify areas of service coverage?

![Health facilities coverage zone .png](Doc%2FCase%20Studies%2FLevel%201%2FHealth%20facilities%20coverage%20zone%20.png)

### Case 1.2: Generate contour lines from the DEM of Puerto Rico with a 50-meter interval.

![Contour lines from DEM.png](Doc%2FCase%20Studies%2FLevel%201%2FContour%20lines%20from%20DEM.png)

### Case 1.3: Select the USA counties that have more than 50,000 population.

![Selection of high population counties.png](Doc%2FCase%20Studies%2FLevel%201%2FSelection%20of%20high%20population%20counties.png)

## Case 1.4: Clip the land cover data of the USA to the Pennsylvania boundary.

![Extracting land cover information.png](Doc%2FCase%20Studies%2FLevel%201%2FExtracting%20land%20cover%20information.png)

### Case 1.5: Please highlight the largest county and the smalest county in South Carolina.
![County Selection](Doc/Case%20Studies/Level%202/CountySelection.png)

## Level 2: Intermediate level
### Case 2.1: Perform the following task: 1) Clip the DEM to SC boundary and load the clipped DEM. 2) Show me the histogram for the clipped DEM pixel values. 3) Generate the zonal statistics for every county in SC, focusing on average elevation. 4) Finally create a choropleth map showing the average elevation of each county in SC.

![Zonal statistics.png](Doc%2FCase%20Studies%2FLevel%202%2FZonal%20statistics.png)


### Case 2.2: Merge the four DEMs into a single raster and perform terrain characteristic analysis for Richland County, including slope, aspect, hillshade, terrain ruggedness index (TRI), and topographic Position Index (TPI).

![Richland county terrain analysis.png](Doc%2FCase%20Studies%2FLevel%202%2FRichland%20county%20terrain%20analysis.png)


### Case 2.3: Generate an obesity risk behavior index of each county in the contiguous US by analyzing the rate of visits to unhealthy food retailers (such as convenience store, alcoholic drinking places, and limited service restaurant) and the visit rate to places that support physical activity (e.g., sports centers, parks, fitness centers). Visualize the results in a thematic map to highlight the obesity risk behavior index across counties.

![County level obesity risk behavior index analysis.png](Doc%2FCase%20Studies%2FLevel%202%2FCounty%20level%20obesity%20risk%20behavior%20index%20analysis.png)


## Level 3: Advanced level

### Case 3.1: Could you analyze and visualize the fast food accessibility score for each county based on the number of fast food restaurants and population using a thematic map with blue graduated colors. Then, analyze the correlation between the county-level obesity rate and the fast food accessibility score by drawing a scatter plot with a regression line.
![Figure 11_Fastfood accessibility analysis.png](Doc%2FCase%20Studies%2FLevel%203%2FFigure%2011_Fastfood%20accessibility%20analysis.png)


### Case 3.2: Could you show the spatial distribution of the COVID-19 cases across US counties?

![Covid19 cases.png](Doc%2FCase%20Studies%2FLevel%203%2FCovid19%20cases.png)

### Case 3.3: Generate the Normalized Difference Vegetation Index (NDVI) of Akure from these satellite imageries.

![NDVI.png](Doc%2FCase%20Studies%2FLevel%203%2FNDVI.png)


### Case 3.4: Can you please select the residential area and calculate the total area covered by the residential area in square kilometers

![Selection of Land use.png](Doc%2FCase%20Studies%2FLevel%203%2FSelection%20of%20Land%20use.png)
