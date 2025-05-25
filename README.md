# Assignment 2B
## How to use
### Setup
#### Create the virtual environment.
```bash
python -m venv .venv
```
#### Activate the virtual environment.
- Windows:
```bash
.\.venv\Scripts\Activate.ps1
```
- Linux, MacOS
```bash
source ./.venv/bin/activate
```
#### Install the requirements
```bash
pip install -r requirements.txt
```
#### Select the environment
1. With the any python file open, at the bottom right click the text that says the python version (the right one).
2. Select the virtual environment you just created, you may need to reload visual studio.

### Running the project
While the current directory as the root of the project:
```bash
python guiWindow.py
```
#### While in the GUI
1. Select a SCATS site to start the pathfinding from.
2. Select a SCATS site to end the pathfinding at.
3. Select a time to conduct the pathfinding at.
4. Select a model to use for traffic prediction.
5. Press the button labelled "Calculate route".

The program will calculate five different routes, displaying the result as text on the gui window.

Then it will open a browser tab with the paths all displayed graphically on a map.

You can also press the "Display graph" button to have the program use the currently selected model to create a visual of all the avaiable sites and edges, with the edges having their estimated traffic displayed as a hue shift, the more red, the more traffic.

## Project Description: Machine Learning and Software Integration
- Due 11:59pm Sunday 25 May 2025 (Week 11)
- Contributes 50% to your Assignment 2 result.
- Group of 3-4 students

___
The Part B of Assignment 2 (2B) requires you to work with your group to implement ML algorithms to train ML models for traffic prediction and integrate the traffic predictor with Part A to develop a fully functioned TBRGS.

___
In Part B of Assignment 2, the team can use any machine learning technique or combinations of them. You should take advantage of existing libraries such as PyTorch, tensorflow, Keras, Theano, etc. We will provide you with a small dataset from VicRoads for the city of Boroondara that contains only the traffic flow data (the number of cars passing an intersection every 15 minutes). You should use this dataset for the following purposes: training ML models for predicting traffic conditions and testing/evaluating the performance of your models. At the very least, your TBRGS system will need to include the two basic deep learning techniques LSTM and GRU. Your TBRGS system will also need to implement at least one other technique (to be identified by you and approved by your tutor) to train another machine learning (ML) model and give a comprehensive comparison between different models. Your program will at least need to implement the following features:
- At the minimum, the TBRGS will have to be able to train ML models using the Boroondara dataset and give meaningful predictions based on these models
- A GUI will be available for the user input, parameter settings and visualisation (and a configuration file for the defaults).

### The Traffic-based Route Guidance Problem
In this problem, there are four main tasks for the team:
1. Implement data processing program to extract data from the given dataset and store it in appropriate data structures to enable ML models to be trained/tested;
2. Implement ML algorithms to train ML models for traffic flow prediction using the provided dataset.
3. Implement a travel time estimation for each edge on the map of the Boroondara area.
4. Integrate Part A of Assignment 2 with the TBRGS to replace the nodes by the intersections, the cost of each edge by the travel time and perform the calculation to find the top-k paths to travel from O to D for any given pair (O,D) of origin and destination.

### System requirements
For this assignment, we expect that the team will implement data processing methods and various ML algorithms (including LSTM and GRU) to train ML models for traffic prediction.

The team will need to use the given dataset for training and also testing the ML models to evaluate their performance. The team will also need to port the programs you developed previously for Assignment 2A to enable it to search for optimal paths on the Boroondara map. The edge cost will need to be replaced by the predicted travel time and subsequently, the optimal paths can be calculated and returned.

The basic version of TBRGS will be for the Boroondara area. A user can specify the origin and destination of their trip as the SCATS site number (e.g. origin O = 2000 [intersection WARRIGAL_RD/TOORAK_RD] and destination D = 3002 [intersection DENMARK_ST/BARKERS_RD]). The system then returns up to five (5) routes from O to D with the estimated travel time along each route. To simplify the calculation, you can make a number of assumptions: (i) The speed limit on every link will be the same and set at 60km/h; (ii) the travel time from a SCATS site A to a SCATS site B can be approximated by a simple expression based on the accumulated volume per hour at the SCATS site B and the distance between A and B (We will provide a simplified way to convert from traffic flow to travel time; See the document Traffic Flow to Travel Time Conversion v1.0.PDF on Canvas); and (iii) there is an average delay of 30 seconds to pass each controlled intersection. Note that, the objective is not to better Google Maps but to utilise the AI techniques you have learned (e.g., machine learning for forecasting traffic volume, graph-based search for optimal paths) to solve a real-world problem.

### In simple terms
Use machine learning to create a model that can predict the traffic conditions for a given street at a given time. Then use that data as an edge cost in our pathfinding algorithms.

#### Roles
We have four parts, two frontend, two backend.
- Frontend has the gui for inputing origin and destination, then outputing the resulting path.
- Frontend has integration with OpenStreetMaps to visualise the path.
- Backend has the machine learning part, to take the traffic data and give an edge cost.
- Backend has the pathfinding algorithms, needed to be retrofitted to work with the gui and models.

### Definitions
- TBRGS: Traffic-Based Route Guidance S?
- LSTM: Long Short-Term Memory ai model
- GRU: Gated Recurrent Unit ai model

### Neural Networks (For reference)
![I forgot the original source](./ChartOfNeuralNetworks.png)

## TODO
- Extracting all the important data from the sources.
- Machine learning
	- Linear regression for testing input data (maybe)
	- LSTM model
	- GRU model
- Pathfinding
	- Import graph data into problem
	- Import machine learning data into graph data
- GUI
	- Input
	- Output
		- As text
		- Open street maps
