# RL-based SYCL

## Repository Structure
- CMakeLists.txt : cmake file
- inputs : input files
- src : source code
- input_generator : generator to generate inputs
- python_visualization : script to visualize histogram of time interval

## Input Format
The input files have the following format.
The first line specify the number of vertices(V) and the number of edges(E).
The next V lines denote the information (id, m, n) of each vertex,
where id is the ID of the vertex, m and n are the dimensions of the matrix
processed in that vertex.
The next E lines denote the edges.
For example, in the following, there are 4 vertices and 3 edges.
Vertex 0 has m = 6 and n = 7. Vertex 1 has m = 4 and n = 5.
Then the edge (0, 2) denotes that vertex 0 connecting to vertex 2.    
```
4 3
0 6 7
1 4 5
2 3 3
3 9 5
0 2
1 2
2 3 
```


## Build
To build the executable, please follow the instructions below. The default compiler is clang++.
```
mkdir build
cd build
cmake ../
make
```

## Run
To run the executable, please follow the instruction below.
```
cd build
./main < ../inputs/1.in
```

## Generate new input files
There are two generators. One is to generate big input file with around 10000 tasks.
The other is to generate small input with around 50 tasks.
Default generated task size is 10000 and 50 for big and small generator, respectively.
To generate an input file of 30000 tasks, please follow the instructions below.
The generated file is placed in `inputs` directory with the number of tasks as the file name. 
```
cd input_generator
python3 generator_big.py 30000
``` 

## Visualize the time interval
After running the executable, we can visualize the time interval between two RL inferences.
A `timestamp.csv` is generated in `python_visualization` directory,
please use `visualization.py` to plot a histogram.
```
cd python_visualization
python3 visualization.py
```
