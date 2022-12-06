# RL-based SYCL

## Repository Structure
- CMakeLists.txt : cmake file
- inputs : input files
- src : source code
- env.sh : script to export environment variables


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
The following figure shows the above task graph.
![](figure/demo_task_graph.png)


## SYCL Compiler Installed
Before building the executable, please make sure SYCL compiler is successfully installed.
The link of the installation instructions is 
https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md


## Environment Setup
To build and run, please export some environment variables.
We provide `env.sh` to export these variables.
Simply follow the instructions below.
```
source env.sh
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
