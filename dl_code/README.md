## Deep Q Learning


# Structure
- pdf : An introduction to RL learning
- policy.py : a python script that generates actions and receives states

# How policy.py interacts with the Cpp code (main.cpp)
The channel that policy.py and main.cpp communicate through is a text file named "action_state.txt"
In the beginning, main.cpp waits for an action suggested by policy.py and looks for the following
protocol in action_state.txt

```
2
1
ACTION_READY
```
The first line is the thread id, the second line is the accelerator, and
the third line denotes that the action is successfully writen in action_state.txt by policy.py.

Then, main.cpp reads the action in action_state.txt and writes a corresponding state in action_state.txt
with the following protocol,
```
0 0 0 0 0 0 0 ...
STATE_READY
```
The first line is the state statistics and the second line denotes that the state statistics
is successfully writen in action_state.txt by main.cpp.

When main.cpp finishes all the tasks, it will write the following in action_state.txt
```
DONE
``` 
