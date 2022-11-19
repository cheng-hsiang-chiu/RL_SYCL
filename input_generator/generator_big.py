#
# Randome task generator
#
# Generate num_tasks tasks. 
# Each task has id, m, and n,
# where m is the row dimension of a matrix and n is the column dimension.
#
# We generate tasks level by level. Each level has [50, 200) tasks. 
# Each task has at most 10 joint counters. 
#

import random
import sys


# define the m and n of a task
def define_task_m_n():
  # dimension between 5 and 50 
  m = random.randrange(5, 50, 1);
  n = random.randrange(5, 50, 1);
  
  return m, n;


def generate(output_file, num_tasks):
  ID = 0
  num_edges = 0

  list_tasks = []

  # write the dimension of row and column 
  with open(output_file, "w") as f:
    while ID < num_tasks:
      m, n = define_task_m_n()
      f.write(str(ID) + ' ' + str(m) + ' ' + str(n) + '\n')      
      ID = ID + 1

  # write edges
  num_tasks_this_level = random.randrange(50, 200, 1)
  num_tasks_next_level = random.randrange(50, 200, 1)
  tid = 0
  cnt_edges = 0
  cnt_tasks = 0
  with open(output_file, "a") as f:
    while tid < num_tasks:

      cnt_edges = random.randrange(5, 10, 1)

      set_edges = set()
      while len(set_edges) < cnt_edges :
        fromID = cnt_tasks + num_tasks_this_level
        toID = fromID + num_tasks_next_level
        sucessor = random.randrange(fromID, toID, 1)
        set_edges.add(sucessor)
      
      for e in set_edges:
        if (e < num_tasks):
          f.write(str(tid) + ' ' + str(e) + '\n')
        else:
          cnt_edges = cnt_edges - 1
      
      num_edges = num_edges + cnt_edges
      
      tid = tid + 1
      if tid == cnt_tasks + num_tasks_this_level:
        cnt_tasks = cnt_tasks + num_tasks_this_level
        num_tasks_this_level = num_tasks_next_level
        num_tasks_next_level = random.randrange(50, 200, 1)      
        if cnt_tasks + num_tasks_this_level >= num_tasks:
          break 
                

  # write the num_tasks and num_edges at the beginning
  with open(output_file, "r+") as f:
    lines = f.readlines()
    lines.insert(0, str(num_tasks) + ' ' + str(num_edges) + '\n')
    f.seek(0)
    f.writelines(lines) 


if __name__ == "__main__":
  if len(sys.argv) > 1:
    num_tasks = int(sys.argv[1])
    if (num_tasks < 200):
      sys.exit("Number of Tasks must be greater than 200")
  
  # generate 10000 tasks default
  else:    
    num_tasks = 10000
  
  output_file = "../inputs/" + str(num_tasks) + ".in"
  generate(output_file, num_tasks)
