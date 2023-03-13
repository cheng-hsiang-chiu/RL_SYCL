
file_path = "../build/action_state.txt"


def read_state():
  while 1:
    with open(file_path, "r") as f:
      lines = f.readlines()
      print(lines)
      if len(lines) > 0 and lines[0] == "DONE":
        return 1
      elif len(lines) > 0 and lines[-1] == "STATE_READY\n":
        print (lines)
        return 0
      else:
        continue

def write_action():
  with open(file_path, "w") as f:
    f.write("1\n")
    f.write("1\n")
    f.write("ACTION_READY\n")

def main():

  while 1:
    rtn = read_state()
    if rtn == 1:
      return
    write_action()






if __name__ == "__main__":
  main()
