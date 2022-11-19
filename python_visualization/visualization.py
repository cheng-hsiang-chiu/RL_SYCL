import pandas as pd
import matplotlib.pyplot as plt


def plot():
  df = pd.read_csv("./timestamp.csv")
  
  mybins = list(range(0, 5000, 100))
  df.plot(kind='hist', bins = mybins, legend=None)
  
  #df.plot(kind='hist', bins=50, legend=None)
  plt.savefig('./output.png')


if __name__ == "__main__":
  plot()
