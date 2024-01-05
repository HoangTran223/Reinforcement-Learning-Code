import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
def storage_column_values(link_excel, sheet_name, index):
    a = []
    df = pd.read_excel(link_excel, sheet_name)
    column_values = df.iloc[:, index]

    for value in column_values:
        a.append(value)
    
    return a


font1 ={'color':'red', 'size': 15}                          

xpoints = np.array(storage_column_values('C:/Users/DELL/Desktop/multi_agent_rl_wrsn/log50.xlsx', 'log50', 0))
ypoints = np.array(storage_column_values('C:/Users/DELL/Desktop/multi_agent_rl_wrsn/log50.xlsx', 'log50', 3))
plt.xlabel("Iteration")
plt.ylabel("Average episodic lifetime")   
plt.title("Visualization of average episodic lifetime for a network of 50 targets with 500 iterations", fontdict = font1, loc = 'center')
plt.plot(xpoints, ypoints)
plt.text(5, 5, 'Chú thích dưới đồ thị', ha='center', va='bottom')
plt.show()



xpoints = np.array(storage_column_values('C:/Users/DELL/Desktop/multi_agent_rl_wrsn/log100.xlsx', 'log100', 0))
ypoints = np.array(storage_column_values('C:/Users/DELL/Desktop/multi_agent_rl_wrsn/log100.xlsx', 'log100', 3))
plt.xlabel("Iteration")
plt.ylabel("Average episodic lifetime")  
plt.plot(xpoints, ypoints)
plt.title("Visualization of average episodic lifetime for a network of 100 targets with 500 iterations", fontdict = font1, loc = 'center')
plt.show()



xpoints = np.array(storage_column_values('C:/Users/DELL/Desktop/multi_agent_rl_wrsn/log50.xlsx', 'log50', 0))
ypoints = np.array(storage_column_values('C:/Users/DELL/Desktop/multi_agent_rl_wrsn/log50.xlsx', 'log50', 5))
plt.xlabel("Iteration")
plt.ylabel("Average reward")  
plt.plot(xpoints, ypoints)
plt.title("Visualization of average reward for a network of 50 targets with 500 iterations", fontdict = font1, loc = 'center')
plt.show()



xpoints = np.array(storage_column_values('C:/Users/DELL/Desktop/multi_agent_rl_wrsn/log100.xlsx', 'log100', 0))
ypoints = np.array(storage_column_values('C:/Users/DELL/Desktop/multi_agent_rl_wrsn/log100.xlsx', 'log100', 5))
plt.xlabel("Iteration")
plt.ylabel("Average reward")  
plt.plot(xpoints, ypoints)
plt.title("Visualization of average reward for a network of 100 targets with 500 iterations", fontdict = font1, loc = 'center')
plt.show()
