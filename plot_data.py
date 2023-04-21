import matplotlib.pyplot as plt

from load_data import load_dataset

list_mat = ['B0005', 'B0006', 'B0007', 'B0018']
data_capacity, data_charge, data_discharge = load_dataset()

def plot_capacity():
    """绘制容量衰减曲线"""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    color_list = ['b:', 'g--', 'r-.', 'c.']
    c = 0
    for name in list_mat:
        data = data_capacity[name]
        color = color_list[c]
        ax.plot(data[0], data[1], color, label=name)
        c += 1
    plt.plot([-1,170],[2.0*0.7,2.0*0.7],c='black',lw=1,ls='--')  # 临界点直线
    ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)', title='Capacity degradation at ambient temperature of 24°C')
    plt.legend()
    plt.show()

def plot_charge(name:str, time:list, attr:str, type:str):
    """绘制充放电的电压、电流和温度曲线"""
    assert attr in ['Current', 'Voltage', 'Temperature'], 'type must be Current, Voltage or Temperature'
    assert type in ['discharge', 'charge'], 'type must be charge or discharge'
    key = attr + '_measured'

    if type == 'discharge':
        dataset = data_discharge
    elif type == 'charge':
        dataset = data_charge
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    for t in time:
        data = dataset[name][t]
        ax.plot(data['Time'], data[key], label='charge time: '+str(t))
    ax.set(xlabel='Time', ylabel=attr, title='Charging Curve')
    plt.legend()
    plt.show()

plot_capacity()
plot_charge('B0005', [0, 50, 100, 150], 'Temperature', 'discharge')