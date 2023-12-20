import numpy as np 
from matplotlib import pyplot as plt 
from scipy.integrate import solve_ivp

#SEIRD model base parameters
beta = 2.28 # The probability of infection
sigma = 0.1 # The speed of recovery: sigma = 1/T(the avarege duration of the disease)
gamma = 0.2 # Incubation period 
mu = 0.034 # Martality rate

#Time parameters 
t_start = 0 
t_end = 200
t_interval = (t_start, t_end)

# Initial coditions 
S0 = 10 # Susceptible  
E0 = 0.3 # Exposed 
I0 = 0.0 # Infected 
R0 = 0.0 # Recovered 
D0 = 0.0 # Dead 


# SEIRD model: Susceptible -> Exposed -> Infected -> Recovered / Dead
def seird(t, y): 
    S, E, I, R, D = y 
    dSdt = -beta * S * I 
    dEdt = beta * S * I - sigma * E 
    dIdt = sigma * E - gamma * I 
    dRdt = gamma * I 
    dDdt = mu * I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

# The solution of the model 
response = solve_ivp(seird, t_interval, [S0, E0, I0, R0, D0], t_eval=np.linspace(t_start, t_end, 1000))

plt.figure(figsize=(12, 6))
plt.plot(response.t, response.y[0], label='S - Susceptible')
plt.plot(response.t, response.y[1], label='E - Exposed')
plt.plot(response.t, response.y[2], label='I - Infected')
plt.plot(response.t, response.y[3], label='R - Recovered')
plt.plot(response.t, response.y[4], label='D - Dead')
plt.xlabel('Duration')
plt.ylabel('Population size')
plt.title('SEIRD model')
plt.legend()
plt.grid(True)
plt.show()



