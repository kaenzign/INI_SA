import pyNN.spiNNaker as sim
import pyNN.utility.plotting as plot
import matplotlib.pyplot as plt
sim.setup(timestep=1.0)

input = sim.Population(10, sim.SpikeSourceArray(spike_times=[0]), label="input")
pop_1 = sim.Population(10, sim.IF_curr_exp(), label="pop_1")
pop_2 = sim.Population(10, sim.IF_curr_exp(), label="pop_2")

input_proj = sim.Projection(input, pop_1, sim.OneToOneConnector())
proj_D = sim.Projection(pop_1, pop_2, sim.AllToAllConnector())

