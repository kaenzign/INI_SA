import pyNN.spiNNaker as sim
import pyNN.utility.plotting as plot
import matplotlib.pyplot as plt
sim.setup(timestep=1.0)

spike_times = [[i] for i in range(1296)]
input = sim.Population(1296, sim.SpikeSourceArray(spike_times=spike_times), label="0_input")
pop_1 = sim.Population(64, sim.IF_curr_exp(), label="1_dense")
pop_2 = sim.Population(4, sim.IF_curr_exp(), label="2_softmax")

input_proj = sim.Projection(input, pop_1, sim.OneToOneConnector(),synapse_type=sim.StaticSynapse(weight=5, delay=1))
proj_D = sim.Projection(pop_1, pop_2, sim.AllToAllConnector())

pop_1.record(["spikes", "v"])

simtime = 100
sim.run(simtime)

neo = pop_1.get_data(variables=["spikes", "v"])
spikes = neo.segments[0].spiketrains
print(spikes)
v = neo.segments[0].filter(name='v')[0]
print (v)

sim.end()

plot.Figure(
# plot voltage for first ([0]) neuron
plot.Panel(v, ylabel="Membrane potential (mV)",
data_labels=[pop_1.label], yticks=True, xlim=(0, simtime)),
# plot spikes (or in this case spike)
plot.Panel(spikes, yticks=True, markersize=5, xlim=(0, simtime)),
title="Simple Example",
annotations="Simulated with {}".format(sim.name())
)
plt.show()