# %% [markdown]
# *Copyright (C) 2021 Intel Corporation*<br>
# *SPDX-License-Identifier: BSD-3-Clause*<br>
# *See: https://spdx.org/licenses/*
# 
# ---
# 
# # Processes
# 
# Learn how to create _Processes_, the fundamental computational units used in Lava to build algorithms and applications.

# %% [markdown]
# ## Recommended tutorials before starting:
# 
# - [Installing Lava](./tutorial01_installing_lava.ipynb "Tutorial on Installing Lava")
# 
# 
# ## What is a _Process_?
# 
# This tutorial will show how to create a _Process_ that simulates a group of leaky integrate-and-fire neurons. But in Lava, the concept of _Processes_ applies widely beyond this example. In general, a _Process_ describes an individual program unit which encapsulates
# <ol>
# <li>data that store its state,</li>
# <li>algorithms that describe how to manipulate the data,</li>
# <li>ports that share data with other Processes, and </li>
# <li>an API that facilitates user interaction.</li>
# </ol>
# 
# A _Process_ can thus be as simple as a single neuron or a synapse, as complex as a full neural network, and as non-neuromorphic as a streaming interface for a peripheral device or an executed instance of regular program code.
# 
# <img src="https://raw.githubusercontent.com/lava-nc/lava-nc.github.io/main/_static/images/tutorial02/fig01_processes.png" width="1000" align="center"/>
# 
# _Processes_ are independent from each other as they primarily operate on their own local memory while they pass messages between each other via channels. Different _Processes_ thus proceed their computations simultaneously and asynchronously, mirroring the high parallelism inherent in neuromorphic hardware. The parallel _Processes_ are furthermore safe against side effects from shared-memory interaction.
# 
# Once a _Process_ has been coded in Python, Lava allows to run it seamlessly across different backends such as a CPU, a GPU, or neuromorphic cores. Developers can thus easily test and benchmark their applications on classical computing hardware and then deploy it to neuromorphic hardware. Furthermore, Lava takes advantage of distributed, heterogeneous hardware such as Loihi as it can run some _Processes_ on neuromorphic cores and in parallel others on embedded conventional CPUs and GPUs. 
# 
# While Lava provides a growing [library of Processes](https://github.com/lava-nc/lava/tree/main/src/lava/proc "Lava's process library"), you can easily write your own processes that suit your needs.

# %% [markdown]
# ## How to build a _Process_?
# 
# #### Overall architecture
# 
# All _Processes_ in Lava share a universal architecture as they inherit from the same _AbstractProcess_ class. Each _Process_ consists of the following four key components.
# <img src="https://raw.githubusercontent.com/lava-nc/lava-nc.github.io/main/_static/images/tutorial02/fig02_architectural_components.png" width="1000"  align="center"/>

# %% [markdown]
# #### _AbstractProcess_: Defining _Vars_, _Ports_, and the API
# 
# When you create your own new process, you need to inherit from the AbstractProcess class. As an example, we will implement the *class LIF*, a group of leaky integrate-and-fire (LIF) neurons.
# 
# <img src="https://raw.githubusercontent.com/lava-nc/lava-nc.github.io/main/_static/images/tutorial02/fig03_lifs.png" width="780"  align="center"/>
# 
# | Component | Name | Python |  | 
# | :- | :- | :- | :-|
# | **Ports** | $a_{in}$ | _Inport_ | Receives spikes from upstream neurons.
# |       | $s_{out}$ | _Outport_ | Transmits spikes to downstream neurons.
# | **State** | $u$ | _Var_ | Synaptic current of the LIF neurons.
# |       | $v$ | _Var_ | Membrane voltage of the LIF neurons.
# |       | $du$ | _Var_ | A time constant describing the current leakage.
# |       | $dv$ | _Var_ | A time constant describing the voltage leakage.
# |       | $bias$ | _Var_ | A bias value.
# |       | $vth$ | _Var_ | A constant threshold that the membrane voltage needs to exceed for a spike.
# | **API**   | All Vars | _Var_ | All public _Vars_ are considered part of the _Process_ API.
# |       | All Ports | _AbstractPort_ | All _Ports_ are considered part of the _Process_ API.
# |       | print_vars | _def_ | A function that prints all internal variables to help the user see if the LIF neuron has correctly been set up.
# 
# The following code implements the class _LIF_ that you can also find in Lava's _Process_ library, but extends it by an additional API method that prints the state of the LIF neurons.

# %%
import numpy as np

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


class LIF(AbstractProcess):
    """Leaky-Integrate-and-Fire neural process with activation input and spike
    output ports a_in and s_out.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)
        self.du = Var(shape=(1,), init=kwargs.pop("du", 0))
        self.dv = Var(shape=(1,), init=kwargs.pop("dv", 0))
        self.bias = Var(shape=shape, init=kwargs.pop("bias", 0))
        self.vth = Var(shape=(1,), init=kwargs.pop("vth", 10))

    def print_vars(self):
        """Prints all variables of a LIF process and their values."""

        sp = 3 * "  "
        print("Variables of the LIF:")
        print(sp + "u:    {}".format(str(self.u.get())))
        print(sp + "v:    {}".format(str(self.v.get())))
        print(sp + "du:   {}".format(str(self.du.get())))
        print(sp + "dv:   {}".format(str(self.dv.get())))
        print(sp + "bias: {}".format(str(self.bias.get())))
        print(sp + "vth:  {}".format(str(self.vth.get())))
        

# %% [markdown]
# You may have noticed that most of the _Vars_ were initialized by scalar integers. But the synaptic current _u_ illustrates that _Vars_ can in general be initialized with numeric objects that have a dimensionality equal or less than specified by its _shape_ argument. The initial value will be scaled up to match the _Var_ dimension at run time.
# 
# There are two further important things to notice about the _Process_ class:
# <ol>
#   <li>It only defines the interface of the LIF neuron, but not its temporal behavior.</li>
#   <li>It is fully agnostic to the computing backend and will thus remain the same if you want to run your code, for example, once on a CPU and once on Loihi.</li>
# </ol>

# %% [markdown]
# #### _ProcessModel_: Defining the behavior of a _Process_
# 
# The behavior of a _Process_ is defined by its _ProcessModel_. For the specific example of LIF neuron, the _ProcessModel_ describes how their current and voltage react to a synaptic input, how these states evolve with time, and when the neurons should emit a spike.
# 
# A single _Process_ can have several _ProcessModels_, one for each backend that you want to run it on.
# 
# The following code implements a _ProcessModel_ that defines how a CPU should run the LIF _Process_. Please do not worry about the precise implementation here&mdash;the code will be explained in detail in the next [Tutorial on ProcessModels](./tutorial03_process_models.ipynb "Tutorial on ProcessModels").

# %%
import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel

@implements(proc=LIF, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyLifModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    u: np.ndarray = LavaPyType(np.ndarray, np.float)
    v: np.ndarray = LavaPyType(np.ndarray, np.float)
    bias: np.ndarray = LavaPyType(np.ndarray, np.float)
    du: float = LavaPyType(float, np.float)
    dv: float = LavaPyType(float, np.float)
    vth: float = LavaPyType(float, np.float)

    def run_spk(self):
        a_in_data = self.a_in.recv()
        self.u[:] = self.u * (1 - self.du)
        self.u[:] += a_in_data
        bias = self.bias
        self.v[:] = self.v * (1 - self.dv) + self.u + bias
        s_out = self.v >= self.vth
        self.v[s_out] = 0  # Reset voltage to 0
        self.s_out.send(s_out)
def main():
    # %% [markdown]
    # #### Instantiating the _Process_
    # 
    # Now we can create an instance of our _Process_, in this case a group of 3 LIF neurons.

    # %%
    n_neurons = 3

    lif = LIF(shape=(3,), du=0, dv=0, bias=3, vth=10)

    # %% [markdown]
    # ## Interacting with _Processes_
    # 
    # Once you have instantiated a group of LIF neurons, you can easily interact with them.
    # 
    # #### Accessing _Vars_
    # 
    # You can always read out the current values of the process _Vars_ to determine the _Process_ state. For example, all three neurons should have been initialized with a zero membrane voltage.

    # %%
    print(lif.v.get())

    # %% [markdown]
    # As described above, the _Var_ _v_ has in this example been initialized as a scalar value that describes the membrane voltage of all three neurons simultaneously.

    # %% [markdown]
    # #### Using custom APIs
    # 
    # To facilitate how users can interact with your _Process_, they can use the custom APIs that you provide them with. For LIF neurons, you defined a custom function that allows the user to inspect the internal _Vars_ of the LIF _Process_. Have a look if all _Vars_ have been set up correctly.

    # %%
    lif.print_vars()

    # %% [markdown]
    # #### Executing a _Process_
    # 
    # Once the _Process_ is instantiated and you are satisfied with its state, you can run the _Process_. As long as a _ProcessModel_ has been defined for the desired backend, the _Process_ can run seamlessly across computing hardware. Do not worry about the details here&#8212;you will learn all about how Lava builds, compiles, and runs _Processes_ in a [separate tutorial](./tutorial04_execution.ipynb "Tutorial on Executing Processes").
    # 
    # To run a _Process_, specify the number of steps to run for and select the desired backend.

    # %%
    from lava.magma.core.run_configs import Loihi1SimCfg
    from lava.magma.core.run_conditions import RunSteps

    lif.run(condition=RunSteps(num_steps=1), run_cfg=Loihi1SimCfg())

    # %% [markdown]
    # The voltage of each LIF neuron should now have increased by the bias value, 3, from their initial values of 0. Check if the neurons have evolved as expected.

    # %%
    print(lif.v.get())

    # %% [markdown]
    # #### Update _Vars_
    # 
    # You can furthermore update the internal _Vars_ manually. You may, for example, set the membrane voltage to new values between two runs.

    # %%
    lif.v.set(np.array([1, 2, 3]) )
    print(lif.v.get())

    # %% [markdown]
    # Note that the _set()_ method becomes available once the _Process_ has been run. Prior to the first run, use the *\_\_init\_\_* function of the _Process_ to set _Vars_.
    # 
    # Later tutorials will illustrate more sophisticated ways to access, store, and change variables during run time using _Process_ code.
    # 
    # In the end, stop the process to terminate its execution. 

    # %%
    lif.stop()

    # %% [markdown]
    # ## How to learn more?
    # 
    # Learn how to implement the behavior of _Processes_ in the [next tutorial on ProcessModels](./tutorial03_process_models.ipynb "Tutorial on ProcessModels").
    # 
    # If you want to find out more about _Processes_, have a look at the [Lava documentation](https://lava-nc.org/ "Lava Documentation") or dive into the [source code](https://github.com/lava-nc/lava/tree/main/src/lava/magma/core/process/process.py "Process Source Code").
    # 
    # To receive regular updates on the latest developments and releases of the Lava Software Framework please subscribe to the [INRC newsletter](http://eepurl.com/hJCyhb "INRC Newsletter").


if __name__ == '__main__':
    main()  

#0
#Variables of the LIF:
#      u:    0
#      v:    0
#      du:   0
#      dv:   0
#      bias: 3
#      vth:  10
#Loaded 'lava.magma.compiler.channels.pypychannel'
#Loaded 'threading'
#[3. 3. 3.]
#[1. 2. 3.]
#The thread 'MainThread' (0x1) has exited with code 0 (0x0).
#The program 'python.exe' has exited with code 0 (0x0).