# %% [markdown]
# *Copyright (C) 2021 Intel Corporation*<br>
# *SPDX-License-Identifier: BSD-3-Clause*<br>
# *See: https://spdx.org/licenses/*
# 
# ---
# 
# # _ProcessModels_
# 
# This tutorial explains how Lava _ProcessModels_ implement the behavior of Lava _Processes_. Each Lava _Process_ must have one or more _ProcessModels_, which provide the instructions for how to execute a Lava _Process_. Lava _ProcessModels_ allow a user to specify a Process's behavior in one or more languages (like Python, C, or the Loihi neurocore interface) and for various compute resources (like CPUs, GPUs, or Loihi chips). In this way, _ProcessModels_ enable seamles cross-platform execution of _Processes_ and allow users to build applications and algorithms agonostic of platform-specific implementations.
# 
# There are two broad classes of _ProcessModels_: _LeafProcessModel_ and _SubProcessModel_. _LeafProcessModels_, which will be the focus of this tutorial, implement the behavior of a process directly. _SubProcessModels_ allow users to implement and compose the behavior of a _Process_ using other _Processes_, thus enabling the creation of Hierarchical _Processes_.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/lava-nc/lava-nc.github.io/main/_static/images/tutorial03/fig01_leafprocessmodel.png"/>

# %% [markdown]
# In this tutorial, we walk through the creation of multiple _LeafProcessModels_ that could be used to implement the behavior of a Leaky Integrate-and-Fire (LIF) neuron _Process_.

# %% [markdown]
# ## Recommended tutorials before starting: 
# - [Installing Lava](./tutorial01_installing_lava.ipynb "Tutorial on Installing Lava")
# - [Processes](./tutorial02_processes.ipynb "Tutorial on Processes")

# %% [markdown]
# ## Create a LIF _Process_

# %% [markdown]
# First, we will define our LIF _Process_ exactly as it is defined in the `Magma` core library of Lava. (For more information on defining Lava Processes, see the [previous tutorial](./tutorial02_processes.ipynb).) Here the LIF neural _Process_ accepts activity from synaptic inputs via _InPort_ `a_in` and outputs spiking activity via _OutPort_ `s_out`.

# %%
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

class LIF(AbstractProcess):
    """Leaky-Integrate-and-Fire (LIF) neural Process.

    LIF dynamics abstracts to:
    u[t] = u[t-1] * (1-du) + a_in         # neuron current
    v[t] = v[t-1] * (1-dv) + u[t] + bias  # neuron voltage
    s_out = v[t] > vth                    # spike if threshold is exceeded
    v[t] = 0                              # reset at spike

    Parameters
    ----------
    du: Inverse of decay time-constant for current decay.
    dv: Inverse of decay time-constant for voltage decay.
    bias: Mantissa part of neuron bias.
    bias_exp: Exponent part of neuron bias, if needed. Mostly for fixed point
              implementations. Unnecessary for floating point
              implementations. If specified, bias = bias * 2**bias_exp.
    vth: Neuron threshold voltage, exceeding which, the neuron will spike.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1,))
        du = kwargs.pop("du", 0)
        dv = kwargs.pop("dv", 0)
        bias = kwargs.pop("bias", 0)
        bias_exp = kwargs.pop("bias_exp", 0)
        vth = kwargs.pop("vth", 10)

        self.shape = shape
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)
        self.du = Var(shape=(1,), init=du)
        self.dv = Var(shape=(1,), init=dv)
        self.bias = Var(shape=shape, init=bias)
        self.bias_exp = Var(shape=shape, init=bias_exp)
        self.vth = Var(shape=(1,), init=vth)

# %% [markdown]
# ## Create a Python _LeafProcessModel_ that implements the LIF _Process_

# %% [markdown]
# Now, we will create a Python _ProcessModel_, or _PyProcessModel_, that runs on a CPU compute resource and implements the LIF _Process_ behavior.

# %% [markdown]
# #### Setup

# %% [markdown]
# We begin by importing the required Lava classes.
# First, we setup our compute resources (CPU) and our _SyncProtocol_. A _SyncProtocol_ defines how and when parallel _Processes_ synchronize. Here we use the _LoihiProtoicol_ which defines the synchronization phases required for execution on the Loihi chip, but users may also specify a completely asynchronous protocol or define a custom _SyncProtocol_. The decorators imported will be necessary to specify the resource _Requirements_ and _SyncProtocol_ of our _ProcessModel_. 

# %%
import numpy as np
from lava.magma.core.decorator import implements, requires
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

# %% [markdown]
# Now we import the parent class from which our _ProcessModel_ inherits, as well as our required _Port_ and _Variable_ types. _PyLoihiProcessModel_ is the abstract class for a Python _ProcessModel_ that implements the _LoihiProtocol_. Our _ProcessModel_ needs _Ports_ and _Variables_ that mirror those the LIF _Process_. The in-ports and out-ports of a Python _ProcessModel_ have types _PyInPort_ and _PyOutPort_, respectively, while variables have type _LavaPyType_.

# %%
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.proc.lif.process import LIF

# %% [markdown]
# #### Defining a _PyLifModel_ for LIF

# %% [markdown]
# We now define a _LeafProcessModel_ `PyLifModel` that implements the behavior of the LIF _Process_.
# 
# The `@implements` decorator specifies the _SyncProtocol_ (`protocol=LoihiProtocol`) and the class of the _Process_ (`proc=LIF`) corresponding to the _ProcessModel_. The `@requires` decorator specifies the CPU compute resource required by the _ProcessModel_. The `@tag` decorator specifies the precision of the _ProcessModel_. Here we illustrate a _ProcessModel_ with standard, floating point precision.
# 
# Next we define the _ProcessModel_ variables and ports. The variables and ports defined in the _ProcessModel_ must exactly match (by name and number) the variables and ports defined in the corresponding _Process_ for compilation. Our LIF example _Process_ and `PyLifModel` each have 1 input port, 1 output port, and variables for `u`, `v`, `du`, `dv`, `bias`, `bias_exp`, and `vth`. Variables and ports in a _ProcessModel_ must be initialized with _LavaType_ objects specific to the language of the _LeafProcessModel_ implementation. Here, variables are initialized with the `LavaPyType` to match our Python _LeafProcessModel_ implementation. In general, _LavaTypes_ specify the class-types of variables and ports, including their numeric d_type, precision and dynamic range. The Lava Compiler reads these _LavaTypes_ to initialize concrete class objects from the initial values provided in the _Process_.
# 
# We then fill in the `run_spk()` method to execute the LIF neural dynamics. `run_spk()` is a method specific to _LeafProcessModels_ of type `PyLoihiProcessModel` that executes user-defined neuron dynamics with correct handling of all phases our `LoihiProtocol` _SyncProtocol_. In this example, `run_spike` will accept activity from synaptic inputs via _PyInPort_ `a_in`, and, after integrating current and voltage according to current-based (CUBA) dynamics, output spiking activity via _PyOutPort_ `s_out`. `recv()` and `send()` are the methods that support the channel based communication of the inputs and outputs to our _ProcessModel_. For more detailed information about Ports and channel-based communication, see the [Ports Tutorial](./tutorial05_ports.ipynb).

# %%
import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.proc.lif.process import LIF


@implements(proc=LIF, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyLifModel1(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    u: np.ndarray = LavaPyType(np.ndarray, np.float)
    v: np.ndarray = LavaPyType(np.ndarray, np.float)
    bias: np.ndarray = LavaPyType(np.ndarray, np.float)
    bias_exp: np.ndarray = LavaPyType(np.ndarray, np.float)
    du: float = LavaPyType(float, np.float)
    dv: float = LavaPyType(float, np.float)
    vth: float = LavaPyType(float, np.float)

    def run_spk(self):
        a_in_data = self.a_in.recv()
        self.u[:] = self.u * (1 - self.du)
        self.u[:] += a_in_data
        bias = self.bias * (2**self.bias_exp)
        self.v[:] = self.v * (1 - self.dv) + self.u + bias
        s_out = self.v >= self.vth
        self.v[s_out] = 0  # Reset voltage to 0
        self.s_out.send(s_out)

def main():
    # %% [markdown]
    # #### Compile and run _PyLifModel_

    # %%
    from lava.magma.core.run_configs import Loihi1SimCfg
    from lava.magma.core.run_conditions import RunSteps

    lif = LIF(shape=(3,), du=0, dv=0, bias=3, vth=10)

    run_cfg = Loihi1SimCfg()
    lif.run(condition=RunSteps(num_steps=10), run_cfg=run_cfg)
    print(lif.v.get())

    # %% [markdown]
    # ## Create an _NcProcessModel_ that implements the LIF _Process_ 

    # %% [markdown]
    # _Processes_ can have more than one _ProcessModel_, and different _ProcessModels_ can enable execution on different compute resources. The Lava Compiler will soon support the execution of _Processes_ on Loihi Neurocores using the _AbstractNcProcessModel_ class. Below is an example _NcLifModel_ that implements our same LIF _Process_.

    # %% [markdown]
    #    ```python
    #    from lava.proc.lif.process import LIF
    #    from lava.magma.core.decorator import implements, requires
    #    from lava.magma.core.resources import Loihi1NeuroCore
    #    from lava.magma.core.model.nc.model import NcLoihiProcessModel
    #    from lava.magma.core.model.nc.ports import NcInPort, NcOutPort
    #    from lava.magma.core.model.nc.type import LavaNcType, NcVar
    #    
    #    @implements(proc=LIF) #Note that the NcLoihiProcessModel class implies the useage of the Loihi SyncProtcol
    #    @requires(Loihi1NeuroCore)
    #    class NcLifModel(NcLoihiProcessModel):
    #        # Declare port implementation
    #        a_in: InPort =   LavaNcType(NcInPort, precision=16)
    #        s_out: OutPort = LavaNcType(NcOutPort, precision=1)
    #        # Declare variable implementation
    #        u: NcVar =         LavaNcType(NcVar, precision=24)
    #        v: NcVar =         LavaNcType(NcVar, precision=24)
    #        b: NcVar =         LavaNcType(NcVar, precision=12)
    #        du: NcVar =        LavaNcType(NcVar, precision=12)
    #        dv: NcVar =        LavaNcType(NcVar, precision=12)
    #        vth: NcVar =       LavaNcType(NcVar, precision=8)
    # 
    #        def allocate(self, net: mg.Net):
    #            """Allocates neural resources in 'virtual' neuro core."""
    #            num_neurons = self.in_args['shape'][0]
    #            # Allocate output axons
    #            out_ax = net.out_ax.alloc(size=num_neurons)
    #            net.connect(self.s_out, out_ax)
    #            # Allocate compartments
    #            cx_cfg = net.cx_cfg.alloc(size=1,
    #                                   du=self.du,
    #                                   dv=self.dv,
    #                                   vth=self.vth)
    #            cx = net.cx.alloc(size=num_neurons,
    #                                       u=self.u,
    #                                       v=self.v,
    #                                  b_mant=self.b,
    #                                  cfg=cx_cfg)
    #            cx.connect(out_ax)
    #            # Allocate dendritic accumulators
    #            da = net.da.alloc(size=num_neurons)
    #            da.connect(cx)
    #            net.connect(self.a_in, da)
    # 
    #    ```
    # 
    #     
    # 

    # %% [markdown]
    # ## Selecting 1 _ProcessModel_: More on _LeafProcessModel_ attributes and relations

    # %% [markdown]
    # We have demonstrated multiple _ProcessModel_ implementations of a single LIF _Process_. How is one of several _ProcessModels_ then selected as the implementation of a _Process_ during runtime? To answer that question, we take a deeper dive into the attributes  of a _LeafProcessModel_ and the relationship between a _LeafProcessModel_, a _Process_, and a _SyncProtocol_. 
    # 
    # As shown below, a _LeafProcessModel_ implements both a Process (in our example, LIF) and a _SyncProtocol_ (in our example, the _LoihiProtocol_). A _LeafProcessModel_ has a single _Type_. In this tutorial `PyLifModel` has Type `PyLoihiProcessModel`, while `NcLifModel` has Type `NcLoihiProcessModel`. A _LeafProcessModel_ also has one or more resource _Requirements_ that specify the compute resources (for example, a CPU, a GPU, or Loihi Neurocores) or peripheral resources (like access to a camera) that are required for execution. Finally, a _LeafProcessModel_ can have one and more user-defineable _Tags_. _Tags_ can be used, among other customizable reasons, to group multiple _ProcessModels_ for a multi- _Process_ application or to distinguish between multiple _LeafProcessModel_ implementations with the same _Type_ and _SyncProtocol_. As an example, we illustrated above a `PyLoihiProcessModel` for LIF that uses floating point precision and has the tag `@tag('floating_pt')`. There also exists a `PyLoihiProcessModel` that uses fixed point precision and has behavior that is bit-accurate with LIF execution on a Loihi chip; this _ProcessModel_ is distinguished by the tag `@tag('fixed_pt')`. Together, the _Type_, _Tag_ and _Requirement_ attributes of  a _LeafProcessModel_ allow users to define a _RunConfig_ that chooses which of several _LeafProcessModels_ is used to implement a _Process_ at runtime. The Core Lava Library will also provide several preconfigured _RunConfigs_. 
    # 
    # <img src="https://raw.githubusercontent.com/lava-nc/lava-nc.github.io/main/_static/images/tutorial03/fig02_processmodel_tags_reqs_syncprotocols.png"/>

    # %% [markdown]
    # ## How to learn more?
    # 
    # Learn how to execute single _Processes_ and networks of _Processes_ in the [next tutorial](./tutorial04_execution.ipynb).
    # 
    # If you want to find out more about _ProcessModels_, have a look at the [Lava documentation](https://lava-nc.org/) or dive into the [source code](https://github.com/lava-nc/lava/tree/main/src/lava/magma/core/model/model.py).
    # 
    # To receive regular updates on the latest developments and releases of the Lava Software Framework please subscribe to [our newsletter](http://eepurl.com/hJCyhb).



if __name__ == '__main__':
    main()  

