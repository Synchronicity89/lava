# %% [markdown]
# *Copyright (C) 2021 Intel Corporation*<br>
# *SPDX-License-Identifier: BSD-3-Clause*<br>
# *See: https://spdx.org/licenses/*
# 
# ---
#


def main(): 
    # # Execution
    # 
    # This tutorial covers how to execute single _Processes_ and networks of _Processes_, how to configure execution, how to pause, resume, and stop execution, and how to manually set up a _Compiler_ and _RunTime_ for more fine-grained control.
    # 
    # ## Recommended tutorials before starting:
    # 
    # - [Installing Lava](./tutorial01_installing_lava.ipynb "Tutorial on Installing Lava")
    # - [Processes](./tutorial02_processes.ipynb "Tutorial on Processes")
    # - [ProcessModel](./tutorial03_process_models.ipynb "Tutorial on ProcessModels")
    # 
    # ## Configuring and starting execution
    # To start executing a _Process_ call its method `run(condition=..., run_cfg=...)`. The execution must be configured by passing in both a _RunCondition_ and a _RunConfiguration_.
    # 
    # #### Run conditions
    # A _RunCondition_ specifies how long a _Process_ is executed.
    # 
    # The run condition _RunSteps_ executes a _Process_ for a specified number time steps, here 42 in the example below. The execution will automatically pause after the specified number of time steps.
    # You can also specify whether or not the call to `run()` will block the program flow.

    # %%
    from lava.magma.core.run_conditions import RunSteps

    run_condition = RunSteps(num_steps=42, blocking=False)

    # %% [markdown]
    # The run condition _RunContinuous_ enables you to run a _Process_ continuously. In this case, the _Process_ will run indefinitely until you explicitly call `pause()` or `stop()` (see below). This call never blocks the program flow (blocking=False).

    # %%
    from lava.magma.core.run_conditions import RunContinuous

    run_condition = RunContinuous()

    # %% [markdown]
    # #### Run configurations
    # A _RunConfig_ specifies on what devices the _Processes_ should be executed.
    # Based on the _RunConfig_, a _Process_ selects and initializes exactly one
    # of its associated [_ProcessModels_](./tutorial03_process_models.ipynb "Tutorial on ProcessModels"), which implement the behavior of the _Process_ in a particular programming language and for a particular computing resource.
    # If the _Process_ has a _SubProcessModel_ composed of other _Processes_, the _RunConfig_ chooses the appropriate _ProcessModel_ implementation of the child _Process_.
    # 
    # Since Lava currently only supports execution in simulation on a single CPU,
    # the only predefined _RunConfig_ is _Loihi1SimCfg_, which simulates executing _Processes_ on Loihi.
    # We will make more predefined run configurations available with the upcoming support for Loihi 1 and 2 and
    # other devices such as GPUs.
    # 
    # The example below specifies that the _Process_ (and all its connected _Processes_
    # and _SubProcesses_) are executed in Python on a CPU.

    # %%
    from lava.magma.core.run_configs import Loihi1SimCfg

    run_cfg = Loihi1SimCfg()

    # %% [markdown]
    # We can now use both a _RunCondition_ and a _RunConfig_ to execute a simple leaky integrate-and-fire (LIF) neuron.

    # %%
    from lava.proc.lif.process import LIF
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.core.run_configs import Loihi1SimCfg

    # create a Process for a LIF neuron
    lif = LIF()

    # execute that Process for 42 time steps in simulation
    lif.run(condition=RunSteps(num_steps=42), run_cfg=Loihi1SimCfg())

    # %% [markdown]
    # ## Running multiple _Processes_
    # 
    # Calling `run()` on a _Process_ will also execute all _Processes_ that are connected to it. In the example below, three _Processes_ _lif1_, _dense_, and _lif2_ are connected in a sequence. We call `run()` on _Process_ _lif2_. Since _lif2_ is connected to _dense_ and _dense_ is connected to _lif1_, all three _Processes_ will be executed. As demonstrated here, the execution will cover the entire connected network of _Processes_, irrespective of the direction in which the _Processes_ are connected.

    # %%
    from lava.proc.lif.process import LIF
    from lava.proc.dense.process import Dense
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.core.run_configs import Loihi1SimCfg

    # create processes
    lif1 = LIF()
    dense = Dense()
    lif2 = LIF()

    # connect the OutPort of lif1 to the InPort of dense
    lif1.s_out.connect(dense.s_in)
    # connect the OutPort of dense to the InPort of lif2
    dense.a_out.connect(lif2.a_in)

    # execute Process lif2 and all Processes connected to it (dense, lif1)
    lif2.run(condition=RunSteps(num_steps=42), run_cfg=Loihi1SimCfg())

    # %% [markdown]
    # We will add more on running multiple _Processes_ in the future, including synchronization and running networks of _Processes_ on different devices.

    # %% [markdown]
    # ## Pausing, resuming, and stopping execution
    # 
    # > **Important Note**:
    # >
    # > Right now, Lava does not support `pause()` and _RunContinuous_. These features will be enabled soon in a feature release.
    # > Nevertheless, the following example illustrates how to pause, resume, and stop a process in Lava.
    # 
    # Calling the `pause()` method of a _Process_ pauses execution but preserves its state.
    # The _Process_ can then be inspected and manipulated by the user, as shown in the example below.
    # 
    # Afterward, execution can be resumed by calling `run()` again.
    # 
    # Calling the `stop()` method of a _Process_ completely terminates its execution.
    # Contrary to pausing execution, `stop()` does not preserve the state of the
    # _Process_. If a _Process_ executed on a hardware device, the connection between
    # the _Process_ and the device is terminated as well.

    # %%
    """
    from lava.proc.lif.process import LIF
    from lava.magma.core.run_conditions import RunContinuous
    from lava.magma.core.run_configs import Loihi1SimCfg

    lif3 = LIF()

    # start continuous execution
    lif3.run(condition=RunContinuous(), run_cfg=Loihi1SimCfg())

    # pause execution
    lif3.pause()

    # inspect the state of the Process, here, the voltage variable 'v'
    print(lif.v.get())
    # manipulate the state of the Process, here, resetting the voltage to zero
    lif3.v.set(0)

    # resume continuous execution
    lif3.run(condition=RunContinuous(), run_cfg=Loihi1SimCfg())

    # terminate execution;
    # after this, you no longer have access to the state of lif
    lif3.stop()
    """

    # %% [markdown]
    # ## Manual compilation and execution
    # 
    # In many cases, creating an instance of a _Process_ and calling its `run()`
    # method is all you need to do. Calling `run()` internally first compiles
    # the _Process_ and then starts execution. These steps can also be manually
    # invoked in sequence, for instance to inspect or manipulate the _Process_ before
    # starting execution.
    # 
    # 1. Instantiation stage: This is the call to the init-method of a _Process_,
    # which instantiates an object of the _Process_.

    # %%
    from lava.proc.lif.process import LIF
    from lava.proc.dense.process import Dense

    lif1 = LIF()
    dense = Dense()
    lif2 = LIF()

    # %% [markdown]
    # 2. Configuration stage: After a _Process_ has been instantiated, it can be
    # configured further through its public API and connected to other _Processes_ via
    # its _Ports_. In addition, probes can be defined for Lava _Vars_ in order to
    # record a time series of its evolution during execution (probing will be 
    # supported in an upcoming Lava release).

    # %%
    # connect the processes
    lif1.s_out.connect(dense.s_in)
    dense.a_out.connect(lif2.a_in)

    # %% [markdown]
    # 3. Compile stage: After a _Process_ has been configured, it needs to be compiled to
    # become executable. After the compilation stage, the state of the _Process_ can
    # still be manipulated and inspected.

    # %%
    from lava.magma.compiler.compiler import Compiler
    from lava.magma.core.run_configs import Loihi1SimCfg

    # create a compiler
    compiler = Compiler()

    # compile the Process (and all connected Processes) into an executable
    executable = compiler.compile(lif2, run_cfg=Loihi1SimCfg())

    # %% [markdown]
    # 4. Execution stage: When compilation is complete, _Processes_ can be
    # executed. The execution stage ensures that the (prior) compilation stage has
    # been completed and otherwise invokes it.

    # %%
    from lava.magma.runtime.runtime import Runtime
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.core.process.message_interface_enum import ActorType

    # create and initialize a runtime
#    runtime = Runtime(run_cond=run_condition, exe=executable)
    runtime = Runtime(executable, ActorType.MultiProcessing)
    runtime.initialize()

    # start execution
    runtime.start(run_condition=RunSteps(num_steps=42))

    # stop execution
    runtime.stop()

    # %% [markdown]
    # The following does all of the above automatically:

    # %%
    from lava.proc.lif.process import LIF
    from lava.proc.dense.process import Dense
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.core.run_configs import Loihi1SimCfg

    # create Processes
    lif = LIF()
    dense = Dense()

    # connect Processes
    lif.s_out.connect(dense.s_in)

    # execute Processes
    lif.run(condition=RunSteps(num_steps=42), run_cfg=Loihi1SimCfg())

    # stop Processes
    lif.stop()

    # %% [markdown]
    # ## How to learn more?
    # 
    # In upcoming releases, we will continually publish more and more tutorials, covering, for example, how to transfer data between _Processes_ and how to compose the behavior of a process using other processes.
    # 
    # If you want to find out more about how to compile and execute _Processes_, have a look at the [Lava documentation](https://lava-nc.org/ "Lava Documentation") or dive into the [Compiler](https://github.com/lava-nc/lava/tree/main/src/lava/magma/compiler/ "Compiler Source Code") and [RunTime source code](https://github.com/lava-nc/lava/tree/main/src/lava/magma/runtime/ "Runtime Source Code").
    # 
    # To receive regular updates on the latest developments and releases of the Lava Software Framework please subscribe to the [INRC newsletter](http://eepurl.com/hJCyhb "INRC Newsletter").



if __name__ == '__main__':
    main()  

