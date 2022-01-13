"""
Contains function defs for running HVQE.py
"""
class Name: # Simple namespace class that is used for dumping and restarting the program.
    pass

import numpy # For the cases where GPU==True and we still want to use numpy.
import qem
import chainer as ch
from datetime import datetime
import argparse
import scipy.optimize
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

try: # Use GPU if CuPy installation is available. 
    import cupy as xp
except ImportError:
    import numpy as xp


def Heisenberg_energy_from_parameters(complete_graph,init_reg,layers,n,par_multiplicity,parameters,reg_psi_list):
    """
    Return the energy of a state as defined via the init state, the ansatz and a setting for the parameters. Ansatz must already be mapped to ints and given as regular python list.
    
    Returns
    -------
    E : chainer.Variable
    
    """
    reg=qem.EmptyReg(n)
    reg.psi=init_reg.psi

    edges=[edge for layer in layers for edge in layer]

    for i in range(len(parameters)):
        gate=qem.Heisenberg_exp(parameters[i])
        for j in range(par_multiplicity):
            edge=edges[(i*par_multiplicity+j)%len(edges)]
            action=qem.Action(edge,gate)
            qem.apply_action(action,reg)

    E,reg_psi=qem.Heisenberg_energy(complete_graph,reg,reg_psi_list)
    return E,reg_psi

def infidelity_from_parameters(init_reg,layers,n,par_multiplicity,parameters,gs_reg):
    reg=qem.EmptyReg(n)
    reg.psi=init_reg.psi    

    edges=[edge for layer in layers for edge in layer]

    for i in range(len(parameters)):
        gate=qem.Heisenberg_exp(parameters[i])
        for j in range(par_multiplicity):
            edge=edges[(i*par_multiplicity+j)%len(edges)] 
            action=qem.Action(edge,gate)
            qem.apply_action(action,reg)

    inf=qem.infidelity(reg,gs_reg)
    return inf

def run_VQE(cmd_args,run_args,init_reg,gs_reg,reg_psi_list):
    """
    Run the VQE.
    """
    global reg_psi
    # global reg_psi_list
    
    vqe_out=Name()
    vqe_out.n_fn_calls=0
    vqe_out.local_min_list=[]
    vqe_out.local_min_parameters_list=[]
    vqe_out.local_min_accept_list=[]
    
    def calc_cost(parameters):
        nonlocal vqe_out
        nonlocal cmd_args
        nonlocal run_args
        global reg_psi
        tmp=Name()
        parameters=ch.Variable(xp.array(parameters))
        
        if cmd_args.cost_fn=='energy':
            cost,reg_psi=Heisenberg_energy_from_parameters(run_args.complete_graph,init_reg,run_args.layers,run_args.n,cmd_args.par_multiplicity,parameters,reg_psi_list)
        elif cmd_args.cost_fn=='infidelity':
            cost=infidelity_from_parameters(init_reg,run_args.layers,run_args.n,cmd_args.par_multiplicity,parameters,gs_reg)
        else:
            raise ValueError('Not a valid cost function')
        cost.backward()
        g=parameters.grad
        vqe_out.n_fn_calls+=1
        print('.',end='',flush=True) #Progress indicator. One dot per function call. Here one function call is defined as one forward and one backward evaluation. 
        if run_args.GPU==True:
            cost=cost.array.get()
            g=g.get()
        elif run_args.GPU==False:
            cost=cost.array

        ### Dump state of the prograpm. Restart has to be done by hand by running another HVQE.py from the command line. 
        if cmd_args.dump_interval!=None:
            if vqe_out.n_fn_calls%cmd_args.dump_interval==0:
                tmp=Name()
                tmp.parameters=parameters.array.tolist()
                tmp.cost=float(cost)
                tmp.g=g.tolist()
                date_dump=str(datetime.utcnow()) # Current time in UTC.
                vqe_out.init_par=list(vqe_out.init_par)
                dump=[vars(cmd_args),vars(run_args),vars(vqe_out),vars(tmp)]
                with open(cmd_args.path+'/dump.txt', 'a') as file:
                    file.write(str(dump)+'\n\n')
                print('Data dump on', date_dump)
        ###
        return cost, g

    def callback(x,f,accept): # Due to a bug in the scipy (version 1.3.1) basinhopping routine, this function is not called after the first local minimum. Hence the lists local_min_list, local_min_parameters_list and local_min_accept_list will not contain entries for the first local minimum found. This issue will be solved in version 1.6.0 of schipy. See https://github.com/scipy/scipy/pull/13029
        #If basinhopping is run with n_iter=0, only a single local minimum is found, and in this case the value of the cost function, and the parameters, are in fact stored, because this minimum is the optimal minimum found and delivers the data for the output of the bassinhopping routine as a whole.
        nonlocal vqe_out
        print('\nNew local min for', vars(cmd_args))
        print('cost=',float(f),'accepted=',accept,'parameters=',list(x))
        vqe_out.local_min_list.append(float(f))   
        vqe_out.local_min_parameters_list.append(list(x))
        vqe_out.local_min_accept_list.append(accept)
       
    if cmd_args.init_par is None:
        vqe_out.init_par=numpy.random.rand(cmd_args.n_par)/1000-1/2000
    else:
        assert len(cmd_args.init_par)==cmd_args.n_par, 'List of initial parameters must be of length n_par.'
        vqe_out.init_par=numpy.array(cmd_args.init_par)
    if cmd_args.n_par==0: # If there is no circuit, just output the energy of the init state.
        if cmd_args.cost_fn=='energy':
            vqe_out.cost_VQE,reg_psi=Heisenberg_energy_from_parameters(run_args.complete_graph,init_reg,run_args.layers,run_args.n,cmd_args.par_multiplicity,[],reg_psi_list) # Still a Chainer.Variable
            print(reg_psi)
            vqe_out.cost_VQE=float(vqe_out.cost_VQE.array)
            vqe_out.opt_parameters=[]
            vqe_out.init_par=[]
        if cmd_args.cost_fn=='infidelity':
            vqe_out.cost_VQE=infidelity_from_parameters(init_reg,run_args.layers,run_args.n,cmd_args.par_multiplicity,[],gs_reg)
            vqe_out.cost_VQE=float(vqe_out.cost_VQE.array)
            vqe_out.opt_parameters=[]
            vqe_out.init_par=[]
            
    else:
        sol=scipy.optimize.basinhopping(calc_cost,vqe_out.init_par,stepsize=cmd_args.stepsize,minimizer_kwargs={'jac':True},niter=cmd_args.n_iter,interval=25,callback=callback,T=cmd_args.temperature)
        vqe_out.cost_VQE=float(sol.fun)
        vqe_out.opt_parameters=sol.x.tolist()
        vqe_out.init_par=list(vqe_out.init_par)

    return vqe_out,reg_psi
     
def plot_VQE_data(path,fn,par_multiplicity,gates_per_cycle):
    # Import data
    with open(path+'/output.txt','r') as f:
        f.readline()
        data=f.readlines()
        data=[line for line in data if line != '\n']
        data=[eval(x.strip()) for x in data]
        
    E=0
    if fn=='energy':    
        with open(path+'/lowest_energies.txt','r') as f:
            E=f.readlines()
            E=[eval(x.strip()) for x in E]
            E[0]=E[0]

    # Sort data into sublists based on the value of n_iter
    n_iter_set=set([line[0]['n_iter'] for line in data])
    data_=[]
    for n_iter in n_iter_set:
        n_iter_array=[line for line in data if line[0]['n_iter']==n_iter]
        n_iter_array.sort(key=lambda x: x[0]['n_par']) # Put datapoints in order of increasing number of parameters. 
        data_.append(n_iter_array)

    data=data_
    n_iter_set=list(n_iter_set)
    
    # Make one plot for every possible val of n_iter, all in one figure.
    fig, ax = plt.subplots()
    for n_iter_class in data:
        n_par_list=[line[0]['n_par'] for line in n_iter_class]
        p_list=[n_par*par_multiplicity/gates_per_cycle for n_par in n_par_list]
        if fn=='energy':
            E_VQE_list=[line[1]['E_VQE'] for line in n_iter_class]
            E_VQE_list=[-(E_VQE-E[0])/E[0] for E_VQE in E_VQE_list] # The relative error in the energy is going to be plotted.
            ax.semilogy(p_list,E_VQE_list,'-o')
        elif fn=='infidelity':
            inf_VQE_list=[line[1]['inf_VQE'] for line in n_iter_class]
            ax.semilogy(p_list,inf_VQE_list,'-o')
        elif fn=='wall_clock':
            wall_clock_list=[line[1]['wall_clock'] for line in n_iter_class]
            ax.semilogy(p_list,wall_clock_list,'-o')

    if fn=='energy':
        ax.axhline(y=-(E[1]-E[0])/E[0]) # Plot a horizontal line at the first excited state.
        ax.axhline(y=-(E[1]-E[0])/E[0]/2,ls='--') # Plot a horizontal dashed line halfway the ground state and the first excited state.
        ax.set_ylabel('Relative energy error')
    elif fn=='infidelity':
        ax.set_ylabel('Infidelity')
    elif fn=='wall_clock':
        ax.set_ylabel('Wall-clock time (h)')

    # On the x-axis, put the number of cycles rather then the number of parameters. 
    ax.set_xlabel('p') 
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)
    ax.legend(n_iter_set, title='n_iter')
    plt.title(path)


    # Write to disk.
    if fn=='energy':
        plt.savefig(path+'/E_VQE.pdf')
    if fn=='infidelity':
        plt.savefig(path+'/inf_VQE.pdf')    
    if fn=='wall_clock':
        plt.savefig(path+'/wall_clock.pdf')
