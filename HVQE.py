import qem
import _HVQE
from time import time
import os,sys,pickle
import chainer as ch 
import numpy as np
from datetime import datetime

usage = """
Usage:
python HVQE.py [job_name] [job_type] [parameters]
job_name: kagome_open, kagome_periodic, triangular, test
job_type: classical, ground_state, 1_excite_state ... [n]_exicte_state
For classical calculation parameters: 
[num of eigenstates (recommend 10)]
For ground state calculation parameters:
[number of parameters (recommend n*30)]
"""

# Definations
job_name = sys.argv[1]     # choices: kagome_open, kagome_periodic, chain_10, triangular, test
job_type = sys.argv[2]     # choices: classical, ground_state, 1_excite_state ... [n]_excite_state
input_graph = "./input/graph_input_%s.txt"%job_name
output_path = "./output/"+job_name
if job_type == "classical":
    k = int(sys.argv[3])   # Number of lowest energy eigenstates (for ground state)
elif job_type[-5:] == "state":
    n_par = int(sys.argv[3])
else:
    print(usage)
    exit(0)

# if the output directory does not exist, create it
if not os.path.exists(output_path):
    os.mkdir(output_path)

if job_type == "classical":
    return_state=True
    with open(input_graph, 'r') as file:
        exec(file.read())

    complete_graph=complete_graph_input
    del complete_graph_input

    f_out = open(output_path+"/output_classical.txt","w")

    # Run ground state
    print('Computing the %d lowest energies of %s\n'%(k,job_name))
    start=time()
    output=qem.ground_state(complete_graph,k,return_state)
    if qem.GPU==True: qem.sync()
    end=time()
    if return_state==False:
        f_out.write('The %d lowest energies are %s\n'%(k,output))
    if return_state==True:
        for i in range(len(output[0])):
            f_out.write("%s_state_%d,%.8f\n"%(job_name,i,output[0][i]))
        
    print('Solutions found with ARPACK for %.3f seconds\n'%(end-start))

    ## Write gs energy to disk
    if return_state==False:
        np.savetxt(output_path+'/lowest_energies.txt',output)
    else:
        np.savetxt(output_path+'/lowest_energies.txt',output[0])
        # Write gs itself to disk if return_state is True
        with open(output_path+'/gs.dat', 'wb') as file:
            pickle.dump(output[1][:,0],file,protocol=4)
    f_out.close()

elif job_type == "ground_state":
    return_state=True
    
    class Name:
        pass
    class Parameters:
        pass

    # cmd_args=_HVQE.get_command_line_input()
    run_args=Name()
    cmd_args=Parameters()
    cmd_args.n_par = n_par         # number of parameters to be used in the VQE
    cmd_args.par_multiplicity = 1  # The parameter multiplicity
    cmd_args.n_iter = 1            # Number of iterations of the basinhopping routine
    cmd_args.cost_fn = 'energy'    # or 'infidelity'
    cmd_args.temperature = 1.      # Temperature for the metropolis creterion in the scipy basinhopping routine
    cmd_args.stepsize = 1.         # max stepsize of random displacement per parameter after each local optimization in the scipy basinhopping routine
    cmd_args.init_par = None       # A list of initial parameters from which the basinhopping routine starts
    cmd_args.dump_interval = None  # Dump the state of the program to path/dump.dat every dump_interval function calls

    try: # Use GPU if CuPy installation is available. 
        import cupy as xp
        run_args.GPU=True
    except ImportError:
        import numpy as xp
        run_args.GPU=False

    # Make timestamp in UTC of start
    run_args.date_start=str(datetime.utcnow())

    # Load the ansatz from graph_input.txt
    with open(input_graph, 'r') as file:
        exec(file.read())
    run_args.complete_graph=complete_graph_input
    run_args.init_layer=init_layer_input
    run_args.layers =layers_input
    del complete_graph_input
    del init_layer_input
    del layers_input

    # Get the number of qubits from the complete_graph.
    nodes=[node for edge in run_args.complete_graph for node in edge]
    nodes=set(nodes)
    run_args.n=len(nodes)
    del nodes

    # Load the true ground state into memory for computation of infidelities. 
    gs_reg=qem.Reg(run_args.n)
    with open(output_path+'/gs.dat','rb') as file:
        gs_reg.psi.re=xp.array(pickle.load(file)).reshape((2,)*run_args.n)  

    # Print info about current run to stdout.    
    print('Started basinhopping at',run_args.date_start, 'UTC')


    # Prepare the init_reg, whose state is a dimer covering of run_args.complete_graph, as specified by run_args.init_layer.
    init_reg=qem.Reg(run_args.n)   
    for edge in run_args.init_layer:
        qem.apply_prepare_singlet(edge,init_reg)

    #RUN THE VQE
    run_args.start=time()

    global reg_psi_list
    reg_psi_list = []

    global reg_psi
    reg_psi = None

    vqe_out,reg_psi=_HVQE.run_VQE(cmd_args,run_args,init_reg,gs_reg,reg_psi_list)
    reg_psi_list.append(reg_psi)

    # save the reg_psi_list
    with open(output_path+'/reg_psi_list_0.pkl', 'wb') as file_list:
        pickle.dump(reg_psi_list,file_list,protocol=4)

    if run_args.GPU==True:
        qem.sync()
    run_args.end=time() 
    # Wall-clock time of VQE (hours) 
    run_args.wall_clock=(run_args.end-run_args.start)/60/60 

    # Get the infidelity and the energy of the final state irrespective of whether we used the energy or the infidelity as the cost functon.
    vqe_out.opt_parameters=ch.Variable(xp.array(vqe_out.opt_parameters))
    if cmd_args.cost_fn=='energy':
        run_args.E_VQE=vqe_out.cost_VQE #Already a float
        run_args.inf_VQE=_HVQE.infidelity_from_parameters(init_reg,run_args.layers,run_args.n,cmd_args.par_multiplicity,vqe_out.opt_parameters,gs_reg)
        run_args.inf_VQE=float(run_args.inf_VQE.array)
    if cmd_args.cost_fn=='infidelity':
        run_args.inf_VQE=vqe_out.cost_VQE #Already a float
        run_args.E_VQE,reg_psi=_HVQE.Heisenberg_energy_from_parameters(run_args.complete_graph,init_reg,run_args.layers,run_args.n,cmd_args.par_multiplicity,vqe_out.opt_parameters)
        run_args.E_VQE=float(run_args.E_VQE.array)

    vqe_out.opt_parameters=vqe_out.opt_parameters.array.tolist() #Convert for printing and storing.
    run_args.date_end=str(datetime.utcnow()) # End time in UTC


    # Write input and results to disk. If no former output exists, print a line explaining the data in the output file.

    output=str([vars(cmd_args),vars(run_args),vars(vqe_out)])

    if not os.path.exists(output_path+'/output_ground.txt'):
        f=open(output_path+'/output_ground.txt', 'w')
    with open(output_path+'/output_ground.txt', 'a') as f:
        f.write(output+'\n\n')
    
    print('Finished basinhopping of ',output_path, 'at',run_args.date_end,'UTC, with')

    f_out = open(output_path+"/output_HVQE_0_parm%d.txt"%(cmd_args.n_par),"w")

    print(vars(cmd_args))
    print('init_par =', vqe_out.init_par)
    print(' ')
    f_out.write('E_VQE,%.8f\n'%(run_args.E_VQE))
    f_out.write('inf_VQE,%.8f\n'%(run_args.inf_VQE))
    f_out.write('n_fn_calls,%s\n'%(vqe_out.n_fn_calls))
    f_out.write('Wall-clock time(hours),%.3f\n'%(run_args.wall_clock))
    f_out.close()

elif job_type[-13:] == "_excite_state":
    return_state=True
    
    class Name:
        pass
    class Parameters:
        pass

    # cmd_args=_HVQE.get_command_line_input()
    run_args=Name()
    cmd_args=Parameters()
    cmd_args.n_par = n_par         # number of parameters to be used in the VQE
    cmd_args.par_multiplicity = 1  # The parameter multiplicity
    cmd_args.n_iter = 1            # Number of iterations of the basinhopping routine
    cmd_args.cost_fn = 'energy'    # or 'infidelity'
    cmd_args.temperature = 1.      # Temperature for the metropolis creterion in the scipy basinhopping routine
    cmd_args.stepsize = 1.         # max stepsize of random displacement per parameter after each local optimization in the scipy basinhopping routine
    cmd_args.init_par = None       # A list of initial parameters from which the basinhopping routine starts
    cmd_args.dump_interval = None  # Dump the state of the program to path/dump.dat every dump_interval function calls

    try: # Use GPU if CuPy installation is available. 
        import cupy as xp
        run_args.GPU=True
    except ImportError:
        import numpy as xp
        run_args.GPU=False

    # Make timestamp in UTC of start
    run_args.date_start=str(datetime.utcnow())

    # Load the ansatz from graph_input.txt
    with open(input_graph, 'r') as file:
        exec(file.read())
    run_args.complete_graph=complete_graph_input
    run_args.init_layer=init_layer_input
    run_args.layers =layers_input
    del complete_graph_input
    del init_layer_input
    del layers_input

    # Get the number of qubits from the complete_graph.
    nodes=[node for edge in run_args.complete_graph for node in edge]
    nodes=set(nodes)
    run_args.n=len(nodes)
    del nodes

    # Load the true ground state into memory for computation of infidelities. 
    gs_reg=qem.Reg(run_args.n)
    with open(output_path+'/gs.dat','rb') as file:
        gs_reg.psi.re=xp.array(pickle.load(file)).reshape((2,)*run_args.n)  

    # Print info about current run to stdout.    
    print('Started basinhopping at',run_args.date_start, 'UTC')


    # Prepare the init_reg, whose state is a dimer covering of run_args.complete_graph, as specified by run_args.init_layer.
    init_reg=qem.Reg(run_args.n)   
    for edge in run_args.init_layer:
        qem.apply_prepare_singlet(edge,init_reg)

    #RUN THE VQE
    run_args.start=time()

    with open(output_path+'/reg_psi_list_%d.pkl'%(int(job_type[0:-13])-1), 'rb') as file_list_pre:
        reg_psi_list = pickle.load(file_list_pre)

    reg_psi = None

    vqe_out,reg_psi=_HVQE.run_VQE(cmd_args,run_args,init_reg,gs_reg,reg_psi_list)
    reg_psi_list.append(reg_psi)

    # save the reg_psi_list
    with open(output_path+'/reg_psi_list_%d.pkl'%(int(job_type[0:-13])), 'wb') as file_list:
        pickle.dump(reg_psi_list,file_list,protocol=4)

    if run_args.GPU==True:
        qem.sync()
    run_args.end=time() 
    # Wall-clock time of VQE (hours) 
    run_args.wall_clock=(run_args.end-run_args.start)/60/60 

    # Get the infidelity and the energy of the final state irrespective of whether we used the energy or the infidelity as the cost functon.
    vqe_out.opt_parameters=ch.Variable(xp.array(vqe_out.opt_parameters))
    if cmd_args.cost_fn=='energy':
        run_args.E_VQE=vqe_out.cost_VQE #Already a float
        run_args.inf_VQE=_HVQE.infidelity_from_parameters(init_reg,run_args.layers,run_args.n,cmd_args.par_multiplicity,vqe_out.opt_parameters,gs_reg)
        run_args.inf_VQE=float(run_args.inf_VQE.array)
    if cmd_args.cost_fn=='infidelity':
        run_args.inf_VQE=vqe_out.cost_VQE #Already a float
        run_args.E_VQE,reg_psi=_HVQE.Heisenberg_energy_from_parameters(run_args.complete_graph,init_reg,run_args.layers,run_args.n,cmd_args.par_multiplicity,vqe_out.opt_parameters)
        run_args.E_VQE=float(run_args.E_VQE.array)

    vqe_out.opt_parameters=vqe_out.opt_parameters.array.tolist() #Convert for printing and storing.
    run_args.date_end=str(datetime.utcnow()) # End time in UTC


    # Write input and results to disk. If no former output exists, print a line explaining the data in the output file.

    output=str([vars(cmd_args),vars(run_args),vars(vqe_out)])

    if not os.path.exists(output_path+'/output_ground.txt'):
        f=open(output_path+'/output_ground.txt', 'w')
    with open(output_path+'/output_ground.txt', 'a') as f:
        f.write(output+'\n\n')
    
    print('Finished basinhopping of ',output_path, 'at',run_args.date_end,'UTC, with')

    f_out = open(output_path+"/output_HVQE_%d_parm%d.txt"%(int(job_type[0:-13]),cmd_args.n_par),"w")

    print(vars(cmd_args))
    print('init_par =', vqe_out.init_par)
    print(' ')
    f_out.write('E_VQE,%.8f\n'%(run_args.E_VQE))
    f_out.write('inf_VQE,%.8f\n'%(run_args.inf_VQE))
    f_out.write('n_fn_calls,%s\n'%(vqe_out.n_fn_calls))
    f_out.write('Wall-clock time(hours),%.3f\n'%(run_args.wall_clock))
    f_out.close()

else:
    pass

