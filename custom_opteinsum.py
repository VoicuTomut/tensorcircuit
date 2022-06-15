


import opt_einsum as oem
from opt_einsum.path_random import ssa_path_compute_cost  # cost of a path
import numpy as np

import tensorcircuit as tc

import qiskit
import pyzx as zx

import time

from my_optimizer import MansikkaOptimizer

class MyOptimizer(oem.paths.PathOptimizer):

    def __call__(self, inputs, output, size_dict, memory_limit=None):

        print("inputs:",inputs)
        print("outputs:", output)
        # print("size_dic:", size_dict)

        fs_inputs = [frozenset(x) for x in inputs]
        output = frozenset(output) | frozenset.intersection(*fs_inputs)

        print("fs_inputs", fs_inputs)
        print("outputs:", output)

        contraction_order = [(0, 1)] * (len(inputs) - 1)
        # print("contraction_order", contraction_order)



        return contraction_order



def  get_tensot_circuit(circuit_path):
    circuit = zx.Circuit.load(circuit_path)
    # print(circuit.qubits)
    qasm_circuit = circuit.to_qasm()
    qiskit_circuit = qiskit.QuantumCircuit.from_qasm_str(qasm_circuit)
    tensor_circuit = tc.Circuit.from_qiskit(qiskit_circuit)
    return tensor_circuit

def  contract_circuit(circuit_name, circuit_folder = "circuite/"):

    circuit_path = circuit_folder+circuit_name
    tensor_circuit =get_tensot_circuit(circuit_path)
    def net():
        return tensor_circuit.matrix()

    tc.set_contractor(
        "custom_stateful",
        optimizer=MansikkaOptimizer
    )

    #


    tic = time.perf_counter()
    opteinsum1_mat = net()
    print("oprteinsum_mat", opteinsum1_mat)
    toc = time.perf_counter()
    opteinsum1_time = toc - tic

    # Basic greedy
    tc.set_contractor("greedy", contraction_info=False)
    tic = time.perf_counter()
    basic_mat = net()
    toc = time.perf_counter()
    basic_time = toc - tic
    print("basic mat", basic_mat)
    #print("basic_time:", basic_time)



    s = 0
    for i in range(len(basic_mat)):
        for j in range(len(basic_mat)):
            s=s+abs(basic_mat[i][j]-opteinsum1_mat[i][j])**2
    print("dif ", s)

    print("done")


#####
contract_circuit(circuit_name="000_test_circuit.qasm", circuit_folder = "circuite/")