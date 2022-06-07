
import tensorcircuit as tc
import cotengra as ctg
import opt_einsum as oem
from my_optimizer import MansikkaOptimizer
import qiskit
import pyzx as zx
import time


def run_code(arg):
    return get_time_circuit(arg)



def get_time_circuit(circuit_name):

    basic_time = -1
    contegra1_time = -1
    contegra2_time = -1
    opteinsum1_time = -1

    K = tc.set_backend("tensorflow")

    circuit_folder = "circuite/"
    circuit_path = circuit_folder + circuit_name
    circuit = zx.Circuit.load(circuit_path)
    nr_q = circuit.qubits
    nr_gates = len(circuit.gates)
    if nr_q>16:
        print("{} > 16Q".format(circuit_name))
        return "\n{} > 16Q".format(circuit_name)
    print("work on : {}".format(circuit_name))
    qasm_circuit = circuit.to_qasm()

    qiskit_circuit = qiskit.QuantumCircuit.from_qasm_str(qasm_circuit)
    tensor_circuit = tc.Circuit.from_qiskit(qiskit_circuit)
    print("##", circuit_name)


    def net():
        return tensor_circuit.matrix()

    # Basic greedy
    tc.set_contractor("greedy", debug_level=1, contraction_info=False)
    tic = time.perf_counter()
    basic_mat = net()
    toc = time.perf_counter()
    basic_time = toc - tic
    print("basic_time:", basic_time)

    # # Contegra 1  greedy and kohypar
    # opt = ctg.ReusableHyperOptimizer(
    #     methods=["greedy", "kahypar"],
    #     parallel=True,
    #     minimize="combo",
    #     max_time=2,
    #     max_repeats=2,
    #     progbar=True,
    # )
    #
    # # Caution: for now, parallel only works for "ray" in newer version of python
    #
    # tc.set_contractor(
    #     "custom", optimizer=opt, preprocessing=True, contraction_info=False, debug_level=0
    # )
    #
    # tic = time.perf_counter()
    # contegra1_mat = net()
    # toc = time.perf_counter()
    # contegra1_time = toc - tic
    #
    # print("contrgra1_time", contegra1_time)
    #
    # # Contegra 2
    # opt = ctg.ReusableHyperOptimizer(
    #     minimize="combo",
    #     max_repeats=2,
    #     max_time=2,
    #     progbar=True,
    # )
    #
    # def opt_reconf(inputs, output, size, **kws):
    #     tree = opt.search(inputs, output, size)
    #     tree_r = tree.subtree_reconfigure_forest(
    #         progbar=True, num_trees=10, num_restarts=20, subtree_weight_what=("size",)
    #     )
    #     return tree_r.get_path()
    #
    # tc.set_contractor(
    #     "custom",
    #     optimizer=opt_reconf,
    #     contraction_info=False,
    #     preprocessing=True,
    #     debug_level=0,
    # )
    # tic = time.perf_counter()
    # contegra2_mat = net()
    # toc = time.perf_counter()
    # contegra2_time = toc - tic
    #
    # print("contrgra2_time", contegra2_time)

    # opteinsum
    tc.set_contractor(
        "custom_stateful",
        optimizer=MansikkaOptimizer,
    )

    tic = time.perf_counter()
    opteinsum1_mat = net()
    toc = time.perf_counter()
    opteinsum1_time = toc - tic

    print("opteinsum1_time", opteinsum1_time)
    print("@ {}, {}, {}, {}, {}, {}, {} ".format(circuit_name,nr_q,nr_gates, basic_time, contegra1_time, contegra2_time, opteinsum1_time))
    return "\n@ {}, {}, {}, {}, {}, {}, {} ".format(circuit_name,nr_q,nr_gates, basic_time, contegra1_time, contegra2_time, opteinsum1_time)