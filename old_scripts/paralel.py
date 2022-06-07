import subprocess
from os import listdir
from os.path import isfile, join






def main():
    folder_path = "../mini_circuite/"
    filenames = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    print("filenames :",filenames)

    outfile = open("output_0.txt", "w")
    outfile.write("@ circuit, nr_q, nr_gates, basic_greedy, contegra1, contegra2, opteinsum\n")

    for i, file in enumerate(filenames):


        proc = subprocess.Popen("python", stdin=subprocess.PIPE, stdout=outfile, stderr=outfile, shell=True)
        print("file_name:{} | file_number: {}/{}".format(file, i + 1, len(filenames)))
        proc.stdin.write('from paralel_suport import run_code\n'.encode())
        proc.stdin.write("arg = '{}'\n".format(file).encode())
        proc.stdin.write("run_code(arg)\n".encode())


    outfile.close()
    print("Done and file close!")


if __name__ == '__main__':
    main()
