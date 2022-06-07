import subprocess
from os import listdir
from os.path import isfile, join

from benchmark_code import run_code




def main():
    folder_path = "circuite/"
    filenames = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    print("filenames :",filenames)

    outfile = open("output_1.txt", "w")
    outfile.write("@ circuit, nr_q, nr_gates, basic_greedy, contegra1, contegra2, opteinsum\n")
    outfile.close()

    for i, file in enumerate(filenames):



        print("\nfile_name:{} | file_number: {}/{}".format(file, i + 1, len(filenames)))
        outfile = open("output_1.txt", "a")
        try:
            rez = run_code(file)
        except:
            rez ="\nsome problem with  " + file
        outfile.write(rez)
        outfile.close()
        # print("wait")


    outfile.close()
    print("Done and file close!")


if __name__ == '__main__':
    main()
