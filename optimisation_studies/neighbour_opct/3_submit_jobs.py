import argparse
from subprocess import call
from os.path import abspath, dirname, join

DIR = abspath(dirname(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='pe_paths', nargs='+',
                        help="paths to camera files")
    parser.add_argument('-c', dest='camera_paths', nargs='+',
                        help="paths to camera files")
    parser.add_argument('--dry', dest='dry', action="store_true")
    args = parser.parse_args()
    pe_paths = args.pe_paths
    camera_paths = args.camera_paths
    dry = args.dry

    script_path = join(DIR, "2_define_job.py")
    python_cmd = "python {} -i {} -c {} -o {} --nsb 100 --trigger 600\n"
    correct_permissions = "getfacl -d . | setfacl --set-file=- {}\n"

    for ifile, pe_path in enumerate(pe_paths):
        for camera_path in camera_paths:
            h5_path = camera_path.replace(".pkl", f"_{ifile}_events.h5")
            shell_path = camera_path.replace(".pkl", f"_{ifile}.sh")

            with open(shell_path, 'w') as file:
                file.write("source $HOME/.bash_profile\n")
                file.write("source activate cta\n")
                file.write("export NUMBA_NUM_THREADS=6\n")
                file.write(f"cd {DIR}\n")
                file.write("pwd\n")
                file.write(python_cmd.format(script_path, pe_path, camera_path, h5_path))
                file.write(correct_permissions.format(h5_path))
                file.write(f"if [ -f {h5_path} ]; then\n")
                file.write(f"\trm -f {shell_path}\n")
                file.write("fi\n")
            call("chmod +x {}".format(shell_path), shell=True)

            qsub_cmd = "qsub -cwd -V -q lfc.q {}".format(shell_path)
            print(qsub_cmd)
            if not dry:
                call(qsub_cmd, shell=True)


if __name__ == '__main__':
    main()
