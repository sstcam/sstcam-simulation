import argparse
from subprocess import call
from os.path import abspath, dirname, join

DIR = abspath(dirname(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='paths', nargs='+',
                        help="paths to event files")
    parser.add_argument('--dry', dest='dry', action="store_true")
    args = parser.parse_args()
    paths = args.paths
    dry = args.dry

    script_path = join(DIR, "3_summarise_runs.py")
    python_cmd = "python {} -i {}\n"
    correct_permissions = "getfacl -d . | setfacl --set-file=- {}\n"

    for path in paths:
        output_path = path.replace("_events.h5", "_summary.h5")
        shell_path = path.replace("_events.h5", "_summary.sh")

        with open(shell_path, 'w') as file:
            file.write("source $HOME/.bash_profile\n")
            file.write("source activate cta\n")
            file.write("export NUMBA_NUM_THREADS=6\n")
            file.write(f"cd {DIR}\n")
            file.write("pwd\n")
            file.write(python_cmd.format(script_path, path))
            file.write(correct_permissions.format(output_path))
            file.write(f"if [ -f {output_path} ]; then\n")
            file.write(f"\trm -f {shell_path}\n")
            file.write("fi\n")
        call("chmod +x {}".format(shell_path), shell=True)

        qsub_cmd = "qsub -cwd -V -q lfc.q {}".format(shell_path)
        print(qsub_cmd)
        if not dry:
            call(qsub_cmd, shell=True)


if __name__ == '__main__':
    main()
