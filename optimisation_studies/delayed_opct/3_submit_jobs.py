import argparse
from subprocess import call
from os.path import abspath, dirname, join

DIR = abspath(dirname(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='camera_paths', nargs='+',
                        help="paths to camera files")
    parser.add_argument(
        '--50peref', default=False, action='store_true', dest='use_50pe_ref',
        help='Use a 50pe reference pulse for the charge extraction'
    )
    parser.add_argument('--dry', dest='dry', action="store_true")
    args = parser.parse_args()
    camera_paths = args.camera_paths
    use_50pe_ref = args.use_50pe_ref
    dry = args.dry

    script_path = join(DIR, "2_define_job.py")
    python_cmd = "python {} -i {} --nsb 100 --trigger 600\n"
    if use_50pe_ref:
        python_cmd = "python {} -i {} --nsb 100 --trigger 600 --50peref\n"
    correct_permissions = "getfacl -d . | setfacl --set-file=- {}\n"

    for camera_path in camera_paths:
        h5_path = camera_path.replace(".pkl", "_lab.h5")
        shell_path = camera_path.replace(".pkl", ".sh")

        with open(shell_path, 'w') as file:
            file.write("source $HOME/.bash_profile\n")
            file.write("source activate cta\n")
            file.write("export NUMBA_NUM_THREADS=6\n")
            file.write(f"cd {DIR}\n")
            file.write("pwd\n")
            file.write(python_cmd.format(script_path, camera_path))
            file.write(correct_permissions.format(h5_path))
            file.write(f"if [ -f {h5_path} ]; then\n")
            file.write(f"\trm -f {shell_path}\n")
            file.write("fi\n")
        call("chmod +x {}".format(shell_path), shell=True)

        qsub_cmd = "qsub -cwd -V -q -j lfc.q {}".format(shell_path)
        print(qsub_cmd)
        if not dry:
            call(qsub_cmd, shell=True)


if __name__ == '__main__':
    main()
