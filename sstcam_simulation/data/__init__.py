from os.path import join, dirname, exists


def get_data(path):
    return join(dirname(__file__), path)


def get_cherenkov_data():
    gamma_path = get_data("cherenkov/gamma.h5")
    proton_path = get_data("cherenkov/gamma.h5")
    if not exists(gamma_path) or not exists(proton_path):
        msg = '''
        Cherenkov files have not been downloaded to sstcam_simulation/data/cherenkov. 
        The files can be downloaded from Nextcloud 
        https://pcloud.mpi-hd.mpg.de/index.php/f/142621
        '''
        raise ValueError(msg)
    return [gamma_path, proton_path]
