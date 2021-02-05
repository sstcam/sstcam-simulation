from os.path import join, dirname
from os import environ
import requests


def get_data(path):
    return join(dirname(__file__), path)


def download_camera_efficiency_data():
    """
    Download the camera efficiency data from the mpi-hd CTA webserver

    Obtains KONRAD_USERNAME and KONRAD_PASSWORD from the environment
    """
    path = get_data("datasheet/p4eff_ASTRI-CHEC.lis")
    r = requests.get(
        'https://www.mpi-hd.mpg.de/hfm/CTA/MC/Prod4/Config/'
        'Efficiencies/p4eff_ASTRI-CHEC.lis',
        auth=(environ["KONRAD_USERNAME"], environ["KONRAD_PASSWORD"])
    )
    with open(path, 'wb') as f:
        f.write(r.content)


if __name__ == '__main__':
    download_camera_efficiency_data()