import os
#import zipfile
import urllib
import tarfile
import argparse
import urllib.request

def parse_arguments():
    parser = argparse.ArgumentParser()
    # naming / file handling
    parser.add_argument(
        '--task', type=str, default='all', help='type of data to download')
    return parser.parse_args()

def download_dataset(source_url, target_dir, target_file):
    global downloaded
    downloaded = 0
    def show_progress(count, block_size, total_size):
        global downloaded
        downloaded += block_size
        print('downloading ... %d%%' % round(((downloaded*100.0) / total_size)))

    print('downloading ... ')
    urllib.request.urlretrieve(source_url, filename=target_file, reporthook=show_progress)

    print('downloading ... done')

    print('extracting ...')
    tar = tarfile.open(target_file, "r:xz")
    tar.extractall()
    tar.close()
    os.remove(target_file)
    print('extracting ... done')

def download_training():
    source_url = 'https://nuage.lix.polytechnique.fr/index.php/s/gmnGHjNq7WKipRA/download'
    target_dir = os.path.dirname(os.path.abspath(__file__))
    target_file = os.path.join(target_dir, 'training_data.tar.xz')
    download_dataset(source_url,  target_dir, target_file)

def download_testing():
    source_url = 'https://nuage.lix.polytechnique.fr/index.php/s/3ZcFtqKm6Z27ZJ6/download'
    target_dir = os.path.dirname(os.path.abspath(__file__))
    target_file = os.path.join(target_dir, 'pointCleanNetDataset.tar.gz')
    download_dataset(source_url,  target_dir, target_file)

def download_models():
    source_url = 'https://nuage.lix.polytechnique.fr/index.php/s/FTCyp5WHg7Z68EM/download'
    target_dir = os.path.dirname(os.path.abspath(__file__))
    target_file = os.path.join(target_dir, 'dse_meshing_pretrained_models.tar.xz')
    download_dataset(source_url,  target_dir, target_file)

if __name__ == '__main__':
    opt = parse_arguments()
    if opt.task == "training" or opt.task == "all":
        download_training()
    elif opt.task == "testing":
        download_testing()
    elif opt.task == "models":
        download_models()
    else:
        print('unknown data type. Choose between training, testing, models or all')
