"""Download data from our server
"""
import requests
import shutil
import os
import zipfile

SERVER_URL = 'http://icarus.cs.weber.edu/~hvalle/cs4580/data/'


def download_file(url, file_name):
    # TODO: Download to pwd
    local_filename = os.path.join(os.getcwd(), file_name)
    with requests.get(url + file_name, steam=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    print(f"Downloaded {file_name} to {local_filename}")

    # TODO: Check extension from file_name, if it is zip
    # Call unzip_file()
    # unzip_file(data01)
    if file_name.endswith('.zip'):
        unzip_file(local_filename)

def unzip_file(file_name):
    # TODO: Unzip file
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        extract_path = os.path.join(os.getcwd(), os.path.splitext(file_name)[0])
        zip_ref.extractall(extract_path)
        print(f"Unzipped {file_name} to {extract_path}")

def main():
    """Driven Function
    """
    data01 = 'pandas01Data.zip'
    download_file(SERVER_URL, data01)

if __name__ == '__main__':
    main()