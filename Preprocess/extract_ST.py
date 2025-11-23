import os
import json
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)

    args = parser.parse_args()

    dir = args.path
    FileList = os.listdir(dir)
    for file in FileList:
        if file.find('AP') >= 0 and file.find('.json') >= 0:
            with open(os.path.join(dir, file), 'r') as f:
                json_data = json.load(f)
            f.close()
            SliceTiming = json_data['SliceTiming']
            targetdir = os.path.join(dir, 'ST_AP.txt')
            with open(targetdir, 'w') as f:
                for item in SliceTiming:
                    f.write("%s\n" % item)
            f.close()

        if file.find('PA') >= 0 and file.find('.json') >= 0:
            with open(os.path.join(dir, file), 'r') as f:
                json_data = json.load(f)
            f.close()
            SliceTiming = json_data['SliceTiming']
            targetdir = os.path.join(dir, 'ST_PA.txt')
            with open(targetdir, 'w') as f:
                for item in SliceTiming:
                    f.write("%s\n" % item)
            f.close()
            