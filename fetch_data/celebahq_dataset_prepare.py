import os
import zipfile
import random
import argparse
import subprocess
import sys
from bin import gen_mask_dataset

class Gen_Mask_Args:

    def __init__(self,config, indir, outdir):
        self.config = config
        self.indir = indir
        self.outdir = outdir
        self.n_jobs = 0
        self.ext = 'jpg'

class Celeb_Prepare:

    def __init__(self,repoPath,dataPath):

        self.repoPath = repoPath
        self.dataPath = dataPath

        # repo files
        self.celeb_yaml = self.repoPath + "/configs/training/location/celeba.yaml"
        self.train_shuffled = self.repoPath + "/fetch_data/train_shuffled.flist"
        self.val_shuffled = self.repoPath + "/fetch_data/val_shuffled.flist"

        # data folders provided or built
        self.celeb_zip = self.dataPath + "/data256x256.zip"
        self.dataset_dir = self.dataPath + "/celeba-hq-dataset"
        self.celeb_unzipped = self.dataset_dir + "/data256x256"
        self.celeba_train_shuffled = self.dataset_dir + "/train_shuffled.flist"
        self.celeba_val_shuffled = self.dataset_dir + "/val_shuffled.flist"
        self.celeba_visual_test_shuffled = self.dataset_dir + "/visual_test_shuffled.flist"

        self.celeba_train_256 = self.dataset_dir + "/train_256"
        self.celeba_val_source_256 = self.dataset_dir + "/val_source_256"
        self.celeba_vis_test_source_256 = self.dataset_dir + "/visual_test_source_256"

        # masks data
        self.thick_mask = self.repoPath + "/configs/data_gen/random_thick_256.yaml"
        self.thin_mask = self.repoPath + "/configs/data_gen/random_thin_256.yaml"
        self.med_mask = self.repoPath + "/configs/data_gen/random_medium_256.yaml"

        self.val_thick = self.dataset_dir + "/val_256/random_thick_256/"
        self.val_thin = self.dataset_dir + "/val_256/random_thin_256/"
        self.val_med = self.dataset_dir + "/val_256/random_medium_256/"

        self.visual_test_thick = self.dataset_dir + "/visual_test_256/random_thick_256/"
        self.visual_test_thin = self.dataset_dir + "/visual_test_256/random_thin_256/"
        self.visual_test_med = self.dataset_dir + "/visual_test_256/random_medium_256/"

    def run(self):
        self.extractZip()
        self.reIndex()
        self.splitTrainAndVal()
        self.createConfig()
        self.genMasks()
    
    
    def extractZip(self):
        with zipfile.ZipFile(self.celeb_zip) as z:
            z.extractall(self.dataset_dir)
            print("Extracted all files in: " + self.celeb_zip + " to: " + self.dataset_dir)

    def reIndex(self):
        for n in range(0, 30000):
            currName = self.celeb_unzipped + '/' + str((n + 1)).zfill(5) + '.jpg'
            newName = self.celeb_unzipped + '/' + str(n) + '.jpg'
            if os.path.exists(currName):
                os.rename(currName, newName)
    
    def splitTrainAndVal(self):

        with open(self.train_shuffled, 'r') as f:

            lines = [line.rstrip() for line in f]
            random.shuffle(lines)

            with open(self.celeba_train_shuffled, 'w+') as f2:
                for item in lines[2001:]:
                    f2.write("%s\n" % item)
            
            with open(self.celeba_val_shuffled, 'w+') as f2:
                for item in lines[0:2000]:
                    f2.write("%s\n" % item)

        with open(self.val_shuffled, 'r') as f:
            lines = [line.rstrip() for line in f]
            with open(self.celeba_visual_test_shuffled, 'w+') as f2:
                for item in lines:
                    f2.write("%s\n" % item)

        self.createFolderIfNotExists(self.celeba_train_256)
        self.createFolderIfNotExists(self.celeba_val_source_256)
        self.createFolderIfNotExists(self.celeba_vis_test_source_256)
        
        self.moveCelebFilesToFolder(self.celeba_train_shuffled, self.celeba_train_256)
        self.moveCelebFilesToFolder(self.celeba_val_shuffled, self.celeba_val_source_256)
        self.moveCelebFilesToFolder(self.celeba_visual_test_shuffled, self.celeba_vis_test_source_256)
    
    def createFolderIfNotExists(self, folder):
        if not os.path.exists(folder):
            os.mkdir(folder)

    def moveCelebFilesToFolder(self, inFile, outFolder):
        with open(inFile, 'r') as f:
            for line in f:
                lineStripped = line.rstrip()
                os.rename(self.celeb_unzipped + "/" + lineStripped, outFolder + "/" + lineStripped)

    def createConfig(self):
        with open(self.celeb_yaml, 'w+') as f:
            f.write("# @package _group_\n")
            f.write("data_root_dir: %s/\n" % self.dataset_dir)
            f.write("out_root_dir: %s/experiments/\n" % self.repoPath)
            f.write("tb_dir: %s/tb_logs/\n" % self.repoPath)
            f.write("pretrained_models: %s/\n" % self.repoPath)

    def genMasks(self):
        gen_mask_dataset.main(Gen_Mask_Args(self.thick_mask, self.celeba_val_source_256, self.val_thick))
        gen_mask_dataset.main(Gen_Mask_Args(self.thin_mask, self.celeba_val_source_256, self.val_thin))
        gen_mask_dataset.main(Gen_Mask_Args(self.med_mask, self.celeba_val_source_256, self.val_med))

        gen_mask_dataset.main(Gen_Mask_Args(self.thick_mask, self.celeba_vis_test_source_256, self.visual_test_thick))
        gen_mask_dataset.main(Gen_Mask_Args(self.thin_mask, self.celeba_vis_test_source_256, self.visual_test_thin))
        gen_mask_dataset.main(Gen_Mask_Args(self.med_mask, self.celeba_vis_test_source_256, self.visual_test_med))
        
        #subprocess.call([sys.executable, self.repoPath + '/bin/gen_mask_dataset.py',  self.thick_mask, self.celeba_val_source_256, self.val_thick])
        #subprocess.call([sys.executable, self.repoPath + '/bin/gen_mask_dataset.py',  self.thin_mask, self.celeba_val_source_256, self.val_thin])
        #subprocess.call([sys.executable, self.repoPath + '/bin/gen_mask_dataset.py',  self.med_mask, self.celeba_val_source_256, self.val_med])

        #subprocess.call([sys.executable, self.repoPath + '/bin/gen_mask_dataset.py',  self.thick_mask, self.celeba_vis_test_source_256, self.visual_test_thick])
        #subprocess.call([sys.executable, self.repoPath + '/bin/gen_mask_dataset.py',  self.thin_mask, self.celeba_vis_test_source_256, self.visual_test_thin])
        #subprocess.call([sys.executable, self.repoPath + '/bin/gen_mask_dataset.py',  self.med_mask, self.celeba_vis_test_source_256, self.visual_test_med])


def main():
    parser = argparse.ArgumentParser(description='Prepares Celeb Dataset')

    # args
    parser.add_argument("-rp",
                        "--repoPath",
                        default=os.getcwd(),
                        help="""path to root 'lama' folder of repo""")
    
    parser.add_argument("-dp",
                        "--dataPath",
                        default=os.getcwd(),
                        help="""folder where the celeb zip is manged """)

    # parse
    args = parser.parse_args()

    # run
    Celeb_Prepare(args.repoPath, args.dataPath).run()


if __name__ == '__main__':
    main()