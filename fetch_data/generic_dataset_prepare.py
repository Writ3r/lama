import os
import zipfile
import random
import argparse
import math
from os import walk
from bin import gen_mask_dataset

class Gen_Mask_Args:

    def __init__(self, config, indir, outdir, ext):
        self.config = config
        self.indir = indir
        self.outdir = outdir
        self.n_jobs = 0
        self.ext = 'jpg'

class Generic_Prepare:

    def __init__(self, repoPath, dataPath, zipPath, imageSize):

        self.repoPath = repoPath
        self.dataPath = dataPath
        self.zipPath = zipPath
        self.imageSize = imageSize
        self.ext = None

        self.datasetName = os.path.splitext(os.path.basename(self.zipPath))[0]
        self.dataset_dir = self.dataPath + "/" + self.datasetName
        self.dataset_unzipped = self.dataset_dir + "/" + self.datasetName + "-unzipped"
        

    def run(self):
        self.extractZip()
        self.reIndex()
        self.splitTrainAndVal()
        self.createConfig()
        self.genMasks()
    
    def extractZip(self):
        self.createFolderIfNotExists(self.dataset_dir)
        self.createFolderIfNotExists(self.dataset_unzipped)
        with zipfile.ZipFile(self.zipPath) as z:
            z.extractall(self.dataset_unzipped)
            print("Extracted all files in: " + self.zipPath + " to: " + self.dataset_dir)

    def reIndex(self):
        index = 0
        for (dirpath, dirnames, filenames) in walk(self.dataset_unzipped):
            for filename in filenames:
                ext = os.path.splitext(filename)[1]
                self.ext = ext
                os.replace(dirpath + '/' + filename, self.dataset_unzipped + '/' + str(index) + ext)
                index = index + 1
        for (dirpath, dirnames, filenames) in walk(self.dataset_unzipped):
            for dirname in dirnames:
                os.rmdir(dirpath + '/' + dirname)
    
    
    def splitTrainAndVal(self):

        self.train_shuffled = self.dataset_dir + "/train_shuffled.flist"
        self.val_shuffled = self.dataset_dir + "/val_shuffled.flist"
        self.visual_test_shuffled = self.dataset_dir + "/visual_test_shuffled.flist"
        
        self.train = self.dataset_dir + "/train"
        self.val_source = self.dataset_dir + "/val_source_" + self.imageSize
        self.vis_test_source = self.dataset_dir + "/visual_test_source_" + self.imageSize

        # 85% train 10% val 5% visual val

        filesInDataset = []
        for (dirpath, dirnames, filenames) in walk(self.dataset_unzipped):
            filesInDataset.extend(filenames)

        random.shuffle(filesInDataset)
        length = len(filesInDataset)
        
        # TODO: am I missing one or two here?
        train = filesInDataset[0:math.floor(0.85 * length)]
        val = filesInDataset[len(train):len(train) + math.floor(0.1 * length)]
        visualTest = filesInDataset[(len(train) + len(val)):length]

        self.writeToFilelist(train, self.train_shuffled)
        self.writeToFilelist(val, self.val_shuffled)
        self.writeToFilelist(visualTest, self.visual_test_shuffled)

        self.createFolderIfNotExists(self.train)
        self.createFolderIfNotExists(self.val_source)
        self.createFolderIfNotExists(self.vis_test_source)
        
        self.moveFilesToFolder(self.train_shuffled, self.train)
        self.moveFilesToFolder(self.val_shuffled, self.val_source)
        self.moveFilesToFolder(self.visual_test_shuffled, self.vis_test_source)
    
    def writeToFilelist(self, datalist, filelist):
        with open(filelist, 'w+') as f:
            for item in datalist:
                f.write("%s\n" % item)
    
    def createFolderIfNotExists(self, folder):
        if not os.path.exists(folder):
            os.mkdir(folder)

    def moveFilesToFolder(self, inFile, outFolder):
        with open(inFile, 'r') as f:
            for line in f:
                lineStripped = line.rstrip()
                os.replace(self.dataset_unzipped + "/" + lineStripped, outFolder + "/" + lineStripped)

    def createConfig(self):
        self.configYaml = self.repoPath + "/configs/training/location/" + self.datasetName + ".yaml"
        with open(self.configYaml, 'w+') as f:
            f.write("# @package _group_\n")
            f.write("data_root_dir: %s/\n" % self.dataset_dir)
            f.write("out_root_dir: %s/experiments/\n" % self.repoPath)
            f.write("tb_dir: %s/tb_logs/\n" % self.repoPath)
            f.write("pretrained_models: %s/\n" % self.repoPath)

    def genMasks(self):
        
        self.thick_mask = self.repoPath + "/configs/data_gen/random_thick_" + self.imageSize + ".yaml"
        self.thin_mask = self.repoPath + "/configs/data_gen/random_thin_" + self.imageSize + ".yaml"
        self.med_mask = self.repoPath + "/configs/data_gen/random_medium_" + self.imageSize + ".yaml"

        self.val_thick = self.dataset_dir + "/val/random_thick_" + self.imageSize + "/"
        self.val_thin = self.dataset_dir + "/val/random_thin_" + self.imageSize + "/"
        self.val_med = self.dataset_dir + "/val/random_medium_" + self.imageSize + "/"

        self.visual_test_thick = self.dataset_dir + "/visual_test/random_thick_" + self.imageSize + "/"
        self.visual_test_thin = self.dataset_dir + "/visual_test/random_thin_"+ self.imageSize + "/"
        self.visual_test_med = self.dataset_dir + "/visual_test/random_medium_" + self.imageSize + "/"

        gen_mask_dataset.main(Gen_Mask_Args(self.thick_mask, self.val_source, self.val_thick, self.ext))
        gen_mask_dataset.main(Gen_Mask_Args(self.thin_mask, self.val_source, self.val_thin, self.ext))
        gen_mask_dataset.main(Gen_Mask_Args(self.med_mask, self.val_source, self.val_med, self.ext))

        gen_mask_dataset.main(Gen_Mask_Args(self.thick_mask, self.vis_test_source, self.visual_test_thick, self.ext))
        gen_mask_dataset.main(Gen_Mask_Args(self.thin_mask, self.vis_test_source, self.visual_test_thin, self.ext))
        gen_mask_dataset.main(Gen_Mask_Args(self.med_mask, self.vis_test_source, self.visual_test_med, self.ext))


def main():
    parser = argparse.ArgumentParser(description='Prepares Dataset')

    # args
    parser.add_argument("dataPath",
                        help="""folder where the data is manged""")
    
    parser.add_argument("zipPath",
                        help="""folder where the zip exists""")
    
    parser.add_argument("imageSize",
                        help="""size of images running against. Ex 256""")
    
    parser.add_argument("-rp",
                        "--repoPath",
                        default=os.getcwd(),
                        help="""path to root 'lama' folder of repo""")

    # parse
    args = parser.parse_args()

    # run
    Generic_Prepare(args.repoPath, args.dataPath, args.zipPath, args.imageSize).run()


if __name__ == '__main__':
    main()