# Reference: https://universe.roboflow.com/cv-hub/yolox-kbgp1
# To be executed under data directory

echo "Downloading and Preparing COCO dataset..."
mkdir COCO
cd COCO
mkdir annotations

wget https://storage.googleapis.com/peekingduck/data/yolox.v1i.coco.zip

unzip yolox.v1i.coco.zip
rm yolox.v1i.coco.zip

cp train/_annotations.coco.json annotations/instances_train2017.json
cp valid/_annotations.coco.json annotations/instances_val2017.json

mv train train2017
mv valid valid2017

cd ..