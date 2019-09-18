# After cloning the repo simply run this
mkdir -p ~/projects/RecoTour/datasets/Ponpare/data
cd ~/projects/RecoTour/datasets/Ponpare/data
kaggle competitions download -c coupon-purchase-prediction
mkdir zip_files
mv *.zip zip_files
cd zip_files
find . -name "*.zip" | while read filename; do unzip -o -d "`dirname "$filename"`" "$filename"; done;
cd ..
mv zip_files/*.csv .
mv zip_files/documentation .
cp ~/projects/RecoTour/Ponpare/prefecture.txt  .
cd ~/projects/RecoTour/Ponpare/