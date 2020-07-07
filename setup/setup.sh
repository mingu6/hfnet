read -p "Path of the directory where datasets are stored and read: " dir
echo "DATA_PATH = '$dir'" >> ./hfnet/settings.py

read -p "Path of the directory where experiments data (logs, checkpoints, configs) are written: " dir
echo "EXPER_PATH = '$dir'" >> ./hfnet/settings.py

read -p "Path of the directory containing the original RobotCar traverse data: " dir
echo "RAW_PATH = '$dir'" >> ./hfnet/settings.py

read -p "Insert RobotCar dataset website username: " name
read -p "Insert website password: " pword

echo pwd
python ./thirdparty/RobotCarDataset-Scraper-master/scrape_mrgdatashare.py --downloads_dir $dir --datasets_file ./thirdparty/RobotCarDataset-Scraper-master/vloc_challenges.csv --username $name --password $pword
