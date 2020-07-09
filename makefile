all: download env associate build

build:
	python3 solution/main.py --input-dir 'solution/data' --output-dir 'solution/output'

env:
	python3 -m pip install -r requirements.txt

associate:
	python2 solution/associate.py 'solution/data/rgb.txt' 'solution/data/depth.txt' > solution/data/rgbd.txt

evaluate:
	python2 solution/evaluate_rpe.py 'solution/data/freiburg2_desk-rgbdslam.txt' 'solution/data/estimate.txt'

evaluate_ate:
	python2 solution/evaluate_rpe.py 'solution/data/groundtruth.txt' 'solution/data/freiburg2_desk-rgbdslam.txt'

evaluate_rpe:
	python2 solution/evaluate_rpe.py 'solution/data/groundtruth.txt' 'solution/data/freiburg2_desk-rgbdslam.txt'

download:
	mkdir -p solution
	#axel -an 10 https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz
	#tar -xf data.tgz
	mv rgbd_dataset_freiburg2_desk solution/data
	mkdir tmp
	mv *.tgz tmp

clean:
	#rm -vf data.tgz
	#rm -rf solution/data
	rm -vf solution/data/rgbd.txt

