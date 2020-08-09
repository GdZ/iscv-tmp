all: download env associate build

build:
	python3 solution/main2.py --input-dir 'solution/data' --output-dir 'solution/output'

env:
	python3 -m pip install -r requirements.txt

associate:
	python2 solution/associate.py 'solution/data/rgb.txt' 'solution/data/depth.txt' > solution/data/rgbd.txt

evaluate: evaluate_rpe evaluate_ate

evaluate_ate:
	#python2 solution/evaluate_ate_v2.py solution/data/groundtruth.txt solution/data/freiburg2_desk-rgbdslam.txt  --plot solution/output/figure-slam.png --offset 0 --scale 1 --verbose
	python2 solution/evaluate_ate_v2.py solution/data/groundtruth.txt solution/output/kf_estimate_b.txt  --plot solution/output/figure-b.png --offset 0 --scale 1 --verbose
	python2 solution/evaluate_ate_v2.py solution/data/groundtruth.txt solution/output/kf_estimate_c.txt  --plot solution/output/build/figure-c.png --offset 0 --scale 1 --verbose
	#python2 solution/evaluate_ate_v2.py solution/data/groundtruth.txt solution/output/kf_estimate_d.txt  --plot solution/output/figure-d.png --offset 0 --scale 1 --verbose
	#python2 solution/evaluate_ate_v2.py solution/data/groundtruth.txt solution/output/kf_estimate_e.txt  --plot solution/output/figure-e.png --offset 0 --scale 1 --verbose

evaluate_rpe:
	#python2 solution/evaluate_rpe.py 'solution/data/groundtruth.txt' 'solution/data/freiburg2_desk-rgbdslam.txt'
	python2 solution/evaluate_rpe.py solution/data/groundtruth.txt solution/output/kf_estimate_b.txt
	python2 solution/evaluate_rpe.py solution/data/groundtruth.txt solution/output/kf_estimate_c.txt
	#python2 solution/evaluate_rpe.py solution/data/groundtruth.txt solution/output/kf_estimate_d.txt
	#python2 solution/evaluate_rpe.py solution/data/groundtruth.txt solution/output/kf_estimate_e.txt

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

