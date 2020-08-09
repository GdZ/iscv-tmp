all: download env associate build

build:
	python3 solution/main2.py --input-dir 'solution/data' --output-dir 'solution/output'

env:
	python3 -m pip install -r requirements.txt

associate:
	python2 solution/associate.py 'solution/data/rgb.txt' 'solution/data/depth.txt' > solution/data/rgbd.txt

evaluate: evaluate_rpe evaluate_ate

evaluate_ate:
	#python2 solution/evaluate_ate.py solution/data/groundtruth.txt solution/data/estimate_ab.txt  --plot solution/output/figure-ab-0.052-$(shell date +"%Y%m%d.%H%M%M").png --offset 0 --scale 1 --verbose
	python2 solution/evaluate_ate_v2.py solution/data/groundtruth.txt solution/data/estimate_c.txt  --plot solution/output/figure-c-$(shell date +"%Y%m%d.%H%M%M").png --offset 0 --scale 1 --verbose
	#python2 solution/evaluate_ate_v2.py solution/data/groundtruth.txt solution/data/estimate_d.txt  --plot solution/output/figure-d-$(shell date +"%Y%m%d.%H%M%M").png --offset 0 --scale 1 --verbose
	#python2 solution/evaluate_ate_v2.py solution/data/groundtruth.txt solution/data/estimate_e.txt  --plot solution/output/figure-e-$(shell date +"%Y%m%d.%H%M%M").png --offset 0 --scale 1 --verbose

evaluate_rpe:
	#python2 solution/evaluate_rpe.py 'solution/data/groundtruth.txt' 'solution/data/freiburg2_desk-rgbdslam.txt'
	python2 solution/evaluate_rpe.py solution/data/groundtruth.txt solution/data/estimate_ab_0.052.txt
	python2 solution/evaluate_rpe.py solution/data/groundtruth.txt solution/data/estimate_ab.txt
	python2 solution/evaluate_rpe.py solution/data/groundtruth.txt solution/data/estimate_c.txt

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

