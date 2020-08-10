all: evaluate

evaluate: evaluate_rpe evaluate_ate

evaluate_ate:
	#python2 evaluate_ate_v1.py groundtruth.txt freiburg2_desk-rgbdslam.txt  --plot figure-slam.png --offset 0 --scale 1 --verbose
	python2 evaluate_ate_v1.py groundtruth.txt kf_estimate_b.txt  --plot figure-b.png --offset 0 --scale 1 --verbose
	python2 evaluate_ate_v2.py groundtruth.txt kf_estimate_c.txt  --plot figure-c.png --offset 0 --scale 1 --verbose
	#python2 evaluate_ate_v2.py groundtruth.txt kf_estimate_d.txt  --plot figure-d.png --offset 0 --scale 1 --verbose
	#python2 evaluate_ate_v2.py groundtruth.txt kf_estimate_e.txt  --plot figure-e.png --offset 0 --scale 1 --verbose

evaluate_rpe:
	#python2 evaluate_rpe.py 'groundtruth.txt' 'freiburg2_desk-rgbdslam.txt'
	python2 evaluate_rpe.py groundtruth.txt kf_estimate_b.txt
	python2 evaluate_rpe.py groundtruth.txt kf_estimate_c.txt
	#python2 evaluate_rpe.py groundtruth.txt kf_estimate_d.txt
	#python2 evaluate_rpe.py groundtruth.txt kf_estimate_e.txt

