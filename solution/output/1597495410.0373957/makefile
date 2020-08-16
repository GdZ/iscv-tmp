all: evaluate

evaluate: evaluate_rpe evaluate_ate

evaluate_ate:
	#python2 evaluate_ate_v1.py groundtruth.txt freiburg2_desk-rgbdslam.txt  --plot figure-slam.png --offset 0 --scale 1 --verbose
	python2 evaluate_ate_v1.py groundtruth.txt pose_estimate_1-5.txt  --plot pose-figure-1.png
	python2 evaluate_ate_v1.py groundtruth.txt pose_estimate_2-5.txt  --plot pose-figure-2.png
	python2 evaluate_ate_v1.py groundtruth.txt kf_estimate_1-5.txt  --plot kf-figure-1.png
	python2 evaluate_ate_v1.py groundtruth.txt kf_estimate_2-5.txt  --plot kf-figure-2.png
	python2 evaluate_ate_v1.py groundtruth.txt kf_estimate_3-5.txt  --plot kf-figure-3.png
	python2 evaluate_ate_v2.py groundtruth.txt kf_estimate_d.txt  --plot kf-figure-d-pose-graph.png
	python2 evaluate_ate_v2.py groundtruth.txt kf_estimate_e.txt  --plot kf-figure-e-loop-closure.png

evaluate_rpe:
	#python2 evaluate_rpe.py 'groundtruth.txt' 'freiburg2_desk-rgbdslam.txt'
	python2 evaluate_rpe.py groundtruth.txt pose_estimate_1-5.txt
	python2 evaluate_rpe.py groundtruth.txt pose_estimate_2-5.txt
	python2 evaluate_rpe.py groundtruth.txt kf_estimate_1-5.txt
	python2 evaluate_rpe.py groundtruth.txt kf_estimate_2-5.txt
	python2 evaluate_rpe.py groundtruth.txt kf_estimate_3-5.txt
	python2 evaluate_rpe.py groundtruth.txt kf_estimate_d.txt
	python2 evaluate_rpe.py groundtruth.txt kf_estimate_e.txt
