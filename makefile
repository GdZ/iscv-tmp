download:
	mkdir -p solution
	axel -an 10 https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz
	tar -xf rgbd_dataset_freiburg2_desk.tgz
	mv rgbd_dataset_freiburg2_desk solution/data
	mkdir tmp
	mv *.tgz tmp

clean:
	rm -vf rgbd_dataset_freiburg2_desk.tgz

