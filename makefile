all: download env associate build

build:
	python solution/main.py --input-dir 'solution/data' --output-dir 'solution/output'

env:
	python3 -m pip install -r requirements.txt

associate:
	solution/associate.py solution/data/rgb.txt solution/data/depth.txt > solution/data/rgbd.txt

download:
	mkdir -p solution
	axel -an 10 https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz
	tar -xf rgbd_dataset_freiburg2_desk.tgz
	mv rgbd_dataset_freiburg2_desk solution/data
	mkdir tmp
	mv *.tgz tmp

docker:
	docker run -d -v $(PWD):/apps --name iscv -it zgd521/vnc:latest /bin/bash

clean:
	#rm -vf rgbd_dataset_freiburg2_desk.tgz
	rm -rf solution/data
	rm -vf solution/data/rgbd.txt

