all: openni2

GIT_DIR = openni2-git
GIT_URL = https://github.com/OpenNI/OpenNI2.git 
GIT_PATCH = disable_softfp_arm.patch
GIT_REVISION = develop

include $(shell rospack find mk)/git_checkout.mk

openni2: $(SVN_DIR) patched
	cd $(GIT_DIR) && make -j4
	mkdir -p lib include
	cp $(GIT_DIR)/Bin/*-Release/libOpenNI2.so lib/
	cp -r $(GIT_DIR)/Bin/*-Release/OpenNI2 lib/
	cp -r $(GIT_DIR)/Include/* include/
	cp -r $(GIT_DIR)/Config/OpenNI.ini lib/
	cp PS1080.ini lib/OpenNI2/Drivers/PS1080.ini
	touch openni2

clean:
	cd $(GIT_DIR) && make clean
	rm -rf openni2 patched

wipe: clean
	rm -rf $(GIT_DIR) include lib include bin rospack_nosubdirs
