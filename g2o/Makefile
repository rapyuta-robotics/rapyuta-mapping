all: g2o

#	GIT_DIR: name of directory into which you want to clone to 
GIT_DIR = g2o_src
GIT_URL = git://github.com/RainerKuemmerle/g2o.git
GIT_REVISION = f5950e28f9cf56108968de77a3a7d9be641c4e74

include $(shell rospack find mk)/git_checkout.mk

g2o: download
	mkdir -p build && cd build && \
	cmake -DCMAKE_INSTALL_PREFIX=`pwd`/..  ../$(GIT_DIR) && \
  make install

clean: 
	rm -rf $(GIT_DIR) patched build include lib rospack_nosubdirs

