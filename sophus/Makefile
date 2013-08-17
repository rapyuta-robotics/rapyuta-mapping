all: sophus

GIT_DIR = sophus_src
GIT_URL=git://github.com/strasdat/Sophus.git

include $(shell rospack find mk)/git_checkout.mk

sophus: download
	mkdir -p build && cd build && \
	cmake -DCMAKE_INSTALL_PREFIX=`pwd`/..  ../$(GIT_DIR) && \
	make install

clean: 
	rm -rf $(GIT_DIR) patched build include lib rospack_nosubdirs

