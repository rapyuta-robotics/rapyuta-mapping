/*
 * convert.h
 *
 *  Created on: Sep 26, 2013
 *      Author: vsu
 */

#ifndef CONVERT_H_
#define CONVERT_H_

#include <tbb/blocked_range.h>

struct convert {
	const uint8_t * yuv;
	uint8_t * intencity;

	convert(const uint8_t * yuv, uint8_t * intencity) :
			yuv(yuv), intencity(intencity) {
	}

	void operator()(const tbb::blocked_range<int>& range) const {
		for (int i = range.begin(); i != range.end(); i++) {
			intencity[i] = yuv[2 * i + 1];
		}

	}

};


#endif /* CONVERT_H_ */
