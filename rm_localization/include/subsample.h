/*
 * subsample.h
 *
 *  Created on: Sep 26, 2013
 *      Author: vsu
 */

#ifndef SUBSAMPLE_H_
#define SUBSAMPLE_H_

#include <tbb/blocked_range.h>

struct subsample {
	const uint8_t * prev_intencity;
	const uint16_t * prev_depth;
	int cols;
	int rows;
	uint8_t * current_intencity;
	uint16_t * current_depth;

	subsample(const uint8_t * prev_intencity, const uint16_t * prev_depth,
			int cols, int rows, uint8_t * current_intencity,
			uint16_t * current_depth) :
			prev_intencity(prev_intencity), prev_depth(prev_depth), cols(cols), rows(
					rows), current_intencity(current_intencity), current_depth(
					current_depth) {
	}

	void operator()(const tbb::blocked_range<int>& range) const {
		for (int i = range.begin(); i != range.end(); i++) {
			int u = i % cols;
			int v = i / cols;

			int p1 = 4 * v * cols + 2 * u;
			int p2 = p1 + 2 * cols;
			int p3 = p1 + 1;
			int p4 = p2 + 1;

			int val = prev_intencity[p1];
			val += prev_intencity[p2];
			val += prev_intencity[p3];
			val += prev_intencity[p4];

			current_intencity[i] = val / 4;

			uint16_t values[4];
			values[0] = prev_depth[p1];
			values[1] = prev_depth[p2];
			values[2] = prev_depth[p3];
			values[3] = prev_depth[p4];
			std::sort(values, values + 4);

			current_depth[i] = values[2];

		}

	}

};


#endif /* SUBSAMPLE_H_ */
