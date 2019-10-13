#pragma once
#include<vector>
#include"coord.h"
#include"param.h"

struct Path : std::vector<COORD> {
    int nearest(const COORD pos) const;
    double mindist(COORD pos);
    int forward(int i, double len) const;
	double getYaw(int i) const;
	Path interpolate(double max_len = 10000.0) const;
	void cutjoin(const Path& p);
	double getlength();
	double getCurDir(int pos_along = 0);

	COORD GetCrossDir(int, bool);

	void text();

	Path copy_without_travelled_points(double dist_to_remove);

	void copy_to(Path& des){
//		des.resize(size());
		des.assign(begin(),end());
	}
};
