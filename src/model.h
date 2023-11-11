#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <bitset>
#include <array>


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#define K 20

#define IDX(i, j) ((i)*K + (j))

inline int modK(int a) {
  return (a > K-1) ? a-K : a;
}


namespace py = pybind11;
using std::vector;
using std::array;
using std::bitset;

typedef double Real;

class Model {
    public:
    bitset<K*K> spins;
    double J;
    double hmu;
    double energy;

    
    Model(double J, double hmu, vector<int> spins_input) : J(J), hmu(hmu) {
        for (int i = 0; i < K*K; i++) {
            spins[i] = (spins_input[i] > 0) ? true : false;
        }
        energy = calc_energy();
    }


    vector<int> get_spins() {
        vector<int> spins_output(K*K);
        for (int i = 0; i < K*K; i++) {
            spins_output[i] = this->spins[i] ? 1 : -1;
        }
        return spins_output;
    }

    double calc_energy() {
        double E = 0.0;
        for(int i = 0; i < K; i++) {
            for(int j = 0; j < K; j++) {
                int s = spins[IDX(i, j)];
                int sup = spins[IDX(modK(i+(K-1)), j)];
                int sdown = spins[IDX(modK(i+1), j)];
                int sleft = spins[IDX(i, modK(j+(K-1)))];
                int sright = spins[IDX(i, modK(j+1))];
                int sumspins = (sup + sdown + sleft + sright)*2 - 4;
                E += -((0.5*J)*sumspins + hmu)*(2*s-1);
            }
        }
        return E;
    }

    vector<Real> random_mc(vector<int>& x_rand, vector<int>& y_rand, vector<Real>& samp_rand, Real beta, int samp_freq) {
        int next_samp = 0;
        vector<Real> energies;
        for (int k = 0; k < x_rand.size(); k++) {
            if (k == next_samp) {
                energies.push_back(this->energy);
                next_samp += samp_freq;
            }
            int i = x_rand[k];
            int j = y_rand[k];
            int s = spins[IDX(i, j)];
            int sup = spins[IDX(modK(i+(K-1)), j)];
            int sdown = spins[IDX(modK(i+1), j)];
            int sleft = spins[IDX(i, modK(j+(K-1)))];
            int sright = spins[IDX(i, modK(j+1))];
            int sumspins = (sup + sdown + sleft + sright)*2 - 4;
            Real dE = 2*(2*s-1)*(J*sumspins + hmu);
            if(dE < 0) {
                spins[IDX(i, j)] = !s;
                this->energy += dE;
            } else {
                Real p = exp(-beta*dE);
                if (samp_rand[k] < p) {
                    spins[IDX(i, j)] = !s;
                    this->energy += dE;
                }
            }
        }
        return energies;
    }

    std::pair<Real, Real> random_mc_meanstd(vector<int>& x_rand, vector<int>& y_rand, vector<Real>& samp_rand, Real beta, int samp_freq) {
        int next_samp = 0;
        Real sumsamp = 0.0;
        Real sumsq = 0.0;
        for (int k = 0; k < x_rand.size(); k++) {
            if (k == next_samp) {
                Real samp = this->energy;
                sumsamp += samp;
                sumsq += samp*samp;
                next_samp += samp_freq;
            }
            int i = x_rand[k];
            int j = y_rand[k];
            int s = spins[IDX(i, j)];
            int sup = spins[IDX(modK(i+(K-1)), j)];
            int sdown = spins[IDX(modK(i+1), j)];
            int sleft = spins[IDX(i, modK(j+(K-1)))];
            int sright = spins[IDX(i, modK(j+1))];
            int sumspins = (sup + sdown + sleft + sright)*2 - 4;
            double dE = 2*(2*s-1)*(J*sumspins + hmu);
            if(dE < 0) {
                spins[IDX(i, j)] = !s;
                this->energy += dE;
            } else {
                double p = exp(-beta*dE);
                if (samp_rand[k] < p) {
                    spins[IDX(i, j)] = !s;
                    this->energy += dE;
                }
            }
        }
        int n = x_rand.size()/samp_freq;
        Real mean = sumsamp/((double) n);
        Real std = sqrt( sumsq/((double)(n-1)) - mean*mean*((double) n ) / ( (double) (n-1) ) );
        return std::make_pair(mean, std);
    }
};
