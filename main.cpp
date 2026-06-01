#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <string>
#include <algorithm>
#include <vector>
#include <omp.h>

using namespace std;

const int NUM_RUNS = 5;  // Average over this many runs

void naive_matmul(double *A, double *B, double *C, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i*p + j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i*p + j] += A[i*n + k] * B[k*p + j];
            }
        }
    }
}

void blocked_matmul(double *A, double *B, double *C, int m, int n, int p, int bs) {
    for (int i = 0; i < m*p; i++) C[i] = 0.0;
    for (int ii = 0; ii < m; ii += bs) {
        for (int jj = 0; jj < p; jj += bs) {
            for (int kk = 0; kk < n; kk += bs) {
                int ie = min(ii + bs, m);
                int je = min(jj + bs, p);
                int ke = min(kk + bs, n);
                for (int i = ii; i < ie; i++) {
                    for (int j = jj; j < je; j++) {
                        for (int k = kk; k < ke; k++) {
                            C[i*p + j] += A[i*n + k] * B[k*p + j];
                        }
                    }
                }
            }
        }
    }
}

void parallel_matmul(double *A, double *B, double *C, int m, int n, int p) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i*p + j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i*p + j] += A[i*n + k] * B[k*p + j];
            }
        }
    }
}

double* read_matrix(const string& filename, int& rows, int& cols) {
    ifstream file(filename);
    if (!file) { cerr << "Cannot open " << filename << endl; exit(1); }
    file >> rows >> cols;
    double* M = (double*)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows * cols; i++) file >> M[i];
    return M;
}

bool validate(double *C, double *Cref, int m, int p, double eps = 1e-2) {
    for (int i = 0; i < m * p; i++)
        if (fabs(C[i] - Cref[i]) > eps) return false;
    return true;
}

double avg(const vector<double>& v) {
    double s = 0;
    for (double t : v) s += t;
    return s / v.size();
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <case 0-9> [mode]" << endl;
        cerr << "  mode: default | blocks | threads" << endl;
        return 1;
    }

    int cn = atoi(argv[1]);
    string mode = (argc > 2) ? argv[2] : "default";

    string pA = "data/" + to_string(cn) + "/input0.raw";
    string pB = "data/" + to_string(cn) + "/input1.raw";
    string pC = "data/" + to_string(cn) + "/output.raw";

    int m, nA, nB, p, mo, po;
    double *A = read_matrix(pA, m, nA);
    double *B = read_matrix(pB, nB, p);
    double *Cref = read_matrix(pC, mo, po);
    int n = nA;

    if (nA != nB) { cerr << "Dimension mismatch" << endl; return 1; }

    double *Cn = (double*)malloc(m * p * sizeof(double));
    double *Cb = (double*)malloc(m * p * sizeof(double));
    double *Cp = (double*)malloc(m * p * sizeof(double));

    if (mode == "blocks") {
        cout << "=== Block Size Sweep, Case " << cn
             << ": A(" << m << "x" << n << ") * B(" << n << "x" << p
             << "), avg of " << NUM_RUNS << " runs ===" << endl;
        vector<double> tn;
        for (int r = 0; r < NUM_RUNS; r++) {
            double t0 = omp_get_wtime();
            naive_matmul(A, B, Cn, m, n, p);
            tn.push_back(omp_get_wtime() - t0);
        }
        double na = avg(tn);
        cout << "Naive baseline: " << na << " s" << endl;
        cout << "Block Size | Time (s) | Speedup | Valid" << endl;
        int sizes[] = {16, 32, 64, 128};
        for (int bs : sizes) {
            vector<double> tb;
            bool ok = true;
            for (int r = 0; r < NUM_RUNS; r++) {
                double t0 = omp_get_wtime();
                blocked_matmul(A, B, Cb, m, n, p, bs);
                tb.push_back(omp_get_wtime() - t0);
                if (r == 0) ok = validate(Cb, Cref, m, p);
            }
            double a = avg(tb);
            cout << "    " << bs << "     | " << a << " | "
                 << na/a << "x | " << (ok ? "OK" : "FAIL") << endl;
        }
    }
    else if (mode == "threads") {
        cout << "=== Thread Count Sweep, Case " << cn
             << ": A(" << m << "x" << n << ") * B(" << n << "x" << p
             << "), avg of " << NUM_RUNS << " runs ===" << endl;
        vector<double> tn;
        for (int r = 0; r < NUM_RUNS; r++) {
            double t0 = omp_get_wtime();
            naive_matmul(A, B, Cn, m, n, p);
            tn.push_back(omp_get_wtime() - t0);
        }
        double na = avg(tn);
        cout << "Naive baseline: " << na << " s" << endl;
        cout << "Threads | Time (s) | Speedup | Valid" << endl;
        int threads[] = {1, 2, 4, 8};
        for (int tc : threads) {
            omp_set_num_threads(tc);
            vector<double> tp;
            bool ok = true;
            for (int r = 0; r < NUM_RUNS; r++) {
                double t0 = omp_get_wtime();
                parallel_matmul(A, B, Cp, m, n, p);
                tp.push_back(omp_get_wtime() - t0);
                if (r == 0) ok = validate(Cp, Cref, m, p);
            }
            double a = avg(tp);
            cout << "   " << tc << "    | " << a << " | "
                 << na/a << "x | " << (ok ? "OK" : "FAIL") << endl;
        }
    }
    else {
        // Default mode: all 3 implementations with averaging
        omp_set_num_threads(4);
        vector<double> tn, tb, tp;
        bool ok_n = true, ok_b = true, ok_p = true;

        for (int r = 0; r < NUM_RUNS; r++) {
            double t0;
            t0 = omp_get_wtime();
            naive_matmul(A, B, Cn, m, n, p);
            tn.push_back(omp_get_wtime() - t0);
            if (r == 0) ok_n = validate(Cn, Cref, m, p);

            t0 = omp_get_wtime();
            blocked_matmul(A, B, Cb, m, n, p, 64);
            tb.push_back(omp_get_wtime() - t0);
            if (r == 0) ok_b = validate(Cb, Cref, m, p);

            t0 = omp_get_wtime();
            parallel_matmul(A, B, Cp, m, n, p);
            tp.push_back(omp_get_wtime() - t0);
            if (r == 0) ok_p = validate(Cp, Cref, m, p);
        }

        double an = avg(tn), ab = avg(tb), ap = avg(tp);

        cout << "Case " << cn << ": A(" << m << "x" << n << ") * B("
             << n << "x" << p << "), avg of " << NUM_RUNS << " runs" << endl;
        cout << "  Naive:    " << an << " s (" << (ok_n ? "OK" : "FAIL") << ")" << endl;
        cout << "  Blocked:  " << ab << " s (" << (ok_b ? "OK" : "FAIL")
             << ") speedup: " << an/ab << "x" << endl;
        cout << "  Parallel: " << ap << " s (" << (ok_p ? "OK" : "FAIL")
             << ") speedup: " << an/ap << "x" << endl;

        string rp = "data/" + to_string(cn) + "/result.raw";
        ofstream out(rp);
        out << m << " " << p << endl;
        for (int i = 0; i < m * p; i++) {
            out << Cp[i];
            if ((i + 1) % p == 0) out << endl;
            else out << " ";
        }
    }

    free(A); free(B); free(Cref);
    free(Cn); free(Cb); free(Cp);
    return 0;
}