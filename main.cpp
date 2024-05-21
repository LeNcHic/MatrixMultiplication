#include <iostream>
#include <vector>
#include <arm_neon.h>
#include <chrono>
#include <cstdlib>
#include <ctime>
using namespace std;
using namespace std::chrono;


vector<vector<int>> multiplyMatrixSimd(const vector<vector<int>>& mat1, const vector<vector<int>>& mat2) {
  int rows1 = mat1.size();
  int cols1 = mat1[0].size();
  int rows2 = mat2.size();
  int cols2 = mat2[0].size();

  vector<vector<int>> result(rows1, vector<int>(cols2, 0));

  for (int i = 0; i < rows1; ++i) {
    for (int j = 0; j < cols2; ++j) {
      int32x4_t sum_vec = vmovq_n_s32(0);

      int k;

      for (k = 0; k <= cols1 - 4; k += 4) {
        int32x4_t mat1_vec = vld1q_s32(&mat1[i][k]);
        int32x4_t mat2_vec = {mat2[k][j], mat2[k + 1][j], mat2[k + 2][j], mat2[k + 3][j]};
        int32x4_t prod_vec = vmulq_s32(mat1_vec, mat2_vec);
        sum_vec = vaddq_s32(sum_vec, prod_vec);
      }

      int32x2_t sum_pair = vadd_s32(vget_low_s32(sum_vec), vget_high_s32(sum_vec));
      int32_t final_sum = vget_lane_s32(vpadd_s32(sum_pair, sum_pair), 0);

      for (; k < cols1; ++k) {
        final_sum += mat1[i][k] * mat2[k][j];
      }

      result[i][j] = final_sum;
    }
  }

  return result;
}

vector<vector<int>> generateRandomMatrix(int rows, int cols) {
  vector<vector<int>> matrix(rows, vector<int>(cols));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix[i][j] = rand() % 100;
    }
  }
  return matrix;
}

vector<vector<int>> multiplyMatrix(const vector<vector<int>>& mat1, const vector<vector<int>>& mat2) {
  int rows1 = mat1.size();
  int cols1 = mat1[0].size();
  int cols2 = mat2[0].size();

  vector<vector<int>> result(rows1, vector<int>(cols2, 0));

  for (int i = 0; i < rows1; ++i) {
    for (int j = 0; j < cols2; ++j) {
      for (int k = 0; k < cols1; ++k) {
        result[i][j] += mat1[i][k] * mat2[k][j];
      }
    }
  }

  return result;
}

vector<vector<int>> multiplyMatrixVinograd(const vector<vector<int>>& mat1, const vector<vector<int>>& mat2) {
  int rows1 = mat1.size();
  int cols1 = mat1[0].size();
  int rows2 = mat2.size();
  int cols2 = mat2[0].size();

  vector<vector<int>> result(rows1, vector<int>(cols2, 0));

  vector<int> row_factor(rows1, 0);
  vector<int> col_factor(cols2, 0);

  for (int i = 0; i < rows1; ++i) {
    for (int j = 0; j < cols1 / 2; ++j) {
      row_factor[i] += mat1[i][2 * j] * mat1[i][2 * j + 1];
    }
  }

  for (int i = 0; i < cols2; ++i) {
    for (int j = 0; j < rows2 / 2; ++j) {
      col_factor[i] += mat2[2 * j][i] * mat2[2 * j + 1][i];
    }
  }

  for (int i = 0; i < rows1; ++i) {
    for (int j = 0; j < cols2; ++j) {
      int sum = -row_factor[i] - col_factor[j];
      for (int k = 0; k < cols1 / 2; ++k) {
        sum += (mat1[i][2 * k] + mat2[2 * k + 1][j]) * (mat1[i][2 * k + 1] + mat2[2 * k][j]);
      }
      result[i][j] = sum;
    }
  }

  if (cols1 % 2 != 0) {
    for (int i = 0; i < rows1; ++i) {
      for (int j = 0; j < cols2; ++j) {
        result[i][j] += mat1[i][cols1 - 1] * mat2[cols1 - 1][j];
      }
    }
  }

  return result;
}

void printMatrix(const vector<vector<int>>& matrix) {
  for (const auto& row : matrix) {
    for (int elem : row) {
      cout << elem << " ";
    }
    cout << endl;
  }
}

int main() {
  srand(time(nullptr));

  int rows1 = 3;
  int cols1 = 2;
  int cols2 = 3;

  vector<double> res_default, res_simd, res_vinograd;
  for (int cols: {30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500}) {
    rows1 = cols;
    cols1 = cols;
    cols2 = cols;
//    std::cout << "TESTING mat1: " << rows1 << "x" << cols1 << " mat2 " << cols1 << "x"
//              << cols2 << std::endl;
    vector<vector<int>> matrix1 = generateRandomMatrix(rows1, cols1);
    vector<vector<int>> matrix2 = generateRandomMatrix(cols1, cols2);

//        cout << "Первая матрица:" << endl;
//        printMatrix(matrix1);
//
//        cout << "Вторая матрица:" << endl;
//        printMatrix(matrix2);

    auto start1 = high_resolution_clock::now();

    vector<vector<int>> result1 = multiplyMatrix(matrix1, matrix2);

    auto stop1 = high_resolution_clock::now();
    auto duration1 = duration_cast<microseconds>(stop1 - start1);

//    cout << "Результат умножения без ускрорений:" << endl;
//    printMatrix(result1);

    res_default.push_back(duration1.count());

    auto start = high_resolution_clock::now();

    vector<vector<int>> result = multiplyMatrixSimd(matrix1, matrix2);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

//        cout << "Результат умножения с помощью Simd:" << endl;
//        printMatrix(result);

    res_simd.push_back(duration.count());

    auto start2 = high_resolution_clock::now();

    vector<vector<int>> result2 = multiplyMatrixVinograd(matrix1, matrix2);

    auto stop2 = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop2 - start2);

//    cout << "Результат умножения с помощью Винограда:" << endl;
//    printMatrix(result2);

    res_vinograd.push_back(duration2.count());
  }

  for (auto i: res_default) {
    cout << i << ", ";
  }
  cout << endl;

  for (auto i: res_simd) {
    cout << i << ", ";
  }
  cout << endl;

  for (auto i: res_vinograd) {
    cout << i << ", ";
  }
  cout << endl;

  return 0;
}
