#ifndef AUTOML_ZERO_OPSRESULTS_H_
#define AUTOML_ZERO_OPSRESULTS_H_

#include <array>
#include <memory>
#include <string>
#include <vector>
#include <utility>

#include <stdlib.h>
#include <iostream>

// #include "mysql_connection.h"
// #include <cppconn/driver.h>
// #include <cppconn/exception.h>
// #include <cppconn/prepared_statement.h>

#include "definitions.h"
#include "algorithm.h"

namespace automl_zero {


class OpsResults{
  public:

    OpsResults();
    OpsResults(int o1, int o2, int o3, int o4, int o5, int o6);

  int total_ops;
  int arith_ops;
  int trig_ops;
  int precalc_ops;
  int linearalg_ops;
  int probstat_ops;

};

}
#endif