#ifndef AUTOML_ZERO_DB_CONNECTION_H_
#define AUTOML_ZERO_DB_CONNECTION_H_

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
#include "CppSQLite3.h"
#include "algorithm.h"

namespace automl_zero {

class DB_Connection{
 public:
    DB_Connection(const char* db_loc);
  
  void Delete(int evol_id);
  void Insert(int evol_id, std::vector<std::shared_ptr<const Algorithm>> algs);
  std::vector<std::shared_ptr<const Algorithm>> Migrate(int evol_id, std::vector<std::shared_ptr<const Algorithm>> algs);
//   vector<Algorithm> migrate(vector<Algorithm> algs);

  const char* db_loc_;
  
};

}
#endif