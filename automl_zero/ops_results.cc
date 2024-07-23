#include "ops_results.h"
#include "algorithm.h"
#include "instruction.h"
#include "definitions.h"

#include <stdlib.h>
#include <ctime>
#include <iostream>

namespace automl_zero {

namespace {

using ::std::abs;
using ::std::cout;
using ::std::endl;
using ::std::fixed;
using ::std::pair;
using ::std::vector;
using ::std::string;
using ::std::shared_ptr;
using ::std::ostringstream;
using ::std::ostream;
using ::std::make_shared;

} //namespace

OpsResults::OpsResults():
    total_ops(0), 
    arith_ops(0), 
    trig_ops(0), 
    precalc_ops(0), 
    linearalg_ops(0), 
    probstat_ops(0) {}

OpsResults::OpsResults(int o1, int o2, int o3, int o4, int o5, int o6): 
    total_ops(o1), 
    arith_ops(o2), 
    trig_ops(o3), 
    precalc_ops(o4), 
    linearalg_ops(o5), 
    probstat_ops(o6) {}

}