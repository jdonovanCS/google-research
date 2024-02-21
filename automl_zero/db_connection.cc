#include "db_connection.h"
// #include "CppSQLite.h"
#include "algorithm.h"
#include "instruction.h"
#include "definitions.h"


#include <stdlib.h>
#include <ctime>
#include <iostream>
// #include "stdafx.h"

// #include "mysql_connection.h"
// #include <cppconn/driver.h>
// #include <cppconn/exception.h>
// #include <cppconn/prepared_statement.h>

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

} //namespace

DB_Connection::DB_Connection(const char* db_loc)
    : db_loc_(db_loc) {
        try {
            CppSQLite3DB db;
            cout << "SQLite Version: " << db.SQLiteVersion() << endl;
            cout << endl << "Creating database for run." << endl;
            remove(db_loc);
            db.open(db_loc);

            db.execDML("Create table algs(id integer not null, evol_id integer not null, setup varchar(2000), learn varchar(2000), predict varchar(2000), PRIMARY KEY (id))");
            // cout << endl << "DML tests" << endl;
            // int nRows = db.execDML("insert into algs (evol_id, setup, learn, predict) values (0, null, null, null)");
            // cout << nRows << " rows inserted" << endl;
        }
        catch (CppSQLite3Exception& e) {
            std::cerr << e.errorCode() << ":" << e.errorMessage() << endl;
        }
    }   

void DB_Connection::Insert(int evol_id, std::vector<shared_ptr<const Algorithm>> algs){
    
    try{
        CppSQLite3DB db;

        cout << "SQLite Version: " << db.SQLiteVersion() << endl;

        db.open(db_loc_);
        ostringstream stmt;
        stmt << "insert into algs (evol_id, setup, predict, learn) values ";
        bool first = true;
        for (shared_ptr<const Algorithm>& next_algorithm : algs){
            ostringstream setup;
            ostringstream learn;
            ostringstream predict;
            for (const shared_ptr<const Instruction>& instruction : next_algorithm->setup_) {
                setup << instruction->ToString();
            }
            for (const shared_ptr<const Instruction>& instruction : next_algorithm->learn_) {
                learn << instruction->ToString();
            }
            for (const shared_ptr<const Instruction>& instruction : next_algorithm->predict_) {
                predict << instruction->ToString();
            }
            if (first == true){
                stmt << "(" << evol_id << ",\'"; 
            }
            else {
                stmt << ",(" << evol_id << ",\'";
            }
            stmt << setup.str(); 
            stmt << "\',\'"; 
            stmt << learn.str();
            stmt << "\',\'"; 
            stmt << predict.str(); 
            stmt << "\')";
            first = false;
        }
        // cout << "query: " << stmt.str() << endl;
        int nRows = db.execScalar(stmt.str().c_str());
        cout << nRows << " rows inserted" << endl;
        db.close();
    }
    catch (CppSQLite3Exception& e) {
        std::cerr << e.errorCode() << ":" << e.errorMessage() << endl;
    }
}

std::vector<shared_ptr<const Algorithm>> DB_Connection::Migrate(int evol_id, std::vector<shared_ptr<const Algorithm>> algs) {
    // this function should look at the evol_id and query the database for the entries that are not associated with it.
    // TODO (Jdonovan): figure out how to store the algorithms so that they can easily be translated to and from C++. Until
    // this is done, replacing them will not work correctly.
    // It should then return a random selection from those algorithms
    return algs;
}

}


