CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -fPIC -I/opt/homebrew/include
LDFLAGS = -L/opt/homebrew/lib
LIBS = -lsymengine -lflint -lgmp -lmpfr

TARGET = main
SRC = $(wildcard src/*.cpp)
OBJ = $(SRC:.cpp=.o)

# Test-specific variables
TEST_SRC = test/test_symbolic_utils.cpp src/symbolic_utils.cpp
TEST_TARGET = test_symbolic_utils
RSCRIPT = /opt/homebrew/bin/Rscript
TEST_CXXFLAGS = $(CXXFLAGS) -w $(shell $(RSCRIPT) -e "Rcpp:::CxxFlags()")
TEST_LDFLAGS = $(LDFLAGS) $(shell $(RSCRIPT) -e "Rcpp:::LdFlags()")
TEST_LIBS = $(LIBS) -lR

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OBJ) -o $(TARGET) $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): $(TEST_SRC)
	$(CXX) $(TEST_CXXFLAGS) $^ -o $@ $(TEST_LDFLAGS) $(TEST_LIBS)

clean:
	rm -f $(OBJ) $(TARGET) $(TEST_TARGET)

rebuild: clean all

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean rebuild run test
